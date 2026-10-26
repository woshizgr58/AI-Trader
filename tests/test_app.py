"""Offline integration fixtures persist only in temporary pytest directories."""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import math
from types import SimpleNamespace

import pandas as pd
import pytest

from paper_trader import app, backtest, calendar, market, models
from paper_trader.config import Config
from paper_trader.store import Store


NOW = datetime(2024, 2, 1, 14, 35, tzinfo=timezone.utc)


class FixtureDataClient:
    """Mock only provider IO, so ingestion, validation, storage and sizing run."""
    def __init__(self, frame=None):
        self.historical_requests = []
        self.quote_requests = []
        self.missing_latest_bar = False
        self.frame = frame

    def get_stock_bars(self, request):
        self.historical_requests.append(request)
        first = pd.Timestamp(request.start).tz_localize("UTC").tz_convert("America/New_York").date().isoformat()
        last = pd.Timestamp(request.end).tz_localize("UTC").tz_convert("America/New_York").date().isoformat()
        if self.frame is not None:
            rows = self.frame.loc[self.frame.session.between(first, last)].drop(columns="session")
            return SimpleNamespace(df=rows.set_index(["symbol", "timestamp"]))
        rows = []
        for label in calendar.sessions(first, last):
            for symbol in request.symbol_or_symbols:
                if self.missing_latest_bar and label == last:
                    continue
                rows.append({"symbol": symbol, "timestamp": pd.Timestamp(label, tz="America/New_York").tz_convert("UTC"),
                             "open": 100.0, "high": 102.0, "low": 98.0, "close": 101.0, "volume": 10000.0})
        return SimpleNamespace(df=pd.DataFrame(rows).set_index(["symbol", "timestamp"]))

    def get_stock_latest_trade(self, request):
        self.quote_requests.append(request)
        return {symbol: SimpleNamespace(price=100.0, timestamp=NOW) for symbol in request.symbol_or_symbols}


class ReadOnlyFixtureBroker:
    def __init__(self):
        self.reads = 0
        self.posts = []

    def snapshot(self):
        self.reads += 1
        return {"account_id": "fixture-paper-account", "paper": True, "market_open": True,
                "timestamp": NOW.isoformat(), "broker_timestamp": NOW.isoformat(),
                "cash": 10000.0, "buying_power": 10000.0,
                "equity": 10000.0, "positions": {}, "open_orders": [], "recent_orders": []}

    def get_order(self, _):
        return None

    def submit_order(self, order):
        self.posts.append(deepcopy(order))
        raise AssertionError("Dry-run integration must never submit any broker order")


@pytest.fixture
def integration(tmp_path, monkeypatch):
    config = Config(data_dir=tmp_path, symbols=("SPY", "QQQ"), history_start="2024-01-02",
                    api_key="fixture-key", secret_key="fixture-secret", cash_buffer=0.01)
    store = Store(config.db_path)
    client = FixtureDataClient()
    monkeypatch.setattr(market, "_client", lambda _: client)
    prediction_calls = []

    def predict(_, bars, signal_session):
        prediction_calls.append(signal_session)
        assert bars.session.max() == signal_session == "2024-01-31"
        return {"SPY": 0.015, "QQQ": -0.005}, {
            "history_fingerprint": "fixture-audit-fingerprint", "signal_session": signal_session,
            "features": {"SPY": {"return_1": 0.01}, "QQQ": {"return_1": -0.01}},
        }, {"version": "fixture-model-v1"}

    monkeypatch.setattr(app.models, "predict_active", predict)
    return config, store, client, ReadOnlyFixtureBroker(), prediction_calls


def test_real_ingestion_sizing_and_dry_run_audit_with_mocked_provider(integration):
    config, store, client, broker, calls = integration
    result = app.run_once(config, store, now=NOW, broker=broker)
    assert result["status"] == "dry_run"
    assert result["session"] == "2024-02-01"
    assert result["data_timestamp"] == "2024-01-31"
    assert result["model_version"] == "fixture-model-v1"
    assert result["features"]["history_fingerprint"] == "fixture-audit-fingerprint"
    assert result["predictions"] == {"SPY": 0.015, "QQQ": -0.005}
    assert result["targets"] == {"SPY": 0.25, "QQQ": 0.0}
    assert result["current_positions"] == {}
    assert result["risk_checks"]["sale_proceeds_assumed"] is False
    orders = result["execution"]["orders"]
    assert len(orders) == 1 and orders[0]["symbol"] == "SPY"
    assert orders[0]["qty"] == 24  # Whole shares with adverse-price allowance.
    assert result["execution"]["responses"] == []
    assert broker.posts == [] and broker.reads >= 1
    assert calls == ["2024-01-31"] and len(client.historical_requests) == 1
    assert market.data_status(config, store)["end_session"] == "2024-01-31"
    assert store.decisions()[0] == json.loads(json.dumps(result))
    assert store.get_control("last_successful_run") == NOW.isoformat()
    assert "fixture-key" not in str(store.decisions())
    assert "fixture-secret" not in str(store.events())


def test_daily_decision_is_immutable_across_restart_and_model_promotion(integration, monkeypatch):
    config, store, client, broker, calls = integration
    first = app.run_once(config, store, now=NOW, broker=broker)

    def forbidden_prediction(*args):
        raise AssertionError("A same-session model promotion must not replace frozen predictions")

    monkeypatch.setattr(app.models, "predict_active", forbidden_prediction)
    restarted_store = Store(config.db_path)
    second = app.run_once(config, restarted_store, now=NOW + timedelta(seconds=30), broker=broker)
    for field in ("data_timestamp", "model_version", "features", "predictions", "targets", "configuration"):
        assert second[field] == json.loads(json.dumps(first[field]))
    assert len(client.historical_requests) == 1
    assert len(calls) == 1 and broker.posts == []
    assert len(restarted_store.decisions()) == 2


@pytest.mark.parametrize("instant", [
    datetime(2024, 2, 1, 14, 29, tzinfo=timezone.utc),
    datetime(2024, 2, 1, 14, 34, 59, tzinfo=timezone.utc),
    datetime(2024, 2, 1, 21, 0, tzinfo=timezone.utc),
    datetime(2024, 7, 4, 15, 0, tzinfo=timezone.utc),
    datetime(2024, 2, 3, 15, 0, tzinfo=timezone.utc),
])
def test_before_decision_after_close_and_holidays_never_touch_broker(integration, instant):
    config, store, client, broker, calls = integration
    result = app.run_once(replace(config, paper_execution_enabled=True), store,
                          dry_run=False, now=instant, broker=broker)
    assert result["status"] == "skipped"
    assert result["next_run"] > instant.isoformat()
    assert not client.historical_requests and not client.quote_requests and not calls
    assert not broker.posts and broker.reads == 0
    assert store.decisions()[0]["status"] == "skipped"


def test_missing_latest_session_blocks_predictions_and_orders_with_persistent_error(integration):
    config, store, client, broker, calls = integration
    client.missing_latest_bar = True
    with pytest.raises(market.DataValidationError, match="Missing or stale"):
        app.run_once(config, store, dry_run=False, now=NOW, broker=broker)
    assert not broker.posts and broker.reads == 0 and not calls
    assert market.load_bars(config, store).empty
    assert store.decisions()[0]["status"] == "error"
    assert "Missing or stale" in store.decisions()[0]["error"]
    assert store.events()[0]["level"] == "error"


def test_live_order_flag_does_not_override_disabled_execution(integration):
    config, store, _, broker, _ = integration
    result = app.run_once(config, store, dry_run=False, now=NOW, broker=broker)
    assert result["status"] == "disabled"
    assert result["execution"]["orders"] and broker.posts == []


def test_overlapping_runs_never_fetch_data_or_access_account(integration):
    config, store, client, broker, _ = integration
    with store.lock("trading"):
        with pytest.raises(RuntimeError, match="already running"):
            app.run_once(config, Store(config.db_path), now=NOW, broker=broker)
    assert not client.historical_requests and broker.reads == 0


def scheduler_clock(monkeypatch, instants):
    index = {"value": 0}

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return instants[index["value"]]

    def next_poll(_):
        index["value"] += 1
        if index["value"] == len(instants):
            raise KeyboardInterrupt

    monkeypatch.setattr(app, "datetime", Clock)
    monkeypatch.setattr(app.time, "sleep", next_poll)


def test_scheduler_waits_for_session_then_runs_once_and_keeps_reconciling(integration, monkeypatch):
    config, store, _, broker, _ = integration
    scheduler_clock(monkeypatch, [NOW - timedelta(seconds=1), NOW, NOW + timedelta(seconds=30)])
    runs, observations = [], []
    monkeypatch.setattr(app, "run_once", lambda *args, **kwargs: runs.append(kwargs))
    monkeypatch.setattr(app, "AlpacaBroker", lambda _: broker)
    monkeypatch.setattr(app, "reconcile", lambda *args: observations.append(args))
    app.scheduler(config, store, poll_seconds=30, dry_run=True)
    assert len(runs) == 1 and runs[0]["dry_run"] is True
    assert len(observations) == 2
    assert store.get_control("scheduler_last_decision")["session"] == "2024-02-01"
    assert store.get_control("scheduler_heartbeat") is None
    assert broker.posts == []


def test_scheduler_retries_failed_reads_without_marking_session_completed(integration, monkeypatch):
    config, store, _, broker, _ = integration
    scheduler_clock(monkeypatch, [NOW, NOW + timedelta(seconds=30), NOW + timedelta(seconds=60)])
    attempts = []

    def run(*args, **kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise market.MarketDataError("Fixture provider temporarily unavailable")
        return {"status": "dry_run"}

    monkeypatch.setattr(app, "run_once", run)
    monkeypatch.setattr(app, "AlpacaBroker", lambda _: broker)
    monkeypatch.setattr(app, "reconcile", lambda *args: None)
    app.scheduler(config, store, poll_seconds=30, dry_run=True)
    assert len(attempts) == 2
    assert store.get_control("scheduler_last_decision")["session"] == "2024-02-01"
    assert any("will retry" in event["message"] for event in store.events())
    assert broker.posts == []


def test_scheduler_lock_prevents_two_active_schedulers(integration):
    config, store, client, broker, _ = integration
    with store.lock("scheduler"):
        with pytest.raises(RuntimeError, match="already running"):
            app.scheduler(config, Store(config.db_path), poll_seconds=30, dry_run=True)
    assert not client.historical_requests and broker.reads == 0


def test_complete_offline_vertical_training_evaluation_promotion_and_daily_dry_run(tmp_path, monkeypatch):
    """Actual app components end to end; only the two external services are mocked."""
    config = Config(data_dir=tmp_path, symbols=("SPY", "QQQ"), history_start="2022-11-01",
                    api_key="fixture-key", secret_key="fixture-secret", threshold=-0.1)
    rows = []
    for day, label in enumerate(calendar.sessions(config.history_start, "2024-01-31")):
        for offset, symbol in enumerate(config.symbols):
            close = 100 + offset * 25 + day * 0.03 + 2 * math.sin(day / 9 + offset)
            rows.append({"session": label, "symbol": symbol,
                         "timestamp": pd.Timestamp(label, tz="America/New_York").tz_convert("UTC"),
                         "open": close * 0.998, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": float(10000 + day * 10 + offset * 100)})
    client = FixtureDataClient(pd.DataFrame(rows))
    monkeypatch.setattr(market, "_client", lambda _: client)
    store = Store(config.db_path)
    historical = market.update_data(config, store, end_session="2023-12-29")
    metadata = models.train(config, historical)
    assert not (config.models_dir / "active.json").exists()
    report = backtest.evaluate(config, historical, version=metadata["version"])
    assert report["current_data_matches_snapshot"] is True
    assert set(report["metrics"]) == {"strategy", "cash", "equal_weight_buy_hold", "trend"}
    assert report["execution_start"] > report["test_signal_start"]
    models.promote(config, metadata["version"])

    broker = ReadOnlyFixtureBroker()
    decision = app.run_once(config, store, now=NOW, broker=broker)
    assert decision["status"] == "dry_run"
    assert decision["model_version"] == metadata["version"]
    assert decision["data_timestamp"] == "2024-01-31" > metadata["data_end"]
    assert set(decision["predictions"]) == set(config.symbols)
    assert set(decision["features"]["features"]) == set(config.symbols)
    assert all(math.isfinite(prediction) for prediction in decision["predictions"].values())
    assert {order["symbol"] for order in decision["execution"]["orders"]} == set(config.symbols)
    assert decision["risk_checks"]["long_only"] is True
    assert broker.posts == []
    assert len(client.historical_requests) == 3  # Initial history plus increment and sentinel.
    assert market.data_status(config, store)["refresh_mode"] == "incremental"
    assert models.load_active(config)[1]["version"] == metadata["version"]
    assert store.decisions()[0]["model_version"] == metadata["version"]
