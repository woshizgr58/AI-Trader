"""Dashboard smoke tests use temporary empty storage and no network access."""
from pathlib import Path
import json
import sys

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from paper_trader import dashboard
from paper_trader.config import Config
from paper_trader.store import Store


@pytest.fixture
def empty_dashboard(tmp_path, monkeypatch):
    config = Config(data_dir=tmp_path, api_key="", secret_key="")
    # Imported load_config is rebound by the executed module, so patch its origin.
    monkeypatch.setattr("paper_trader.config.load_config", lambda _: config)
    monkeypatch.setattr(sys, "argv", ["dashboard.py"])
    return AppTest.from_file(str(Path(dashboard.__file__)), default_timeout=15).run()


def test_empty_dashboard_has_honest_states_and_paper_banner(empty_dashboard):
    view = empty_dashboard
    assert not view.exception
    assert any("PAPER TRADING — VIRTUAL MONEY" in element.value for element in view.warning)
    messages = "\n".join(element.value for element in view.info)
    assert "No training data" in messages
    assert "No deployed model" in messages
    assert "No historical evaluation" in messages
    assert "No forward paper history" in messages
    metrics = {element.label: element.value for element in view.metric}
    assert metrics["Paper connection"] == "Not connected"
    assert metrics["Automatic orders"] == "Disabled"
    assert metrics["Data feed"] == "No data"


def test_dashboard_disables_account_io_without_credentials(empty_dashboard):
    buttons = {button.label: button for button in empty_dashboard.button}
    for label in ("Refresh paper account", "Preview today's orders", "Cancel all open paper orders"):
        assert buttons[label].disabled
    assert not buttons["Enable automatic paper orders"].disabled
    assert not buttons["Pause new orders"].disabled


def test_forward_benchmark_uses_only_common_post_close_observations():
    # Store supplies snapshots newest first. Intraday marks cannot enter this comparison.
    snapshots = [
        {"timestamp": "2024-03-11T20:05:00+00:00", "equity": 11000},
        {"timestamp": "2024-03-11T18:00:00+00:00", "equity": 99999},
        {"timestamp": "2024-03-08T21:05:00+00:00", "equity": 10000},
        {"timestamp": "2024-03-07T21:05:00+00:00", "equity": 8000},
    ]
    bars = pd.DataFrame([
        {"session": "2024-03-08", "symbol": "SPY", "close": 100},
        {"session": "2024-03-08", "symbol": "QQQ", "close": 200},
        {"session": "2024-03-11", "symbol": "SPY", "close": 110},
        {"session": "2024-03-11", "symbol": "QQQ", "close": 210},
    ])
    curve = dashboard.forward_curve(snapshots, bars, ("SPY", "QQQ"))
    assert list(curve.index) == ["2024-03-08", "2024-03-11"]
    assert curve.iloc[0].eq(0).all()
    assert curve.iloc[-1]["Observed paper equity"] == pytest.approx(10)
    assert curve.iloc[-1]["Equal-weight total-return benchmark"] == pytest.approx(7.5)
    assert curve["Cash (0% interest)"].eq(0).all()
    assert dashboard.forward_curve(snapshots[:1], bars, ("SPY", "QQQ")).empty


def test_populated_dashboard_renders_audit_account_and_historical_report(tmp_path, monkeypatch):
    config = Config(data_dir=tmp_path, api_key="", secret_key="")
    store = Store(config.db_path)
    store.save_snapshot({"timestamp": "2024-02-01T14:35:00+00:00", "account_id": "fixture-account",
                         "equity": 10000.0, "cash": 7500.0, "buying_power": 7500.0,
                         "positions": {"SPY": {"qty": 25, "market_value": 2500.0, "current_price": 100.0}},
                         "open_orders": [], "recent_orders": [{"symbol": "SPY", "filled_qty": 25, "status": "filled"}]})
    store.save_decision({"timestamp": "2024-02-01T14:35:00+00:00", "session": "2024-02-01",
                         "status": "dry_run", "dry_run": True, "data_timestamp": "2024-01-31",
                         "model_version": "fixture-v1", "predictions": {"SPY": 0.01},
                         "targets": {"SPY": 0.25}, "configuration": config.public_dict()})
    store.event("info", "Fixture account observation", {"component": "dashboard-test"})
    config.models_dir.mkdir()
    (config.models_dir / "active.json").write_text(json.dumps({"version": "fixture-v1"}))
    report_dir = tmp_path / "backtests" / "fixture-v1"
    report_dir.mkdir(parents=True)
    (report_dir / "report.json").write_text(json.dumps({"metrics": {"strategy": {"net_return": 0.01}, "cash": {"net_return": 0.0}}}))
    pd.DataFrame([
        {"session": "2024-01-30", "portfolio": "strategy", "equity": 10000.0, "exposure": 0.25},
        {"session": "2024-01-31", "portfolio": "strategy", "equity": 10100.0, "exposure": 0.25},
        {"session": "2024-01-30", "portfolio": "cash", "equity": 10000.0, "exposure": 0.0},
        {"session": "2024-01-31", "portfolio": "cash", "equity": 10000.0, "exposure": 0.0},
    ]).to_csv(report_dir / "curves.csv", index=False)
    monkeypatch.setattr("paper_trader.config.load_config", lambda _: config)
    monkeypatch.setattr(sys, "argv", ["dashboard.py"])
    view = AppTest.from_file(str(Path(dashboard.__file__)), default_timeout=15).run()
    assert not view.exception
    metrics = {element.label: element.value for element in view.metric}
    assert metrics["Active model"] == "fixture-v1"
    assert metrics["Virtual portfolio"] == "$10,000.00"
    assert metrics["Cash"] == "$7,500.00"
    assert metrics["Exposure"] == "25.0%"
    assert any("HISTORICAL BACKTEST" in caption.value for caption in view.caption)
