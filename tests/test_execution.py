"""Broker fixtures only: these tests never contact a provider or place orders."""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from paper_trader.config import Config
from paper_trader.execution import ExecutionError, audit_session, execute, reconcile
from paper_trader.store import Store


NOW = datetime(2026, 10, 13, 14, 0, tzinfo=timezone.utc)
SESSION = "2026-10-13"


@pytest.fixture
def setup(tmp_path):
    config = Config(data_dir=tmp_path, symbols=("SPY", "QQQ"), paper_execution_enabled=True,
                    slippage_bps=0, fee_bps=0, cash_buffer=0, max_asset_weight=0.5)
    store = Store(config.db_path)
    prices = {s: {"price": 100.0, "timestamp": NOW.isoformat()} for s in config.symbols}
    return config, store, prices


class FakeBroker:
    def __init__(self, cash=10000):
        self.state = {"account_id": "paper-account", "cash": cash, "buying_power": cash,
                      "equity": 10000.0, "positions": {}, "open_orders": [],
                      "timestamp": NOW.isoformat(), "broker_timestamp": NOW.isoformat(),
                      "market_open": True, "paper": True}
        self.orders = {}
        self.posts = []
        self.lookups = []
        self.mode = "new"
        self.cancelled = False

    def snapshot(self):
        result = deepcopy(self.state)
        result["open_orders"] = [deepcopy(o) for o in self.orders.values()
                                 if o["status"] not in {"filled", "rejected", "canceled", "expired"}]
        return result

    def get_order(self, client_id):
        self.lookups.append(client_id)
        return deepcopy(self.orders.get(client_id))

    def submit_order(self, intent):
        self.posts.append(deepcopy(intent))
        response = {**intent, "id": f"broker-{len(self.posts)}", "status": "new", "filled_qty": 0}
        if self.mode == "timeout_missing":
            raise TimeoutError("fixture only")
        if self.mode == "rejected":
            response["status"] = "rejected"
        self.orders[intent["client_order_id"]] = response
        if self.mode == "timeout_accepted":
            raise TimeoutError("fixture only")
        return deepcopy(response)


def run(setup, broker, targets=None, **kwargs):
    config, store, prices = setup
    return execute(config, store, broker, SESSION, targets or {"SPY": 0.5, "QQQ": 0.5}, prices,
                   now=NOW, **kwargs)


def test_dry_run_is_default_and_disabled_blocks_posts(setup):
    broker = FakeBroker()
    assert run(setup, broker)["status"] == "dry_run"
    assert broker.posts == []
    config, store, prices = setup
    store.set_control("execution_enabled", False)
    assert run(setup, broker, dry_run=False)["status"] == "disabled"
    assert broker.posts == []
    with store.connection() as db:
        assert db.execute("SELECT count(*) FROM execution_plans").fetchone()[0] == 0


def test_durable_no_duplicate_on_rerun_or_process_restart(setup):
    broker = FakeBroker()
    first = run(setup, broker, dry_run=False)
    assert len(first["orders"]) == len(broker.posts) == 2
    config, _, prices = setup
    restarted = (config, Store(config.db_path), prices)
    # Even a changed recommendation cannot mutate a submitted session plan.
    second = run(restarted, broker, targets={"SPY": 0.0, "QQQ": 0.0}, dry_run=False)
    assert second["status"] == "reconciled"
    assert len(broker.posts) == 2
    assert first["orders"] == second["orders"]
    assert all(len(o["client_order_id"]) <= 48 for o in broker.posts)


def test_accepted_timeout_is_queried_then_continued_without_duplicate(setup):
    broker = FakeBroker()
    broker.mode = "timeout_accepted"
    with pytest.raises(ExecutionError, match="ambiguous"):
        run(setup, broker, dry_run=False)
    lost_id = broker.posts[0]["client_order_id"]
    broker.mode = "new"
    run(setup, broker, dry_run=False)
    assert lost_id in broker.lookups
    assert [o["client_order_id"] for o in broker.posts].count(lost_id) == 1
    assert len(broker.posts) == 2


def test_missing_after_timeout_never_blindly_retries_even_after_restart(setup):
    broker = FakeBroker()
    broker.mode = "timeout_missing"
    with pytest.raises(ExecutionError, match="ambiguous"):
        run(setup, broker, dry_run=False)
    config, _, prices = setup
    for _ in range(2):
        with pytest.raises(ExecutionError, match="Unresolved prior submission"):
            run((config, Store(config.db_path), prices), broker, dry_run=False)
    assert len(broker.posts) == 1


def test_unknown_previous_day_submission_blocks_new_session(setup):
    broker = FakeBroker()
    broker.mode = "timeout_missing"
    with pytest.raises(ExecutionError):
        run(setup, broker, dry_run=False)
    config, store, prices = setup
    tomorrow = NOW + timedelta(days=1)
    broker.state["timestamp"] = tomorrow.isoformat()
    broker.state["broker_timestamp"] = tomorrow.isoformat()
    prices = {s: {"price": 100, "timestamp": tomorrow.isoformat()} for s in prices}
    with pytest.raises(ExecutionError, match="Unresolved prior"):
        execute(config, store, broker, "2026-10-14", {"SPY": 0.5, "QQQ": 0.5}, prices, False, tomorrow)
    assert len(broker.posts) == 1


def test_partial_fill_updates_durable_state_without_replacing_order(setup):
    broker = FakeBroker()
    run(setup, broker, dry_run=False)
    cid = broker.posts[0]["client_order_id"]
    broker.orders[cid].update(status="partially_filled", filled_qty=10)
    symbol = broker.orders[cid]["symbol"]
    broker.state["positions"][symbol] = {"qty": 10, "market_value": 1000, "current_price": 100}
    broker.state["cash"] -= 1000
    broker.state["buying_power"] -= 1000
    run(setup, broker, dry_run=False)
    assert len(broker.posts) == 2
    with setup[1].connection() as db:
        assert db.execute("SELECT state FROM execution_orders WHERE client_order_id=?", (cid,)).fetchone()[0] == "partially_filled"


def test_rejection_stops_session_and_is_never_resubmitted(setup):
    broker = FakeBroker()
    broker.mode = "rejected"
    assert run(setup, broker, dry_run=False)["status"] == "blocked"
    assert run(setup, broker, dry_run=False)["status"] == "blocked"
    assert len(broker.posts) == 1


def test_later_rejection_blocks_unsubmitted_remainder(setup):
    broker = FakeBroker()
    broker.mode = "timeout_accepted"
    with pytest.raises(ExecutionError):
        run(setup, broker, dry_run=False)
    broker.orders[broker.posts[0]["client_order_id"]]["status"] = "rejected"
    assert run(setup, broker, dry_run=False)["status"] == "blocked"
    assert len(broker.posts) == 1


def test_pending_partial_buy_counts_toward_target(setup):
    broker = FakeBroker()
    broker.state["positions"] = {"SPY": {"qty": 10, "market_value": 1000, "current_price": 100}}
    broker.state["cash"] = broker.state["buying_power"] = 9000
    broker.orders["outside"] = {"id": "outside", "client_order_id": "outside", "symbol": "SPY",
                                "side": "buy", "qty": 50, "filled_qty": 10,
                                "status": "partially_filled", "limit_price": 100}
    preview = run(setup, broker)
    assert not any(o["symbol"] == "SPY" for o in preview["orders"])
    assert preview["risk_checks"]["pending_buy_reserve"] == 4000


def test_sales_are_not_counted_as_spendable_cash(setup):
    broker = FakeBroker(cash=0)
    broker.state["positions"] = {"SPY": {"qty": 100, "market_value": 10000, "current_price": 100}}
    result = run(setup, broker, targets={"SPY": 0, "QQQ": 0.5}, dry_run=False)
    assert [o["side"] for o in result["orders"]] == ["sell"]
    assert len(broker.posts) == 1


def test_pause_and_drawdown_prevent_orders_without_liquidating(setup):
    broker = FakeBroker()
    setup[1].set_control("equity_peak:paper-account", 12000)
    result = run(setup, broker, dry_run=False)
    assert result["status"] == "paused"
    assert setup[1].get_control("paused") is True
    assert broker.posts == [] and not broker.cancelled


def test_stale_quote_and_stale_account_fail_closed(setup):
    broker = FakeBroker()
    setup[2]["SPY"]["timestamp"] = (NOW - timedelta(minutes=10)).isoformat()
    with pytest.raises(ExecutionError, match="stale"):
        run(setup, broker, dry_run=False)
    assert not broker.posts
    setup[2]["SPY"]["timestamp"] = NOW.isoformat()
    broker.state["timestamp"] = (NOW - timedelta(minutes=10)).isoformat()
    with pytest.raises(ExecutionError, match="stale"):
        run(setup, broker, dry_run=False)


def test_market_closed_blocks_new_orders(setup):
    broker = FakeBroker()
    broker.state["market_open"] = False
    with pytest.raises(ExecutionError, match="current open"):
        run(setup, broker, dry_run=False)
    assert broker.posts == []


def test_stale_broker_clock_cannot_be_hidden_by_fresh_local_timestamp(setup):
    broker = FakeBroker()
    broker.state["broker_timestamp"] = (NOW - timedelta(minutes=10)).isoformat()
    with pytest.raises(ExecutionError, match="market clock is stale"):
        run(setup, broker, dry_run=False)
    assert broker.posts == []


def test_read_failure_before_submit_never_posts(setup):
    broker = FakeBroker()
    broker.get_order = lambda _: (_ for _ in ()).throw(TimeoutError("fixture secret"))
    with pytest.raises(ExecutionError, match="TimeoutError") as exc:
        run(setup, broker, dry_run=False)
    assert "fixture secret" not in str(exc.value)
    assert broker.posts == []


def test_order_notional_limit_whole_shares_and_cash(setup):
    config, store, prices = setup
    bounded = replace(config, max_order_notional=2250, cash_buffer=0.1)
    broker = FakeBroker(cash=3500)
    result = run((bounded, store, prices), broker, dry_run=False)
    assert sum(o["qty"] * o["limit_price"] for o in broker.posts) <= 2500
    assert all(o["qty"] * o["limit_price"] <= 2250 and isinstance(o["qty"], int) for o in broker.posts)
    assert result["status"] == "submitted"


def test_read_only_reconcile_tracks_fills(setup):
    broker = FakeBroker()
    run(setup, broker, dry_run=False)
    cid = broker.posts[0]["client_order_id"]
    broker.orders[cid].update(status="filled", filled_qty=50)
    result = reconcile(setup[0], setup[1], broker, now=NOW)
    assert any(o["status"] == "filled" for o in result["orders"])
    assert len(broker.posts) == 2


def test_missing_known_open_order_fails_closed(setup):
    broker = FakeBroker()
    broker.mode = "timeout_accepted"
    with pytest.raises(ExecutionError):
        run(setup, broker, dry_run=False)
    snapshot = broker.snapshot
    def hidden():
        result = snapshot()
        result["open_orders"] = []
        return result
    broker.snapshot = hidden
    with pytest.raises(ExecutionError, match="missing from the broker"):
        run(setup, broker, dry_run=False)
    assert len(broker.posts) == 1


def test_overlapping_execution_rejected(setup):
    with setup[1].lock("execution"):
        with pytest.raises(RuntimeError, match="already running"):
            run(setup, FakeBroker(), dry_run=False)


@pytest.mark.parametrize("field,value", [
    ("timestamp", (NOW - timedelta(minutes=10)).isoformat()),
    ("broker_timestamp", (NOW - timedelta(minutes=10)).isoformat()),
    ("cash", -1), ("equity", 0), ("equity", float("nan")),
    ("buying_power", float("inf")), ("paper", False),
])
def test_periodic_reconcile_rejects_invalid_account_without_touching_peak(setup, field, value):
    config, store, _ = setup
    broker = FakeBroker()
    broker.state[field] = value
    with pytest.raises(ExecutionError):
        reconcile(config, store, broker, now=NOW)
    assert store.get_control("equity_peak:paper-account") is None
    assert store.latest_snapshot() is None
    assert broker.posts == []


def test_fill_quantity_regression_fails_closed(setup):
    broker = FakeBroker()
    run(setup, broker, dry_run=False)
    cid = broker.posts[0]["client_order_id"]
    broker.orders[cid].update(status="partially_filled", filled_qty=10)
    broker.state["positions"][broker.orders[cid]["symbol"]] = {"qty": 10, "market_value": 1000, "current_price": 100}
    broker.state["cash"] = broker.state["buying_power"] = 9000
    reconcile(setup[0], setup[1], broker, now=NOW)
    broker.orders[cid]["filled_qty"] = 5
    with pytest.raises(ExecutionError, match="regression"):
        reconcile(setup[0], setup[1], broker, now=NOW)
    assert len(broker.posts) == 2


def test_filled_status_requires_full_quantity(setup):
    broker = FakeBroker()
    run(setup, broker, dry_run=False)
    cid = broker.posts[0]["client_order_id"]
    broker.orders[cid].update(status="filled", filled_qty=1)
    with pytest.raises(ExecutionError, match="complete fill quantity"):
        reconcile(setup[0], setup[1], broker, now=NOW)


def test_rejected_order_discovered_for_planned_intent_blocks_remainder(setup):
    broker = FakeBroker()
    lookup = broker.get_order
    # Interrupt after durable plan creation, before sending its first order.
    broker.get_order = lambda _: (_ for _ in ()).throw(TimeoutError())
    with pytest.raises(ExecutionError):
        run(setup, broker, dry_run=False)
    broker.get_order = lookup
    import json
    with setup[1].connection() as db:
        intent = json.loads(db.execute("SELECT body FROM execution_orders ORDER BY rowid LIMIT 1").fetchone()[0])
    broker.orders[intent["client_order_id"]] = {**intent, "id": "previous", "status": "rejected", "filled_qty": 0}
    assert run(setup, broker, dry_run=False)["status"] == "blocked"
    assert broker.posts == []


def test_sells_never_rely_on_unfilled_buy_inventory(setup):
    broker = FakeBroker()
    broker.orders["outside-buy"] = {"id": "buy", "client_order_id": "outside-buy", "symbol": "SPY",
                                     "qty": 50, "filled_qty": 0, "side": "buy", "status": "new", "limit_price": 100}
    result = run(setup, broker, targets={"SPY": 0, "QQQ": 0}, dry_run=False)
    assert result["orders"] == []
    assert broker.posts == []


def test_pending_buys_cannot_hide_naked_pending_sells(setup):
    broker = FakeBroker()
    for side in ("buy", "sell"):
        broker.orders[side] = {"id": side, "client_order_id": side, "symbol": "SPY", "qty": 50,
                               "filled_qty": 0, "side": side, "status": "new", "limit_price": 100}
    with pytest.raises(ExecutionError, match="confirmed held shares"):
        run(setup, broker, dry_run=False)
    assert broker.posts == []


def test_cash_change_before_second_post_stops_remaining_order(setup):
    broker = FakeBroker()
    submit = broker.submit_order
    def cash_changed(intent):
        response = submit(intent)
        broker.state["cash"] = broker.state["buying_power"] = 5000
        return response
    broker.submit_order = cash_changed
    with pytest.raises(ExecutionError, match="Confirmed cash"):
        run(setup, broker, dry_run=False)
    assert len(broker.posts) == 1


def test_pause_during_submissions_stops_remaining_without_cancel(setup):
    broker = FakeBroker()
    submit = broker.submit_order
    def pause_after_first(intent):
        response = submit(intent)
        setup[1].set_control("paused", True)
        return response
    broker.submit_order = pause_after_first
    assert run(setup, broker, dry_run=False)["status"] == "paused"
    assert len(broker.posts) == 1
    assert not broker.cancelled


def test_reconcile_updates_drawdown_pause_without_submitting(setup):
    config, store, _ = setup
    broker = FakeBroker()
    store.set_control("equity_peak:paper-account", 12000)
    result = reconcile(config, store, broker, now=NOW)
    assert result["snapshot"]["drawdown"] == pytest.approx(1 - 10000 / 12000)
    assert store.get_control("paused") is True
    assert broker.posts == []


def test_audit_session_without_execution_tables_is_read_only(setup):
    config, store, _ = setup
    assert audit_session(store, config.strategy_id, SESSION) == []
    with store.connection() as db:
        assert db.execute("SELECT count(*) FROM sqlite_master WHERE name LIKE 'execution_%'").fetchone()[0] == 0


def test_audit_session_recovers_success_and_ambiguous_order_after_restart(setup):
    config, _, _ = setup
    broker = FakeBroker()
    submit = broker.submit_order
    def lose_second_response(intent):
        if broker.posts:
            broker.mode = "timeout_missing"
        return submit(intent)
    broker.submit_order = lose_second_response
    with pytest.raises(ExecutionError, match="ambiguous"):
        run(setup, broker, dry_run=False)
    restarted = Store(config.db_path)
    records = audit_session(restarted, config.strategy_id, SESSION)
    assert len(records) == 1
    record = records[0]
    assert record["account_id"] == "paper-account"
    assert record["status"] == "active"
    assert record["plan"]["risk_checks"]["sale_proceeds_assumed"] is False
    first, second = record["orders"]
    assert first["state"] == "new"
    assert first["response"]["client_order_id"] == first["intent"]["client_order_id"]
    assert second["state"] == "ambiguous" and second["response"] is None
    assert len(broker.posts) == 2
    assert audit_session(restarted, "different-strategy", SESSION) == []
    assert audit_session(restarted, config.strategy_id, "2026-10-14") == []
    assert audit_session(restarted, config.strategy_id, SESSION) == records
