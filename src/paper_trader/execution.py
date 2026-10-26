"""Durable once-per-session paper order intents and fail-closed reconciliation."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math

from .broker import BrokerError, ORDER_STATUSES, TERMINAL_STATUSES
from .calendar import decision_session
from .strategy import plan_orders


class ExecutionError(RuntimeError):
    pass


def _utcnow():
    return datetime.now(timezone.utc)


def _timestamp(value):
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if result.tzinfo is None:
            raise ValueError
        return result.astimezone(timezone.utc)
    except (TypeError, ValueError):
        raise ExecutionError("Missing or malformed timezone-aware timestamp; refresh account and quotes.") from None


def _init(store):
    with store.connection() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS execution_plans (
                plan_key TEXT PRIMARY KEY, account_id TEXT NOT NULL,
                strategy_id TEXT NOT NULL, session TEXT NOT NULL,
                body TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS execution_orders (
                client_order_id TEXT PRIMARY KEY, plan_key TEXT NOT NULL,
                body TEXT NOT NULL, state TEXT NOT NULL, response TEXT,
                updated_at TEXT NOT NULL
            );
        """)


def _rows(store, query, params=()):
    with store.connection() as db:
        cursor = db.execute(query, params)
        columns = [column[0] for column in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]


def audit_session(store, strategy_id, session) -> list[dict]:
    """Read durable plans and order outcomes, including incomplete submissions.

    This accessor neither creates execution tables nor contacts the broker. A
    decision's error record can retain this evidence after a partial batch or
    restart, even when execute() could not return its normal result.
    """
    tables = _rows(store, "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('execution_plans','execution_orders')")
    if len(tables) != 2:
        return []
    plans = _rows(store, "SELECT * FROM execution_plans WHERE strategy_id=? AND session=? ORDER BY created_at,plan_key", (strategy_id, session))
    return [
        {
            "plan_key": row["plan_key"], "account_id": row["account_id"],
            "strategy_id": row["strategy_id"], "session": row["session"],
            "status": row["status"], "created_at": row["created_at"],
            "plan": json.loads(row["body"]),
            "orders": [
                {"client_order_id": order["client_order_id"],
                 "intent": json.loads(order["body"]), "state": order["state"],
                 "response": json.loads(order["response"]) if order["response"] else None,
                 "updated_at": order["updated_at"]}
                for order in _rows(store, "SELECT * FROM execution_orders WHERE plan_key=? ORDER BY rowid", (row["plan_key"],))
            ],
        }
        for row in plans
    ]


def _save_order(store, client_id, state, response=None):
    with store.connection() as db:
        db.execute("UPDATE execution_orders SET state=?,response=?,updated_at=? WHERE client_order_id=?",
                   (state, json.dumps(response) if response is not None else None, _utcnow().isoformat(), client_id))


def _identity_matches(intent, actual):
    return (actual.get("client_order_id") == intent["client_order_id"]
            and actual.get("symbol") == intent["symbol"]
            and actual.get("side") == intent["side"]
            and float(actual.get("qty", -1)) == intent["qty"]
            and abs(float(actual.get("limit_price") or 0) - intent["limit_price"]) < 1e-8
            and actual.get("type", "limit") == "limit"
            and actual.get("time_in_force", "day") == "day")


def _validate_order_state(actual, previous=None):
    qty = float(actual.get("qty", -1))
    filled = float(actual.get("filled_qty", -1))
    status = actual.get("status")
    if (status not in ORDER_STATUSES or not math.isfinite(qty) or not math.isfinite(filled)
            or qty <= 0 or not 0 <= filled <= qty):
        raise ExecutionError("Broker returned an invalid order status or fill quantity; reconciliation required.")
    if status == "filled" and filled != qty:
        raise ExecutionError("Broker marked an order filled without its complete fill quantity; reconciliation required.")
    if status == "partially_filled" and not 0 < filled < qty:
        raise ExecutionError("Broker partial-fill status disagrees with the filled quantity; reconciliation required.")
    if previous and filled < float(previous.get("filled_qty", 0)):
        raise ExecutionError("Broker reported a regression in filled quantity; wait for consistent reconciliation.")


def _block_plan(store, key):
    with store.connection() as db:
        db.execute("UPDATE execution_plans SET status='blocked' WHERE plan_key=?", (key,))


def _reconcile(store, broker, account_id):
    rows = _rows(store, """SELECT o.* FROM execution_orders o JOIN execution_plans p USING(plan_key)
        WHERE p.account_id=? AND o.state NOT IN ('planned','filled','canceled','expired','rejected','replaced','abandoned')""", (account_id,))
    reconciled = []
    for row in rows:
        intent = json.loads(row["body"])
        actual = broker.get_order(row["client_order_id"])
        if actual is None:
            raise ExecutionError("Unresolved prior submission: broker has no visible matching order. New orders stopped; inspect Alpaca orders and local execution records. Automatic POST retry is intentionally disabled.")
        if not _identity_matches(intent, actual):
            raise ExecutionError("Broker order differs from its persisted intent; reconcile account manually before trading.")
        _validate_order_state(actual, json.loads(row["response"]) if row["response"] else None)
        _save_order(store, row["client_order_id"], actual["status"], actual)
        if actual["status"] == "rejected":
            _block_plan(store, row["plan_key"])
        reconciled.append(actual)
    return reconciled


def _validate_known_pending(store, snapshot):
    """Broker open-order view must contain every acknowledged local open order."""
    rows = _rows(store, """SELECT o.* FROM execution_orders o JOIN execution_plans p USING(plan_key)
        WHERE p.account_id=? AND o.state NOT IN ('planned','filled','canceled','expired','rejected','replaced','abandoned')""", (snapshot["account_id"],))
    visible = {order.get("client_order_id"): order for order in snapshot["open_orders"]}
    for row in rows:
        actual = visible.get(row["client_order_id"])
        if actual is None or not _identity_matches(json.loads(row["body"]), actual):
            raise ExecutionError("An acknowledged order is missing from the broker open-order view; wait for consistent reconciliation before submitting more.")
        if row["response"] is not None:
            previous = json.loads(row["response"])
            if float(actual.get("filled_qty", 0)) < float(previous.get("filled_qty", 0)):
                raise ExecutionError("Broker order views disagree on partial fills; wait for reconciliation.")


def _validate_account_snapshot(snapshot, config, now):
    if not snapshot.get("account_id"):
        raise ExecutionError("Account identifier is missing; cannot establish idempotent order ownership.")
    if snapshot.get("paper") is not True:
        raise ExecutionError("Only paper account snapshots are accepted.")
    age = (now - _timestamp(snapshot.get("timestamp"))).total_seconds()
    if age < -5 or age > config.max_quote_age_seconds:
        raise ExecutionError("Account snapshot is stale; refresh broker state before trading.")
    broker_age = (now - _timestamp(snapshot.get("broker_timestamp"))).total_seconds()
    if broker_age < -5 or broker_age > config.max_quote_age_seconds:
        raise ExecutionError("Broker market clock is stale; refresh broker state before trading.")
    for field in ("cash", "buying_power", "equity"):
        value = float(snapshot[field])
        if not math.isfinite(value) or value < 0 or (field == "equity" and value == 0):
            raise ExecutionError("Account cash/equity is invalid or leveraged; new submissions stopped.")
    positions = snapshot["positions"]
    for symbol, position in positions.items():
        qty = float(position["qty"])
        if not math.isfinite(qty) or qty < 0:
            raise ExecutionError("Short or malformed positions detected; this application is long-only.")
        if symbol not in config.symbols and qty:
            raise ExecutionError("Account holds a symbol outside the configured universe; reconcile it or expand the universe first.")
        for field in ("market_value", "current_price"):
            value = float(position[field])
            if not math.isfinite(value) or value < 0 or (field == "current_price" and qty > 0 and value == 0):
                raise ExecutionError("Position valuation is malformed; refresh the broker account.")
    seen_orders = set()
    pending_sells = {}
    for order in snapshot["open_orders"]:
        _validate_order_state(order)
        if order["status"] in TERMINAL_STATUSES:
            raise ExecutionError("Broker open-order view contains a terminal order; refresh account state.")
        identifier = order.get("client_order_id")
        if not identifier or identifier in seen_orders:
            raise ExecutionError("Broker open orders have missing or duplicate identifiers; reconciliation required.")
        seen_orders.add(identifier)
        if order.get("symbol") not in config.symbols or order.get("side") not in {"buy", "sell"} or order.get("legs"):
            raise ExecutionError("Unsupported outside/open order prevents complete risk reconciliation.")
        # Market buys have no finite maximum cash reservation. Stop rather than
        # guess how much cash they will consume before the next read.
        if order["side"] == "buy":
            price = float(order.get("limit_price") or 0)
            if not math.isfinite(price) or price <= 0:
                raise ExecutionError("An outstanding buy has no bounded limit price; wait for it to finish or cancel it.")
        else:
            symbol = order["symbol"]
            pending_sells[symbol] = pending_sells.get(symbol, 0) + float(order["qty"]) - float(order["filled_qty"])
    if any(qty > float(positions.get(symbol, {}).get("qty", 0)) + 1e-8 for symbol, qty in pending_sells.items()):
        raise ExecutionError("Outstanding sells exceed confirmed held shares; new submissions stopped.")


def _validate_snapshot(snapshot, config, prices, now):
    _validate_account_snapshot(snapshot, config, now)
    for symbol in set(config.symbols) | set(snapshot["positions"]):
        quote = prices.get(symbol)
        if not quote:
            raise ExecutionError(f"Current price missing for {symbol}; no orders submitted.")
        price = float(quote["price"])
        age = (now - _timestamp(quote.get("timestamp"))).total_seconds()
        if not math.isfinite(price) or price <= 0 or age < -5 or age > config.max_quote_age_seconds:
            raise ExecutionError(f"Price for {symbol} is stale or invalid; refresh real market quotes.")


def _drawdown(store, snapshot, config):
    key = "equity_peak:" + snapshot["account_id"]
    equity = float(snapshot["equity"])
    peak = max(float(store.get_control(key, equity)), equity)
    if not math.isfinite(peak) or peak <= 0:
        raise ExecutionError("Persisted equity peak is invalid; inspect account controls before continuing.")
    store.set_control(key, peak)
    drawdown = 1 - equity / peak
    snapshot["peak_equity"] = peak
    snapshot["drawdown"] = drawdown
    if drawdown >= config.max_drawdown:
        store.set_control("paused", True)
        store.set_control("pause_reason", "Configured drawdown threshold reached. Review the account before resuming.")
    return drawdown


def _client_id(account, strategy, session, order):
    payload = json.dumps([account, strategy, session, order["symbol"], order["side"], order["qty"], order["limit_price"]], separators=(",", ":"))
    return "pt-" + hashlib.sha256(payload.encode()).hexdigest()[:40]


def _persist_plan(store, key, account, config, session, plan):
    with store.connection() as db:
        db.execute("INSERT INTO execution_plans VALUES (?,?,?,?,?,?,?)",
                   (key, account, config.strategy_id, session, json.dumps(plan), "active", _utcnow().isoformat()))
        for order in plan["orders"]:
            db.execute("INSERT INTO execution_orders VALUES (?,?,?,?,?,?)",
                       (order["client_order_id"], key, json.dumps(order), "planned", None, _utcnow().isoformat()))


def _check_intent(intent, snapshot, prices, targets, config):
    # Re-size against freshly confirmed cash and holdings before EVERY POST.
    # Existing pending buys/sells are included by the shared sizing policy.
    allowed = plan_orders(targets, snapshot, prices, config)
    symbol = intent["symbol"]
    projected_qty = float(snapshot["positions"].get(symbol, {}).get("qty", 0))
    sellable_qty = projected_qty
    for order in snapshot["open_orders"]:
        if order["symbol"] == symbol:
            projected_qty += (float(order["qty"]) - float(order.get("filled_qty", 0))) * (1 if order["side"] == "buy" else -1)
            if order["side"] == "sell":
                sellable_qty -= float(order["qty"]) - float(order.get("filled_qty", 0))
    difference = allowed["target_quantities"][symbol] - projected_qty
    if intent["qty"] > abs(difference) + 1e-8 or (difference > 0) != (intent["side"] == "buy"):
        raise ExecutionError("Persisted order no longer fits current target holdings; no further orders this run.")
    if intent["side"] == "sell" and intent["qty"] > sellable_qty:
        raise ExecutionError("Persisted sell would exceed current shares after outstanding sells.")
    notional = intent["qty"] * intent["limit_price"]
    if notional > config.max_order_notional + 1e-8:
        raise ExecutionError("Persisted order exceeds the configured order-size limit.")
    if intent["side"] == "buy":
        available = allowed["risk_checks"]["cash_budget"]
        cost = notional * (1 + config.fee_bps / 10_000)
        if cost > available + 1e-8:
            raise ExecutionError("Confirmed cash is insufficient for the persisted limit order; sale proceeds are not anticipated.")
        asset_values = {
            s: max(float(snapshot["positions"].get(s, {}).get("market_value", 0)),
                   float(snapshot["positions"].get(s, {}).get("qty", 0)) *
                   max(float(prices[s]["price"]), float(snapshot["positions"].get(s, {}).get("current_price", 0))))
            for s in config.symbols
        }
        for order in snapshot["open_orders"]:
            if order["side"] == "buy":
                s = order["symbol"]
                asset_values[s] += (float(order["qty"]) - float(order.get("filled_qty", 0))) * max(
                    float(order["limit_price"]), float(prices[s]["price"]),
                    float(snapshot["positions"].get(s, {}).get("current_price", 0)))
        exposure = intent["qty"] * max(intent["limit_price"], float(prices[symbol]["price"]),
                                        float(snapshot["positions"].get(symbol, {}).get("current_price", 0)))
        post_fee_equity = allowed["risk_checks"]["fee_adjusted_equity_floor"]
        if (asset_values[symbol] + exposure > post_fee_equity * config.max_asset_weight + 1e-8 or
                sum(asset_values.values()) + exposure > post_fee_equity * config.max_total_weight + 1e-8):
            raise ExecutionError("Persisted buy exceeds current asset or total exposure limits.")


def execute(config, store, broker, session, targets, prices, dry_run=True, now=None) -> dict:
    """Preview or execute one immutable account/strategy/session plan.

    Unknown submissions remain blocked even after a 404, because broker order
    visibility may lag acceptance. A later read finding the order resolves them.
    """
    with store.lock("execution"):
        _init(store)
        try:
            return _execute(config, store, broker, session, targets, prices, dry_run, now)
        except (ExecutionError, BrokerError) as exc:
            store.event("ERROR", str(exc), {"session": session, "component": "execution"})
            raise ExecutionError(str(exc)) from None
        except Exception as exc:
            message = f"Execution stopped during reconciliation ({type(exc).__name__}); inspect account and local records before rerunning."
            store.event("ERROR", message, {"session": session, "component": "execution"})
            raise ExecutionError(message) from None


def reconcile(config, store, broker, now=None) -> dict:
    """Read-only periodic order/fill synchronization, including drawdown pause."""
    with store.lock("execution"):
        _init(store)
        try:
            snapshot = broker.snapshot()
            _validate_account_snapshot(snapshot, config, now or _utcnow())
            account = snapshot["account_id"]
            orders = _reconcile(store, broker, snapshot["account_id"])
            if orders:
                snapshot = broker.snapshot()
                _validate_account_snapshot(snapshot, config, now or _utcnow())
                if snapshot["account_id"] != account:
                    raise ExecutionError("Account identity changed during reconciliation; account state was not saved.")
            _validate_known_pending(store, snapshot)
            _drawdown(store, snapshot, config)
            store.save_snapshot(snapshot)
            return {"snapshot": snapshot, "orders": orders, "status": "reconciled"}
        except Exception as exc:
            message = str(exc) if isinstance(exc, (ExecutionError, BrokerError)) else f"Paper reconciliation failed ({type(exc).__name__}); no orders submitted."
            store.event("ERROR", message, {"component": "reconciliation"})
            raise ExecutionError(message) from None


def _execute(config, store, broker, session, targets, prices, dry_run, now):
    snapshot = broker.snapshot()
    current_time = now or _utcnow()
    _validate_snapshot(snapshot, config, prices, current_time)
    initial_account = snapshot["account_id"]
    reconciled = _reconcile(store, broker, initial_account)
    if reconciled:
        snapshot = broker.snapshot()
        current_time = now or _utcnow()
        _validate_snapshot(snapshot, config, prices, current_time)
        if snapshot["account_id"] != initial_account:
            raise ExecutionError("Account identity changed during reconciliation; new submissions stopped.")
    _validate_known_pending(store, snapshot)
    _drawdown(store, snapshot, config)
    store.save_snapshot(snapshot)
    preview = plan_orders(targets, snapshot, prices, config)
    result = {"status": "dry_run", "session": session, "snapshot": snapshot,
              **preview, "responses": reconciled, "dry_run": dry_run}
    enabled = bool(store.get_control("execution_enabled", config.paper_execution_enabled))
    paused = bool(store.get_control("paused", False))
    result["execution_enabled"] = enabled
    result["paused"] = paused
    if paused:
        result["status"] = "paused"
        return result
    if dry_run:
        return result
    if not enabled:
        result["status"] = "disabled"
        return result
    if decision_session(current_time, config.decision_delay_minutes) != session or not snapshot.get("market_open", False):
        raise ExecutionError("Order session is not the current open exchange session after the decision delay.")
    account = snapshot["account_id"]
    key = hashlib.sha256(json.dumps([account, config.strategy_id, session]).encode()).hexdigest()
    existing = _rows(store, "SELECT * FROM execution_plans WHERE plan_key=?", (key,))
    if existing:
        plan = json.loads(existing[0]["body"])
        if existing[0]["status"] == "blocked":
            result.update(status="blocked", orders=plan["orders"])
            return result
    else:
        plan = {**preview, "targets": targets}
        for order in plan["orders"]:
            order["limit_price"] = order["estimated_price"]
            order["client_order_id"] = _client_id(account, config.strategy_id, session, order)
        _persist_plan(store, key, account, config, session, plan)
    result["orders"] = plan["orders"]
    result["plan_key"] = key
    submitted = False
    for row in _rows(store, "SELECT * FROM execution_orders WHERE plan_key=? ORDER BY rowid", (key,)):
        if row["state"] != "planned":
            continue
        intent = json.loads(row["body"])
        actual = broker.get_order(intent["client_order_id"])
        if actual is not None:
            if not _identity_matches(intent, actual):
                raise ExecutionError("Client order ID collision with a different broker order; trading stopped.")
            _validate_order_state(actual)
            _save_order(store, intent["client_order_id"], actual["status"], actual)
            result["responses"].append(actual)
            if actual["status"] == "rejected":
                _block_plan(store, key)
                result["status"] = "blocked"
                return result
            continue
        result["responses"].extend(_reconcile(store, broker, account))
        if _rows(store, "SELECT status FROM execution_plans WHERE plan_key=?", (key,))[0]["status"] == "blocked":
            result["status"] = "blocked"
            return result
        fresh = broker.snapshot()
        check_time = now or _utcnow()
        _validate_snapshot(fresh, config, prices, check_time)
        _validate_known_pending(store, fresh)
        if fresh["account_id"] != account or not fresh.get("market_open", False):
            raise ExecutionError("Account identity or market-open state changed; trading stopped.")
        if decision_session(check_time, config.decision_delay_minutes) != session:
            raise ExecutionError("The trading window ended; remaining orders were not submitted.")
        _drawdown(store, fresh, config)
        if store.get_control("paused", False) or not store.get_control("execution_enabled", config.paper_execution_enabled):
            result["status"] = "paused" if store.get_control("paused", False) else "disabled"
            return result
        _check_intent(intent, fresh, prices, plan["targets"], config)
        # Commit before the network call. A process crash now is deliberately
        # treated as ambiguous, sacrificing availability to prevent duplication.
        _save_order(store, intent["client_order_id"], "submitting")
        try:
            actual = broker.submit_order(intent)
        except Exception:
            _save_order(store, intent["client_order_id"], "ambiguous")
            store.event("ERROR", "Paper submission uncertain; do not retry POST. Reconcile stable client order ID.", {"client_order_id": intent["client_order_id"]})
            raise ExecutionError("Paper submission is ambiguous; the next run will query its stable ID before proceeding. No automatic resubmission is permitted.") from None
        if not _identity_matches(intent, actual):
            _save_order(store, intent["client_order_id"], "ambiguous", actual)
            raise ExecutionError("Paper submission response differs from its persisted intent; reconciliation required.")
        _validate_order_state(actual)
        _save_order(store, intent["client_order_id"], actual["status"], actual)
        submitted = True
        result["responses"].append(actual)
        if actual["status"] == "rejected":
            _block_plan(store, key)
            store.event("ERROR", "Paper order rejected; remaining session orders stopped.", {"client_order_id": intent["client_order_id"]})
            result["status"] = "blocked"
            return result
    result["status"] = "submitted" if submitted else ("reconciled" if plan["orders"] else "no_orders")
    return result
