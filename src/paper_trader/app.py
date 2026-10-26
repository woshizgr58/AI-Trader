"""The vertical workflow and a small calendar-aware local scheduler."""
from __future__ import annotations

import time
from datetime import datetime, timezone

from . import calendar, market, models
from .broker import AlpacaBroker
from .config import Config
from .execution import audit_session, execute, reconcile
from .store import Store, redact, utc_now
from .strategy import target_weights


def refresh_account(config: Config, store: Store, broker=None) -> dict:
    broker = broker or AlpacaBroker(config)
    try:
        snapshot = reconcile(config, store, broker)["snapshot"]
        store.set_control("connection_status", {"ok": True, "timestamp": utc_now()})
        return snapshot
    except Exception as exc:
        store.set_control("connection_status", {"ok": False, "timestamp": utc_now(), "error": redact(str(exc))})
        raise


def run_once(config: Config, store: Store, *, dry_run=True, now=None, broker=None) -> dict:
    injected_now = now
    now = now or datetime.now(timezone.utc)
    record = {"timestamp": now.isoformat(), "dry_run": dry_run, "strategy_id": config.strategy_id}
    with store.lock("trading"):
        try:
            session = calendar.decision_session(now, config.decision_delay_minutes)
            record["session"] = session
            if session is None:
                record.update(status="skipped", reason="Outside the market session or before the decision time", next_run=calendar.next_run(now, config.decision_delay_minutes).isoformat())
                store.save_decision(record)
                return record
            signal_session = calendar.previous_session(now)
            # Freeze each session's decision. A new promotion takes effect next session.
            key = f"decision:{config.strategy_id}:{session}"
            decision = store.get_control(key)
            bars = market.load_bars(config, store) if decision else market.update_data(config, store, end_session=signal_session)
            market.validate_bars(bars, tuple(dict.fromkeys((*config.symbols, "SPY"))), end_session=signal_session)
            if decision is None:
                predictions, feature_reference, metadata = models.predict_active(config, bars, signal_session)
                decision = {
                    "data_timestamp": signal_session,
                    "last_bar_timestamp_utc": bars.timestamp.max().isoformat(),
                    "data_provenance": market.data_status(config, store),
                    "model_version": metadata["version"],
                    "features": feature_reference,
                    "predictions": predictions,
                    "targets": target_weights(predictions, config),
                    "configuration": config.public_dict(),
                }
                store.set_control(key, decision)
            record.update(decision)
            prices = market.latest_prices(config, config.symbols, now=injected_now)
            record["current_prices"] = prices
            broker = broker or AlpacaBroker(config)
            result = execute(config, store, broker, session, decision["targets"], prices, dry_run=dry_run, now=injected_now)
            record["execution"] = result
            record["status"] = result["status"]
            snapshot = result.get("snapshot")
            if snapshot:
                record["current_positions"] = snapshot.get("positions", {})
                store.save_snapshot(snapshot)
            record["risk_checks"] = result.get("risk_checks", {})
            store.save_decision(record)
            if record["status"] not in {"blocked", "error"}:
                store.set_control("last_successful_run", record["timestamp"])
            store.event("info", "Daily paper decision recorded", {"session": session, "status": record["status"], "model_version": record["model_version"]})
            return record
        except Exception as exc:
            message = redact(str(exc))
            record.update(status="error", error=message)
            record["risk_checks"] = {"passed": False, "failure": message}
            if record.get("session"):
                record["execution_audit"] = audit_session(store, config.strategy_id, record["session"])
            store.save_decision(record)
            store.event("error", "Daily workflow stopped; no further submissions", {"error": message})
            raise


def set_execution(store: Store, enabled: bool) -> None:
    store.set_control("execution_enabled", bool(enabled))
    store.event("info", "Automatic paper execution enabled" if enabled else "Automatic paper execution disabled")


def set_pause(store: Store, paused: bool) -> None:
    store.set_control("paused", bool(paused))
    store.event("info", "New orders paused; existing orders remain open" if paused else "Pause cleared; risk checks still apply")


def cancel_open_orders(config: Config, store: Store) -> dict:
    with store.lock("trading"):
        result = AlpacaBroker(config).cancel_open_orders()
        store.event("info", "Open paper order cancellation requested", {"responses": result})
        return {"responses": result, "note": "Cancellation requests can race with fills; refresh to confirm final status."}


def scheduler(config: Config, store: Store, *, poll_seconds=30, dry_run=False) -> None:
    """One frozen decision/session; retry/reconcile the durable plan while open."""
    if poll_seconds < 10:
        raise ValueError("Scheduler poll interval must be at least 10 seconds")
    with store.lock("scheduler"):
        store.event("info", "Local paper scheduler started", {"dry_run": dry_run})
        try:
            while True:
                now = datetime.now(timezone.utc)
                session = calendar.decision_session(now, config.decision_delay_minutes)
                store.set_control("scheduler_heartbeat", now.isoformat())
                store.set_control("next_scheduled_run", calendar.next_run(now, config.decision_delay_minutes).isoformat())
                enabled = store.get_control("execution_enabled", config.paper_execution_enabled)
                paused = store.get_control("paused", False)
                state = {"session": session, "enabled": enabled, "paused": paused, "dry_run": dry_run}
                try:
                    if session and state != store.get_control("scheduler_last_decision"):
                        run_once(config, store, dry_run=dry_run)
                        store.set_control("scheduler_last_decision", state)
                    elif config.has_credentials:
                        # Observe partial fills, rejections and late fills even after close.
                        with store.lock("trading"):
                            reconcile(config, store, AlpacaBroker(config))
                except Exception as exc:
                    store.event("error", "Scheduler operation stopped; will retry on next poll", {"error": redact(str(exc))})
                time.sleep(poll_seconds)
        except KeyboardInterrupt:
            store.event("info", "Local paper scheduler stopped")
        finally:
            store.set_control("scheduler_heartbeat", None)
