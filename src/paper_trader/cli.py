"""Explicit operational commands. No command can use live trading."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import subprocess
import sys

from .config import load_config
from .store import Store, redact


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="PAPER TRADING — VIRTUAL MONEY")
    parser.add_argument("--config", default="config.toml")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="Fetch/update real completed daily bars")
    fetch.add_argument("--end", help="Last completed session YYYY-MM-DD")
    sub.add_parser("train", help="Train versioned models; does not promote")
    evaluate = sub.add_parser("evaluate", help="Evaluate the saved untouched test window")
    evaluate.add_argument("--version", help="Saved version; default active version")
    promote = sub.add_parser("promote", help="Explicitly select a saved model")
    promote.add_argument("version")
    once = sub.add_parser("run", help="Run one daily decision; defaults to dry-run")
    mode = once.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true", help="Allow paper orders only if execution is also enabled")
    schedule = sub.add_parser("schedule", help="Run the local calendar-aware scheduler")
    schedule.add_argument("--dry-run", action="store_true")
    schedule.add_argument("--poll-seconds", type=int, default=60)
    for name in ("enable", "disable", "pause", "resume", "cancel", "status", "dashboard"):
        sub.add_parser(name)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        config = load_config(args.config)
        store = Store(config.db_path)
        from . import app, backtest, market, models

        if args.command == "fetch":
            with store.lock("trading"):
                bars = market.update_data(config, store, end_session=args.end)
            result = {"rows": len(bars), "last_session": bars.session.max(), "feed": config.feed}
        elif args.command == "train":
            with store.lock("trading"):
                result = models.train(config, market.load_bars(config, store))
        elif args.command == "evaluate":
            with store.lock("trading"):
                result = backtest.evaluate(config, market.load_bars(config, store), version=args.version)
        elif args.command == "promote":
            with store.lock("trading"):
                models.promote(config, args.version)
            result = {"promoted": args.version, "note": "Takes effect on the next unfrozen session decision"}
        elif args.command == "run":
            result = app.run_once(config, store, dry_run=not args.execute)
        elif args.command == "schedule":
            app.scheduler(config, store, poll_seconds=args.poll_seconds, dry_run=args.dry_run)
            return 0
        elif args.command in {"enable", "disable"}:
            app.set_execution(store, args.command == "enable")
            result = {"execution_enabled": args.command == "enable", "paper_only": True}
        elif args.command in {"pause", "resume"}:
            app.set_pause(store, args.command == "pause")
            result = {"paused": args.command == "pause"}
        elif args.command == "cancel":
            result = app.cancel_open_orders(config, store)
        elif args.command == "status":
            result = app.refresh_account(config, store)
        else:
            dashboard = Path(__file__).with_name("dashboard.py")
            return subprocess.call([sys.executable, "-m", "streamlit", "run", str(dashboard), "--server.address=127.0.0.1", "--browser.gatherUsageStats=false", "--", "--config", str(Path(args.config).resolve())])
        print(json.dumps(redact(result), indent=2, default=str, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "error", "message": redact(str(exc))}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
