"""SQLite audit records and OS process locks (released automatically on exit)."""
from __future__ import annotations

import fcntl
import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def redact(value):
    """Defense in depth: never persist credentials embedded in exception messages."""
    secrets = [os.getenv(name) for name in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "APCA_API_KEY_ID", "APCA_API_SECRET_KEY")]
    if isinstance(value, dict):
        return {k: ("[REDACTED]" if any(s in k.lower() for s in ("secret", "api_key", "authorization")) else redact(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
    return value


def dumps(value) -> str:
    return json.dumps(redact(value), default=str, allow_nan=False, sort_keys=True)


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS controls (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS decisions (id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS snapshots (id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL, body TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL, body TEXT NOT NULL);
            """)

    @contextmanager
    def connection(self):
        con = sqlite3.connect(self.path, timeout=15)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA busy_timeout=15000")
        try:
            yield con
            con.commit()
        except BaseException:
            con.rollback()
            raise
        finally:
            con.close()

    def get_control(self, key, default=None):
        with self.connection() as con:
            row = con.execute("SELECT value FROM controls WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_control(self, key, value):
        with self.connection() as con:
            con.execute("INSERT INTO controls VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, dumps(value)))

    def _save(self, table, body):
        body = dict(body)
        body.setdefault("timestamp", utc_now())
        with self.connection() as con:
            con.execute(f"INSERT INTO {table}(timestamp,body) VALUES (?,?)", (body["timestamp"], dumps(body)))

    def _read(self, table, limit):
        with self.connection() as con:
            rows = con.execute(f"SELECT body FROM {table} ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def event(self, level, message, details=None):
        body = {"timestamp": utc_now(), "level": level, "message": message, "details": details or {}}
        self._save("events", body)
        logging.getLogger("paper_trader").log(getattr(logging, level.upper(), logging.INFO), dumps(body))

    def save_decision(self, body):
        self._save("decisions", body)

    def decisions(self, limit=100):
        return self._read("decisions", limit)

    def save_snapshot(self, body):
        self._save("snapshots", body)

    def snapshots(self, limit=1000):
        return self._read("snapshots", limit)

    def latest_snapshot(self):
        rows = self.snapshots(1)
        return rows[0] if rows else None

    def events(self, limit=100):
        return self._read("events", limit)

    @contextmanager
    def lock(self, name):
        if not name.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Invalid lock name")
        path = self.path.with_suffix(f".{name}.lock")
        with path.open("a+") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Another {name} operation is already running") from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
