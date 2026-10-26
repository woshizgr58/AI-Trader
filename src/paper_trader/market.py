"""Real Alpaca daily bars and current trades; fail closed on incomplete data.

Historical data always uses ``adjustment=all`` and an explicitly selected feed.
Incremental refreshes include ten prior sessions and the oldest cached bar. Any
revision triggers a full reload, so a split/dividend cannot leave a mixed price
basis in SQLite. Weekly full reloads also pick up older vendor corrections.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame

from . import calendar

if TYPE_CHECKING:
    from .config import Config
    from .store import Store

BAR_COLUMNS = ["session", "symbol", "timestamp", "open", "high", "low", "close", "volume"]
VALUE_COLUMNS = ["open", "high", "low", "close", "volume"]
ADJUSTMENT = "all"
OVERLAP_SESSIONS = 10
FULL_REFRESH_DAYS = 7


class MarketDataError(RuntimeError):
    """Unavailable provider data, credentials, or entitlement."""


class DataValidationError(MarketDataError):
    """A data set is unsafe for signals or current order sizing."""


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=BAR_COLUMNS)


def data_symbols(config: Config) -> tuple[str, ...]:
    """Traded universe plus SPY, the nontrade broad-market feature reference."""
    return tuple(dict.fromkeys((*config.symbols, "SPY")))


def _client(config: Config) -> StockHistoricalDataClient:
    if not config.api_key or not config.secret_key:
        raise MarketDataError("Set ALPACA_API_KEY and ALPACA_SECRET_KEY to your paper-account credentials before fetching market data.")
    if config.feed not in {"iex", "sip"}:
        raise MarketDataError("Data feed must be iex or sip. Real-time SIP requires the provider's subscription.")
    client = StockHistoricalDataClient(config.api_key, config.secret_key)
    # SDK GET retries are safe; bound transport waits so outages do not hang a run.
    client._retry = 2
    client._retry_wait = 1
    client._retry_codes = [429, 500, 502, 503, 504]
    original_request = client._session.request

    def bounded_request(*args, **kwargs):
        kwargs.setdefault("timeout", (5, 30))
        return original_request(*args, **kwargs)

    client._session.request = bounded_request
    return client


def _provider_error(exc: Exception, operation: str) -> MarketDataError:
    # Do not echo request objects, HTTP bodies, or exception strings containing secrets.
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    status_text = f" (HTTP {status})" if isinstance(status, int) else ""
    return MarketDataError(
        f"Alpaca {operation} failed{status_text}. Check paper credentials, data-feed entitlement, network connectivity and provider status; no generated data was substituted."
    )


def validate_bars(
    bars: pd.DataFrame,
    symbols: tuple[str, ...] | list[str],
    start_session: str | None = None,
    end_session: str | None = None,
) -> None:
    """Require a finite OHLCV row for every symbol and exchange session.

    Explicit bounds enforce freshness and coverage. Without bounds, check the
    complete panel between the earliest and latest rows; do not require a saved
    research data set to reach today's date.
    """
    missing_columns = sorted(set(BAR_COLUMNS) - set(bars.columns))
    if missing_columns:
        raise DataValidationError(f"Daily bars are missing columns: {', '.join(missing_columns)}.")
    if bars.empty:
        raise DataValidationError("No daily bars were returned. Check symbols, date range and feed access.")
    if not symbols or len(set(symbols)) != len(symbols):
        raise DataValidationError("Expected symbols must be a nonempty unique list.")
    if bars[BAR_COLUMNS].isna().any().any():
        raise DataValidationError("Daily bars contain missing values.")
    if set(bars["symbol"]) != set(symbols):
        raise DataValidationError("Daily bars do not contain exactly the configured symbols.")
    if bars.duplicated(["symbol", "session"]).any():
        raise DataValidationError("Duplicate daily bars found for a symbol/session.")
    try:
        labels = bars["session"].map(lambda value: date.fromisoformat(value).isoformat())
        if not labels.equals(bars["session"]):
            raise ValueError("Noncanonical dates")
        for timestamp in bars["timestamp"]:
            parsed = pd.Timestamp(timestamp)
            if parsed.tzinfo is None:
                raise ValueError("Timezone-naive timestamp")
        timestamps = pd.to_datetime(bars["timestamp"], utc=True)
        timestamp_sessions = timestamps.dt.tz_convert(calendar.NEW_YORK).dt.strftime("%Y-%m-%d")
        if not (timestamp_sessions.to_numpy() == labels.to_numpy()).all():
            raise ValueError("Timestamp/session mismatch")
        values = bars[VALUE_COLUMNS].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError, OverflowError):
        raise DataValidationError("Daily bars contain malformed dates, timestamps or numeric values.") from None
    if not np.isfinite(values).all() or (values <= 0).any():
        raise DataValidationError("Daily OHLC prices and volume must be finite and positive.")
    o, h, low, c, _ = values.T
    if ((h < np.maximum(o, c)) | (low > np.minimum(o, c)) | (h < low)).any():
        raise DataValidationError("Daily bars contain impossible OHLC ranges.")
    first = start_session or min(labels)
    last = end_session or max(labels)
    try:
        expected_sessions = calendar.sessions(first, last)
    except (TypeError, ValueError):
        raise DataValidationError("Invalid daily-bar date bounds.") from None
    expected = {(label, symbol) for label in expected_sessions for symbol in symbols}
    actual = set(zip(bars["session"], bars["symbol"]))
    if actual - expected:
        raise DataValidationError("Daily bars include non-session dates or rows outside the requested range.")
    missing = sorted(expected - actual)
    if missing:
        example = ", ".join(f"{symbol} {label}" for label, symbol in missing[:5])
        raise DataValidationError(f"Missing or stale daily bars ({len(missing)} rows; first: {example}). Fetch a complete feed before making decisions.")


def _init_tables(store: Store) -> None:
    with store.connection() as connection:
        connection.execute("""CREATE TABLE IF NOT EXISTS market_bars (
            feed TEXT NOT NULL, adjustment TEXT NOT NULL,
            session TEXT NOT NULL, symbol TEXT NOT NULL, timestamp TEXT NOT NULL,
            open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL,
            close REAL NOT NULL, volume REAL NOT NULL,
            PRIMARY KEY(feed, adjustment, symbol, session))""")
        connection.execute("""CREATE TABLE IF NOT EXISTS market_metadata (
            dataset TEXT PRIMARY KEY, payload TEXT NOT NULL)""")


def load_bars(config: Config, store: Store) -> pd.DataFrame:
    """Load only the configured feed, universe and requested historical range."""
    _init_tables(store)
    placeholders = ",".join("?" for _ in data_symbols(config))
    with store.connection() as connection:
        rows = connection.execute(
            f"SELECT {','.join(BAR_COLUMNS)} FROM market_bars "
            f"WHERE feed=? AND adjustment=? AND symbol IN ({placeholders}) AND session>=? "
            "ORDER BY session,symbol",
            (config.feed, ADJUSTMENT, *data_symbols(config), config.history_start),
        ).fetchall()
    if not rows:
        return _empty()
    result = pd.DataFrame([tuple(row) for row in rows], columns=BAR_COLUMNS)
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    return result


def _dataset_key(config: Config) -> str:
    value = json.dumps([config.feed, ADJUSTMENT, sorted(data_symbols(config)), config.history_start])
    return hashlib.sha256(value.encode()).hexdigest()


def data_status(config: Config, store: Store) -> dict:
    """Persisted provenance and refresh time for the dashboard and audit log."""
    _init_tables(store)
    with store.connection() as connection:
        row = connection.execute("SELECT payload FROM market_metadata WHERE dataset=?", (_dataset_key(config),)).fetchone()
    return json.loads(row[0]) if row else {}


def _fetch(client: StockHistoricalDataClient, config: Config, first: str, last: str) -> pd.DataFrame:
    start = datetime.combine(date.fromisoformat(first), time.min, calendar.NEW_YORK)
    end = datetime.combine(date.fromisoformat(last) + timedelta(days=1), time.min, calendar.NEW_YORK) - timedelta(microseconds=1)
    request = StockBarsRequest(
        symbol_or_symbols=list(data_symbols(config)), timeframe=TimeFrame.Day,
        start=start, end=end, adjustment=Adjustment.ALL, feed=DataFeed(config.feed),
    )
    try:
        # With limit=None, alpaca-py automatically consumes next_page_token.
        response = client.get_stock_bars(request)
        frame = response.df.reset_index()
    except Exception as exc:
        raise _provider_error(exc, "historical-bars request") from None
    if frame.empty:
        raise DataValidationError(f"Alpaca returned no {config.feed} bars for {first} through {last}.")
    try:
        for timestamp in frame["timestamp"]:
            if pd.Timestamp(timestamp).tzinfo is None:
                raise ValueError("Naive provider timestamp")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame["session"] = frame["timestamp"].dt.tz_convert(calendar.NEW_YORK).dt.strftime("%Y-%m-%d")
        frame = frame[BAR_COLUMNS].sort_values(["session", "symbol"]).reset_index(drop=True)
    except (KeyError, TypeError, ValueError):
        raise DataValidationError("Alpaca returned malformed daily bars.") from None
    validate_bars(frame, data_symbols(config), first, last)
    return frame


def _revised(cached: pd.DataFrame, refreshed: pd.DataFrame) -> bool:
    common = cached.merge(refreshed, on=["session", "symbol"], suffixes=("_old", "_new"))
    for column in VALUE_COLUMNS:
        if not np.array_equal(common[f"{column}_old"].to_numpy(), common[f"{column}_new"].to_numpy()):
            return True
    return False


def update_data(config: Config, store: Store, end_session: str | None = None) -> pd.DataFrame:
    """Incrementally fetch verified, fully completed adjusted daily history.

    Nothing is written unless the entire candidate data set passes validation.
    A provider failure keeps the previous cache intact and raises an error.
    """
    now = calendar.utc_now()
    completed = calendar.previous_session(now)
    last = end_session or completed
    try:
        date.fromisoformat(last)
        expected = calendar.sessions(config.history_start, last)
    except (ValueError, TypeError):
        raise DataValidationError("Invalid historical date range.") from None
    if last > completed:
        raise DataValidationError("Requested daily bar is incomplete. Signals use only sessions before today's New York date.")
    if not expected or expected[-1] != last:
        raise DataValidationError("end_session must be a completed exchange session within the history range.")
    client = _client(config)
    with store.lock("market-data"):
        cached = load_bars(config, store)
        if not cached.empty and cached["session"].max() > last:
            raise DataValidationError("The cache already contains newer sessions than end_session. Use a separate data directory for an earlier snapshot, so adjusted price bases cannot be mixed.")
        status = data_status(config, store)
        full_refresh = cached.empty or not status
        if not full_refresh:
            try:
                validate_bars(cached, data_symbols(config), expected[0], cached["session"].max())
                refreshed_at = datetime.fromisoformat(status["last_full_refresh"])
                full_refresh = now - refreshed_at >= timedelta(days=FULL_REFRESH_DAYS)
            except (DataValidationError, KeyError, TypeError, ValueError):
                full_refresh = True
        if full_refresh:
            candidate = _fetch(client, config, expected[0], last)
        else:
            stored_sessions = sorted(cached["session"].unique())
            overlap_start = stored_sessions[max(0, len(stored_sessions) - OVERLAP_SESSIONS)]
            incremental = _fetch(client, config, overlap_start, last)
            revised = _revised(cached, incremental)
            if not revised and stored_sessions[0] < overlap_start:
                sentinel = _fetch(client, config, stored_sessions[0], stored_sessions[0])
                revised = _revised(cached, sentinel)
            if revised:
                candidate = _fetch(client, config, expected[0], last)
                full_refresh = True
            else:
                candidate = pd.concat([cached[cached["session"] < overlap_start], incremental], ignore_index=True)
        validate_bars(candidate, data_symbols(config), expected[0], last)
        candidate = candidate.sort_values(["session", "symbol"]).reset_index(drop=True)
        metadata = {
            "provider": "Alpaca", "feed": config.feed, "adjustment": ADJUSTMENT,
            "symbols": list(data_symbols(config)), "start_session": expected[0], "end_session": last,
            "last_update": now.isoformat(), "rows": len(candidate),
            "last_full_refresh": now.isoformat() if full_refresh else status["last_full_refresh"],
            "refresh_mode": "full" if full_refresh else "incremental",
        }
        records = [
            (config.feed, ADJUSTMENT, row.session, row.symbol, row.timestamp.isoformat(),
             float(row.open), float(row.high), float(row.low), float(row.close), float(row.volume))
            for row in candidate.itertuples(index=False)
        ]
        with store.connection() as connection:
            placeholders = ",".join("?" for _ in data_symbols(config))
            connection.execute(
                f"DELETE FROM market_bars WHERE feed=? AND adjustment=? AND symbol IN ({placeholders}) AND session>=? AND session<=?",
                (config.feed, ADJUSTMENT, *data_symbols(config), config.history_start, last),
            )
            connection.executemany(
                "INSERT OR REPLACE INTO market_bars VALUES (?,?,?,?,?,?,?,?,?,?)", records
            )
            connection.execute(
                "INSERT OR REPLACE INTO market_metadata VALUES (?,?)",
                (_dataset_key(config), json.dumps(metadata, sort_keys=True)),
            )
        store.event("info", "Market data refreshed", metadata)
        return candidate


def latest_prices(config: Config, symbols: tuple[str, ...] | list[str], now: datetime | None = None) -> dict:
    """Current unadjusted traded prices for sizing; reject old or future trades."""
    if not symbols:
        return {}
    client = _client(config)
    request = StockLatestTradeRequest(symbol_or_symbols=list(symbols), feed=DataFeed(config.feed))
    try:
        trades = client.get_stock_latest_trade(request)
    except Exception as exc:
        raise _provider_error(exc, "latest-trades request") from None
    # Assess freshness at receipt. A slow request must not make a current trade
    # appear future-dated relative to the time the request started.
    instant = calendar.utc_now(now)
    result = {}
    for symbol in symbols:
        if symbol not in trades:
            raise DataValidationError(f"No current {config.feed} trade for {symbol}; new orders stopped.")
        trade = trades[symbol]
        try:
            price, timestamp = float(trade.price), pd.Timestamp(trade.timestamp)
            if timestamp.tzinfo is None:
                raise ValueError("Naive trade timestamp")
            timestamp = timestamp.tz_convert("UTC")
            age = (instant - timestamp.to_pydatetime()).total_seconds()
            if not math.isfinite(price) or price <= 0 or not math.isfinite(age):
                raise ValueError("Invalid trade")
        except (TypeError, ValueError, AttributeError, OverflowError):
            raise DataValidationError(f"Malformed current trade for {symbol}; new orders stopped.") from None
        if age < -5:
            raise DataValidationError(f"Future-dated trade for {symbol}; check the local clock.")
        if age > config.max_quote_age_seconds:
            raise DataValidationError(f"Stale {symbol} trade ({age:.0f}s old; limit {config.max_quote_age_seconds}s); wait for fresh data.")
        result[symbol] = {"price": price, "timestamp": timestamp.isoformat()}
    return result
