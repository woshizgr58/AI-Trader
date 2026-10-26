"""Causal daily features. The five-session label is never an input feature."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .calendar import sessions

HORIZON = 5
FEATURE_COLUMNS = [
    "return_1", "return_5", "return_20", "volatility_20",
    "ma_distance_20", "ma_distance_60", "relative_volume_20",
    "market_return_1", "market_return_5",
]
SCHEMA_VERSION = 1


def data_fingerprint(bars: pd.DataFrame) -> str:
    """Stable checksum of economic inputs, independent of incoming row order."""
    cols = ["session", "symbol", "open", "high", "low", "close", "volume"]
    ordered = bars[cols].sort_values(["session", "symbol"]).reset_index(drop=True)
    payload = ordered.to_csv(index=False, float_format="%.12g").encode()
    return hashlib.sha256(payload).hexdigest()


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    required = {"session", "symbol", "close", "volume"}
    if missing := required.difference(bars.columns):
        raise ValueError(f"Bars are missing required columns: {sorted(missing)}")
    if bars.empty:
        raise ValueError("No historical bars. Fetch real market data before training.")
    frame = bars.copy().sort_values(["symbol", "session"]).reset_index(drop=True)
    frame["session"] = frame["session"].astype(str)
    if frame.duplicated(["session", "symbol"]).any():
        raise ValueError("Duplicate symbol/session bars cannot be used for features.")
    values = frame[["close", "volume"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (frame["close"] <= 0).any() or (frame["volume"] < 0).any():
        raise ValueError("Feature inputs contain malformed prices or volumes.")
    observed_sessions = sorted(frame["session"].unique())
    if observed_sessions != sessions(observed_sessions[0], observed_sessions[-1]):
        raise ValueError("Historical bars must contain every exchange session; refresh missing or invalid sessions.")
    # A common grid is essential: shift(-5) must mean the same sessions for all assets.
    counts = frame.groupby("session")["symbol"].nunique()
    if (counts != frame["symbol"].nunique()).any():
        raise ValueError("Missing asset bars on the common session grid; refresh market data.")
    pieces = []
    for _, part in frame.groupby("symbol", sort=True):
        part = part.copy()
        close = part["close"]
        for lag in (1, 5, 20):
            part[f"return_{lag}"] = close.pct_change(lag, fill_method=None)
        part["volatility_20"] = part["return_1"].rolling(20, min_periods=20).std(ddof=1)
        for window in (20, 60):
            part[f"ma_distance_{window}"] = close / close.rolling(window, min_periods=window).mean() - 1
        avg_volume = part["volume"].rolling(20, min_periods=20).mean()
        part["relative_volume_20"] = part["volume"] / avg_volume.replace(0, np.nan)
        part["target"] = close.shift(-HORIZON) / close - 1
        part["label_end"] = part["session"].shift(-HORIZON)
        pieces.append(part)
    result = pd.concat(pieces, ignore_index=True)
    # SPY is ingested as a market reference even if it is outside the traded universe.
    if "SPY" not in set(result["symbol"]):
        raise ValueError("SPY bars are required for broad-market features (also for custom universes).")
    market = result.loc[result["symbol"] == "SPY", ["session", "return_1", "return_5"]]
    market = market.rename(columns={"return_1": "market_return_1", "return_5": "market_return_5"})
    result = result.merge(market, on="session", validate="many_to_one")
    result[FEATURE_COLUMNS] = result[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return result.sort_values(["session", "symbol"]).reset_index(drop=True)
