"""Validated configuration. Credentials never enter persisted configuration."""
from __future__ import annotations

import math
import os
import re
import tomllib
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

PAPER_URL = "https://paper-api.alpaca.markets"


def _bool(value: str) -> bool:
    if value.lower() not in {"true", "false", "1", "0"}:
        raise ValueError("Boolean environment settings must be true or false")
    return value.lower() in {"true", "1"}


def validate_paper_environment() -> None:
    for name in ("APCA_API_BASE_URL", "ALPACA_BASE_URL", "ALPACA_TRADING_URL", "ALPACA_TRADING_ENDPOINT"):
        value = os.environ.get(name)
        if value and value.rstrip("/") != PAPER_URL:
            raise ValueError(f"{name}: only the Alpaca paper endpoint is allowed")
    for name in ("ALPACA_PAPER", "APCA_PAPER"):
        if name in os.environ and not _bool(os.environ[name]):
            raise ValueError(f"{name}: live trading is prohibited")
    for name in ("TRADING_MODE", "ALPACA_TRADING_MODE"):
        if name in os.environ and os.environ[name].lower() != "paper":
            raise ValueError(f"{name}: only paper trading is supported")


@dataclass(frozen=True)
class Config:
    symbols: tuple[str, ...] = ("SPY", "QQQ", "IWM", "DIA")
    data_dir: Path = Path("data")
    history_start: str = "2020-07-27"
    feed: str = "iex"
    threshold: float = 0.002
    max_asset_weight: float = 0.25
    max_total_weight: float = 1.0
    max_order_notional: float = 25000.0
    max_drawdown: float = 0.15
    cash_buffer: float = 0.01
    slippage_bps: float = 10.0
    fee_bps: float = 1.0
    initial_cash: float = 100000.0
    seed: int = 42
    decision_delay_minutes: int = 5
    max_quote_age_seconds: int = 120
    strategy_id: str = "five-day-etf-v1"
    api_key: str = field(default="", repr=False)
    secret_key: str = field(default="", repr=False)
    paper_execution_enabled: bool = False

    def __post_init__(self) -> None:
        validate_paper_environment()
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        object.__setattr__(self, "symbols", tuple(self.symbols))
        if not self.symbols or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("symbols must be a nonempty unique universe")
        if any(not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", s) for s in self.symbols):
            raise ValueError("symbols must be uppercase US equity ticker symbols")
        if self.feed not in {"iex", "sip"}:
            raise ValueError("feed must be iex or sip; delayed feeds are unsuitable for execution")
        date.fromisoformat(self.history_start)
        for name in ("max_asset_weight", "max_total_weight", "max_drawdown"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if not 0 <= self.cash_buffer < 1:
            raise ValueError("cash_buffer must be in [0, 1)")
        for name in ("initial_cash", "max_order_notional", "max_quote_age_seconds"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("slippage_bps", "fee_bps"):
            if not math.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) < 1000:
                raise ValueError(f"{name} must be in [0, 1000)")
        if not math.isfinite(self.threshold):
            raise ValueError("threshold must be finite")
        if not isinstance(self.decision_delay_minutes, int) or not 1 <= self.decision_delay_minutes < 180:
            raise ValueError("decision_delay_minutes must be an integer in [1, 180)")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", self.strategy_id):
            raise ValueError("strategy_id must be 1–40 letters, numbers, underscores or hyphens")

    @property
    def db_path(self) -> Path:
        return self.data_dir / "paper_trader.sqlite3"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def has_credentials(self) -> bool:
        return bool(self.api_key and self.secret_key)

    def public_dict(self) -> dict:
        values = asdict(self)
        values.pop("api_key")
        values.pop("secret_key")
        values["data_dir"] = str(self.data_dir)
        return values


def load_config(path: str | Path = "config.toml") -> Config:
    path = Path(path).resolve()
    load_dotenv(path.parent / ".env", override=False)
    validate_paper_environment()
    values = tomllib.loads(path.read_text()) if path.exists() else {}
    for forbidden in ("api_key", "secret_key", "paper_execution_enabled"):
        if forbidden in values:
            raise ValueError(f"{forbidden} must be configured through environment variables or operational controls")
    allowed = set(Config.__dataclass_fields__) - {"api_key", "secret_key", "paper_execution_enabled"}
    if set(values) - allowed:
        raise ValueError("Unsupported configuration keys (endpoint overrides and live mode are prohibited): " + ", ".join(sorted(set(values) - allowed)))
    if "data_dir" in values:
        values["data_dir"] = path.parent / values["data_dir"]
    else:
        values["data_dir"] = path.parent / "data"
    values["api_key"] = os.getenv("ALPACA_API_KEY", os.getenv("APCA_API_KEY_ID", ""))
    values["secret_key"] = os.getenv("ALPACA_SECRET_KEY", os.getenv("APCA_API_SECRET_KEY", ""))
    values["paper_execution_enabled"] = _bool(os.getenv("PAPER_EXECUTION_ENABLED", "false"))
    return Config(**values)
