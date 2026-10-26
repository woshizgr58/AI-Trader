"""Deterministic generated fixtures are tests only, never a market-data fallback."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from paper_trader import calendar, market
from paper_trader.config import Config
from paper_trader.store import Store


@pytest.fixture
def config(tmp_path):
    return Config(data_dir=tmp_path, symbols=("SPY", "QQQ"), history_start="2024-01-02", api_key="test-key", secret_key="test-secret")


@pytest.fixture
def store(config):
    return Store(config.db_path)


def fixture_bars(first="2024-01-02", last="2024-01-31", symbols=("SPY", "QQQ")):
    rows = []
    for i, session in enumerate(calendar.sessions(first, last)):
        for j, symbol in enumerate(symbols):
            price = 100 + i + j * 20
            rows.append({"session": session, "symbol": symbol,
                         "timestamp": pd.Timestamp(session, tz="America/New_York").tz_convert("UTC"),
                         "open": float(price), "high": float(price + 2), "low": float(price - 2),
                         "close": float(price + 1), "volume": 100000.0})
    return pd.DataFrame(rows)


class FixtureClient:
    def __init__(self, frame):
        self.frame = frame
        self.requests = []

    def get_stock_bars(self, request):
        self.requests.append(request)
        # alpaca-py normalizes request datetime fields into naive UTC internally.
        begin = pd.Timestamp(request.start).tz_localize("UTC").tz_convert("America/New_York").strftime("%Y-%m-%d")
        end = pd.Timestamp(request.end).tz_localize("UTC").tz_convert("America/New_York").strftime("%Y-%m-%d")
        rows = self.frame[(self.frame.session >= begin) & (self.frame.session <= end)]
        return SimpleNamespace(df=rows.drop(columns="session").set_index(["symbol", "timestamp"]))


def fixed_clock(monkeypatch, iso="2024-02-02T15:00:00+00:00"):
    instant = datetime.fromisoformat(iso)
    original = calendar.utc_now
    monkeypatch.setattr(calendar, "utc_now", lambda now=None: original(instant if now is None else now))


def test_complete_panel_accepts_holiday_and_dst():
    bars = fixture_bars("2024-03-07", "2024-03-12")
    market.validate_bars(bars, ("SPY", "QQQ"), "2024-03-07", "2024-03-12")
    assert str(bars.timestamp.dtype).startswith("datetime64")
    assert bars.iloc[0].timestamp.hour == 5
    assert bars.iloc[-1].timestamp.hour == 4


def test_rejects_stale_missing_duplicate_and_wrong_session_data():
    bars = fixture_bars(last="2024-01-05")
    with pytest.raises(market.DataValidationError, match="Missing or stale"):
        market.validate_bars(bars.iloc[:-1], ("SPY", "QQQ"), "2024-01-02", "2024-01-05")
    with pytest.raises(market.DataValidationError, match="Missing or stale"):
        market.validate_bars(bars[bars.session < "2024-01-05"], ("SPY", "QQQ"), "2024-01-02", "2024-01-05")
    with pytest.raises(market.DataValidationError, match="Duplicate"):
        market.validate_bars(pd.concat([bars, bars.iloc[:1]]), ("SPY", "QQQ"))
    bad = bars.copy()
    bad.loc[0, "session"] = "2024-01-06"
    with pytest.raises(market.DataValidationError, match="malformed"):
        market.validate_bars(bad, ("SPY", "QQQ"))


@pytest.mark.parametrize("column,value", [("close", np.inf), ("open", -1), ("volume", 0), ("high", 1), ("low", 999)])
def test_rejects_malformed_ohlcv(column, value):
    bars = fixture_bars(last="2024-01-05")
    bars.loc[0, column] = value
    with pytest.raises(market.DataValidationError):
        market.validate_bars(bars, ("SPY", "QQQ"))


def test_missing_credentials_never_generate_bars(config, store):
    with pytest.raises(market.MarketDataError, match="ALPACA_API_KEY"):
        market.update_data(replace(config, api_key="", secret_key=""), store, "2024-01-31")
    assert market.load_bars(config, store).empty


def test_initial_and_incremental_updates_persist_real_provider_response(config, store, monkeypatch):
    fixed_clock(monkeypatch)
    client = FixtureClient(fixture_bars())
    monkeypatch.setattr(market, "_client", lambda _: client)
    first = market.update_data(config, store, "2024-01-30")
    assert market.data_status(config, store)["refresh_mode"] == "full"
    second = market.update_data(config, store, "2024-01-31")
    assert len(second) == len(first) + 2
    assert len(client.requests) == 3  # Initial, overlapping increment, oldest sentinel.
    assert client.requests[1].start > client.requests[0].start
    assert client.requests[0].adjustment.value == "all"
    assert client.requests[0].feed.value == "iex"
    assert client.requests[0].limit is None
    assert market.data_status(config, store)["refresh_mode"] == "incremental"
    pd.testing.assert_frame_equal(second, market.load_bars(config, store))


def test_corporate_action_refresh_replaces_entire_price_basis(config, store, monkeypatch):
    fixed_clock(monkeypatch)
    client = FixtureClient(fixture_bars())
    monkeypatch.setattr(market, "_client", lambda _: client)
    market.update_data(config, store, "2024-01-30")
    client.frame.loc[:, ["open", "high", "low", "close"]] *= 0.5
    client.frame.loc[:, "volume"] *= 2
    result = market.update_data(config, store, "2024-01-31")
    assert len(client.requests) == 3  # Changed overlap forces full instead of sentinel.
    assert client.requests[-1].start == client.requests[0].start
    assert market.data_status(config, store)["refresh_mode"] == "full"
    pd.testing.assert_frame_equal(result, client.frame.sort_values(["session", "symbol"]).reset_index(drop=True))
    pd.testing.assert_frame_equal(result, market.load_bars(config, store))


def test_oldest_sentinel_detects_old_adjustment_outside_increment(config, store, monkeypatch):
    fixed_clock(monkeypatch)
    client = FixtureClient(fixture_bars())
    monkeypatch.setattr(market, "_client", lambda _: client)
    market.update_data(config, store, "2024-01-30")
    early = client.frame.session < "2024-01-08"
    client.frame.loc[early, ["open", "high", "low", "close"]] *= 0.98
    market.update_data(config, store, "2024-01-31")
    assert len(client.requests) == 4
    assert market.data_status(config, store)["refresh_mode"] == "full"


def test_failure_does_not_partially_overwrite_good_data(config, store, monkeypatch):
    fixed_clock(monkeypatch)
    client = FixtureClient(fixture_bars())
    monkeypatch.setattr(market, "_client", lambda _: client)
    before = market.update_data(config, store, "2024-01-30")
    client.frame = client.frame.iloc[:-1]
    with pytest.raises(market.DataValidationError, match="Missing or stale"):
        market.update_data(config, store, "2024-01-31")
    pd.testing.assert_frame_equal(before, market.load_bars(config, store))
    assert market.data_status(config, store)["end_session"] == "2024-01-30"


def test_weekly_full_refresh_catches_older_vendor_revisions(config, store, monkeypatch):
    fixed_clock(monkeypatch)
    client = FixtureClient(fixture_bars())
    monkeypatch.setattr(market, "_client", lambda _: client)
    market.update_data(config, store, "2024-01-30")
    fixed_clock(monkeypatch, "2024-02-10T15:00:00+00:00")
    market.update_data(config, store, "2024-01-31")
    assert len(client.requests) == 2
    assert client.requests[-1].start == client.requests[0].start
    assert market.data_status(config, store)["refresh_mode"] == "full"


def test_incomplete_day_is_rejected_even_after_regular_close(config, store, monkeypatch):
    fixed_clock(monkeypatch, "2024-02-01T23:00:00+00:00")
    with pytest.raises(market.DataValidationError, match="incomplete"):
        market.update_data(config, store, "2024-02-01")


@pytest.mark.parametrize("seconds,price,message", [(121, 100, "Stale"), (-6, 100, "Future"), (0, float("nan"), "Malformed")])
def test_stale_future_and_malformed_current_prices_stop_sizing(config, monkeypatch, seconds, price, message):
    now = datetime(2024, 2, 1, 15, tzinfo=timezone.utc)
    client = SimpleNamespace(get_stock_latest_trade=lambda _: {"SPY": SimpleNamespace(price=price, timestamp=now - timedelta(seconds=seconds))})
    monkeypatch.setattr(market, "_client", lambda _: client)
    with pytest.raises(market.DataValidationError, match=message):
        market.latest_prices(config, ["SPY"], now)


def test_latest_prices_use_unadjusted_trade_and_same_feed(config, monkeypatch):
    now = datetime(2024, 2, 1, 15, tzinfo=timezone.utc)

    def response(request):
        assert request.feed.value == config.feed
        return {"SPY": SimpleNamespace(price=481.2, timestamp=now - timedelta(seconds=2))}

    monkeypatch.setattr(market, "_client", lambda _: SimpleNamespace(get_stock_latest_trade=response))
    prices = market.latest_prices(config, ["SPY"], now)
    assert prices["SPY"]["price"] == 481.2
    assert prices["SPY"]["timestamp"] == "2024-02-01T14:59:58+00:00"


def test_provider_errors_do_not_reveal_credentials(config, monkeypatch):
    def response(_):
        raise RuntimeError(f"Connection error {config.api_key} {config.secret_key}")

    monkeypatch.setattr(market, "_client", lambda _: SimpleNamespace(get_stock_latest_trade=response))
    with pytest.raises(market.MarketDataError) as caught:
        market.latest_prices(config, ["SPY"])
    assert config.api_key not in str(caught.value)
    assert config.secret_key not in str(caught.value)


def test_spy_is_fetched_as_reference_without_adding_a_traded_asset(config, store, monkeypatch):
    fixed_clock(monkeypatch)
    config = replace(config, symbols=("QQQ",))
    client = FixtureClient(fixture_bars())
    monkeypatch.setattr(market, "_client", lambda _: client)
    bars = market.update_data(config, store, "2024-01-31")
    assert config.symbols == ("QQQ",)
    assert set(client.requests[0].symbol_or_symbols) == {"QQQ", "SPY"}
    assert set(bars.symbol) == {"QQQ", "SPY"}
    assert set(market.load_bars(config, store).symbol) == {"QQQ", "SPY"}


def test_switching_feeds_does_not_mix_historical_observations(config, store, monkeypatch):
    fixed_clock(monkeypatch)
    client = FixtureClient(fixture_bars())
    monkeypatch.setattr(market, "_client", lambda _: client)
    market.update_data(config, store, "2024-01-31")
    sip_config = replace(config, feed="sip")
    assert market.load_bars(sip_config, store).empty
    assert market.data_status(sip_config, store) == {}


def test_earlier_snapshot_cannot_create_a_mixed_adjustment_basis(config, store, monkeypatch):
    fixed_clock(monkeypatch)
    client = FixtureClient(fixture_bars())
    monkeypatch.setattr(market, "_client", lambda _: client)
    original = market.update_data(config, store, "2024-01-31")
    with pytest.raises(market.DataValidationError, match="newer sessions"):
        market.update_data(config, store, "2024-01-30")
    pd.testing.assert_frame_equal(original, market.load_bars(config, store))


def test_latest_trade_freshness_uses_response_receipt_time(config, monkeypatch):
    started = datetime(2024, 2, 1, 15, tzinfo=timezone.utc)
    clock = {"now": started}

    def delayed_response(_):
        clock["now"] = started + timedelta(seconds=30)
        return {"SPY": SimpleNamespace(price=100.0, timestamp=clock["now"])}

    monkeypatch.setattr(market, "_client", lambda _: SimpleNamespace(get_stock_latest_trade=delayed_response))
    monkeypatch.setattr(calendar, "utc_now", lambda now=None: now or clock["now"])
    assert market.latest_prices(config, ["SPY"])["SPY"]["timestamp"] == clock["now"].isoformat()
