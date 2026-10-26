"""Generated bars in this module are fixtures only, never application market data."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from paper_trader.backtest import evaluate, simulate
from paper_trader.config import Config
from paper_trader.calendar import sessions as exchange_sessions
from paper_trader.features import FEATURE_COLUMNS, build_features
from paper_trader.models import chronological_split, load_active, load_version, predict_active, promote, train
from paper_trader.strategy import plan_orders, target_weights


def fixture_bars(count=280, symbols=("SPY", "QQQ")):
    sessions = pd.to_datetime(exchange_sessions("2020-01-02", "2022-12-31")[:count])
    rows = []
    for index, symbol in enumerate(symbols):
        for day, session in enumerate(sessions):
            close = 100 + index * 25 + day * .03 + 2 * np.sin(day / 9 + index)
            rows.append({"session": session.date().isoformat(), "symbol": symbol,
                         "timestamp": session.tz_localize("UTC").isoformat(),
                         "open": close * .998, "high": close * 1.01, "low": close * .99,
                         "close": close, "volume": 10000 + day * 10 + 100 * index})
    return pd.DataFrame(rows)


@pytest.fixture
def config(tmp_path):
    return Config(data_dir=tmp_path, symbols=("SPY", "QQQ"), max_order_notional=100000)


def snapshot(cash=10000, positions=None, equity=None, orders=None):
    positions = positions or {}
    return {"cash": cash, "buying_power": cash, "equity": equity or cash + sum(p["market_value"] for p in positions.values()),
            "positions": positions, "open_orders": orders or []}


def test_features_are_causal_and_labels_five_sessions_forward():
    bars = fixture_bars(100)
    features = build_features(bars)
    for symbol in bars.symbol.unique():
        group = features.loc[features.symbol == symbol].reset_index(drop=True)
        assert group.loc[70, "target"] == pytest.approx(group.loc[75, "close"] / group.loc[70, "close"] - 1)
        assert group.loc[70, "label_end"] == group.loc[75, "session"]
        assert group.tail(5).target.isna().all()
        assert group.tail(5).label_end.isna().all()
    cutoff = sorted(bars.session.unique())[80]
    truncated = build_features(bars.loc[bars.session <= cutoff])
    expected = features.loc[features.session <= cutoff, ["session", "symbol"] + FEATURE_COLUMNS].reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, truncated[["session", "symbol"] + FEATURE_COLUMNS])


def test_feature_grid_rejects_missing_and_duplicate_rows():
    bars = fixture_bars()
    with pytest.raises(ValueError, match="Missing asset"):
        build_features(bars.iloc[1:])
    missing_day = bars.loc[bars.session != sorted(bars.session.unique())[50]]
    with pytest.raises(ValueError, match="every exchange session"):
        build_features(missing_day)
    with pytest.raises(ValueError, match="Duplicate"):
        build_features(pd.concat([bars, bars.iloc[:1]]))


def test_chronological_partitions_purge_five_session_future_overlap():
    split = chronological_split(build_features(fixture_bars()))
    assert split["train"].label_end.max() < split["validation"].session.min()
    assert split["validation"].label_end.max() < split["test"].session.min()
    assert set(split["train"].session).isdisjoint(split["test"].session)
    assert set(split["validation"].session).isdisjoint(split["test"].session)
    assert split["test"].target.notna().all()


def test_training_scaler_and_explicit_fixed_promotion(config):
    bars = fixture_bars()
    metadata = train(config, bars)
    assert metadata["test_evaluated"] is False
    assert all(set(item) == {"validation"} for item in metadata["evaluation"].values())
    with pytest.raises(ValueError, match="No active"):
        load_active(config)
    artifact, _ = load_version(config, metadata["version"])
    expected = chronological_split(build_features(bars))["train"][FEATURE_COLUMNS].mean().to_numpy()
    for candidate in artifact["candidates"].values():
        np.testing.assert_allclose(candidate.named_steps["scale"].mean_, expected)
    promote(config, metadata["version"])
    second = train(config, bars)
    assert second["version"] != metadata["version"]
    assert load_active(config)[1]["version"] == metadata["version"]
    predictions, reference, active = predict_active(config, bars, bars.session.max())
    assert set(predictions) == set(config.symbols)
    assert active["version"] == metadata["version"]
    assert reference["signal_session"] == bars.session.max()
    with pytest.raises(ValueError, match="leak future"):
        predict_active(config, bars, metadata["partitions"]["train"]["start"])


def test_test_data_cannot_influence_training_or_selection(config):
    bars = fixture_bars()
    first = train(config, bars)
    changed = bars.copy()
    after = changed.session >= first["partitions"]["test"]["start"]
    changed.loc[after, ["open", "high", "low", "close"]] *= 1.8
    second = train(config, changed)
    assert first["selected_model"] == second["selected_model"]
    assert first["evaluation"] == second["evaluation"]
    a, _ = load_version(config, first["version"])
    b, _ = load_version(config, second["version"])
    inputs = build_features(bars).dropna(subset=FEATURE_COLUMNS)[FEATURE_COLUMNS]
    for name in a["candidates"]:
        np.testing.assert_array_equal(a["candidates"][name].predict(inputs), b["candidates"][name].predict(inputs))


def test_backtest_heldout_alignment_and_artifacts(config):
    bars = fixture_bars()
    metadata = train(config, bars)
    report = evaluate(config, bars, metadata["version"])
    assert report["test_signal_start"] == metadata["partitions"]["test"]["start"]
    assert report["execution_start"] > report["test_signal_start"]
    assert report["execution_end"] > report["test_signal_end"]
    assert report["unlabelled_signal_rows_excluded"] == 5 * len(config.symbols)
    curves = pd.read_csv(report["curve_path"])
    session_lists = [group.session.tolist() for _, group in curves.groupby("portfolio")]
    assert all(sessions == session_lists[0] for sessions in session_lists)
    assert report["metrics"]["cash"]["net_return"] == 0
    assert report["metrics"]["cash"]["sharpe"] is None
    assert report["current_data_matches_snapshot"]
    assert json.loads((config.data_dir / "backtests" / "latest.json").read_text())["version"] == metadata["version"]


def test_signal_execution_is_next_session_and_costs_are_charged(config):
    bars = fixture_bars(70)
    dates = sorted(bars.session.unique())
    cfg = replace(config, threshold=-1, fee_bps=10, slippage_bps=20)
    targets = {dates[65]: {"SPY": .25, "QQQ": 0}, dates[66]: {"SPY": 0, "QQQ": 0}}
    curve, trades = simulate(cfg, bars, targets, dates[66:68])
    assert trades[0]["signal_session"] == dates[65]
    assert trades[0]["execution_session"] == dates[66]
    actual_open = bars.loc[(bars.session == dates[66]) & (bars.symbol == "SPY"), "open"].iloc[0]
    assert trades[0]["estimated_price"] >= actual_open * 1.002
    assert all(trade["fee"] > 0 for trade in trades)
    assert curve.fees.sum() == pytest.approx(sum(trade["fee"] for trade in trades))


def test_allocation_threshold_equality_and_caps(config):
    assert target_weights({"SPY": config.threshold, "QQQ": -.1}, config) == {"SPY": 0, "QQQ": 0}
    assert target_weights({"SPY": .1, "QQQ": .2}, config) == {"SPY": .25, "QQQ": .25}
    with pytest.raises(ValueError, match="finite"):
        target_weights({"SPY": np.nan, "QQQ": .1}, config)


def test_cash_exposure_and_order_limits_are_respected(config):
    cfg = replace(config, max_asset_weight=.5, max_total_weight=.7, cash_buffer=.1, max_order_notional=1234, fee_bps=10)
    state = snapshot(cash=2000, equity=10000)
    state["buying_power"] = 1500
    plan = plan_orders({"SPY": .35, "QQQ": .35}, state, {"SPY": 101.12, "QQQ": 75.01}, cfg)
    spend = sum(order["estimated_notional"] * 1.001 for order in plan["orders"])
    assert spend <= 500
    assert all(isinstance(order["qty"], int) and order["qty"] > 0 for order in plan["orders"])
    assert all(order["estimated_notional"] <= 1234 for order in plan["orders"])
    assert plan["projected_exposure"] <= .7


def test_sale_proceeds_never_increase_buying_budget(config):
    cfg = replace(config, max_asset_weight=1, max_total_weight=1, cash_buffer=0)
    positions = {"SPY": {"qty": 100, "market_value": 10000, "current_price": 100}}
    plan = plan_orders({"SPY": 0, "QQQ": 1}, snapshot(cash=0, positions=positions), {"SPY": 100, "QQQ": 100}, cfg)
    assert [order["side"] for order in plan["orders"]] == ["sell"]
    assert plan["risk_checks"]["sale_proceeds_assumed"] is False


def test_partial_pending_buy_reserves_cash_and_prevents_duplicate_quantity(config):
    cfg = replace(config, slippage_bps=0, fee_bps=0, cash_buffer=0)
    positions = {"SPY": {"qty": 3, "market_value": 300, "current_price": 100}}
    orders = [{"symbol": "SPY", "side": "buy", "qty": 10, "filled_qty": 3, "limit_price": 100, "status": "partially_filled"}]
    state = snapshot(cash=3700, positions=positions, orders=orders)
    plan = plan_orders({"SPY": .25, "QQQ": 0}, state, {"SPY": 100, "QQQ": 100}, cfg)
    assert plan["orders"] == []
    assert plan["risk_checks"]["pending_buy_reserve"] == 700
    assert plan["projected_cash"] == 3000


def test_rejected_pending_order_no_longer_reserves_quantity(config):
    cfg = replace(config, slippage_bps=0, fee_bps=0, cash_buffer=0)
    state = snapshot(cash=4000, orders=[{"symbol": "SPY", "side": "buy", "qty": 10, "filled_qty": 0, "status": "rejected"}])
    plan = plan_orders({"SPY": .25, "QQQ": 0}, state, {"SPY": 100, "QQQ": 100}, cfg)
    assert plan["orders"][0]["qty"] == 10


def test_drawdown_pause_never_liquidates_or_resumes(config):
    bars = fixture_bars(70)
    dates = sorted(bars.session.unique())
    bars.loc[(bars.session >= dates[66]) & (bars.symbol == "SPY"), ["open", "close"]] = 100
    bars.loc[(bars.session >= dates[67]) & (bars.symbol == "SPY"), ["open", "close"]] = 40
    cfg = replace(config, max_asset_weight=1, cash_buffer=0, fee_bps=0, slippage_bps=0)
    targets = {date: {"SPY": 1, "QQQ": 0} for date in dates[65:69]}
    curve, trades = simulate(cfg, bars, targets, dates[66:70])
    assert curve.paused.iloc[-1]
    assert len(trades) == 1
    assert curve.exposure.iloc[-1] > 0


def test_fee_costs_cannot_push_new_positions_through_exposure_cap(config):
    cfg = replace(config, symbols=("SPY", "QQQ"), max_asset_weight=.5,
                  max_total_weight=.6, cash_buffer=0, fee_bps=100, slippage_bps=0)
    plan = plan_orders({"SPY": .3, "QQQ": .3}, snapshot(cash=10000), {"SPY": 1, "QQQ": 1}, cfg)
    gross = sum(order["estimated_notional"] for order in plan["orders"])
    fee = gross * .01
    assert gross / (10000 - fee) <= .6
    assert all(order["estimated_notional"] / (10000 - fee) <= .5 for order in plan["orders"])


def test_custom_trade_universe_can_use_spy_reference_without_trading_it(config):
    cfg = replace(config, symbols=("QQQ",))
    bars = fixture_bars()
    metadata = train(cfg, bars)
    promote(cfg, metadata["version"])
    predictions, _, _ = predict_active(cfg, bars, bars.session.max())
    assert set(predictions) == {"QQQ"}
    report = evaluate(cfg, bars, metadata["version"])
    assert all(trade["symbol"] == "QQQ" for entries in report["trades"].values() for trade in entries)


def test_unfilled_buys_are_never_sellable_inventory(config):
    cfg = replace(config, slippage_bps=0, fee_bps=0, cash_buffer=0)
    pending = [{"symbol": "SPY", "side": "buy", "qty": 10, "filled_qty": 0,
                "limit_price": 100, "status": "new"}]
    plan = plan_orders({"SPY": 0, "QQQ": 0}, snapshot(cash=4000, orders=pending),
                       {"SPY": 100, "QQQ": 100}, cfg)
    assert plan["orders"] == []
    positions = {"SPY": {"qty": 3, "market_value": 300, "current_price": 100}}
    plan = plan_orders({"SPY": 0, "QQQ": 0}, snapshot(cash=3700, positions=positions, orders=pending),
                       {"SPY": 100, "QQQ": 100}, cfg)
    assert plan["orders"][0]["qty"] == 3
    pending.append({"symbol": "SPY", "side": "sell", "qty": 4, "filled_qty": 0, "status": "new"})
    with pytest.raises(ValueError, match="Outstanding sells exceed"):
        plan_orders({"SPY": 0, "QQQ": 0}, snapshot(cash=3700, positions=positions, orders=pending),
                    {"SPY": 100, "QQQ": 100}, cfg)


def test_new_buy_exposure_uses_conservative_account_mark(config):
    cfg = replace(config, slippage_bps=0, fee_bps=0, cash_buffer=0)
    positions = {"SPY": {"qty": 10, "market_value": 2000, "current_price": 200}}
    plan = plan_orders({"SPY": .25, "QQQ": 0}, snapshot(cash=8000, positions=positions),
                       {"SPY": 100, "QQQ": 100}, cfg)
    assert plan["orders"][0]["qty"] == 2
    assert plan["projected_exposure"] == pytest.approx(.24)


def test_training_rejects_missing_entire_configured_asset(config):
    with pytest.raises(ValueError, match="missing configured assets"):
        train(config, fixture_bars(symbols=("SPY",)))
