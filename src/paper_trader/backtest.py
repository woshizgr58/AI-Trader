"""Held-out historical evaluation. Never submits orders or changes the active model."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, build_features, data_fingerprint
from .models import _write_json, load_active, load_version, prediction_metrics
from .strategy import plan_orders, target_weights


def performance_metrics(curve):
    equity = curve["equity"].astype(float)
    returns = equity.pct_change().dropna()
    standard_deviation = returns.std(ddof=1)
    sharpe = float(np.sqrt(252) * returns.mean() / standard_deviation) if len(returns) > 1 and standard_deviation > 1e-12 else None
    drawdown = equity / equity.cummax() - 1
    return {"net_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
            "max_drawdown": float(-drawdown.min()), "sharpe": sharpe,
            "annualized_volatility": float(standard_deviation * np.sqrt(252)) if len(returns) > 1 else None,
            "turnover": float(curve["traded_notional"].sum() / equity.iloc[0]),
            "average_exposure": float(curve["exposure"].iloc[1:].mean()),
            "ending_equity": float(equity.iloc[-1]), "sessions": len(returns),
            "fees": float(curve["fees"].sum()), "paused": bool(curve["paused"].any())}


def simulate(config, bars, signal_targets, execution_sessions, *, buy_and_hold=False):
    """Completed signal t executes on session t+1, approximated by its daily open.

    Both live and historical execution use plan_orders. Historical limits are
    assumed filled at their adverse slippage price and charged configured fees.
    Existing holdings are marked to each daily close; no terminal liquidation.
    """
    sessions = sorted(bars.session.unique())
    previous = {sessions[i]: sessions[i - 1] for i in range(1, len(sessions))}
    indexed = bars.set_index(["session", "symbol"])
    cash = float(config.initial_cash)
    quantities = {s: 0 for s in config.symbols}
    high_water = cash
    paused = False
    first_signal = previous[execution_sessions[0]]
    curve = [{"session": first_signal, "signal_session": None, "equity": cash, "cash": cash,
              "exposure": 0.0, "traded_notional": 0.0, "fees": 0.0, "paused": False}]
    trades = []
    for count, session in enumerate(execution_sessions):
        signal_session = previous[session]
        opens = {s: float(indexed.loc[(session, s), "open"]) for s in config.symbols}
        closes = {s: float(indexed.loc[(session, s), "close"]) for s in config.symbols}
        equity = cash + sum(quantities[s] * opens[s] for s in config.symbols)
        if 1 - equity / high_water >= config.max_drawdown:
            paused = True
        targets = signal_targets.get(signal_session)
        traded_notional = fees = 0.0
        if targets is not None and not paused and (not buy_and_hold or count == 0):
            snapshot = {"equity": equity, "cash": cash, "buying_power": cash,
                        "positions": {s: {"qty": quantities[s], "market_value": quantities[s] * opens[s], "current_price": opens[s]} for s in config.symbols},
                        "open_orders": []}
            planned = plan_orders(targets, snapshot, opens, config)
            for order in planned["orders"]:
                notional = order["qty"] * order["estimated_price"]
                fee = notional * config.fee_bps / 10000
                if order["side"] == "buy":
                    cash -= notional + fee
                    quantities[order["symbol"]] += order["qty"]
                else:
                    cash += notional - fee
                    quantities[order["symbol"]] -= order["qty"]
                traded_notional += notional
                fees += fee
                trades.append({"signal_session": signal_session, "execution_session": session, **order, "fee": fee})
        if cash < -1e-7 or min(quantities.values()) < 0:
            raise RuntimeError("Backtest cash or long-only invariant failed.")
        invested = sum(quantities[s] * closes[s] for s in config.symbols)
        equity = cash + invested
        high_water = max(high_water, equity)
        if 1 - equity / high_water >= config.max_drawdown:
            paused = True
        curve.append({"session": session, "signal_session": signal_session, "equity": equity, "cash": cash,
                      "exposure": invested / equity, "traded_notional": traded_notional, "fees": fees, "paused": paused})
    return pd.DataFrame(curve), trades


def evaluate(config, bars, version=None):
    if version is None:
        _, active_metadata = load_active(config)
        version = active_metadata["version"]
    artifact, metadata = load_version(config, version)
    if list(config.symbols) != metadata["symbols"] or config.feed != metadata["config"]["feed"]:
        raise ValueError("Evaluation universe/feed must match the model training configuration.")
    snapshot_path = Path(config.models_dir) / version / "bars.csv.gz"
    saved = pd.read_csv(snapshot_path, dtype={"session": str, "symbol": str}, float_precision="round_trip")
    if data_fingerprint(saved) != metadata["data_fingerprint"]:
        raise ValueError("Saved training data checksum mismatch; cannot reproduce held-out evaluation.")
    features = build_features(saved)
    test_sessions = metadata["partitions"]["test"]["sessions"]
    test = features.loc[features.session.isin(test_sessions) & features.symbol.isin(config.symbols)].copy()
    if test[FEATURE_COLUMNS + ["target"]].isna().any().any():
        raise ValueError("Held-out test contains unavailable labels/features.")
    test_results = {}
    for name, model in artifact["candidates"].items():
        predictions = model.predict(test[FEATURE_COLUMNS])
        test_results[name] = prediction_metrics(test.target, predictions)
        if name == artifact["selected"]:
            test["prediction"] = predictions
    allocations = {str(session): target_weights(dict(zip(frame.symbol, frame.prediction, strict=True)), config)
                   for session, frame in test.groupby("session", sort=True)}
    all_sessions = sorted(saved.session.unique())
    following = {all_sessions[i]: all_sessions[i + 1] for i in range(len(all_sessions) - 1)}
    execution_sessions = [following[session] for session in sorted(allocations)]
    if not execution_sessions:
        raise ValueError("No held-out sessions with available next-session execution bars.")
    equal = {s: min(config.max_asset_weight, config.max_total_weight / len(config.symbols)) for s in config.symbols}
    trend = {}
    for session, group in test.groupby("session", sort=True):
        eligible = set(group.loc[group.ma_distance_60 > 0, "symbol"])
        weight = min(config.max_asset_weight, config.max_total_weight / len(eligible)) if eligible else 0.0
        trend[str(session)] = {s: weight if s in eligible else 0.0 for s in config.symbols}
    targets = {"strategy": allocations,
               "cash": {session: {s: 0.0 for s in config.symbols} for session in allocations},
               "equal_weight_buy_hold": {session: equal for session in allocations}, "trend": trend}
    metrics, curves, trades = {}, [], {}
    for name, weights in targets.items():
        curve, entries = simulate(config, saved, weights, execution_sessions, buy_and_hold=name == "equal_weight_buy_hold")
        metrics[name] = performance_metrics(curve)
        curve["portfolio"] = name
        curves.append(curve)
        trades[name] = entries
    directory = Path(config.data_dir) / "backtests" / version
    directory.mkdir(parents=True, exist_ok=True)
    curve_path, report_path = directory / "curves.csv", directory / "report.json"
    pd.concat(curves, ignore_index=True).to_csv(curve_path, index=False)
    current_matches = data_fingerprint(bars.loc[bars.session <= metadata["data_end"]]) == metadata["data_fingerprint"] if not bars.empty else False
    report = {
        "kind": "historical_held_out_backtest", "version": version, "selected_model": artifact["selected"],
        "created_at": datetime.now(timezone.utc).isoformat(), "config": config.public_dict(),
        "test_signal_start": min(allocations), "test_signal_end": max(allocations),
        "execution_start": execution_sessions[0], "execution_end": execution_sessions[-1],
        "metrics": metrics, "prediction_metrics": test_results, "trades": trades,
        "source": "Immutable historical snapshot saved at training; no generated data.",
        "data_fingerprint": metadata["data_fingerprint"], "current_data_matches_snapshot": current_matches,
        "unlabelled_signal_rows_excluded": int(features.loc[features.symbol.isin(config.symbols), "target"].isna().sum()),
        "curve_path": str(curve_path), "report_path": str(report_path),
        "assumptions": [
            "Completed session t features execute at session t+1 daily open, plus adverse slippage and fees. Live runs after open+configured delay; daily open is only an approximation of that executable price.",
            "Historical limit orders are assumed completely filled at cent-rounded adverse slippage prices. No intraday liquidity, partial fills, or queue simulation; forward paper orders may differ.",
            "Whole shares, cash buffer, order notional/exposure limits, no unconfirmed sale proceeds, and persistent drawdown pause use the live sizing policy. Pause never liquidates positions.",
            "Equal-weight buy-and-hold purchases once at the first execution open, subject to the same initial caps/cash/whole-share limits; it never rebalances. Its realized starting weights can differ from equal targets.",
            "Trend holds assets above their trailing 60-session mean, equally weighted within the same limits. Cash earns zero interest.",
            "All curves share the initial cash point and identical execution sessions. Final five signal sessions without labels are excluded. No forced terminal liquidation or exit costs.",
            "Sharpe uses daily net equity returns, zero risk-free rate, sample standard deviation and sqrt(252); it is not a significance test. Five-session prediction labels overlap within each partition.",
            "Turnover is gross purchases plus sales divided by initial equity; exposure is the arithmetic mean of invested/equity at daily closes. Returns and drawdown include fees and slippage.",
            "Provider split/dividend-adjusted OHLC prices and adjusted share units approximate dividend reinvestment, not actual raw-share corporate-action cashflows. No extra dividend cash is credited; historical whole-share rounding is approximate. Live sizing uses current unadjusted quotes and actual broker positions.",
            "The test was excluded from fitting and model selection. Repeated inspection or configuration tuning on this report consumes that holdout; obtain a new future evaluation period for new claims.",
        ],
    }
    _write_json(report_path, report)
    _write_json(Path(config.data_dir) / "backtests" / "latest.json", {"version": version, "report_path": str(report_path), "curve_path": str(curve_path)})
    metadata["test_evaluated"] = True
    metadata["test_evaluated_at"] = report["created_at"]
    _write_json(Path(config.models_dir) / version / "metadata.json", metadata)
    return report
