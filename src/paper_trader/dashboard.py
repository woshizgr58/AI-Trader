"""Read-only by default local Streamlit console; controls require deliberate clicks."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json

import pandas as pd
import streamlit as st

from paper_trader import app, calendar, market
from paper_trader.config import load_config
from paper_trader.store import Store, redact


def attempt(action):
    try:
        with st.spinner("Working…"):
            result = action()
        st.success("Completed")
        if result:
            st.json(redact(result), expanded=False)
    except Exception as exc:
        st.error(redact(str(exc)))


def forward_curve(snapshots, bars, symbols):
    """Align observed post-close account marks with the same daily benchmark sessions."""
    rows = []
    for snapshot in reversed(snapshots):
        stamp = pd.Timestamp(snapshot["timestamp"])
        day = stamp.tz_convert("America/New_York").date().isoformat()
        try:
            close = calendar.session_close(day)
        except (ValueError, KeyError):
            continue
        if stamp >= close and stamp < close + pd.Timedelta(minutes=30):
            rows.append({"session": day, "equity": float(snapshot["equity"])})
    if not rows or bars.empty:
        return pd.DataFrame()
    account = pd.DataFrame(rows).drop_duplicates("session", keep="last").set_index("session")
    closes = bars[bars.symbol.isin(symbols)].pivot(index="session", columns="symbol", values="close").dropna()
    aligned = account.join(closes, how="inner").dropna()
    if len(aligned) < 2:
        return pd.DataFrame()
    result = pd.DataFrame(index=aligned.index)
    result["Observed paper equity"] = aligned.equity / aligned.equity.iloc[0] - 1
    result["Equal-weight total-return benchmark"] = (aligned[list(symbols)] / aligned[list(symbols)].iloc[0]).mean(axis=1) - 1
    result["Cash (0% interest)"] = 0.0
    return result * 100


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default="config.toml")
    args, _ = parser.parse_known_args()
    st.set_page_config(page_title="Paper Trader", page_icon="◈", layout="wide")
    st.markdown("""<style>
      .block-container {padding-top:2rem;max-width:1440px}
      [data-testid="stMetric"] {background:#1222380c;border:1px solid #8795a330;padding:16px;border-radius:10px}
      [data-testid="stMetricValue"] {font-size:1.5rem}
      [data-testid="stSidebar"] {border-right:1px solid #8795a330}
    </style>""", unsafe_allow_html=True)
    st.title("Paper Trader")
    st.warning("PAPER TRADING — VIRTUAL MONEY")
    st.caption("Daily ETF research · real market data · supervised models · local execution")
    try:
        config = load_config(args.config)
        store = Store(config.db_path)
    except Exception as exc:
        st.error(redact(str(exc)))
        st.stop()
    now = datetime.now(timezone.utc)
    enabled = store.get_control("execution_enabled", config.paper_execution_enabled)
    paused = store.get_control("paused", False)
    with st.sidebar:
        st.subheader("Operational controls")
        st.write(f"**Execution:** {'enabled' if enabled else 'disabled'}")
        st.write(f"**New orders:** {'paused' if paused else 'eligible for risk checks'}")
        if st.button("Disable automatic orders" if enabled else "Enable automatic paper orders", width="stretch"):
            app.set_execution(store, not enabled)
            st.rerun()
        if st.button("Resume new orders" if paused else "Pause new orders", width="stretch"):
            app.set_pause(store, not paused)
            st.rerun()
        st.caption("Pause prevents new submissions. It leaves positions and existing orders in place. Drawdown checks still apply after resume.")
        if st.button("Refresh paper account", disabled=not config.has_credentials, width="stretch"):
            attempt(lambda: app.refresh_account(config, store))
        if st.button("Preview today's orders", disabled=not config.has_credentials, width="stretch"):
            attempt(lambda: app.run_once(config, store, dry_run=True))
        with st.expander("Cancel open paper orders"):
            st.caption("Cancel requests can race with fills. Positions are not liquidated.")
            if st.button("Cancel all open paper orders", disabled=not config.has_credentials):
                attempt(lambda: app.cancel_open_orders(config, store))
        st.divider()
        st.write("**Universe**")
        st.write(" · ".join(config.symbols))
        st.caption(f"Feed: {config.feed.upper()} · max asset {config.max_asset_weight:.0%} · max invested {config.max_total_weight:.0%}")
        st.caption("Keep `paper-trader schedule` running in a terminal. Closing it or sleeping this computer interrupts scheduling.")
        if st.button("Reload dashboard", width="stretch"):
            st.rerun()

    if not config.has_credentials:
        st.info("Connect your paper account: copy .env.example to .env and set ALPACA_API_KEY and ALPACA_SECRET_KEY. No account data or performance is fabricated.")
    pointer_path = config.models_dir / "active.json"
    active = json.loads(pointer_path.read_text()) if pointer_path.exists() else {}
    bars = market.load_bars(config, store)
    snapshot = store.latest_snapshot()
    data_status = market.data_status(config, store)
    top = st.columns(4)
    top[0].metric("Paper connection", "Observed" if snapshot else "Not connected")
    top[1].metric("Data feed", config.feed.upper() if not bars.empty else "No data")
    top[2].metric("Automatic orders", "Paused" if paused else ("Enabled" if enabled else "Disabled"))
    top[3].metric("Active model", active.get("version", "Not promoted"))
    st.caption(f"Last successful decision: {store.get_control('last_successful_run', 'None')}  |  Next session decision: {calendar.next_run(now, config.decision_delay_minutes).isoformat()}  |  All displayed timestamps are UTC unless labeled.")
    heartbeat = store.get_control("scheduler_heartbeat")
    running = heartbeat and (pd.Timestamp(now) - pd.Timestamp(heartbeat)).total_seconds() < 150
    st.caption(f"Scheduler: {'heartbeat observed' if running else 'not running / heartbeat stale'}" + (f" · heartbeat {heartbeat}" if heartbeat else ""))
    if not bars.empty:
        expected = calendar.previous_session(now)
        last_session = str(bars.session.max())
        if last_session != expected:
            st.error(f"Stored market data is stale: {last_session}; required completed session: {expected}. New orders will stop until refreshed.")
        else:
            st.caption(f"Completed data through {last_session} · feed {config.feed.upper()} · corporate-action adjustment: all")
    else:
        st.info("No training data yet. Run `paper-trader fetch`, then `paper-trader train`.")
    if not active:
        st.info("No deployed model. Train and evaluate a saved version, then explicitly promote it.")

    portfolio, decisions_tab, history, forward, operations = st.tabs(["Portfolio", "Decisions", "Historical evaluation", "Forward paper performance", "Operations"])
    with portfolio:
        if not snapshot:
            st.info("Refresh the paper account to see virtual portfolio value, positions, orders, and recent fills.")
        else:
            equity = float(snapshot["equity"])
            positions = snapshot.get("positions", {})
            invested = sum(abs(float(p.get("market_value", 0))) for p in positions.values())
            snapshots = store.snapshots()
            same_account = [s for s in snapshots if s.get("account_id") == snapshot.get("account_id")]
            peak = max([equity, float(store.get_control("equity_peak:" + snapshot["account_id"], equity))] + [float(s["equity"]) for s in same_account])
            metrics = st.columns(4)
            metrics[0].metric("Virtual portfolio", f"${equity:,.2f}")
            metrics[1].metric("Cash", f"${float(snapshot['cash']):,.2f}")
            metrics[2].metric("Exposure", f"{invested / equity:.1%}" if equity > 0 else "Unavailable")
            metrics[3].metric("Observed drawdown", f"{1 - equity / peak:.1%}" if peak > 0 else "Unavailable")
            st.caption(f"Account snapshot: {snapshot['timestamp']} · buying power ${float(snapshot['buying_power']):,.2f}. Snapshot is not a live stream.")
            if (pd.Timestamp(now) - pd.Timestamp(snapshot["timestamp"])).total_seconds() > 180:
                st.warning("Account snapshot is stale. Refresh to retrieve current broker state.")
            st.subheader("Positions")
            if positions:
                st.dataframe(pd.DataFrame.from_dict(positions, orient="index"), width="stretch")
            else:
                st.info("No paper positions.")
            st.subheader("Open orders")
            orders = snapshot.get("open_orders", [])
            if orders:
                st.dataframe(pd.DataFrame(orders), width="stretch")
            else:
                st.info("No open paper orders.")
            recent = snapshot.get("recent_orders", [])
            fills = [o for o in recent if float(o.get("filled_qty", 0) or 0) > 0]
            st.subheader("Recent fills")
            if fills:
                st.dataframe(pd.DataFrame(fills), width="stretch")
            else:
                st.info("No recorded recent fills.")
    with decisions_tab:
        decisions = store.decisions(30)
        usable = [d for d in decisions if "predictions" in d]
        if usable:
            decision = usable[0]
            st.write(f"Signal session **{decision['data_timestamp']}** · model **{decision['model_version']}**")
            rows = []
            for symbol, prediction in decision["predictions"].items():
                weight = decision["targets"].get(symbol, 0)
                rows.append({"Symbol": symbol, "Predicted five-session return": prediction, "Target weight": weight, "Allocation reason": "Prediction exceeds threshold; equal allocation capped by limits" if weight else "Prediction does not exceed threshold; allocation held in cash"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={"Predicted five-session return": st.column_config.NumberColumn(format="percent"), "Target weight": st.column_config.NumberColumn(format="percent")})
            st.caption(f"Threshold: {decision.get('configuration', {}).get('threshold', config.threshold):.2%}. Experimental defaults; prediction quality alone does not establish profitability.")
            with st.expander("Features, risk checks, proposed orders, and broker responses"):
                st.json(decision)
        else:
            st.info("No decisions yet. A run uses only completed bars from the preceding session and is eligible after the next session opens.")
        if decisions:
            st.dataframe(pd.DataFrame([{k: d.get(k) for k in ("timestamp", "session", "status", "dry_run", "error")} for d in decisions]), hide_index=True, width="stretch")
    with history:
        reports = sorted((config.data_dir / "backtests").glob("*/report.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not reports:
            st.info("No historical evaluation. Run `paper-trader evaluate --version VERSION`. The dashboard will show the saved untouched test-window results.")
        else:
            report_path = st.selectbox("Saved historical evaluation", reports, format_func=lambda p: p.parent.name)
            report = json.loads(report_path.read_text())
            st.caption("HISTORICAL BACKTEST · theoretical fills and costs · separate from forward paper results")
            if report.get("metrics"):
                st.dataframe(pd.DataFrame(report["metrics"]).T, width="stretch")
            curves_path = report_path.parent / "curves.csv"
            if curves_path.exists():
                curves = pd.read_csv(curves_path)
                equity_curves = curves.pivot(index="session", columns="portfolio", values="equity")
                st.line_chart((equity_curves / equity_curves.iloc[0] - 1) * 100, y_label="Net return (%)")
            with st.expander("Evaluation assumptions and complete report"):
                st.json(report)
    with forward:
        if not snapshot:
            st.info("No forward paper history. Only observed broker account snapshots will appear here.")
        else:
            snapshots = [s for s in store.snapshots(10000) if s.get("account_id") == snapshot.get("account_id")]
            values = pd.DataFrame([{"timestamp": s["timestamp"], "Virtual equity": s["equity"]} for s in reversed(snapshots)]).drop_duplicates("timestamp").set_index("timestamp")
            st.subheader("Observed virtual account value")
            st.line_chart(values)
            st.caption("Raw account value; deposits, withdrawals, and account resets affect this curve. This is not cash-flow-adjusted strategy attribution.")
            aligned = forward_curve(snapshots, bars, config.symbols)
            st.subheader("Aligned session comparison")
            if aligned.empty:
                st.info("Comparison needs at least two post-close account snapshots (within 30 minutes of the close) with matching completed market bars. Keep the scheduler running through the close, then refresh data on the following session.")
            else:
                st.line_chart(aligned, y_label="Return since common start (%)")
                st.caption(f"Aligned {aligned.index.min()} through {aligned.index.max()}. Account marks sampled within 30 minutes after regular close; benchmark uses provider adjusted daily closes (may include extended hours), fractional total-return normalization, zero fees. Timing differences remain; historical strategy evaluation uses its separately documented costs.")
    with operations:
        st.subheader("Errors and activity")
        events = store.events(100)
        if events:
            st.dataframe(pd.DataFrame(events), hide_index=True, width="stretch")
        else:
            st.info("No operational events recorded.")
        with st.expander("Data-feed metadata"):
            st.json(data_status or {})
        with st.expander("Configuration (credentials excluded)"):
            st.json(config.public_dict())


if __name__ == "__main__":
    main()
