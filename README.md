# AI Paper Trading Platform

**PAPER TRADING — VIRTUAL MONEY**

A local Python application that ingests actual Alpaca market data, trains small supervised models, evaluates a daily allocation strategy, and can place simulated orders through Alpaca's paper API. SQLite holds data and operational records; Streamlit displays the account, decisions, evaluations, and controls. There is no live-trading mode, generated market-data fallback, cloud infrastructure, or LLM dependency.

Order submission is **disabled by default** for new installations. The application supports real IEX data ingestion, chronological model training and evaluation, dry-run previews, and paper-order execution with persistent reconciliation. Automated tests use labeled fixtures and mocked broker responses; historical evaluations and forward paper-account results are tracked separately. See [verification notes](docs/VERIFICATION.md) for dated test and evaluation results.

## Features

- Daily allocation across SPY, QQQ, IWM, and DIA using actual market data.
- Ridge and random forest models with nine causal features and purged chronological splits.
- Historical evaluation against cash, equal-weight buy-and-hold, and trend benchmarks.
- Durable order intents, duplicate-order prevention, position limits, and restart recovery.
- Local dashboard for account balances, positions, orders, predictions, and trading controls.
- Exchange-calendar scheduling with holiday, early-close, and daylight-saving handling.

## Setup

Python 3.12 on macOS or Linux is required. Process locking uses `fcntl`; native Windows is not supported (use WSL). Dependencies were resolved and tested on macOS arm64. From this directory:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
cp .env.example .env
```

Set `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` in `.env` to keys from an **Alpaca paper account**. The supported `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` aliases also work. Existing environment variables override `.env`. Keep `.env` private; it and all local data are gitignored. Never paste credentials into code or `config.toml`.

The supported SDK is `alpaca-py`, pinned with the scientific and dashboard dependencies in `pyproject.toml`. `requirements.lock` pins the complete resolved application/test dependency set. Install the lock before the package for reproducibility. The build backend is separately pinned in `pyproject.toml`.

## Complete workflow

Run commands from the repository root with the virtual environment activated. The optional `--config PATH` belongs **before** the subcommand. Data paths resolve relative to that configuration file.

```sh
# 1. Fetch real completed daily bars; rerunning updates the cache.
paper-trader fetch

# 2. Train Ridge and random forest candidates; retain the printed version.
paper-trader train

# 3. Evaluate the saved version on its untouched chronological test window.
paper-trader evaluate --version VERSION

# 4. Explicitly deploy the selected candidate from that version.
paper-trader promote VERSION

# 5. Check the paper account (read-only), then preview an eligible daily run.
paper-trader status
paper-trader run --dry-run

# 6. Open the local dashboard in a separate terminal.
paper-trader dashboard
```

Replace `VERSION` with the actual version printed by training, such as the timestamp/identifier in `data/models/`. `evaluate` without `--version` evaluates the active model. `fetch --end YYYY-MM-DD` accepts an earlier completed session for a new dataset; use a separate `data_dir` when requesting a date earlier than an existing cache's end.

`run` defaults to dry-run. Decisions are eligible only after the current exchange session's open plus `decision_delay_minutes` (default 5), and before its close. An out-of-window run returns `skipped` and the next scheduled time. Monday uses Friday's completed bars unless a holiday changes the preceding session. The first start during an eligible session can run immediately; missed sessions are not replayed.

## Automatic paper orders and controls

Start the scheduler in a terminal and keep it running:

```sh
# Observe daily dry-runs regardless of execution control.
paper-trader schedule --dry-run

# OR: honor the persisted execution control, disabled initially.
paper-trader schedule
```

Only enable orders when you are ready to use the paper account:

```sh
paper-trader enable              # Explicitly allow automatic PAPER orders
paper-trader run --execute       # Optional immediate eligible-session run
paper-trader pause               # Prevent new submissions
paper-trader resume              # Clear pause; all risk checks still apply
paper-trader disable             # Persistently disable new submissions
paper-trader cancel              # Separate request to cancel all open paper orders
```

The dashboard provides the same enable/disable, pause/resume, cancel, read-only refresh, and dry-run controls. `PAPER_EXECUTION_ENABLED=false` is the initial fallback. An explicit CLI/dashboard enable or disable is stored in SQLite and takes precedence across restarts. An explicitly requested dry-run always prevents submission. Enabling does not start a scheduler process. Dashboard access is bound to `127.0.0.1`; it is a local control surface, without multi-user authentication.

Pausing or disabling does not sell positions or cancel existing orders. Cancellation is separate, can race with a fill, and must be confirmed by reconciliation. The drawdown pause also leaves positions in place. Resuming while the account is still beyond the configured threshold immediately pauses again. The peak is maintained per account; transfers and account resets affect it, so use a dedicated paper account and investigate such changes.

The scheduler polls every 60 seconds by default, handles exchange holidays and early closes, maintains a heartbeat, and reconciles orders between decisions, including after the close. **Closing the terminal/process or putting the computer to sleep interrupts scheduling.** It is not a background service. A process lock prevents duplicate schedulers; a separate trading lock prevents overlapping runs and administrative operations.

## Data and time conventions

Default universe: SPY, QQQ, IWM, DIA. Edit `config.toml` to change it. SPY is also ingested as the broad-market feature reference even if it is excluded from the traded universe. Use liquid US equities/ETFs on the XNYS calendar; other exchanges, asset classes, and changing historical universes are outside scope.

The default `iex` feed provides actual single-venue IEX prices and volumes, **not consolidated US market volume**. It has no intentional real-time delay, but an inactive venue's latest trade can still be old. The application rejects prices older than `max_quote_age_seconds` (default 120 seconds). It does not substitute delayed SIP or generated prices. `sip` can be selected explicitly but requires suitable real-time entitlement; entitlement and account-specific access have not been tested here. Alpaca documents a latest-15-minute historical restriction for unsubscribed SIP access. The chosen feed is explicit in requests, storage, and model metadata.

The SDK automatically paginates historical bars. Ingestion validates finite positive OHLCV, consistent OHLC ranges, timezone-aware timestamps, duplicate rows, exact symbols, complete exchange-session coverage, and the required final session. It never forward-fills missing bars. A ten-session overlap plus the oldest cached bar detects corporate-action/revision changes; a detected change reloads the complete history before committing. Weekly full reloads catch older vendor corrections. Updates are transactional. API reads have bounded waits; outages stop the affected run with an actionable error.

Historical bars use Alpaca `adjustment=all` (splits, cash dividends, spin-offs). Current execution sizing uses raw latest trades and actual broker positions. Revised adjusted history is refreshed consistently; every trained version also saves its own immutable historical snapshot and checksum for evaluation. Historical adjusted whole-share calculations approximate total-return economics; they do not reproduce raw-share dividend cash accounting. Provider adjustments and corrections are not a point-in-time historical data archive.

All operational timestamps are UTC with an explicit offset. Session keys are New York exchange dates; `exchange_calendars` supplies DST, holidays, and early closes. Alpaca daily bars aggregate the New York calendar day and can contain eligible extended-hours prints. Therefore today's daily bar is **never** used, even after the regular close. A signal uses bars only through the preceding exchange session.

## Models and strategy

Features are 1-, 5-, and 20-session returns; 20-session volatility; distances from 20- and 60-session moving averages; relative volume; and SPY returns. They are calculated by one shared causal function for training and running. The regression target is adjusted `close[t+5] / close[t] - 1`. The last five sessions have no labels and are excluded from supervised evaluation, while their completed features remain usable for current predictions.

Eligible sessions split chronologically 60%/20%/20%. The earlier partition's label end must precede the next partition's first session, purging five boundary signal sessions. A training-only `StandardScaler` precedes Ridge and a modest 64-tree random forest. Validation RMSE selects the candidate; there is no parameter search or refit on validation/test. Test scoring occurs only in `evaluate`. At least roughly 215 input sessions are needed. The default history begins on July 27, 2020, the start of complete shared IEX coverage observed for the default ETF universe during authenticated verification on October 19, 2026; different feeds/universes may have different coverage.

Each version saves both candidates, feature schema, parameters, seed, dates, splits, validation metrics, dependency version, model checksum, input checksum, public configuration, and compressed input bars. Deployment changes only through `promote`. A session's predictions and targets are frozen on its first successful decision, including a dry-run; promotion affects the next unfrozen session. Load only artifacts created by this application—joblib artifacts are executable Python serialization.

Predictions strictly above the configured threshold qualify. Qualifiers receive equal target weights capped by 25% per asset and 100% in total; unallocated weight remains cash. Whole shares, a default 1% cash buffer, and a $25,000 per-order cap can reduce realized weights. The initial defaults are experimental, not optimized investment advice. Prediction accuracy or training performance alone does not establish profitability.

## Historical evaluation

All four portfolios—model strategy, cash, capped equal-weight buy-and-hold, and a 60-session trend rule—share the same test dates and initial cash. The backtest uses the same allocation/sizing and risk rules as paper execution. Completed signal `t` executes at the next daily bar's open with configurable adverse slippage (default 10 bps) and fees (default 1 bp). Live paper execution occurs after open+5 minutes, so the daily open is an approximation and may reflect extended hours. Historical fills are assumed complete; there is no order-book or intraday queue simulation.

Reports include net return, maximum drawdown, annualized volatility, Sharpe, turnover, exposure, fees, trades, and aligned curves. Sharpe assumes 252 sessions/year, zero risk-free rate, and sample daily-return volatility. Cash earns zero. Turnover is gross traded notional divided by initial equity; exposure is average daily invested/equity. No terminal liquidation or exit fee is invented. The equal-weight benchmark buys once subject to initial limits; subsequent drift is retained. Test prediction labels overlap within a partition, so observations are not independent. Repeated tuning after looking at the test report consumes the holdout.

Historical reports live in `data/backtests/VERSION/`. Forward paper views use only broker snapshots and remain separate. Forward aligned comparison needs at least two snapshots taken within 30 minutes after regular closes plus matching completed bars. It normalizes observed account value and a fractional equal-weight adjusted-close benchmark to a common starting session. Timing, cost, and cash-flow differences are disclosed in the dashboard; snapshots are not cash-flow-adjusted strategy attribution. No performance is fabricated when inputs are missing.

## Order safety, persistence, and recovery

The broker adapter constructs `TradingClient(..., paper=True)` with a fixed paper endpoint. It rejects live endpoint/mode environment settings, refuses redirects, and disallows arbitrary endpoint configuration. Credentials are excluded from stored configuration and redacted from audit messages. Paper orders use whole-share regular-session DAY limits; adverse slippage determines the buy ceiling/sell floor, so an order may not fill.

SQLite holds a durable account/strategy/session plan and stable client order IDs for each intended symbol, side, quantity, and price. The intent is committed **before** the network POST. Reruns and restarts query broker state; partially filled quantities and pending orders count toward position differences, cash reservations, and exposure. Buying power is capped by confirmed cash—margin capacity and anticipated sale proceeds are never used. The plan is fixed within the session; sale proceeds can support a new plan in a later session after broker confirmation.

Before each submission, the application rereads account/positions/orders, checks consistency, verifies price and broker-clock freshness, rechecks controls, and resizes against the current limits. Outside-universe holdings/orders, short positions, malformed state, or uncertain reconciliation stop new orders. Market drift can move existing exposure above a cap; additional buys are prevented and eligible risk-reducing sells remain possible unless paused. Use a dedicated account and avoid concurrent manual trading.

The SDK's implicit retry loop is explicitly disabled; only read requests get bounded retries for timeouts, 429, and transient server failures. Internal transport settings are covered by pinned-SDK adapter tests. **An ambiguous submission is never blindly POSTed again.** If its stable ID is not yet visible, further submissions remain blocked, even after a 404. Later discovery reconciles it; rejected orders stop remaining intents for that session. Reconciliation continues while the scheduler is running.

For an unresolved submission: pause, inspect the broker order history and local `execution_orders` record using its `client_order_id`, restore connectivity, and rerun/reconcile. Do not delete the database, change strategy IDs, or clear intents to force a retry; doing so can defeat duplicate prevention. If no matching broker order ever appears, manual investigation is required; there is deliberately no automatic force-resubmit control. Preserve the SQLite database, WAL files, and model directory together when backing up while stopped.

## Architecture and verification

| Module | Responsibility |
| --- | --- |
| `config`, `store` | Validation, credentials, SQLite audits/controls, process locks |
| `calendar`, `market` | Exchange sessions, actual data ingestion, validation, prices |
| `features`, `models` | Causal features, purged splits, model artifacts and promotion |
| `strategy`, `backtest` | Shared allocation/risk sizing, held-out strategy evaluation |
| `broker`, `execution` | Paper SDK adapter, durable intents, reconciliation |
| `app`, `cli`, `dashboard` | Workflow, scheduler, commands, local operational views |

```sh
python -m pytest -q
ruff check src tests
python -m compileall -q src
```

Test fixtures exist only under `tests/` and pytest temporary directories. They exercise causality, target timing, partition purging, model promotion, costs/limits, partial fills/rejections, restarts, ambiguous submissions, stale data, holidays/DST, paper endpoint rejection, structured persistence, the vertical dry-run, and dashboard empty states. Automated tests do not require credentials and do not place broker orders.

Provider/library references checked during implementation:

- [Alpaca TradingClient](https://alpaca.markets/sdks/python/api_reference/trading/trading-client.html), [order requests](https://alpaca.markets/sdks/python/api_reference/trading/requests.html), and [SDK REST transport](https://github.com/alpacahq/alpaca-py/blob/master/alpaca/common/rest.py).
- [Stock historical data SDK](https://alpaca.markets/sdks/python/api_reference/data/stock/historical.html), [historical bars and adjustments](https://docs.alpaca.markets/us/reference/stockbars), [market-data FAQ](https://docs.alpaca.markets/us/docs/market-data-faq), and [feed subscriptions/access](https://docs.alpaca.markets/us/docs/about-market-data-api).
- [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html), [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html), and [Streamlit API](https://docs.streamlit.io/develop/api-reference).
