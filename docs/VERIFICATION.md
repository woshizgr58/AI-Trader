# Completion and verification report

## Authenticated workflow update — 2026-10-19

Paper credentials were configured locally. Verified the actual Alpaca paper account connection with $100,000 virtual cash, no positions, and no open orders. No credentials were printed or stored in this report.

The original 2017 start failed strict data completeness checks. An actual provider coverage query showed a complete common IEX panel beginning **2020-07-27** for SPY, QQQ, IWM, and DIA; earlier data was incomplete. Updated the configured/default start to that observed date, retaining all completeness checks. Downloaded **6,260 actual adjusted daily bars**, through **2026-10-16**, with no generated replacements.

Trained, evaluated, and activated the initial version `20261019T184234-bb0583d4`. Validation RMSE selected the random forest. Held-out execution dates were **2025-08-01 through 2026-10-12**:

| Historical portfolio | Net return | Maximum drawdown | Sharpe |
| --- | ---: | ---: | ---: |
| Model strategy | 17.77% | 5.12% | 1.53 |
| Equal-weight buy-and-hold | 27.19% | 9.22% | 1.51 |
| Trend | 18.46% | 4.95% | 1.32 |
| Cash | 0.00% | 0.00% | Undefined |

These are historical results under the saved report's fill/cost/adjusted-price assumptions, not forward paper results or a guarantee of future returns. The model strategy did not beat buy-and-hold on net return. Reports are saved under `data/backtests/20261019T184234-bb0583d4/`.

An actual-market dry-run during the **2026-10-19** session used completed bars through **2026-10-16**, current IEX trades, the trained model, and broker account reconciliation. All four predictions qualified; the app produced whole-share buy proposals with cash/exposure checks. **No orders were submitted. Execution remains disabled; no automatic scheduler was started.** Actual broker order acceptance, fills, and cancellation behavior are still unverified.

The following report records the earlier implementation verification, before credentials were configured.

Verified on 2026-10-17 with Python 3.12.3 on macOS arm64.

## Delivered

- Working Python source in `src/paper_trader/`, with a CLI and local Streamlit dashboard.
- Exact direct dependency pins in `pyproject.toml`, transitive pins in `requirements.lock`, safe defaults in `config.toml`, and `.env.example`.
- Real Alpaca historical/current-data adapter, incremental SQLite storage, calendar validation, causal features, purged model training, explicit promotion, held-out strategy/benchmark evaluation, durable paper execution, reconciliation, controls, scheduling, and audit records.
- Setup, every operational command, architecture, credential instructions, provider references, assumptions, and recovery limitations in `README.md`.

## Verified locally

| Check | Result |
| --- | --- |
| `python -m pytest -q` | **123 passed**, one upstream deprecation warning |
| `ruff check src tests` | Passed |
| `python -m compileall -q src` | Passed |
| `python -m pip check` | No broken requirements |
| Editable package installation and `paper-trader --help` | Passed |
| `paper-trader fetch` without credentials | Correct actionable missing-credentials error; no generated data |
| `paper-trader run --dry-run` on Saturday, October 17 | Correctly skipped; next scheduled decision Monday, October 19 at 13:35 UTC / 06:35 PDT |
| Streamlit empty and populated AppTests | Passed |
| Actual localhost dashboard in Chrome | Rendered correctly; paper banner, disabled execution, missing-data/model/account states verified |

The offline vertical integration test runs the actual ingestion → training → evaluation → promotion → incremental refresh → predictions → allocation/risk checks → dry-run → persisted decision workflow. Only external provider and broker I/O are mocked. Test fixtures are clearly identified and stored under pytest temporary directories; they are not installed as operational market data or presented as investment performance.

Other tests exercise future-data invariance, target alignment, common-session gaps, train-only preprocessing, chronological purging, untouched-test isolation, deployment immutability, fees and sizing, cash/exposure limits, pending buy/sell reservations, partial fills, rejections, restart idempotency, ambiguous timeouts, no implicit SDK POST retry, per-order control changes, stale broker/quote state, drawdown pauses, live-endpoint rejection, calendar holidays/DST/early closes, scheduler retries/overlap, redaction, and transactional persistence.

The warning is `websockets.legacy` deprecation from the Alpaca dependency import. It does not fail the tests. The application uses REST for its daily workflow.

## Not verified without credentials

No Alpaca keys were available. The following are therefore **not claimed as verified**:

- Authenticated history/current-market-data access or IEX/SIP account entitlement.
- The user's paper-account connectivity, balances, buying power, and broker clock.
- A dry-run using actual downloaded market data and the user's account.
- Paper-order acceptance, actual simulated partial fills, cancellations, and account-specific behavior.

No actual broker order was submitted. Automatic execution remains disabled and no scheduler was started against a broker. The local dashboard is available independently of credentials.

To complete provider verification, configure paper keys in `.env`, run `fetch`, `train`, `evaluate --version VERSION`, `promote VERSION`, and `status`, then `run --dry-run` during an eligible market session. This read-only/dry-run sequence does not require enabling orders. The user must explicitly use `enable` or the dashboard control before automatic paper submissions are permitted.

Historical results use documented daily-open fill, cost, and adjusted-share assumptions. Forward paper results come only from actual observed broker snapshots. Neither mocked tests nor predictive accuracy establish profitability.
