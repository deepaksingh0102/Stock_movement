# QQQ Intraday Walk-forward Study (Streamlit)

A research harness that asks one question: **does a simple model forecast QQQ's next 5/15/30/60 minutes better than a naive baseline, out of sample?** It is not a trading system and not financial advice.

## What it does

- Builds causal features from 1-minute bars (returns, EMA/MACD/RSI/ADX/ATR, VWAP, relative volume, candle shape, 20-bar breakout, time of day).
- Labels each minute DOWN / SIDEWAYS / UP using a **volatility-scaled threshold**: the forward return must beat `±k · σ_t · √horizon`, where `σ_t` is an EWMA of 1-minute volatility known at time *t*.
- Trains a multinomial logistic regression with an **expanding-window walk-forward** (the model that scores session *i* sees only sessions before *i*).
- Compares it with the **training-set class frequencies** (what you would predict knowing nothing but the base rates), using log loss, Brier score and accuracy.
- Reports everything **per session** with a bootstrap-over-days 95% interval, because minute-level forecasts overlap heavily and the honest sample size is the number of days.
- Logs live forecasts and scores them later (forward test).

## Files

| File | Purpose |
|---|---|
| `qqq_core.py` | All logic (no Streamlit). Importable and testable. |
| `app.py` | Streamlit UI: validation tab and live forecast/log tab. |
| `live_log.py` | Headless logger for cron. |
| `run_study.py` | Command-line walk-forward run on a CSV. |
| `test_core.py` | Tests for gaps, causality, leakage and the logger (`python test_core.py`). |

## Key design choices

1. **Exact-minute horizons.** Each session is reindexed to the full 09:30–15:59 grid. "15 minutes ahead" means the close at exactly *t + 15 min*; if that bar does not exist, the row has no label. Nothing is filled with a neighbouring price. Missing bars are carried flat only inside indicator state so EMAs keep running, and such bars are never prediction rows or targets.
2. **Sessions with under 60% of their bars** (half days, partial downloads) are dropped and listed in the app.
3. **First 15 minutes of each session** are not forecast; return features are undefined there.
4. **RSI, ATR and ADX reset at the open**, consistent with the returns and VWAP, so the overnight gap does not leak into some indicators and not others.
5. **Baseline is the prior of the training set actually used**, not the test period's own class mix.
6. **Costs:** the per-day edge table subtracts an assumed round-trip cost (default 1 bp). It is close-to-close, with no fills, slippage model or position sizing.

## Setup

```
pip install -r requirements.txt
QQQ_TEST_CSV=QQQ_1min.csv python test_core.py   # 12 tests; CSV needs ~40 sessions
python run_study.py QQQ_1min.csv         # CLI walk-forward
streamlit run app.py
```

CSV columns: `timestamp,open,high,low,close,volume` (timezone-aware or UTC). For Alpaca, add secrets (never commit them):

```
ALPACA_API_KEY = "your_key"
ALPACA_SECRET_KEY = "your_secret"
```

Use `iex` unless your plan includes consolidated `sip` data. IEX is a subset of volume, so VWAP and relative volume differ from the full tape.

## Forward testing

In the app's **Live forecast & log** tab, or via cron:

```
*/5 9-16 * * 1-5  cd /path/to/repo && python live_log.py
```

Each run resolves earlier forecasts whose resolution minute has arrived, trains on completed sessions, and appends forecasts for the latest bar. A (bar, horizon) pair is never logged twice. Forecasts whose exact resolution minute is missing are marked `NO_BAR` and excluded from scoring. Streamlit Community Cloud storage is ephemeral, so download the log or run `live_log.py` on a machine you control.

## Reading the results

- **Log-loss gain > 0 with a CI that excludes 0** is the minimum evidence of skill. Anything else means the model is no better than the base rates.
- Accuracy alone is misleading with three imbalanced classes.
- The 41-session history needed here is short. Use 1–2 years before drawing conclusions, and treat any result from sweeping `k` or `C` on the same data as exploratory.
- A model that "wins" in a backtest can still lose after costs, and the forward-test log is the check that cannot be tuned against.
