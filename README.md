# QQQ Intraday Replay (Streamlit)

Replay QQQ 1-minute candles on **October 23, 2026**, produce heuristic bullish/bearish/neutral signals at each minute close, and evaluate the next 5/15/30/60 **regular-session trading minutes** without lookahead. Not financial advice or a trained probability model.

## Setup
1. Upload `app.py`, `requirements.txt`, and `.gitignore` to your GitHub repository.
2. Deploy via [Streamlit Community Cloud](https://share.streamlit.io/).
3. Add these secrets in the Streamlit app settings (never commit them):

```toml
ALPACA_API_KEY = "your_key"
ALPACA_SECRET_KEY = "your_secret"
```

4. Choose `iex` if your Alpaca plan only provides IEX historical data, or `sip` if your subscription permits consolidated SIP. IEX is a subset of US market trades and can differ from SIP.
5. Select October 23, 2026, click **Fetch QQQ**, and choose a replay timestamp and horizon.

The app requests several prior calendar days as indicator warmup, fetches paginated Alpaca bars, and restricts evaluation to the chosen day's regular session (09:30–16:00 America/New_York). Predictions are made at a bar's close; future returns use subsequent bar closes, not future inputs. Missing minute bars are not invented: a forecast is evaluated only if the exact future minute exists. The final minutes of the session cannot be evaluated for horizons extending past 16:00.

### Important limitations
- Scores are **heuristics**, not calibrated probabilities. A signal accuracy statistic is not evidence of live profitability.
- VWAP resets at the regular-session open; warmup history is used for EMA/RSI/MACD/ADX/ATR only.
- Entry/exit execution, slippage, commissions, and risk management are not simulated. Forecast outcomes use close-to-close moves.
- Historical market data may differ by feed and adjustments. The script uses raw bars and does not model dividends/corporate actions.
- All historical bars for the selected date are fetched to score outcomes, but feature computation is strictly causal and never uses future bars.
