import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="QQQ Intraday Replay", layout="wide")
ET = ZoneInfo("America/New_York")
HORIZONS = [5, 15, 30, 60]
# Move thresholds in basis points. These are hypotheses, NOT optimized/calibrated probabilities.
MOVE_BPS = {5: 10, 15: 15, 30: 20, 60: 30}


def secret(name):
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_bars(start_iso, end_iso, feed, key, secret_key):
    url = "https://data.alpaca.markets/v2/stocks/QQQ/bars"
    params = {"start": start_iso, "end": end_iso, "timeframe": "1Min", "feed": feed,
              "adjustment": "raw", "limit": 10000, "sort": "asc"}
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret_key}
    records = []
    token = None
    with requests.Session() as session:
        for _ in range(100):
            p = dict(params)
            if token:
                p["page_token"] = token
            try:
                resp = session.get(url, params=p, headers=headers, timeout=35)
                resp.raise_for_status()
                payload = resp.json()
            except requests.RequestException as exc:
                detail = resp.text[:500] if "resp" in locals() else str(exc)
                raise RuntimeError(f"Alpaca request failed: {detail}") from exc
            records.extend(payload.get("bars") or [])
            next_token = payload.get("next_page_token")
            if not next_token:
                break
            if next_token == token:
                raise RuntimeError("Alpaca pagination returned a repeated token.")
            token = next_token
        else:
            raise RuntimeError("Exceeded pagination safety limit.")
    if not records:
        raise RuntimeError("No bars returned. Check API entitlement, date, and feed.")
    df = pd.DataFrame(records).rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(ET)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def rma(s, period):
    return s.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def indicators(df):
    # Features are computed on regular-session minutes only; no forward-looking operations.
    x = df.copy()
    c, h, l = x.close, x.high, x.low
    x["ema9"] = c.ewm(span=9, adjust=False, min_periods=9).mean()
    x["ema20"] = c.ewm(span=20, adjust=False, min_periods=20).mean()
    x["ema50"] = c.ewm(span=50, adjust=False, min_periods=50).mean()
    ema12 = c.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = c.ewm(span=26, adjust=False, min_periods=26).mean()
    x["macd"] = ema12 - ema26
    x["macd_signal"] = x.macd.ewm(span=9, adjust=False, min_periods=9).mean()
    delta = c.diff()
    up = rma(delta.clip(lower=0), 14)
    down = rma((-delta).clip(lower=0), 14)
    rs = up / down.replace(0, np.nan)
    x["rsi"] = 100 - 100 / (1 + rs)
    x.loc[(down == 0) & (up > 0), "rsi"] = 100
    x.loc[(down == 0) & (up == 0), "rsi"] = 50
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    x["atr"] = rma(tr, 14)
    plus_move = h.diff()
    minus_move = -l.diff()
    plus_dm = plus_move.where((plus_move > minus_move) & (plus_move > 0), 0.0)
    minus_dm = minus_move.where((minus_move > plus_move) & (minus_move > 0), 0.0)
    atr_safe = x.atr.replace(0, np.nan)
    x["plus_di"] = 100 * rma(plus_dm, 14) / atr_safe
    x["minus_di"] = 100 * rma(minus_dm, 14) / atr_safe
    dx = 100 * (x.plus_di - x.minus_di).abs() / (x.plus_di + x.minus_di).replace(0, np.nan)
    x["adx"] = rma(dx, 14)
    session = pd.Series(x.index.date, index=x.index)
    typical = (h + l + c) / 3
    cum_pv = (typical * x.volume).groupby(session).cumsum()
    cum_vol = x.volume.groupby(session).cumsum().replace(0, np.nan)
    x["vwap"] = cum_pv / cum_vol
    # Prior-session same minute-of-day volume baseline (up to 20 observations).
    # Uses only completed prior sessions; no future-volume leakage.
    minute = x.index.hour * 60 + x.index.minute
    volume_matrix = x.assign(day=x.index.date, minute=minute).pivot(index="day", columns="minute", values="volume")
    baseline = volume_matrix.shift(1).rolling(20, min_periods=3).mean()
    x["volume_baseline"] = [baseline.loc[d, m] for d, m in zip(x.index.date, minute)]
    x["rvol"] = x.volume / x.volume_baseline.replace(0, np.nan)
    body = (x.close - x.open).abs()
    lower_wick = np.minimum(x.open, x.close) - x.low
    upper_wick = x.high - np.maximum(x.open, x.close)
    x["bull_engulf"] = (x.close > x.open) & (x.close.shift(1) < x.open.shift(1)) & (x.open <= x.close.shift(1)) & (x.close >= x.open.shift(1))
    x["bear_engulf"] = (x.close < x.open) & (x.close.shift(1) > x.open.shift(1)) & (x.open >= x.close.shift(1)) & (x.close <= x.open.shift(1))
    x["hammer"] = (lower_wick >= 2 * body) & (upper_wick <= body) & (body > 0)
    x["shooting_star"] = (upper_wick >= 2 * body) & (lower_wick <= body) & (body > 0)
    # Previous 20 bars, excluding current candle.
    x["breakout"] = x.close > x.high.shift(1).rolling(20, min_periods=20).max()
    x["breakdown"] = x.close < x.low.shift(1).rolling(20, min_periods=20).min()
    return x


def signal(x):
    needed = ["ema9", "ema20", "macd", "macd_signal", "rsi", "adx", "vwap", "rvol"]
    valid = x[needed].notna().all(axis=1)
    bull = (20 * (x.close > x.vwap).astype(int)
            + 15 * (x.ema9 > x.ema20).astype(int)
            + 15 * (x.macd > x.macd_signal).astype(int)
            + 10 * x.rsi.between(50, 70).astype(int)
            + 10 * (x.adx > 20).astype(int)
            + 15 * (x.rvol >= 1.5).astype(int)
            + 15 * (x.bull_engulf | x.hammer | x.breakout).astype(int))
    bear = (20 * (x.close < x.vwap).astype(int)
            + 15 * (x.ema9 < x.ema20).astype(int)
            + 15 * (x.macd < x.macd_signal).astype(int)
            + 10 * x.rsi.between(30, 50).astype(int)
            + 10 * (x.adx > 20).astype(int)
            + 15 * (x.rvol >= 1.5).astype(int)
            + 15 * (x.bear_engulf | x.shooting_star | x.breakdown).astype(int))
    x = x.copy()
    x["bull_score"] = bull.where(valid)
    x["bear_score"] = bear.where(valid)
    # Only score >= 65 and strictly greater than opposing score triggers a signal.
    x["signal"] = np.select([valid & (bull >= 65) & (bull > bear),
                              valid & (bear >= 65) & (bear > bull)], ["BULLISH", "BEARISH"], default="WAIT")
    x.loc[~valid, "signal"] = "INSUFFICIENT DATA"
    return x


def evaluate(day_df, horizon):
    y = day_df.copy()
    # Exact timestamp matching: missing bars cannot silently shorten the horizon.
    future_index = y.index + pd.Timedelta(minutes=horizon)
    future_close = y.close.reindex(future_index).to_numpy()
    y["future_close"] = future_close
    y["return_bps"] = (y.future_close / y.close - 1) * 10000
    threshold = MOVE_BPS[horizon]
    y["actual"] = np.select([y.return_bps > threshold, y.return_bps < -threshold], ["UP", "DOWN"], default="SIDEWAYS")
    y.loc[y.future_close.isna(), "actual"] = "NOT AVAILABLE"
    y["correct"] = ((y.signal.eq("BULLISH") & y.actual.eq("UP")) |
                    (y.signal.eq("BEARISH") & y.actual.eq("DOWN")))
    return y


st.title("QQQ | Intraday Prediction Replay")
st.caption("Historical QQQ only · 1-minute candles · 5/15/30/60-minute outcomes · No live trading")
with st.sidebar:
    st.header("Settings")
    selected_date = st.date_input("Replay date", value=dt.date(2026, 10, 23), max_value=dt.date.today())
    feed = st.selectbox("Alpaca feed", ["iex", "sip"], help="SIP needs appropriate Alpaca market-data entitlement.")
    horizon = st.selectbox("Prediction horizon (minutes)", HORIZONS, index=1)
    api_key = st.text_input("Alpaca API key", value=secret("ALPACA_API_KEY"), type="password")
    api_secret = st.text_input("Alpaca secret key", value=secret("ALPACA_SECRET_KEY"), type="password")
    load = st.button("Fetch QQQ", type="primary", use_container_width=True)

if not load:
    st.info("Set Alpaca credentials, select a feed, and click **Fetch QQQ**. The default test date is October 23, 2026.")
    st.stop()
if not api_key or not api_secret:
    st.error("Add Alpaca credentials in Streamlit Secrets or enter them in the sidebar.")
    st.stop()
if selected_date.weekday() >= 5:
    st.warning("The selected date is a weekend; US equities do not have a regular trading session.")
    st.stop()

# Fetch previous sessions for indicator warmup and same-minute prior-session RVOL.
start = dt.datetime.combine(selected_date - dt.timedelta(days=45), dt.time(0), tzinfo=ET).isoformat()
end = dt.datetime.combine(selected_date + dt.timedelta(days=1), dt.time(0), tzinfo=ET).isoformat()
try:
    with st.spinner("Fetching historical QQQ bars from Alpaca..."):
        raw = fetch_bars(start, end, feed, api_key, api_secret)
except Exception as exc:
    st.error(str(exc))
    st.stop()

regular = raw.between_time("09:30", "15:59")
regular = regular[regular.index.weekday < 5]
if regular.empty:
    st.error("No regular-session bars found.")
    st.stop()
all_data = signal(indicators(regular))
day = all_data[all_data.index.date == selected_date].copy()
if day.empty:
    st.error("No QQQ regular-session data found for this date. Check your feed, permissions, or date.")
    st.stop()
if len(day) < 300:
    st.warning(f"Only {len(day)} one-minute bars returned (typical full session ~390). IEX or missing data may affect results.")

st.caption(f"Feed: {feed.upper()} | Date: {selected_date.isoformat()} | Bars: {len(day)} | Times: US Eastern")
selected_time = st.select_slider("Replay timestamp (prediction after candle close)", options=list(day.index),
                                  value=day.index[min(60, len(day)-1)],
                                  format_func=lambda x: x.strftime("%I:%M %p ET"))
row = day.loc[selected_time]
assessed = evaluate(day, horizon)
result = assessed.loc[selected_time]

c1, c2, c3, c4 = st.columns(4)
c1.metric("QQQ close", f"${row.close:,.2f}")
c2.metric("Signal", str(row.signal))
c3.metric("Bullish score", "N/A" if pd.isna(row.bull_score) else f"{row.bull_score:.0f}/100")
c4.metric("Bearish score", "N/A" if pd.isna(row.bear_score) else f"{row.bear_score:.0f}/100")
st.info("Scores are rule-based signal alignment, **not probabilities**. Outcomes are revealed only for evaluation after the selected timestamp.")

chart_data = day.loc[:selected_time].tail(130)
fig = go.Figure()
fig.add_trace(go.Candlestick(x=chart_data.index, open=chart_data.open, high=chart_data.high,
                             low=chart_data.low, close=chart_data.close, name="QQQ"))
for col, color in [("vwap", "#a855f7"), ("ema9", "#0ea5e9"), ("ema20", "#f59e0b")]:
    fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data[col], name=col.upper(), line=dict(color=color, width=1.5)))
fig.update_layout(height=520, xaxis_rangeslider_visible=False, title="QQQ candles up to selected replay timestamp",
                  xaxis_title="US Eastern", yaxis_title="Price ($)")
st.plotly_chart(fig, use_container_width=True)

st.subheader(f"Actual next {horizon}-minute outcome (revealed after prediction)")
a, b, c = st.columns(3)
a.metric("Actual outcome", str(result.actual))
b.metric("Actual return", "Unavailable" if pd.isna(result.return_bps) else f"{result.return_bps:+.1f} bps")
c.metric("Signal matched direction", "N/A" if result.actual == "NOT AVAILABLE" or result.signal not in ["BULLISH", "BEARISH"] else ("Yes" if result.correct else "No"))

st.subheader("All horizons at selected timestamp")
rows = []
for h in HORIZONS:
    z = evaluate(day, h).loc[selected_time]
    rows.append({"Horizon": f"{h} min", "Signal": z.signal, "Bull score": z.bull_score,
                 "Bear score": z.bear_score, "Actual": z.actual,
                 "Return (bps)": round(z.return_bps, 2) if pd.notna(z.return_bps) else np.nan})
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

st.subheader("October 23 signal evaluation")
summary = []
for h in HORIZONS:
    z = evaluate(day, h)
    trades = z[z.signal.isin(["BULLISH", "BEARISH"]) & z.future_close.notna()]
    summary.append({"Horizon": f"{h} min", "Evaluable signals": len(trades),
                    "Directional hit rate": f"{100 * trades.correct.mean():.1f}%" if len(trades) else "N/A",
                    "Average signed return (bps)": round(np.where(trades.signal.eq("BULLISH"), trades.return_bps, -trades.return_bps).mean(), 2) if len(trades) else np.nan})
st.dataframe(pd.DataFrame(summary), hide_index=True, use_container_width=True)
st.caption("Hit rate = signal direction matched an outcome beyond the horizon's sideways threshold. Signals overlap and are not independent trades. No trading costs, slippage, or stops are included.")

with st.expander("View indicator readings at replay timestamp"):
    cols = ["open", "high", "low", "close", "volume", "vwap", "ema9", "ema20", "ema50", "macd", "macd_signal", "rsi", "adx", "atr", "rvol", "bull_engulf", "bear_engulf", "hammer", "shooting_star", "breakout", "breakdown"]
    st.dataframe(pd.DataFrame({"Indicator": cols, "Value": [row[k] for k in cols]}), hide_index=True)

csv = evaluate(day, horizon).reset_index().to_csv(index=False).encode("utf-8")
st.download_button("Download replay results CSV", csv, file_name=f"QQQ_{selected_date}_{horizon}min.csv", mime="text/csv")
