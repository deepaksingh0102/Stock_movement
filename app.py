"""Stock Movement V5: rule-based Darvas-style box breakout research dashboard."""
import datetime as dt
import re
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

NY = ZoneInfo('America/New_York')
UTC = dt.timezone.utc
st.set_page_config(page_title='Stock Movement | Box Theory V5', layout='wide')


def secret(*names, default=''):
    for name in names:
        v = st.secrets.get(name, '')
        if v and str(v).strip():
            return str(v).strip()
    return default


def clean(raw, naive_tz='America/New_York'):
    df = raw.copy()
    df.columns = [str(c).lower().strip() for c in df.columns]
    if 'timestamp' in df.columns:
        times = pd.to_datetime(df['timestamp'], errors='coerce')
        if times.dt.tz is None:
            times = times.dt.tz_localize(naive_tz, ambiguous='NaT', nonexistent='NaT')
        df.index = pd.DatetimeIndex(times).tz_convert(NY)
    elif isinstance(df.index, pd.DatetimeIndex):
        df.index = (df.index.tz_localize(naive_tz) if df.index.tz is None else df.index).tz_convert(NY)
    else:
        raise ValueError('CSV needs a timestamp column')
    required = ['open', 'high', 'low', 'close', 'volume']
    if any(c not in df for c in required):
        raise ValueError('CSV needs timestamp, open, high, low, close, volume columns')
    df = df[required].apply(pd.to_numeric, errors='coerce').dropna()
    df = df[~df.index.isna()]
    df = df[~df.index.duplicated(keep='last')].sort_index()
    df = df.between_time('09:30', '15:59')
    return df[df.index.dayofweek < 5]


@st.cache_data(ttl=60, show_spinner=False)
def alpaca_bars(symbol, start, end, feed, key, secret_key):
    url = f'https://data.alpaca.markets/v2/stocks/{symbol}/bars'
    params = {'timeframe': '1Min', 'start': start, 'end': end,
              'limit': 10000, 'feed': feed, 'adjustment': 'split', 'sort': 'asc'}
    headers = {'APCA-API-KEY-ID': key, 'APCA-API-SECRET-KEY': secret_key}
    rows = []
    with requests.Session() as session:
        for _ in range(100):
            response = session.get(url, params=params, headers=headers, timeout=40)
            response.raise_for_status()
            body = response.json()
            rows.extend(body.get('bars', []))
            token = body.get('next_page_token')
            if not token:
                break
            params['page_token'] = token
        else:
            raise RuntimeError('Alpaca pagination exceeded 100 pages')
    if not rows:
        raise ValueError('No bars returned: check ticker, feed, market hours, and access')
    return clean(pd.DataFrame(rows).rename(columns={
        't': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low',
        'c': 'close', 'v': 'volume'}))


@st.cache_data(ttl=21600, show_spinner=False)
def history(symbol, feed, key, secret_key, today):
    end_day = dt.date.fromisoformat(today)
    start = dt.datetime.combine(end_day - dt.timedelta(days=85), dt.time.min, UTC)
    end = dt.datetime.combine(end_day, dt.time.min, UTC)
    return alpaca_bars(symbol, start.isoformat(), end.isoformat(), feed, key, secret_key)


def live(symbol, feed, key, secret_key):
    now = dt.datetime.now(UTC) - dt.timedelta(minutes=2)
    end = dt.datetime.fromtimestamp(int(now.timestamp() // 30) * 30, UTC)
    start = end - dt.timedelta(days=4)
    return alpaca_bars(symbol, start.isoformat(), end.isoformat(), feed, key, secret_key)


def merge_data(a, b):
    x = pd.concat([a, b]).sort_index()
    return x[~x.index.duplicated(keep='last')]


def to_five_minute(raw):
    """Five one-minute bars per candle, no partial candles, no overnight mixing."""
    out = []
    for _, d in raw.groupby(raw.index.date):
        agg = d.resample('5min', label='left', closed='left').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'})
        counts = d['close'].resample('5min', label='left', closed='left').count()
        agg = agg[counts.reindex(agg.index, fill_value=0) == 5].dropna()
        if not agg.empty:
            out.append(agg)
    if not out:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    return pd.concat(out).sort_index()


def levels_for_day(raw, date, opening_minutes=15):
    dates = sorted(set(raw.index.date))
    prev = [d for d in dates if d < date]
    result = {'PDH': np.nan, 'PDL': np.nan, 'PDC': np.nan,
              'OR High': np.nan, 'OR Low': np.nan}
    if prev:
        p = raw[raw.index.date == prev[-1]]
        result.update(PDH=float(p.high.max()), PDL=float(p.low.min()), PDC=float(p.close.iloc[-1]))
    today = raw[raw.index.date == date]
    start = pd.Timestamp(date, tz=NY) + pd.Timedelta(hours=9, minutes=30)
    expected = pd.date_range(start, periods=opening_minutes, freq='min')
    opening = today.reindex(expected)
    if opening['close'].notna().all():
        result['OR High'] = float(opening.high.max())
        result['OR Low'] = float(opening.low.min())
    return result


def make_boxes(raw, lookback=12, max_width_pct=0.9, vol_multiplier=1.2,
               require_trend=True, require_volume=True):
    """Rolling Darvas-style boxes: prior N complete 5m candles in same session.
    This is a testable adaptation, not the original Darvas pivot-confirmation method.
    Signal is known only at the end of its 5m candle.
    """
    x = to_five_minute(raw)
    if x.empty:
        return x
    blocks = []
    for _, d in x.groupby(x.index.date):
        d = d.copy()
        n = lookback
        d['box_high'] = d.high.rolling(n, min_periods=n).max().shift(1)
        d['box_low'] = d.low.rolling(n, min_periods=n).min().shift(1)
        d['box_width_pct'] = (d.box_high - d.box_low) / d.close * 100
        d['vol_ratio'] = d.volume / d.volume.rolling(n, min_periods=n).mean().shift(1).replace(0, np.nan)
        d['ema9'] = d.close.ewm(span=9, adjust=False).mean()
        d['ema20'] = d.close.ewm(span=20, adjust=False).mean()
        typical = (d.high + d.low + d.close) / 3
        d['vwap'] = (typical * d.volume).cumsum() / d.volume.cumsum().replace(0, np.nan)
        d['atr'] = pd.concat([
            d.high - d.low,
            (d.high - d.close.shift()).abs(),
            (d.low - d.close.shift()).abs()], axis=1).max(axis=1).ewm(
                alpha=1/14, adjust=False, min_periods=14).mean()
        eligible = d.box_high.notna() & (d.box_width_pct <= max_width_pct)
        long_ok = eligible & (d.close > d.box_high) & (d.close > d.open)
        short_ok = eligible & (d.close < d.box_low) & (d.close < d.open)
        if require_trend:
            long_ok &= (d.ema9 > d.ema20) & (d.close > d.vwap)
            short_ok &= (d.ema9 < d.ema20) & (d.close < d.vwap)
        if require_volume:
            long_ok &= d.vol_ratio >= vol_multiplier
            short_ok &= d.vol_ratio >= vol_multiplier
        d['signal'] = np.select([long_ok, short_ok], [1, -1], default=0)
        d['signal_time'] = d.index + pd.Timedelta(minutes=5)
        blocks.append(d)
    return pd.concat(blocks).sort_index()


def backtest(boxes, hold_bars=6, stop_atr=1.2, reward_risk=1.5, cost_bps=2.0):
    """Next 5m open entry; stop-first when stop and target both touch a bar.
    One position at a time per session; forced same-day exit; no overnight positions.
    All returns net of assumed round-trip cost. No leverage assumed.
    """
    trades = []
    for day, d in boxes.groupby(boxes.index.date):
        d = d.reset_index().rename(columns={'index': 'timestamp'})
        i = 0
        while i < len(d) - 1:
            s = int(d.at[i, 'signal'])
            if not s or pd.isna(d.at[i, 'atr']):
                i += 1
                continue
            entry_i = i + 1
            entry = float(d.at[entry_i, 'open'])
            risk = max(float(d.at[i, 'atr']) * stop_atr, entry * 0.0005)
            stop = entry - s * risk
            target = entry + s * risk * reward_risk
            last_i = min(entry_i + hold_bars - 1, len(d) - 1)
            exit_i = last_i
            exit_price = float(d.at[last_i, 'close'])
            reason = 'time/session exit'
            for j in range(entry_i, last_i + 1):
                lo, hi = float(d.at[j, 'low']), float(d.at[j, 'high'])
                stop_hit = lo <= stop if s == 1 else hi >= stop
                target_hit = hi >= target if s == 1 else lo <= target
                if stop_hit:
                    # Conservative gap handling: worse of stop or bar open.
                    opened = float(d.at[j, 'open'])
                    fill = min(stop, opened) if s == 1 else max(stop, opened)
                    exit_i, exit_price, reason = j, fill, 'stop'
                    break
                if target_hit:
                    exit_i, exit_price, reason = j, target, 'target'
                    break
            gross_bps = s * (exit_price / entry - 1) * 10000
            net_bps = gross_bps - cost_bps
            trades.append({
                'session': str(day), 'signal_time': d.at[i, 'signal_time'],
                'entry_time': d.at[entry_i, 'timestamp'],
                'exit_time': d.at[exit_i, 'timestamp'] + pd.Timedelta(minutes=5),
                'side': 'LONG' if s == 1 else 'SHORT',
                'entry': entry, 'exit': exit_price, 'stop': stop, 'target': target,
                'reason': reason, 'gross_bps': gross_bps, 'net_bps': net_bps,
                'win': net_bps > 0})
            i = exit_i + 1
    return pd.DataFrame(trades)


def summarize(trades):
    if trades.empty:
        return {'Trades': 0, 'Win rate': np.nan, 'Avg net bps': np.nan,
                'Total net bps': np.nan, 'Profit factor': np.nan}
    wins = trades.loc[trades.net_bps > 0, 'net_bps'].sum()
    losses = -trades.loc[trades.net_bps < 0, 'net_bps'].sum()
    return {'Trades': len(trades), 'Win rate': trades.win.mean(),
            'Avg net bps': trades.net_bps.mean(),
            'Total net bps': trades.net_bps.sum(),
            'Profit factor': wins / losses if losses else np.nan}


def chart_for_day(raw, boxes, day, opening_minutes, visible_bars):
    d = raw[raw.index.date == day].copy()
    b = boxes[boxes.index.date == day].copy()
    if visible_bars:
        d = d.tail(visible_bars)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=d.index, open=d.open, high=d.high, low=d.low,
                                 close=d.close, name='Price'))
    if not b.empty:
        shown = b[(b.index + pd.Timedelta(minutes=5)) >= d.index.min()]
        for name, color in [('box_high', '#e67e22'), ('box_low', '#16a085'),
                            ('ema9', '#eab308'), ('ema20', '#2563eb'), ('vwap', '#8b5cf6')]:
            fig.add_trace(go.Scatter(x=shown.index, y=shown[name], mode='lines',
                                     name=name.upper(), line=dict(color=color, width=1.5)))
        for s, color, marker, label in [(1, 'green', 'triangle-up', 'Bullish box breakout'),
                                        (-1, 'red', 'triangle-down', 'Bearish box breakdown')]:
            ev = shown[shown.signal == s]
            fig.add_trace(go.Scatter(x=ev.signal_time,
                                     y=ev.close * (0.9997 if s == 1 else 1.0003),
                                     mode='markers', name=label,
                                     marker=dict(color=color, size=12, symbol=marker)))
    levels = levels_for_day(raw, day, opening_minutes)
    palette = {'PDH': '#16a34a', 'PDL': '#dc2626', 'PDC': '#64748b',
               'OR High': '#0891b2', 'OR Low': '#ea580c'}
    for name, val in levels.items():
        if np.isfinite(val):
            fig.add_hline(y=val, line_dash='dot', line_color=palette[name],
                          annotation_text=f'{name} {val:.2f}', annotation_position='top right')
    # Two confirmed swing pivots on 5-minute bars, two future bars for confirmation.
    if len(b) >= 9:
        for field, color, is_high in [('high', '#ef4444', True), ('low', '#22c55e', False)]:
            arr = b[field].to_numpy()
            pivots = []
            for j in range(2, len(arr)-2):
                left = arr[j-2:j]
                right = arr[j+1:j+3]
                if ((arr[j] > max(left) and arr[j] >= max(right)) if is_high else
                    (arr[j] < min(left) and arr[j] <= min(right))):
                    pivots.append(j)
            # Only the latest pair, drawn after both pivots have confirmed.
            if len(pivots) >= 2:
                a, z = pivots[-2:]
                slope = (arr[z] - arr[a]) / (z - a)
                fig.add_trace(go.Scatter(x=[b.index[a], b.index[-1]],
                    y=[arr[a], arr[a] + slope * (len(b)-1-a)], mode='lines',
                    name='Resistance trendline' if is_high else 'Support trendline',
                    line=dict(color=color, width=2, dash='dash')))
    fig.update_layout(height=600, title=f'{day} | Darvas-style box levels and breakout signals',
                      xaxis_title='Eastern Time', yaxis_title='Price ($)',
                      xaxis_rangeslider_visible=False)
    return fig, levels


st.title('Stock Movement V5 | Darvas Box Breakout')
st.caption('Research and paper-testing only. No orders are placed. WAIT means no confirmed setup.')

with st.sidebar:
    symbol = st.text_input('U.S. stock / ETF symbol', 'QQQ').strip().upper()
    if not re.fullmatch(r'[A-Z][A-Z0-9.-]{0,9}', symbol):
        st.error('Invalid ticker symbol')
        st.stop()
    feed = st.selectbox('Alpaca feed', ['iex', 'sip'],
                        index=1 if secret('ALPACA_FEED', default='iex') == 'sip' else 0)
    opening = st.selectbox('Opening range (minutes)', [5, 15, 30], index=1)
    window = st.selectbox('Chart timeframe', ['5 minutes', '15 minutes', '1 hour',
                                              '2 hours', '4 hours', 'Since market open'], index=5)
    windows = {'5 minutes': 5, '15 minutes': 15, '1 hour': 60,
               '2 hours': 120, '4 hours': 240, 'Since market open': None}
    st.divider()
    st.subheader('Box strategy')
    lookback = st.slider('Box lookback (5m candles)', 6, 24, 12)
    max_width = st.slider('Maximum box width (%)', 0.2, 2.0, 0.9, 0.1)
    vol_mult = st.slider('Breakout relative volume', 1.0, 3.0, 1.2, 0.1)
    hold_minutes = st.selectbox('Maximum holding time', [5, 15, 30, 60], index=2)
    stop_atr = st.slider('Stop distance (ATR multiplier)', 0.5, 3.0, 1.2, 0.1)
    reward_risk = st.slider('Profit target (reward/risk)', 0.5, 4.0, 1.5, 0.1)
    cost_bps = st.number_input('Estimated round-trip cost (bps)', 0.0, 30.0, 2.0, 0.5)
    st.divider()
    uploaded = st.file_uploader('Historical 1-minute CSV (optional)', type=['csv'])
    naive_tz = st.selectbox('Timezone for CSV timestamps without offset',
                            ['America/New_York', 'UTC'], index=0)
    run_test = st.button('Run historical backtest', type='primary')

key = secret('ALPACA_API_KEY', 'APCA_API_KEY_ID', 'ALPACA_KEY')
secret_key = secret('ALPACA_SECRET_KEY', 'APCA_API_SECRET_KEY', 'ALPACA_SECRET')

if uploaded is not None:
    try:
        raw_uploaded = clean(pd.read_csv(uploaded), naive_tz)
        st.success(f'CSV loaded: {len(raw_uploaded):,} regular-session 1-minute bars')
    except Exception as exc:
        st.error(f'CSV parsing error: {exc}')
        raw_uploaded = None
else:
    raw_uploaded = None


@st.fragment(run_every='30s')
def live_dashboard():
    if not key or not secret_key:
        st.info('Configure Alpaca keys in Streamlit Secrets to enable live charts.')
        return
    try:
        today = dt.datetime.now(NY).date()
        old = history(symbol, feed, key, secret_key, today.isoformat())
        recent = live(symbol, feed, key, secret_key)
        raw = merge_data(old, recent)
        if raw.empty:
            st.warning('No recent market data')
            return
        boxes = make_boxes(raw, lookback, max_width, vol_mult)
        latest_day = raw.index[-1].date()
        fig, levels = chart_for_day(raw, boxes, latest_day, opening, windows[window])
        st.subheader(f'{symbol} | Live chart (refresh every 30 seconds)')
        st.metric('Last completed minute close', f'${raw.close.iloc[-1]:.2f}')
        st.caption(f'Latest candle: {raw.index[-1]} | Feed: {feed.upper()} | 1-minute bars; box signals use completed 5-minute candles')
        if (dt.datetime.now(NY) - raw.index[-1]).total_seconds() > 300:
            st.warning('Data is stale or market is closed. Do not treat this as a live entry.')
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pd.DataFrame([levels]).T.rename(columns={0:'Price level'}), use_container_width=True)
        day_boxes = boxes[boxes.index.date == latest_day]
        if not day_boxes.empty:
            latest = day_boxes.iloc[-1]
            st.subheader('Box signal')
            # Signal is a five-minute bar close; no guarantee it is still actionable.
            signal = {1:'BULLISH BREAKOUT', -1:'BEARISH BREAKDOWN', 0:'WAIT'}[int(latest.signal)]
            st.write(f'**{signal}** | Box high: {latest.box_high:.2f} | Box low: {latest.box_low:.2f} | '
                     f'Volume ratio: {latest.vol_ratio:.2f} | EMA9: {latest.ema9:.2f} | EMA20: {latest.ema20:.2f}')
            st.caption('Signal is descriptive, not a probability. Any entry must wait for the next candle and use a defined stop.')
            events = day_boxes[day_boxes.signal != 0]
            if not events.empty:
                st.dataframe(events[['signal_time', 'signal', 'box_high', 'box_low', 'vol_ratio']].tail(20), use_container_width=True)
    except Exception as exc:
        st.error(f'Live data/chart error: {exc}')


live_dashboard()

st.divider()
st.header('Historical box strategy backtest')
st.caption('Tests completed 5-minute breakout signals, next-candle entries, ATR stops, fixed targets, one position at a time, and no overnight trades.')

if run_test:
    try:
        if raw_uploaded is not None:
            test_raw = raw_uploaded
        elif key and secret_key:
            today = dt.datetime.now(NY).date()
            test_raw = merge_data(history(symbol, feed, key, secret_key, today.isoformat()),
                                  live(symbol, feed, key, secret_key))
        else:
            raise ValueError('Upload a CSV or configure Alpaca credentials')
        if len(set(test_raw.index.date)) < 10:
            raise ValueError('At least 10 sessions recommended for backtesting')
        st.session_state['test_raw'] = test_raw
        # Compare the unfiltered box breakout with the trend+volume filtered version.
        comparisons = []
        for label, trend, volume in [
            ('Box breakout only', False, False),
            ('Box + trend', True, False),
            ('Box + trend + volume', True, True)]:
            b = make_boxes(test_raw, lookback, max_width, vol_mult, trend, volume)
            t = backtest(b, hold_minutes//5, stop_atr, reward_risk, cost_bps)
            summary = summarize(t)
            comparisons.append({'Strategy': label, **summary})
            if label == 'Box + trend + volume':
                st.session_state['test_boxes'] = b
                st.session_state['test_trades'] = t
        st.session_state['test_comparison'] = pd.DataFrame(comparisons)
    except Exception as exc:
        st.error(f'Backtest failed: {exc}')

if 'test_comparison' in st.session_state:
    st.subheader('Strategy comparison')
    table = st.session_state['test_comparison']
    st.dataframe(table.style.format({'Win rate':'{:.1%}', 'Avg net bps':'{:.2f}',
                                     'Total net bps':'{:.2f}', 'Profit factor':'{:.2f}'}),
                 hide_index=True, use_container_width=True)
    st.caption('Total net bps is the arithmetic sum of per-trade basis-point returns, not compounded portfolio performance. '
               'The strategy may be unprofitable. IEX and SIP feeds can produce different signals.')
    trades = st.session_state['test_trades']
    if not trades.empty:
        st.download_button('Download backtest trades CSV', trades.to_csv(index=False),
                           file_name=f'{symbol}_box_trades.csv', mime='text/csv')
        st.dataframe(trades.tail(50), use_container_width=True)
    dates = sorted(set(st.session_state['test_raw'].index.date))
    chosen = st.selectbox('Historical chart replay day', dates, index=len(dates)-1)
    fig, levels = chart_for_day(st.session_state['test_raw'], st.session_state['test_boxes'],
                                chosen, opening, windows[window])
    st.plotly_chart(fig, use_container_width=True)
    st.caption('Chart shows completed 5-minute box levels and signals. Trendlines are visual aids, not entry rules.')
