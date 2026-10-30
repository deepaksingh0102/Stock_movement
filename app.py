"""Stock Movement V11: distinct trading-box colors and optional provisional overlays."""
import datetime as dt
import re
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import pandas_market_calendars as mcal

NY = ZoneInfo('America/New_York')
UTC = dt.timezone.utc
st.set_page_config(page_title='Stock Movement | Box Theory V15.1', layout='wide')


def secret(*names, default=''):
    for name in names:
        v = st.secrets.get(name, '')
        if v and str(v).strip():
            return str(v).strip()
    return default


def clean(raw, naive_tz='America/New_York'):
    df = raw.copy()
    df.columns = [str(c).lower().strip() for c in df.columns]
    if 'timestamp_utc' in df.columns:
        times = pd.to_datetime(df['timestamp_utc'], errors='coerce', utc=True)
        df.index = pd.DatetimeIndex(times).tz_convert(NY)
    elif 'timestamp' in df.columns:
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


@st.cache_data(ttl=86400, show_spinner=False)
def market_hours(day_iso):
    """NYSE regular session schedule, including holidays and early closes."""
    schedule = mcal.get_calendar('NYSE').schedule(
        start_date=day_iso, end_date=day_iso)
    if schedule.empty:
        return None
    return (schedule.iloc[0]['market_open'].to_pydatetime(),
            schedule.iloc[0]['market_close'].to_pydatetime())


def is_regular_session_open():
    now = dt.datetime.now(UTC)
    schedule = market_hours(now.astimezone(NY).date().isoformat())
    return schedule is not None and schedule[0] <= now < schedule[1]


def load_market_data(symbol, feed, key, secret_key):
    """Fetch live during session; reuse last snapshot when closed.

    On a first visit outside market hours, fetch once to populate the chart.
    The cache is isolated per symbol, feed, and New York calendar day.
    """
    today = dt.datetime.now(NY).date()
    cache_id = f'market_snapshot:{symbol}:{feed}:{today.isoformat()}'
    if not is_regular_session_open() and cache_id in st.session_state:
        return st.session_state[cache_id].copy()

    old = history(symbol, feed, key, secret_key, today.isoformat())
    recent = live(symbol, feed, key, secret_key)
    raw = merge_data(old, recent)
    st.session_state[cache_id] = raw.copy()
    return raw


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
        # Display-only provisional box, based solely on completed 5m candles.
        # No trade may use these levels before the full lookback exists.
        d['completed_box_candles'] = np.arange(1, len(d) + 1)
        d['developing_high'] = d.high.rolling(n, min_periods=1).max()
        d['developing_low'] = d.low.rolling(n, min_periods=1).min()
        d['developing'] = d['completed_box_candles'] < n
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
        # Explicit audit fields: these are computed from the SAME previous box
        # used for both the live chart and the backtest.
        d['price_break_up'] = eligible & (d.close > d.box_high)
        d['price_break_down'] = eligible & (d.close < d.box_low)
        d['trend_long_ok'] = (d.ema9 > d.ema20) & (d.close > d.vwap)
        d['trend_short_ok'] = (d.ema9 < d.ema20) & (d.close < d.vwap)
        d['volume_ok'] = d.vol_ratio >= vol_multiplier
        d['signal'] = np.select([long_ok, short_ok], [1, -1], default=0)
        d['signal_time'] = d.index + pd.Timedelta(minutes=5)
        d['status'] = np.select(
            [d.signal == 1, d.signal == -1,
             d.price_break_up, d.price_break_down],
            ['CONFIRMED UP', 'CONFIRMED DOWN', 'PRICE ABOVE BOX', 'PRICE BELOW BOX'],
            default='INSIDE BOX / INELIGIBLE')
        blocks.append(d)
    return pd.concat(blocks).sort_index()


def backtest(boxes, hold_bars=6, stop_atr=1.2, reward_risk=1.5, cost_bps=2.0):
    """Next 5m open entry; stop-first when stop and target both touch a bar.
    One position at a time per session; forced same-day exit; no overnight positions.
    All returns net of assumed round-trip cost. No leverage assumed.
    """
    trades = []
    for day, d in boxes.groupby(boxes.index.date):
        d = d.reset_index().rename(columns={d.index.name or 'index': 'timestamp'})
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


def breakout_audit(boxes, day=None):
    """Single-source audit; timestamps are 5-minute candle CLOSE times in ET."""
    b = boxes if day is None else boxes[boxes.index.date == day]
    if b.empty:
        return pd.DataFrame()
    a = b.copy()
    a['candle_start_et'] = a.index
    a['candle_close_et'] = a['signal_time']
    a['side'] = np.select([a.price_break_up, a.price_break_down],
                          ['UP', 'DOWN'], default='INSIDE')
    a['trend_ok'] = np.where(a.price_break_up, a.trend_long_ok,
                              np.where(a.price_break_down, a.trend_short_ok, False))
    cols = ['candle_start_et', 'candle_close_et', 'open', 'close',
            'box_high', 'box_low', 'box_width_pct', 'side',
            'trend_ok', 'vol_ratio', 'volume_ok', 'signal', 'status']
    return a.loc[a.side != 'INSIDE', cols].reset_index(drop=True)


def minute_chart_indicators(raw):
    """Plot-only 1-minute EMA/VWAP. Independent of the 5m signal model.

    Compute before chart-window slicing, reset at each session open, and
    never use a future candle. VWAP uses typical-price approximation.
    """
    if raw.empty:
        return pd.DataFrame(index=raw.index,
                            columns=['ema9_1m', 'ema20_1m', 'vwap_1m'])
    parts = []
    for _, session in raw.groupby(raw.index.date):
        s = session[['high', 'low', 'close', 'volume']].copy()
        s['ema9_1m'] = s['close'].ewm(span=9, adjust=False).mean()
        s['ema20_1m'] = s['close'].ewm(span=20, adjust=False).mean()
        typical = (s['high'] + s['low'] + s['close']) / 3.0
        cumulative_volume = s['volume'].cumsum().replace(0, np.nan)
        s['vwap_1m'] = (typical * s['volume']).cumsum() / cumulative_volume
        parts.append(s[['ema9_1m', 'ema20_1m', 'vwap_1m']])
    return pd.concat(parts).sort_index()


def live_minute_box(raw, day, lookback):
    """Display-only 1-minute rolling box using PRIOR completed 1m bars.

    The window is lookback * 5 minutes, matching the selected box duration.
    It updates with each completed minute, never using that minute's OHLC
    in its own boundary. It does NOT affect 5m backtest/trading signals.
    """
    d = raw[raw.index.date == day].copy()
    if d.empty:
        return pd.DataFrame(index=d.index)
    n = int(lookback) * 5
    # Minimum 1 prior minute allows an early provisional display.
    # Only the 5m prior box from make_boxes() is tradable.
    out = pd.DataFrame(index=d.index)
    out['live_box_high'] = d.high.shift(1).rolling(n, min_periods=1).max()
    out['live_box_low'] = d.low.shift(1).rolling(n, min_periods=1).min()
    out['minutes_ready'] = np.minimum(np.arange(len(d)), n)
    return out




def trend_duration_tracker(boxes, day=None):
    """Track completed-5m indicator alignment after confirmed box signals.

    Uses the same 5m EMA9, EMA20 and VWAP as make_boxes().
    Each signal begins an episode; same-direction repeat signals do not reset it.
    An episode ends at the first completed candle losing full alignment.
    Observations after a loss do not restore that episode's duration.
    """
    if boxes.empty:
        return pd.DataFrame(), pd.DataFrame(), None
    d = boxes if day is None else boxes[boxes.index.date == day]
    if d.empty:
        return pd.DataFrame(), pd.DataFrame(), None
    d = d.sort_index().copy()
    bull = (d['ema9'] > d['ema20']) & (d['close'] > d['vwap'])
    bear = (d['ema9'] < d['ema20']) & (d['close'] < d['vwap'])
    d['alignment'] = np.select([bull, bear], ['BULLISH', 'BEARISH'], default='SIDEWAYS')
    d['alignment_code'] = np.select([bull, bear], [1, -1], default=0)
    episodes = []
    active = None
    prev_alignment = None
    for _, bar in d.iterrows():
        t = bar['signal_time']
        alignment = str(bar['alignment'])
        direction = int(bar['signal'])
        if active is not None and active['first_loss_et'] is None:
            if alignment != active['direction']:
                active['first_loss_et'] = t
                active['aligned_minutes'] = max(0, int((t - active['trigger_et']).total_seconds() // 60))
                active['outcome'] = 'REVERSED' if alignment in ('BULLISH', 'BEARISH') else 'WEAKENED'
        if direction != 0:
            new_direction = 'BULLISH' if direction == 1 else 'BEARISH'
            # Do not restart a still-aligned trend for repeated breakouts.
            if active is None or active['direction'] != new_direction or active['first_loss_et'] is not None:
                active = {
                    'trigger_et': t,
                    'direction': new_direction,
                    'first_loss_et': None,
                    'aligned_minutes': None,
                    'outcome': 'ACTIVE',
                    'repeat_triggers': 0,
                }
                episodes.append(active)
            else:
                active['repeat_triggers'] += 1
        prev_alignment = alignment
    last_t = d['signal_time'].iloc[-1]
    for episode in episodes:
        if episode['first_loss_et'] is None:
            episode['aligned_minutes'] = int((last_t - episode['trigger_et']).total_seconds() // 60)
            episode['outcome'] = 'ACTIVE / CENSORED'
    events = pd.DataFrame(episodes)
    current = d.iloc[-1]
    last_trigger = events.iloc[-1] if not events.empty else None
    if last_trigger is None:
        status = 'NO BOX TRIGGER'
        age = None
        trigger_time = None
    else:
        trigger_time = last_trigger['trigger_et']
        age = int((last_t - trigger_time).total_seconds() // 60)
        if current['alignment'] == last_trigger['direction']:
            status = 'CONTINUING' if pd.isna(last_trigger['first_loss_et']) else 'RE-ALIGNED (NEW TRIGGER NEEDED)'
        elif current['alignment'] == 'SIDEWAYS':
            status = 'WEAKENING / SIDEWAYS'
        else:
            status = 'REVERSING'
    snapshot = {
        'asof_et': last_t,
        'alignment': str(current['alignment']),
        'status': status,
        'trigger_direction': None if last_trigger is None else last_trigger['direction'],
        'trigger_time': trigger_time,
        'minutes_since_trigger': age,
        'ema9': float(current['ema9']),
        'ema20': float(current['ema20']),
        'vwap': float(current['vwap']),
        'close': float(current['close']),
        'volume_ratio': float(current['vol_ratio']) if pd.notna(current['vol_ratio']) else np.nan,
        'total_triggers': int((d['signal'] != 0).sum()),
        'trend_episodes': len(episodes),
        'last_completed_alignment_minutes': None if last_trigger is None else int(last_trigger['aligned_minutes']),
    }
    return d, events, snapshot


def indicator_direction(ema9, ema20, close, vwap):
    """Direction requires BOTH EMA and VWAP agreement; volume is not directional."""
    vals = [ema9, ema20, close, vwap]
    if not all(pd.notna(v) and np.isfinite(v) for v in vals):
        return 'UNAVAILABLE'
    if ema9 > ema20 and close > vwap:
        return 'BULLISH'
    if ema9 < ema20 and close < vwap:
        return 'BEARISH'
    return 'SIDEWAYS'


def dual_timeframe_snapshot(raw, boxes, day=None):
    """Use latest completed 1m candle and latest completed 5m candle, never partial 5m."""
    if raw.empty or boxes.empty:
        return None
    if day is None:
        day = raw.index[-1].date()
    minute = raw[raw.index.date == day].copy()
    five = boxes[boxes.index.date == day].copy()
    if minute.empty or five.empty:
        return None
    indicators = minute_chart_indicators(minute)
    last_minute = minute.iloc[-1]
    last_indicators = indicators.iloc[-1]
    last_five = five.iloc[-1]
    minute_direction = indicator_direction(
        last_indicators['ema9_1m'], last_indicators['ema20_1m'],
        last_minute['close'], last_indicators['vwap_1m'])
    five_direction = indicator_direction(
        last_five['ema9'], last_five['ema20'],
        last_five['close'], last_five['vwap'])
    if minute_direction == five_direction and minute_direction in ('BULLISH', 'BEARISH'):
        overall = minute_direction
    elif minute_direction == 'UNAVAILABLE' or five_direction == 'UNAVAILABLE':
        overall = 'UNAVAILABLE'
    elif minute_direction == 'SIDEWAYS' and five_direction == 'SIDEWAYS':
        overall = 'SIDEWAYS'
    else:
        overall = 'MIXED'
    return {
        'one_minute': minute_direction,
        'five_minute': five_direction,
        'overall': overall,
        'minute_asof': minute.index[-1] + pd.Timedelta(minutes=1),
        'five_asof': last_five['signal_time'],
        'one_ema9': float(last_indicators['ema9_1m']),
        'one_ema20': float(last_indicators['ema20_1m']),
        'one_close': float(last_minute['close']),
        'one_vwap': float(last_indicators['vwap_1m']),
        'five_ema9': float(last_five['ema9']),
        'five_ema20': float(last_five['ema20']),
        'five_close': float(last_five['close']),
        'five_vwap': float(last_five['vwap']),
        'volume_ratio': float(last_five['vol_ratio']) if pd.notna(last_five['vol_ratio']) else np.nan,
        'volume_ok': bool(last_five['volume_ok']),
    }


def render_trend_duration_tracker(boxes, raw, day=None):
    st.subheader('Trend Duration Tracker — 1m + 5m')
    timeline, events, state = trend_duration_tracker(boxes, day)
    dual = dual_timeframe_snapshot(raw, boxes, day)
    if state is None or dual is None:
        st.info('Waiting for completed one-minute and five-minute candles.')
        return

    # Do not call a trend BULLISH if the chart's 1m indicators are bearish.
    a, b, c, d = st.columns(4)
    a.metric('Combined alignment', dual['overall'])
    b.metric('1-minute direction', dual['one_minute'])
    c.metric('5-minute direction', dual['five_minute'])
    d.metric('Confirmed box triggers', state['total_triggers'])

    original = state['trigger_direction']
    if original is None:
        condition = 'NO BOX TRIGGER'
        st.info('No confirmed breakout yet. Indicator direction alone is not a trade signal.')
    elif dual['overall'] == original:
        condition = 'CONTINUING' if state['status'] == 'CONTINUING' else 'RE-ALIGNED / NEW TRIGGER NEEDED'
    elif dual['one_minute'] != original and dual['five_minute'] == original:
        condition = 'WEAKENING — 1m AGAINST 5m'
    elif dual['one_minute'] == dual['five_minute'] and dual['overall'] in ('BULLISH', 'BEARISH') and dual['overall'] != original:
        condition = 'REVERSING — BOTH TIMEFRAMES'
    else:
        condition = 'MIXED / SIDEWAYS'

    st.markdown(f'**Trend condition:** {condition}')
    if dual['overall'] == 'MIXED':
        st.warning('1-minute and 5-minute indicators disagree or are neutral. Do not interpret this as a fully confirmed bullish or bearish trend.')
    if state['trigger_time'] is not None:
        st.write(
            f"**Latest box trigger:** {original} at "
            f"{state['trigger_time'].strftime('%I:%M %p ET')} | "
            f"**Time since trigger:** {state['minutes_since_trigger']} min | "
            f"**Initial 5m alignment:** {state['last_completed_alignment_minutes']} min"
        )
    st.caption(
        '1m values match the chart. 5m values match the breakout/backtest strategy. '
        'The two clocks may differ by up to five minutes. Volume is a separate breakout filter, not a trend direction vote.'
    )

    rows = []
    for timeframe, prefix, direction in [
        ('1 minute', 'one', dual['one_minute']),
        ('5 minutes', 'five', dual['five_minute']),
    ]:
        ema9, ema20 = dual[f'{prefix}_ema9'], dual[f'{prefix}_ema20']
        close, vwap = dual[f'{prefix}_close'], dual[f'{prefix}_vwap']
        rows.append({
            'Timeframe': timeframe,
            'EMA 9': round(ema9, 3), 'EMA 20': round(ema20, 3),
            'EMA direction': 'BULLISH' if ema9 > ema20 else ('BEARISH' if ema9 < ema20 else 'NEUTRAL'),
            'Close': round(close, 3), 'VWAP': round(vwap, 3),
            'VWAP direction': 'BULLISH' if close > vwap else ('BEARISH' if close < vwap else 'NEUTRAL'),
            'Overall': direction,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(
        f"1m completed: {dual['minute_asof'].strftime('%I:%M %p ET')} | "
        f"5m completed: {dual['five_asof'].strftime('%I:%M %p ET')} | "
        f"5m relative volume: {dual['volume_ratio']:.2f}x | "
        f"Volume filter: {'PASS' if dual['volume_ok'] else 'FAIL'}"
    )
    if not events.empty:
        st.markdown('**Confirmed breakout episodes (5-minute strategy)**')
        visible = events.rename(columns={
            'trigger_et': 'Trigger time (ET)', 'direction': 'Trigger',
            'first_loss_et': 'First 5m alignment loss (ET)',
            'aligned_minutes': 'Continuous 5m aligned minutes',
            'outcome': '5m outcome', 'repeat_triggers': 'Repeated triggers',
        })
        st.dataframe(visible.tail(25), hide_index=True, use_container_width=True)
    st.caption('Trend alignment and duration are descriptive; neither measures realized trading profitability.')

def reversal_analysis(raw, boxes, day, reclaim_bars=3):
    """Research-only reversal layer. No changes to backtest entry rules.

    Breakout is a completed 5m CLOSE beyond the prior eligible box, regardless
    of volume/trend confirmation. The original boundary is frozen at crossing.
    Reclaim is recognized only on a later completed 5m candle, within the
    specified window. No future observations are used at any signal time.
    """
    minute = raw[raw.index.date == day].copy()
    five = boxes[boxes.index.date == day].copy()
    if minute.empty or five.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    m = minute_chart_indicators(minute)
    m['close'] = minute['close']
    fast = minute['close'].ewm(span=12, adjust=False).mean()
    slow = minute['close'].ewm(span=26, adjust=False).mean()
    m['macd'] = fast - slow
    m['macd_signal'] = m['macd'].ewm(span=9, adjust=False).mean()
    m['histogram'] = m['macd'] - m['macd_signal']
    m['hist_slope'] = m['histogram'].diff()
    # A 3-bar slope is less sensitive to single-candle noise.
    m['hist_slope_3'] = m['histogram'].diff(3)
    m['momentum'] = np.select(
        [(m.histogram > 0) & (m.hist_slope_3 > 0),
         (m.histogram < 0) & (m.hist_slope_3 < 0),
         (m.histogram < 0) & (m.hist_slope_3 > 0),
         (m.histogram > 0) & (m.hist_slope_3 < 0)],
        ['BULLISH ACCELERATING', 'BEARISH ACCELERATING',
         'BEARISH WEAKENING', 'BULLISH WEAKENING'], default='NEUTRAL')
    f = five.copy()
    fast5 = f.close.ewm(span=12, adjust=False).mean()
    slow5 = f.close.ewm(span=26, adjust=False).mean()
    f['macd'] = fast5 - slow5
    f['macd_signal'] = f.macd.ewm(span=9, adjust=False).mean()
    f['histogram'] = f.macd - f.macd_signal
    f['hist_slope'] = f.histogram.diff()
    f['momentum'] = np.select(
        [(f.histogram < 0) & (f.hist_slope > 0),
         (f.histogram > 0) & (f.hist_slope < 0),
         (f.histogram > 0) & (f.hist_slope > 0),
         (f.histogram < 0) & (f.hist_slope < 0)],
        ['BEARISH WEAKENING', 'BULLISH WEAKENING',
         'BULLISH ACCELERATING', 'BEARISH ACCELERATING'], default='NEUTRAL')
    events = []
    active = None
    for i, (_, bar) in enumerate(f.iterrows()):
        # First evaluate a previously opened candidate; never reclaim on same bar.
        if active is not None:
            elapsed = i - active['start_i']
            reclaimed = ((active['direction'] == 'DOWN' and bar.close > active['frozen_boundary'])
                         or (active['direction'] == 'UP' and bar.close < active['frozen_boundary']))
            if reclaimed and elapsed <= reclaim_bars:
                events.append({'time_et': bar.signal_time, 'event': 'FAILED BREAKDOWN' if active['direction']=='DOWN' else 'FAILED BREAKOUT',
                               'original_break_et': active['break_time'], 'frozen_level': active['frozen_boundary'],
                               'close': float(bar.close), 'bars_to_reclaim': elapsed})
                active = None
            elif elapsed >= reclaim_bars:
                active = None
        # Only one candidate at a time; new break uses strictly previous box.
        if active is None and pd.notna(bar.box_high) and bar.box_width_pct <= max_width_for_reversal:
            direction = 'UP' if bar.close > bar.box_high else ('DOWN' if bar.close < bar.box_low else None)
            if direction:
                active = {'direction': direction,
                          'frozen_boundary': float(bar.box_high if direction=='UP' else bar.box_low),
                          'break_time': bar.signal_time, 'start_i': i}
    return m, f, pd.DataFrame(events)


# This threshold is assigned from the current sidebar before charts are drawn.
max_width_for_reversal = 0.9


def render_reversal_panel(raw, boxes, day, reclaim_bars=3):
    st.subheader('Momentum & Failed-Breakout Monitor (experimental)')
    m, f, events = reversal_analysis(raw, boxes, day, reclaim_bars)
    if m.empty or f.empty:
        st.info('Waiting for completed candles to assess momentum and failed breakouts.')
        return
    last_m, last_f = m.iloc[-1], f.iloc[-1]
    latest_event = events.iloc[-1] if not events.empty else None
    a, b, c = st.columns(3)
    a.metric('1m MACD momentum', str(last_m.momentum))
    b.metric('5m MACD momentum', str(last_f.momentum))
    c.metric('Failed breakouts detected', len(events))
    if latest_event is not None:
        st.warning(f"Latest: {latest_event['event']} at {latest_event['time_et'].strftime('%I:%M %p ET')} "
                   f"| Frozen level ${latest_event['frozen_level']:.2f} "
                   f"| Reclaimed after {int(latest_event['bars_to_reclaim']) * 5} minutes")
    if last_f.ema9 < last_f.ema20 and last_f.close < last_f.vwap and last_m.hist_slope_3 > 0:
        st.info('Bearish EMA/VWAP alignment, but 1m MACD histogram is improving: possible bearish weakening, not a confirmed reversal.')
    elif last_f.ema9 > last_f.ema20 and last_f.close > last_f.vwap and last_m.hist_slope_3 < 0:
        st.info('Bullish EMA/VWAP alignment, but 1m MACD histogram is weakening: possible bullish weakening, not a confirmed reversal.')
    st.caption('MACD slope and box reclaims are diagnostic warnings only. The original 5m box signals and backtest rules are unchanged. Reclaim events require a later completed 5m close across the frozen breakout boundary.')
    hist = go.Figure()
    hist.add_trace(go.Bar(x=m.index, y=m.histogram,
        marker_color=np.where(m.histogram >= 0, '#16a34a', '#dc2626'), name='1m MACD histogram'))
    hist.add_trace(go.Scatter(x=m.index, y=m.hist_slope_3, name='Histogram change (3m)',
                              line=dict(color='#7c3aed', width=1.6)))
    hist.update_layout(height=260, title='MACD histogram and 3-minute change',
                       xaxis_title='Eastern Time', margin=dict(t=40,b=30))
    st.plotly_chart(hist, use_container_width=True)
    if not events.empty:
        st.dataframe(events.tail(20), hide_index=True, use_container_width=True)



# ================================================================
# V15 detection-only engine. Does NOT change make_boxes or backtest.
# ================================================================
def time_of_day_rvol(raw, boxes, min_sessions=5, history_sessions=20):
    """5m volume / mean of the SAME ET time slot in prior sessions only.

    No same-session or future observations are used for a reference.
    Insufficient prior sessions -> NaN, never a fabricated 1.0 ratio.
    """
    if boxes.empty:
        return pd.Series(dtype=float, name='tod_rvol')
    five = to_five_minute(raw)
    if five.empty:
        return pd.Series(np.nan, index=boxes.index, name='tod_rvol')
    days = sorted(set(five.index.date))
    results = pd.Series(np.nan, index=boxes.index, dtype=float, name='tod_rvol')
    for k, day in enumerate(days):
        earlier = days[max(0, k - history_sessions):k]
        if len(earlier) < min_sessions:
            continue
        hist = five[np.isin(five.index.date, earlier)].copy()
        hist['slot'] = hist.index.strftime('%H:%M')
        slot_mean = hist.groupby('slot')['volume'].mean().replace(0, np.nan)
        today = five[five.index.date == day]
        vals = today['volume'].to_numpy() / slot_mean.reindex(today.index.strftime('%H:%M')).to_numpy()
        idx = today.index.intersection(results.index)
        if len(idx):
            results.loc[idx] = pd.Series(vals, index=today.index).reindex(idx)
    return results


def detection_engine(raw, boxes, reclaim_bars=3, sustain_bars=2,
                     min_tod_sessions=5, min_move_atr=0.25):
    """Causal event state machine on completed 5m bars; no execution decisions.

    Stages: CROSSED -> CONFIRMED or UNCONFIRMED -> SUSTAINED,
    FAILED (reclaim) or WINDOW EXPIRED. A frozen boundary never moves.
    Distinct crossings are logged only on transition across the *prior* box.
    A new crossing can start only when the previous candidate resolves.
    """
    if boxes.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    b = boxes.copy().sort_index()
    b['tod_rvol'] = time_of_day_rvol(raw, b, min_sessions=min_tod_sessions)
    b['macd_hist'] = np.nan
    b['macd_slope'] = np.nan
    b['macd_acceleration'] = np.nan
    for day, g in b.groupby(b.index.date):
        fast = g.close.ewm(span=12, adjust=False).mean()
        slow = g.close.ewm(span=26, adjust=False).mean()
        macd = fast - slow
        hist = macd - macd.ewm(span=9, adjust=False).mean()
        b.loc[g.index, 'macd_hist'] = hist
        b.loc[g.index, 'macd_slope'] = hist.diff()
        b.loc[g.index, 'macd_acceleration'] = hist.diff().diff()
    events = []
    candidates = []
    for day, g in b.groupby(b.index.date):
        active = None
        prev_up = prev_down = False
        prev_momentum_warning = None
        rows = list(g.iterrows())
        for i, (idx, bar) in enumerate(rows):
            t = bar.signal_time
            # A warning is emitted only when an opposing momentum condition
            # first appears. No repeated event on every subsequent candle.
            warning = None
            if i >= 26 and pd.notna(bar.macd_slope) and pd.notna(bar.macd_acceleration):
                if (bar.ema9 < bar.ema20 and bar.close < bar.vwap and
                        bar.macd_slope > 0 and bar.macd_acceleration > 0):
                    warning = 'BEARISH MOMENTUM WEAKENING'
                elif (bar.ema9 > bar.ema20 and bar.close > bar.vwap and
                        bar.macd_slope < 0 and bar.macd_acceleration < 0):
                    warning = 'BULLISH MOMENTUM WEAKENING'
            if warning is not None and warning != prev_momentum_warning:
                events.append(dict(time_et=t, event=warning, candidate_id=None,
                                   direction='UP' if warning.startswith('BEARISH') else 'DOWN',
                                   frozen_boundary=np.nan, close=float(bar.close),
                                   bars_since_cross=np.nan))
            prev_momentum_warning = warning
            # Use existing strategy eligibility as the source of truth:
            # price_break_up/down already include configured box-width threshold.
            up = bool(bar.price_break_up)
            down = bool(bar.price_break_down)
            if active is not None:
                age = i - active['start_i']
                direction = active['direction']
                boundary = active['frozen_boundary']
                reclaimed = bool(bar.close < boundary) if direction == 'UP' else bool(bar.close > boundary)
                if reclaimed:
                    active['outcome'] = 'FAILED'
                    active['resolved_at_et'] = t
                    active['bars_to_resolution'] = age
                    events.append(dict(time_et=t, event='FAILED BREAKOUT' if direction == 'UP' else 'FAILED BREAKDOWN',
                                       candidate_id=active['id'], direction=direction,
                                       frozen_boundary=boundary, close=float(bar.close), bars_since_cross=age))
                    active = None
                elif age == sustain_bars:
                    events.append(dict(time_et=t, event='SUSTAINED', candidate_id=active['id'],
                                       direction=direction, frozen_boundary=boundary,
                                       close=float(bar.close), bars_since_cross=age))
                    active['sustained'] = True
                if active is not None and age >= reclaim_bars:
                    active['outcome'] = 'SUSTAINED / WINDOW EXPIRED' if active['sustained'] else 'WINDOW EXPIRED'
                    active['resolved_at_et'] = t
                    active['bars_to_resolution'] = age
                    active = None
            # Prevent repeated triggers while price stays outside a moving box.
            crossed_up = up and not prev_up
            crossed_down = down and not prev_down
            prev_up, prev_down = up, down
            if active is None and (crossed_up or crossed_down):
                direction = 'UP' if crossed_up else 'DOWN'
                boundary = float(bar.box_high if crossed_up else bar.box_low)
                cid = f'{day}-{t.strftime("%H%M")}-{direction}'
                confirmation = bool(bar.signal == (1 if crossed_up else -1))
                candidate = dict(id=cid, day=str(day), crossed_at_et=t,
                                 direction=direction, frozen_boundary=boundary,
                                 crossing_close=float(bar.close), confirmed=confirmation,
                                 volume_ratio=float(bar.vol_ratio) if pd.notna(bar.vol_ratio) else np.nan,
                                 tod_rvol=float(bar.tod_rvol) if pd.notna(bar.tod_rvol) else np.nan,
                                 macd_slope=float(bar.macd_slope) if pd.notna(bar.macd_slope) else np.nan,
                                 macd_acceleration=float(bar.macd_acceleration) if pd.notna(bar.macd_acceleration) else np.nan,
                                 start_i=i, sustained=False, outcome='OPEN / CENSORED',
                                 resolved_at_et=pd.NaT, bars_to_resolution=np.nan,
                                 atr=float(bar.atr) if pd.notna(bar.atr) else np.nan)
                candidates.append(candidate)
                active = candidate
                events.append(dict(time_et=t, event='CROSSED', candidate_id=cid,
                                   direction=direction, frozen_boundary=boundary,
                                   close=float(bar.close), bars_since_cross=0))
                events.append(dict(time_et=t, event='CONFIRMED' if confirmation else 'UNCONFIRMED',
                                   candidate_id=cid, direction=direction, frozen_boundary=boundary,
                                   close=float(bar.close), bars_since_cross=0))
        # End-of-file / session is censored, not a false positive.
    c = pd.DataFrame(candidates)
    e = pd.DataFrame(events)
    if c.empty:
        return b, c, e
    # Validation: forward price observations only after the crossing candle.
    # Horizon outcomes are scored only when a future completed candle exists.
    for minutes in (5, 15, 30):
        c[f'forward_{minutes}m_bps'] = np.nan
        c[f'correct_{minutes}m'] = np.nan
        c[f'meaningful_{minutes}m'] = np.nan
        for day, g in b.groupby(b.index.date):
            for row_i in c.index[c.day == str(day)]:
                start_t = c.at[row_i, 'crossed_at_et']
                future = g[g.signal_time == start_t + pd.Timedelta(minutes=minutes)]
                if future.empty:
                    continue
                sign = 1 if c.at[row_i, 'direction'] == 'UP' else -1
                signed_return = sign * (float(future.close.iloc[0]) / c.at[row_i, 'crossing_close'] - 1)
                c.at[row_i, f'forward_{minutes}m_bps'] = signed_return * 10000
                c.at[row_i, f'correct_{minutes}m'] = int(signed_return > 0)
                atr = c.at[row_i, 'atr']
                if pd.notna(atr) and atr > 0:
                    c.at[row_i, f'meaningful_{minutes}m'] = int(sign * (float(future.close.iloc[0]) - c.at[row_i, 'crossing_close']) >= min_move_atr * atr)
    c = c.drop(columns=['start_i', 'atr'])
    return b, c, e


def detection_metrics(candidates):
    """Matured denominators only; no accuracy claims for censored events."""
    if candidates.empty:
        return pd.DataFrame([{'Metric':'Crossing candidates', 'Value':0}])
    n = len(candidates)
    closed = candidates[candidates.outcome != 'OPEN / CENSORED']
    rows = [
        ('Crossing candidates', n),
        ('Confirmed by original V14 rules', int(candidates.confirmed.sum())),
        ('Failed / matured candidates', int((closed.outcome == 'FAILED').sum())),
        ('Matured candidates', len(closed)),
        ('Failure rate among matured', float((closed.outcome == 'FAILED').mean()) if len(closed) else np.nan),
        ('Still open / censored', n - len(closed)),
        ('Time-of-day RVOL available', int(candidates.tod_rvol.notna().sum())),
    ]
    for horizon in (5, 15, 30):
        eligible = candidates[f'correct_{horizon}m'].dropna()
        meaningful = candidates[f'meaningful_{horizon}m'].dropna()
        rows.extend([
            (f'{horizon}m directional evaluations', len(eligible)),
            (f'{horizon}m directional hit rate', float(eligible.mean()) if len(eligible) else np.nan),
            (f'{horizon}m ATR-move evaluations', len(meaningful)),
            (f'{horizon}m >=0.25 ATR continuation rate', float(meaningful.mean()) if len(meaningful) else np.nan),
        ])
    return pd.DataFrame(rows, columns=['Metric','Value'])


def render_detection_lab(raw, boxes, day, reclaim_bars, sustain_bars, min_tod_sessions):
    st.subheader('V15 | Breakout & Reversal Detection Lab')
    b, candidates, events = detection_engine(raw, boxes, reclaim_bars,
                                             sustain_bars, min_tod_sessions)
    if b.empty:
        st.info('Waiting for complete five-minute bars.')
        return
    day_candidates = candidates[candidates.day == str(day)] if not candidates.empty else candidates
    day_events = events[events.time_et.dt.date == day] if not events.empty else events
    a, z, c, d = st.columns(4)
    a.metric('Box crossings', len(day_candidates))
    z.metric('V14-confirmed crossings', int(day_candidates.confirmed.sum()) if not day_candidates.empty else 0)
    c.metric('Failed / reclaimed', int((day_candidates.outcome == 'FAILED').sum()) if not day_candidates.empty else 0)
    d.metric('Open / censored', int((day_candidates.outcome == 'OPEN / CENSORED').sum()) if not day_candidates.empty else 0)
    latest = b[b.index.date == day]
    if not latest.empty:
        bar = latest.iloc[-1]
        tod = 'N/A (insufficient prior sessions)' if pd.isna(bar.tod_rvol) else f'{bar.tod_rvol:.2f}x'
        st.write(f'**Time-of-day RVOL:** {tod} | **5m MACD histogram slope:** {bar.macd_slope:.5f} | **Acceleration:** {bar.macd_acceleration:.5f}')
        st.caption('Slope = histogram change per completed 5m bar; acceleration = change in slope. '
                   'Time-of-day RVOL uses only earlier sessions at the same Eastern-time five-minute slot.')
    if not day_candidates.empty:
        last = day_candidates.iloc[-1]
        st.write(f"**Latest frozen boundary:** ${last.frozen_boundary:.3f} ({last.direction}) | "
                 f"**Stage:** {last.outcome} | **Crossed:** {last.crossed_at_et.strftime('%I:%M %p ET')}")
    warning_count = int(day_events.event.str.contains('MOMENTUM WEAKENING', na=False).sum()) if not day_events.empty else 0
    st.caption(f'MACD slope + acceleration reversal-warning transitions: {warning_count} (diagnostic, not confirmed reversals).')
    st.markdown('**Detection events (Eastern Time)**')
    if day_events.empty:
        st.info('No qualifying boundary-crossing events for this session.')
    else:
        st.dataframe(day_events.tail(100), hide_index=True, use_container_width=True)
        st.download_button('Download V15 event log CSV', day_events.to_csv(index=False),
                           file_name=f'detection_events_{day}.csv', mime='text/csv',
                           key=f'event_download_{day}')
    st.markdown('**Detection validation (not trading returns)**')
    metrics = detection_metrics(day_candidates)
    st.dataframe(metrics, hide_index=True, use_container_width=True)
    if not day_candidates.empty:
        st.download_button('Download V15 candidate validation CSV', day_candidates.to_csv(index=False),
                           file_name=f'detection_candidates_{day}.csv', mime='text/csv',
                           key=f'candidate_download_{day}')
    st.caption('All event states are computed from completed 5m bars. Forward outcomes are retrospective diagnostics, '
               'never used to generate historical signals. Open cases and missing future candles are excluded from matured rates. '
               'These measures do not demonstrate predictive accuracy without multi-session holdout testing.')


def chart_for_day(raw, boxes, day, opening_minutes, visible_bars, chart_display="Both", box_lookback=12, show_developing_boxes=False):
    # Calculate the indicators on ALL available 1m candles for the day,
    # then slice the display window. This avoids restarting the EMAs when
    # switching from Since market open to 5/15/60 minutes.
    d = raw[raw.index.date == day].copy()
    minute_indicators = minute_chart_indicators(d)
    d = d.join(minute_indicators)
    d = d.join(live_minute_box(raw, day, box_lookback))
    b = boxes[boxes.index.date == day].copy()
    if visible_bars:
        d = d.tail(visible_bars)
    fig = go.Figure()
    _, v15_candidates, v15_events = detection_engine(raw, boxes, reclaim_window,
                                                     sustain_window, min_tod_sessions)
    if not v15_events.empty:
        for kind, color, shape in [
            ('CROSSED', '#f59e0b', 'circle-open'),
            ('SUSTAINED', '#16a34a', 'diamond'),
            ('FAILED BREAKOUT', '#a855f7', 'triangle-down'),
            ('FAILED BREAKDOWN', '#f97316', 'triangle-up')]:
            marked = v15_events[(v15_events.event == kind) &
                                (v15_events.time_et.dt.date == day)]
            if not marked.empty:
                fig.add_trace(go.Scatter(x=marked.time_et, y=marked.close,
                    mode='markers', name='V15 ' + kind,
                    marker=dict(symbol=shape, size=11, color=color),
                    hovertemplate=kind + '<br>%{x}<br>$%{y:.3f}<extra></extra>'))
    _, _, failed_events = reversal_analysis(raw, boxes, day, reclaim_window)
    if chart_display in ('Candlestick', 'Both'):
        fig.add_trace(go.Candlestick(x=d.index, open=d.open, high=d.high, low=d.low,
                                     close=d.close, name='Price'))
    if chart_display in ('Line', 'Both'):
        fig.add_trace(go.Scatter(
            x=d.index, y=d.close, mode='lines', name='1m close price',
            line=dict(color='#16a34a', width=2.3),
            hovertemplate='Time: %{x}<br>Close: $%{y:.3f}<extra></extra>'))
    # Optional, low-contrast reference overlays. Not used for trading.
    if show_developing_boxes:
        for field, color, label in [
            ('live_box_high', '#9CA3AF', 'LIVE BOX_HIGH (1m, display only)'),
            ('live_box_low', '#CBD5E1', 'LIVE BOX_LOW (1m, display only)'),
        ]:
            fig.add_trace(go.Scatter(
                x=d.index, y=d[field], mode='lines', name=label,
                line=dict(color=color, width=1.5, dash='dot', shape='hv'),
                hovertemplate='%{x}<br>' + label + ': $%{y:.3f}<extra></extra>'
            ))
    # Extend the last known prior 5m box to the latest 1m candle on screen.
    # No new five-minute signal is inferred from this extension.
    if not b.empty and not d.empty:
        # All box values and signals become known at the five-minute candle CLOSE.
        # Plot the prior-box boundaries at that timestamp, not at candle start.
        shown = b[(b.signal_time >= d.index.min()) &
                  (b.signal_time <= d.index[-1] + pd.Timedelta(minutes=1))]
        # Developing box is informational only. At 09:35 ET, the first
        # completed 09:30-09:35 candle supplies the first provisional levels.
        # Keep it separate from prior-box boundaries used for trade signals.
        if show_developing_boxes:
            developing = shown[shown['developing']]
            for field, color, label in [
                ('developing_high', '#D1D5DB', 'Developing BOX_HIGH (not tradable)'),
                ('developing_low', '#E5E7EB', 'Developing BOX_LOW (not tradable)')
            ]:
                fig.add_trace(go.Scatter(
                    x=developing.signal_time, y=developing[field],
                    mode='lines+markers', name=label,
                    line=dict(color=color, width=1.5, dash='dot', shape='hv'),
                    marker=dict(size=4),
                    customdata=developing['completed_box_candles'],
                    hovertemplate='%{x}<br>' + label + ': $%{y:.3f}' +
                                  '<br>Completed candles: %{customdata}<extra></extra>'
                ))
        # Draw prior-box levels through the most recent minute so the lines
        # do not appear to stop 1-5 minutes behind the live price.
        # This is a visual hold, NOT a new signal.
        confirmed_events = shown[shown.signal != 0].copy()
        if not shown.empty and d.index[-1] > shown.signal_time.iloc[-1]:
            last_row = shown.iloc[-1].copy()
            last_row['signal_time'] = d.index[-1]
            shown = pd.concat([shown, pd.DataFrame([last_row])], ignore_index=True)
        # Confirmed prior box stays identical to the backtest reference.
        # First eligible signal is the close of candle N+1, so the prior
        # reference becomes visible then. No artificial early signals.
        for name, color, label in [
            ('box_high', '#DC2626', 'BOX_HIGH (Trading)'),
            ('box_low', '#2563EB', 'BOX_LOW (Trading)')
        ]:
            fig.add_trace(go.Scatter(x=shown.signal_time, y=shown[name],
                                     mode='lines+markers', name=label,
                                     line=dict(color=color, width=3.5, shape='hv'),
                                     marker=dict(size=5, color=color),
                                     customdata=shown.close,
                                     hovertemplate=(
                                         '%{x}<br>' + label + ': $%{y:.3f}' +
                                         '<br>5m close: %{customdata:.3f}<extra></extra>')))
        for side, color, marker, label in [(1, 'green', 'triangle-up', 'Confirmed UP'),
                                           (-1, 'red', 'triangle-down', 'Confirmed DOWN')]:
            ev = confirmed_events[confirmed_events.signal == side]
            fig.add_trace(go.Scatter(x=ev.signal_time, y=ev.close,
                                     mode='markers', name=label,
                                     marker=dict(color=color, size=12, symbol=marker)))
    # Chart indicators use 1m candles, unlike the box strategy's 5m
    # EMA/VWAP confirmation. Draw them even before the first box exists.
    for field, color, label in [
        ('ema9_1m', '#eab308', 'EMA9 (1m)'),
        ('ema20_1m', '#2563eb', 'EMA20 (1m)'),
        ('vwap_1m', '#8b5cf6', 'VWAP (1m)'),
    ]:
        fig.add_trace(go.Scatter(
            x=d.index, y=d[field], mode='lines', name=label,
            line=dict(color=color, width=1.7),
            hovertemplate='%{x}<br>' + label + ': $%{y:.3f}<extra></extra>'
        ))
    levels = levels_for_day(raw, day, opening_minutes)
    palette = {'PDH': '#16a34a', 'PDL': '#dc2626', 'PDC': '#64748b',
               'OR High': '#0891b2', 'OR Low': '#ea580c'}
    for name, val in levels.items():
        if np.isfinite(val):
            fig.add_hline(y=val, line_dash='dot', line_color=palette[name],
                          annotation_text=f'{name} {val:.2f}', annotation_position='top right')
    # Trendlines are visual aids only. They do not enter breakout conditions.
    if len(b) >= 9:
        for field, color, is_high in [('high', '#ef4444', True), ('low', '#22c55e', False)]:
            arr = b[field].to_numpy()
            pivots = []
            for j in range(2, len(arr)-2):
                left, right = arr[j-2:j], arr[j+1:j+3]
                if ((arr[j] > max(left) and arr[j] >= max(right)) if is_high else
                    (arr[j] < min(left) and arr[j] <= min(right))):
                    pivots.append(j)
            if len(pivots) >= 2:
                a, z = pivots[-2:]
                slope = (arr[z] - arr[a]) / (z - a)
                fig.add_trace(go.Scatter(x=[b.index[a], b.index[-1]],
                    y=[arr[a], arr[a] + slope * (len(b)-1-a)], mode='lines',
                    name='Resistance trendline' if is_high else 'Support trendline',
                    line=dict(color=color, width=2, dash='dash')))
    fig.update_layout(height=600,
                      title=f'{day} | Previous completed box (5m candle close, ET)',
                      xaxis_title='Eastern Time', yaxis_title='Price ($)',
                      xaxis_rangeslider_visible=False)
    return fig, levels


st.title('Stock Movement V15 | Box Theory & Detection Lab')
st.caption('Research only. V15 frozen-boundary detection, MACD slope/acceleration and time-of-day RVOL are diagnostic only; original trading/backtest signals remain unchanged. No orders are placed.')

with st.sidebar:
    symbol = st.text_input('U.S. stock / ETF symbol', 'QQQ').strip().upper()
    if not re.fullmatch(r'[A-Z][A-Z0-9.-]{0,9}', symbol):
        st.error('Invalid ticker symbol')
        st.stop()
    feed = st.selectbox('Alpaca feed', ['iex', 'sip'],
                        index=1 if secret('ALPACA_FEED', default='iex') == 'sip' else 0)
    opening = st.selectbox('Opening range (minutes)', [5, 15, 30], index=1)
    show_developing_boxes = st.checkbox('Show developing/live boxes', value=False,
                                       help='Optional visual-only box lines. Red and blue trading boxes remain visible.')
    chart_display = st.selectbox('Chart display', ['Candlestick', 'Line', 'Both'], index=2)
    window = st.selectbox('Chart timeframe', ['5 minutes', '15 minutes', '1 hour',
                                              '2 hours', '4 hours', 'Since market open'], index=5)
    windows = {'5 minutes': 5, '15 minutes': 15, '1 hour': 60,
               '2 hours': 120, '4 hours': 240, 'Since market open': None}
    st.divider()
    st.subheader('Box strategy')
    lookback = st.slider('Box lookback (5m candles)', 3, 24, 12)
    max_width = st.slider('Maximum box width (%)', 0.2, 2.0, 0.9, 0.1)
    vol_mult = st.slider('Breakout relative volume', 1.0, 3.0, 1.2, 0.1)
    reclaim_window = st.slider('Failed-breakout reclaim window (5m bars)', 1, 6, 3)
    sustain_window = st.slider('Sustained breakout confirmation (5m bars)', 1, 3, 2)
    min_tod_sessions = st.slider('Minimum prior sessions for time-of-day RVOL', 2, 15, 5)
    hold_minutes = st.selectbox('Maximum holding time', [5, 15, 30, 60], index=2)
    stop_atr = st.slider('Stop distance (ATR multiplier)', 0.5, 3.0, 1.2, 0.1)
    reward_risk = st.slider('Profit target (reward/risk)', 0.5, 4.0, 1.5, 0.1)
    cost_bps = st.number_input('Estimated round-trip cost (bps)', 0.0, 30.0, 2.0, 0.5)
    st.divider()
    uploaded = st.file_uploader('Historical 1-minute CSV (optional)', type=['csv'])
    naive_tz = st.selectbox('Timezone for CSV timestamps without offset',
                            ['America/New_York', 'UTC'], index=0)
    run_test = st.button('Run historical backtest', type='primary')

max_width_for_reversal = max_width

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


# The timer is enabled only when this page is rendered during market hours.
# At the first post-close tick we rerun the app to remove the timer.
@st.fragment(run_every='30s' if is_regular_session_open() else None)
def live_dashboard():
    if not key or not secret_key:
        st.info('Configure Alpaca keys in Streamlit Secrets to enable live charts.')
        return
    try:
        if not is_regular_session_open() and st.session_state.get('was_market_open', False):
            # Rebuild page once with run_every=None; stop all scheduled reruns.
            st.session_state['was_market_open'] = False
            st.rerun(scope='app')
        st.session_state['was_market_open'] = is_regular_session_open()
        raw = load_market_data(symbol, feed, key, secret_key)
        if raw.empty:
            st.warning('No recent market data')
            return
        boxes = make_boxes(raw, lookback, max_width, vol_mult, True, True)
        latest_day = raw.index[-1].date()
        fig, levels = chart_for_day(raw, boxes, latest_day, opening, windows[window], chart_display, lookback, show_developing_boxes)
        st.subheader(f'{symbol} | Live chart')
        if is_regular_session_open():
            st.caption('NYSE regular session OPEN — refreshing every 30 seconds.')
        else:
            st.info('NYSE regular session CLOSED — refresh paused. '
                    'Last cached chart is displayed. Reload after the next market open.')
        st.metric('Last completed minute close', f'${raw.close.iloc[-1]:.2f}')
        st.caption(f'Latest candle: {raw.index[-1]} | Feed: {feed.upper()} | 1-minute chart indicators; box signals use completed 5-minute candles')
        if (dt.datetime.now(NY) - raw.index[-1]).total_seconds() > 300:
            st.warning('Data is stale or market is closed. Do not treat this as a live entry.')
        st.plotly_chart(fig, use_container_width=True)
        st.caption('Trading BOX_HIGH is red; trading BOX_LOW is blue. Optional gray developing/live boxes are visual only. Confirmed 5m boxes remain the trade/backtest reference.')
        render_trend_duration_tracker(boxes, raw, latest_day)
        render_reversal_panel(raw, boxes, latest_day, reclaim_window)
        render_detection_lab(raw, boxes, latest_day, reclaim_window, sustain_window, min_tod_sessions)
        day_boxes = boxes[boxes.index.date == latest_day]
        if not day_boxes.empty:
            latest = day_boxes.iloc[-1]
            st.subheader('Box signal')
            # Signal is a five-minute bar close; no guarantee it is still actionable.
            signal = {1:'BULLISH BREAKOUT', -1:'BEARISH BREAKDOWN', 0:'WAIT'}[int(latest.signal)]
            st.write(f'**{signal}** | Box high: {latest.box_high:.2f} | Box low: {latest.box_low:.2f} | '
                     f'Volume ratio: {latest.vol_ratio:.2f} | EMA9: {latest.ema9:.2f} | EMA20: {latest.ema20:.2f}')
            if pd.isna(latest.box_high):
                st.info(f'Developing box: {int(latest.completed_box_candles)}/{lookback} completed 5-minute candles. Not eligible for breakout signals yet.')
            st.caption('Signal is descriptive, not a probability. Dotted developing levels never trigger trades; only the prior completed box is used for confirmation. Any entry must wait for the next candle and use a defined stop.')
            audit = breakout_audit(boxes, latest_day)
            st.subheader('Breakout audit — previous box vs 5m close')
            st.caption('All times are Eastern. Price crossing is shown even when trend/volume filters reject the trade.')
            if not audit.empty:
                st.dataframe(audit.tail(30), hide_index=True, use_container_width=True)
            else:
                st.info('No price crossings of the previous completed box yet.')
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
            st.warning('Fewer than 10 sessions: diagnostic only, not reliable strategy validation.')
        st.session_state['test_raw'] = test_raw
        st.session_state['test_symbol'] = symbol
        st.session_state['test_settings'] = (lookback, max_width, vol_mult, hold_minutes,
                                             stop_atr, reward_risk, cost_bps, feed)
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
    current_settings = (lookback, max_width, vol_mult, hold_minutes,
                        stop_atr, reward_risk, cost_bps, feed)
    if (current_settings != st.session_state.get('test_settings') or
            symbol != st.session_state.get('test_symbol')):
        st.warning('Backtest results use earlier ticker/settings. Click Run historical backtest to update.')
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
                                chosen, opening, windows[window], chart_display, st.session_state['test_settings'][0], show_developing_boxes)
    st.plotly_chart(fig, use_container_width=True)
    render_trend_duration_tracker(st.session_state['test_boxes'], st.session_state['test_raw'], chosen)
    render_reversal_panel(st.session_state['test_raw'], st.session_state['test_boxes'], chosen, reclaim_window)
    render_detection_lab(st.session_state['test_raw'], st.session_state['test_boxes'], chosen, reclaim_window, sustain_window, min_tod_sessions)
    st.caption('Backtest and confirmed signals use prior 5m boxes. LIVE 1m rolling box is display-only and can differ from the confirmed 5m box. ET signal times mark 5m candle close.')
    st.subheader('Historical breakout audit')
    audit = breakout_audit(st.session_state['test_boxes'], chosen)
    if not audit.empty:
        st.dataframe(audit, hide_index=True, use_container_width=True)
        st.download_button('Download breakout audit CSV', audit.to_csv(index=False),
                           file_name=f'{st.session_state.get("test_symbol", symbol)}_{chosen}_breakout_audit.csv',
                           mime='text/csv')
    else:
        st.info('No previous-box price breakouts detected on this day.')
