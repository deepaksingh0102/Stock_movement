
import datetime as dt
import hashlib
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="QQQ Intraday Predictor",
    layout="wide"
)

NY = ZoneInfo("America/New_York")
UTC = dt.timezone.utc

HORIZONS = [5, 15, 30, 60]

THRESHOLDS = {
    5: 5,
    15: 8,
    30: 12,
    60: 18
}

FEATURES = [
    "ret1", "ret5", "ret15",
    "ema_spread", "ema50_gap",
    "vwap_gap", "rsi", "macd_hist",
    "adx", "atr_pct", "rvol",
    "range_pct", "body_pct",
    "wick_up", "wick_down",
    "break20", "sin_time", "cos_time"
]

CHART_WINDOWS = {
    "5 minutes": 5,
    "15 minutes": 15,
    "1 hour": 60,
    "2 hours": 120,
    "3 hours": 180,
    "4 hours": 240,
    "Since market open": None
}

LABELS = np.array(["DOWN", "SIDEWAYS", "UP"])


# ============================================================
# STREAMLIT SECRETS
# ============================================================

def get_secret(*names):
    for name in names:
        value = st.secrets.get(name, "")
        if value and str(value).strip():
            return str(value).strip()
    return ""


API_KEY = get_secret(
    "ALPACA_API_KEY",
    "APCA_API_KEY_ID",
    "ALPACA_KEY"
)

API_SECRET = get_secret(
    "ALPACA_SECRET_KEY",
    "APCA_API_SECRET_KEY",
    "ALPACA_SECRET"
)

DEFAULT_FEED = get_secret("ALPACA_FEED") or "iex"


# ============================================================
# CLEAN AND NORMALIZE DATA
# ============================================================

def clean(raw):
    x = raw.copy()

    x.columns = [
        str(c).strip().lower()
        for c in x.columns
    ]

    if "timestamp" in x.columns:
        parsed = pd.to_datetime(
            x["timestamp"],
            errors="coerce",
            utc=True
        )
        x["timestamp"] = parsed
        x = x.set_index("timestamp")

    if not isinstance(x.index, pd.DatetimeIndex):
        raise ValueError(
            "Data requires a timestamp column or DatetimeIndex"
        )

    if x.index.tz is None:
        # Assume naive timestamps are UTC.
        x.index = x.index.tz_localize("UTC")

    x.index = x.index.tz_convert(NY)

    required = [
        "open", "high", "low", "close", "volume"
    ]

    missing = [
        col for col in required
        if col not in x.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    x = x[required].apply(
        pd.to_numeric,
        errors="coerce"
    ).dropna()

    x = x[~x.index.isna()]
    x = x[~x.index.duplicated(keep="last")]
    x = x.sort_index()

    x = x.between_time("09:30", "15:59")
    x = x[x.index.dayofweek < 5]

    return x


# ============================================================
# ALPACA API
# ============================================================

@st.cache_data(ttl=30, show_spinner=False)
def fetch_alpaca(start, end, feed):
    if not API_KEY or not API_SECRET:
        raise ValueError(
            "Alpaca credentials missing from Streamlit Secrets"
        )

    url = (
        "https://data.alpaca.markets"
        "/v2/stocks/QQQ/bars"
    )

    headers = {
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": API_SECRET
    }

    params = {
        "start": start,
        "end": end,
        "timeframe": "1Min",
        "feed": feed,
        "adjustment": "split",
        "limit": 10000,
        "sort": "asc"
    }

    rows = []

    with requests.Session() as session:
        for _ in range(200):
            response = session.get(
                url,
                params=params,
                headers=headers,
                timeout=45
            )

            response.raise_for_status()

            payload = response.json()

            rows.extend(
                payload.get("bars", [])
            )

            token = payload.get("next_page_token")

            if not token:
                break

            params["page_token"] = token
        else:
            raise RuntimeError(
                "Alpaca pagination limit reached"
            )

    if not rows:
        raise ValueError(
            "Alpaca returned no QQQ bars"
        )

    df = pd.DataFrame(rows).rename(
        columns={
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume"
        }
    )

    return clean(df)


@st.cache_data(ttl=21600, show_spinner=False)
def historical_data(end_day, feed):
    # Completed sessions only.
    end_date = dt.date.fromisoformat(end_day)
    start_date = end_date - dt.timedelta(days=65)

    return fetch_alpaca(
        dt.datetime.combine(
            start_date,
            dt.time.min,
            UTC
        ).isoformat(),
        dt.datetime.combine(
            end_date,
            dt.time.min,
            UTC
        ).isoformat(),
        feed
    )


def get_live_data(feed):
    now = dt.datetime.now(UTC)

    # Use a short recent-data request.
    end_dt = now - dt.timedelta(minutes=2)
    start_dt = end_dt - dt.timedelta(days=2)

    # Round request boundaries to 30-second intervals
    # so Streamlit's data cache can reuse requests.
    rounded = int(end_dt.timestamp() // 30) * 30
    end_dt = dt.datetime.fromtimestamp(
        rounded, tz=UTC
    )
    start_dt = end_dt - dt.timedelta(days=2)

    return fetch_alpaca(
        start_dt.isoformat(),
        end_dt.isoformat(),
        feed
    )


def combine_data(history, recent):
    x = pd.concat([history, recent])
    x = x[~x.index.duplicated(keep="last")]
    return x.sort_index()


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def rma(series, n):
    return series.ewm(
        alpha=1 / n,
        adjust=False,
        min_periods=n
    ).mean()


def make_features(df):
    x = df.copy()

    c = x["close"]
    h = x["high"]
    l = x["low"]
    o = x["open"]
    v = x["volume"]

    day = pd.Series(x.index.date, index=x.index)

    def day_shift(series, n):
        return series.groupby(day).shift(n)

    x["ret1"] = (
        c / day_shift(c, 1) - 1
    ) * 10000

    x["ret5"] = (
        c / day_shift(c, 5) - 1
    ) * 10000

    x["ret15"] = (
        c / day_shift(c, 15) - 1
    ) * 10000

    e9 = c.ewm(span=9, adjust=False).mean()
    e20 = c.ewm(span=20, adjust=False).mean()
    e50 = c.ewm(span=50, adjust=False).mean()

    x["ema9"] = e9
    x["ema20"] = e20

    x["ema_spread"] = (
        e9 / e20 - 1
    ) * 10000

    x["ema50_gap"] = (
        c / e50 - 1
    ) * 10000

    typical = (h + l + c) / 3

    cumulative_pv = (
        typical * v
    ).groupby(day).cumsum()

    cumulative_volume = (
        v.groupby(day).cumsum()
    ).replace(0, np.nan)

    x["vwap"] = (
        cumulative_pv / cumulative_volume
    )

    x["vwap_gap"] = (
        c / x["vwap"] - 1
    ) * 10000

    delta = c.diff()

    gain = rma(delta.clip(lower=0), 14)
    loss = rma((-delta).clip(lower=0), 14)

    x["rsi"] = (
        100 - 100 / (1 + gain / loss.replace(0, np.nan))
    )

    x.loc[
        (loss == 0) & (gain > 0),
        "rsi"
    ] = 100

    x.loc[
        (loss == 0) & (gain == 0),
        "rsi"
    ] = 50

    macd = (
        c.ewm(span=12, adjust=False).mean()
        - c.ewm(span=26, adjust=False).mean()
    )

    signal = macd.ewm(
        span=9, adjust=False
    ).mean()

    x["macd_hist"] = (
        (macd - signal) / c * 10000
    )

    previous = c.shift()

    tr = pd.concat([
        h - l,
        (h - previous).abs(),
        (l - previous).abs()
    ], axis=1).max(axis=1)

    atr = rma(tr, 14)

    x["atr_pct"] = (
        atr / c * 10000
    )

    up = h.diff()
    down = -l.diff()

    pdm = up.where(
        (up > down) & (up > 0), 0
    )

    mdm = down.where(
        (down > up) & (down > 0), 0
    )

    pdi = (
        100 * rma(pdm, 14)
        / atr.replace(0, np.nan)
    )

    mdi = (
        100 * rma(mdm, 14)
        / atr.replace(0, np.nan)
    )

    dx = (
        100 * (pdi - mdi).abs()
        / (pdi + mdi).replace(0, np.nan)
    )

    x["adx"] = rma(dx, 14)

    minute = (
        x.index.hour * 60
        + x.index.minute
    )

    volume_matrix = x.assign(
        session_date=x.index.date,
        minute_of_day=minute
    ).pivot(
        index="session_date",
        columns="minute_of_day",
        values="volume"
    )

    baseline = (
        volume_matrix.shift(1)
        .rolling(10, min_periods=3)
        .mean()
    )

    baseline_series = baseline.stack().rename(
        "baseline_volume"
    )

    lookup = pd.MultiIndex.from_arrays(
        [x.index.date, minute],
        names=["session_date", "minute_of_day"]
    )

    expected_volume = baseline_series.reindex(
        lookup
    ).to_numpy()

    x["rvol"] = (
        v.to_numpy()
        / np.where(expected_volume > 0,
                   expected_volume, np.nan)
    )

    x["rvol"] = x["rvol"].clip(0, 20)

    x["range_pct"] = (
        (h - l) / c * 10000
    )

    x["body_pct"] = (
        (c - o) / c * 10000
    )

    x["wick_up"] = (
        (h - np.maximum(c, o))
        / c * 10000
    )

    x["wick_down"] = (
        (np.minimum(c, o) - l)
        / c * 10000
    )

    high20 = (
        h.groupby(day)
        .transform(lambda s: s.rolling(20).max())
    )

    low20 = (
        l.groupby(day)
        .transform(lambda s: s.rolling(20).min())
    )

    high20 = day_shift(high20, 1)
    low20 = day_shift(low20, 1)

    x["break20"] = np.where(
        c > high20,
        1,
        np.where(c < low20, -1, 0)
    )

    minute_in_day = minute - 570

    x["sin_time"] = np.sin(
        2 * np.pi * minute_in_day / 390
    )

    x["cos_time"] = np.cos(
        2 * np.pi * minute_in_day / 390
    )

    x["session"] = x.index.date

    for horizon in HORIZONS:
        future = c.groupby(day).shift(-horizon)

        x[f"future_{horizon}"] = (
            future / c - 1
        ) * 10000

    return x


# ============================================================
# MACHINE LEARNING
# ============================================================

def build_model():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            max_iter=350
        )
    )


def prepare_training(x, horizon):
    target = f"future_{horizon}"

    train = x.dropna(
        subset=[target]
    ).copy()

    train = train[
        train[FEATURES].notna().sum(axis=1) >= 12
    ]

    threshold = THRESHOLDS[horizon]

    y = np.where(
        train[target] > threshold,
        2,
        np.where(train[target] < -threshold, 0, 1)
    )

    return train, y


@st.cache_resource(
    show_spinner="Training QQQ models..."
)
def cached_models(training_key, _training_df):
    models = {}

    for horizon in HORIZONS:
        train, y = prepare_training(
            _training_df, horizon
        )

        if len(train) < 100:
            continue

        if len(np.unique(y)) < 3:
            continue

        model = build_model()
        model.fit(train[FEATURES], y)

        models[horizon] = model

    return models


def forecast_latest(features, models, min_confidence):
    latest = features.iloc[[-1]]

    if latest[FEATURES].notna().sum(axis=1).iloc[0] < 12:
        return pd.DataFrame()

    output = []

    for horizon in HORIZONS:
        if horizon not in models:
            continue

        model = models[horizon]

        probabilities = model.predict_proba(
            latest[FEATURES]
        )[0]

        probs = np.zeros(3)

        for j, cls in enumerate(model.classes_):
            probs[int(cls)] = probabilities[j]

        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])

        direction = LABELS[predicted_class]

        if direction != "SIDEWAYS":
            if confidence < min_confidence:
                direction = "WAIT"

        output.append({
            "Horizon": f"{horizon} min",
            "Signal": direction,
            "Confidence": confidence,
            "P(DOWN)": probs[0],
            "P(SIDEWAYS)": probs[1],
            "P(UP)": probs[2]
        })

    return pd.DataFrame(output)


# ============================================================
# WALK-FORWARD BACKTEST
# ============================================================

def fit_predict(
    x,
    horizon,
    min_train_days=15,
    refit_every=5
):
    target = f"future_{horizon}"

    days = np.array(
        sorted(x["session"].unique())
    )

    if len(days) < min_train_days + 2:
        raise ValueError(
            f"Need {min_train_days + 2} sessions; "
            f"found {len(days)}"
        )

    output = []
    model = None

    for i in range(min_train_days, len(days)):

        if (
            model is None
            or (i - min_train_days) % refit_every == 0
        ):
            train_days = days[:i]

            train_data = x[
                x["session"].isin(train_days)
            ]

            train, y = prepare_training(
                train_data, horizon
            )

            if len(np.unique(y)) < 3:
                continue

            model = build_model()
            model.fit(train[FEATURES], y)

        test = x[
            x["session"] == days[i]
        ].copy()

        test = test[
            test[FEATURES].notna().sum(axis=1) >= 12
        ]

        if test.empty or model is None:
            continue

        probabilities = model.predict_proba(
            test[FEATURES]
        )

        probs = np.zeros((len(test), 3))

        for j, cls in enumerate(model.classes_):
            probs[:, int(cls)] = probabilities[:, j]

        test["p_down"] = probs[:, 0]
        test["p_sideways"] = probs[:, 1]
        test["p_up"] = probs[:, 2]

        test["prediction"] = LABELS[
            np.argmax(probs, axis=1)
        ]

        test["confidence"] = probs.max(axis=1)

        threshold = THRESHOLDS[horizon]

        test["actual"] = np.where(
            test[target] > threshold,
            "UP",
            np.where(
                test[target] < -threshold,
                "DOWN",
                "SIDEWAYS"
            )
        )

        test.loc[
            test[target].isna(),
            "actual"
        ] = None

        test["horizon"] = horizon

        output.append(test[[
            "close", "session", "horizon",
            "prediction", "confidence",
            "p_down", "p_sideways",
            "p_up", "actual", target
        ]])

    if not output:
        raise ValueError(
            "Insufficient training examples"
        )

    return pd.concat(output)


def calculate_metrics(predictions):
    z = predictions.dropna(
        subset=["actual"]
    )

    if z.empty:
        return None

    probs = z[
        ["p_down", "p_sideways", "p_up"]
    ].to_numpy()

    actual_ids = z["actual"].map({
        "DOWN": 0,
        "SIDEWAYS": 1,
        "UP": 2
    }).to_numpy()

    baseline = z["actual"].value_counts(
        normalize=True
    ).max()

    return {
        "Evaluated": len(z),
        "Accuracy": accuracy_score(
            z["actual"], z["prediction"]
        ),
        "Baseline": baseline,
        "Log loss": log_loss(
            actual_ids,
            probs,
            labels=[0, 1, 2]
        ),
        "Actionable %": (
            z["prediction"] != "SIDEWAYS"
        ).mean()
    }

# ============================================================
# PRICE ACTION PATTERN RECOGNITION
# ============================================================

def detect_price_patterns(raw):
    """
    Detect price-action events on completed 5-minute
    QQQ candles.

    Uses only current and prior candles.
    No future price data is used.
    """

    if raw.empty:
        return pd.DataFrame()

    # Work within individual regular trading sessions.
    pieces = []

    for session_date, day in raw.groupby(raw.index.date):
        bars = day.resample(
            "5min",
            origin="start_day",
            label="left",
            closed="left"
        ).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna()

        # Only include completed five-minute candles.
        counts = day["close"].resample(
            "5min",
            origin="start_day"
        ).count()

        bars = bars[counts.reindex(
            bars.index, fill_value=0
        ) == 5].copy()

        if not bars.empty:
            pieces.append(bars)

    if not pieces:
        return pd.DataFrame()

    x = pd.concat(pieces).sort_index()

    session = pd.Series(x.index.date, index=x.index)

    def prior(s, n=1):
        return s.groupby(session).shift(n)

    o = x["open"]
    h = x["high"]
    l = x["low"]
    c = x["close"]

    body = (c - o).abs()
    candle_range = (h - l).replace(0, np.nan)

    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l

    prev_open = prior(o)
    prev_close = prior(c)

    # Hammer: small real body and long lower wick.
    hammer = (
        (lower_wick >= 2 * body) &
        (upper_wick <= body) &
        (body / candle_range <= 0.35)
    )

    # Shooting star: small real body and long upper wick.
    shooting_star = (
        (upper_wick >= 2 * body) &
        (lower_wick <= body) &
        (body / candle_range <= 0.35)
    )

    # Bullish engulfing.
    bullish_engulfing = (
        (prev_close < prev_open) &
        (c > o) &
        (o <= prev_close) &
        (c >= prev_open)
    )

    # Bearish engulfing.
    bearish_engulfing = (
        (prev_close > prev_open) &
        (c < o) &
        (o >= prev_close) &
        (c <= prev_open)
    )

    # Previous 12 completed five-minute candles.
    # Shift prevents current-candle leakage.
    resistance = h.groupby(session).transform(
        lambda s: s.rolling(
            12, min_periods=6
        ).max().shift(1)
    )

    support = l.groupby(session).transform(
        lambda s: s.rolling(
            12, min_periods=6
        ).min().shift(1)
    )

    x["support"] = support
    x["resistance"] = resistance

    # Volume relative to previous 12 candles.
    avg_volume = x["volume"].groupby(
        session
    ).transform(
        lambda s: s.rolling(
            12, min_periods=6
        ).mean().shift(1)
    )

    x["volume_ratio"] = (
        x["volume"] / avg_volume.replace(0, np.nan)
    )

    bullish_breakout = (
        (c > resistance) &
        (prev_close <= prior(resistance)) &
        (x["volume_ratio"] >= 1.5)
    )

    bearish_breakdown = (
        (c < support) &
        (prev_close >= prior(support)) &
        (x["volume_ratio"] >= 1.5)
    )

    # Require hammer/star proximity to recent levels.
    tolerance = 0.0015

    near_support = (
        support.notna() &
        (l <= support * (1 + tolerance)) &
        (h >= support * (1 - tolerance))
    )

    near_resistance = (
        resistance.notna() &
        (h >= resistance * (1 - tolerance)) &
        (l <= resistance * (1 + tolerance))
    )

    x["pattern"] = ""
    x["direction"] = ""

    patterns = [
        (
            hammer & near_support,
            "Hammer at support",
            "BULLISH"
        ),
        (
            shooting_star & near_resistance,
            "Shooting star at resistance",
            "BEARISH"
        ),
        (
            bullish_engulfing,
            "Bullish engulfing",
            "BULLISH"
        ),
        (
            bearish_engulfing,
            "Bearish engulfing",
            "BEARISH"
        ),
        (
            bullish_breakout,
            "Resistance breakout",
            "BULLISH"
        ),
        (
            bearish_breakdown,
            "Support breakdown",
            "BEARISH"
        )
    ]

    for condition, name, direction in patterns:
        x.loc[condition, "pattern"] = name
        x.loc[condition, "direction"] = direction

    # A 5-minute bar labeled 10:00 is known
    # only after it completes at 10:05.
    x["detected_at"] = x.index + pd.Timedelta(
        minutes=5
    )

    return x
# ============================================================
# SIDEBAR
# ============================================================

st.title(
    "QQQ | Intraday Prediction & Walk-forward Validation"
)

st.caption(
    "Research tool only. No orders are placed. "
    "Predictions are experimental."
)

with st.sidebar:
    st.header("QQQ Live Chart")

    chart_window = st.selectbox(
        "Chart timeframe",
        list(CHART_WINDOWS.keys()),
        index=6
    )

    feed = st.selectbox(
        "Feed",
        ["iex", "sip"],
        index=0 if DEFAULT_FEED == "iex" else 1
    )

    min_confidence = st.slider(
        "Minimum directional confidence",
        min_value=0.35,
        max_value=0.90,
        value=0.55,
        step=0.05
    )

    st.caption("Auto-refresh: every 30 seconds")

    with st.expander("Alpaca Connection Status"):
        st.write(
            "API Key:",
            "Configured" if API_KEY else "Missing"
        )
        st.write(
            "Secret:",
            "Configured" if API_SECRET else "Missing"
        )

    st.divider()

    st.header("Historical Validation")

    uploaded = st.file_uploader(
        "Optional QQQ 1-minute CSV",
        type=["csv"]
    )

    min_train = st.slider(
        "Initial training sessions",
        10, 35, 15
    )

    refit = st.slider(
        "Refit every N sessions",
        1, 10, 5
    )

    run = st.button(
        "Run walk-forward test",
        type="primary"
    )

show_patterns = st.checkbox(
    "Show price action patterns",
    value=True
)

show_levels = st.checkbox(
    "Show support and resistance",
    value=True
)

show_pattern_table = st.checkbox(
    "Show detected pattern details",
    value=True
)
# ============================================================
# HISTORICAL VALIDATION - RUN ONLY ON BUTTON CLICK
# ============================================================

if run:
    try:
        if uploaded is not None:
            raw = clean(pd.read_csv(uploaded))
        else:
            if not API_KEY or not API_SECRET:
                raise ValueError(
                    "Alpaca credentials missing"
                )

            now = dt.datetime.now(UTC)

            raw = fetch_alpaca(
                (
                    now - dt.timedelta(days=75)
                ).isoformat(),
                (
                    now - dt.timedelta(minutes=2)
                ).isoformat(),
                feed
            )

        if raw.empty:
            raise ValueError(
                "No regular-session QQQ data found"
            )

        st.session_state["backtest_raw"] = raw

        with st.spinner(
            "Computing historical predictions..."
        ):
            features = make_features(raw)

            results = {
                h: fit_predict(
                    features, h, min_train, refit
                )
                for h in HORIZONS
            }

        st.session_state["backtest_results"] = results

    except Exception as exc:
        st.error(
            f"Walk-forward test failed: {exc}"
        )


if "backtest_results" in st.session_state:
    results = st.session_state["backtest_results"]
    raw = st.session_state["backtest_raw"]

    st.subheader("Historical Backtest Results")

    summary = []

    for h, result in results.items():
        metrics = calculate_metrics(result)

        if metrics:
            summary.append({
                "Horizon": f"{h} min",
                **metrics
            })

    if summary:
        summary_df = pd.DataFrame(summary)

        st.dataframe(
            summary_df.style.format({
                "Accuracy": "{:.1%}",
                "Baseline": "{:.1%}",
                "Log loss": "{:.3f}",
                "Actionable %": "{:.1%}"
            }),
            hide_index=True,
            use_container_width=True
        )

    selected_horizon = st.selectbox(
        "Historical prediction horizon",
        HORIZONS,
        index=1
    )

    result = results[selected_horizon]

    days = sorted(result["session"].unique())

    chosen_day = st.selectbox(
        "Historical replay day",
        days,
        index=len(days) - 1
    )

    selected = result[
        result["session"] == chosen_day
    ]

    st.dataframe(
        selected[[
            "close", "prediction",
            "confidence", "p_up",
            "p_down", "p_sideways",
            "actual"
        ]].tail(30),
        use_container_width=True
    )

    all_predictions = pd.concat(
        results.values()
    ).reset_index()

    csv = all_predictions.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download historical predictions",
        data=csv,
        file_name="QQQ_predictions.csv",
        mime="text/csv"
    )


# ============================================================
# LIVE FORECAST - REFRESH EVERY 30 SECONDS
# ============================================================

st.divider()
st.subheader("QQQ Live Chart & Forecast")

@st.fragment(run_every="30s")
def live_dashboard():

    if not API_KEY or not API_SECRET:
        st.warning(
            "Configure Alpaca credentials "
            "in Streamlit Secrets."
        )
        return

    try:
        now_ny = dt.datetime.now(NY)

        # Use midnight of current New York date
        # as the boundary for completed sessions.
        today = now_ny.date()

        with st.spinner("Loading QQQ market data..."):

            history = historical_data(
                today.isoformat(),
                feed
            )

            recent = get_live_data(feed)

            raw = combine_data(
                history, recent
            )

        if raw.empty:
            st.warning("No QQQ market data available")
            return

        latest_ts = raw.index[-1]
        latest_close = raw["close"].iloc[-1]

        st.metric(
            "QQQ Latest Completed Close",
            f"${latest_close:.2f}"
        )

        st.caption(
            f"Feed: {feed.upper()} | "
            f"Latest candle: {latest_ts} | "
            "Refresh: 30 seconds"
        )

        age_minutes = (
            dt.datetime.now(NY) - latest_ts
        ).total_seconds() / 60

        if age_minutes > 5:
            st.warning(
                f"Latest candle is {age_minutes:.0f} "
                "minutes old. Market may be closed "
                "or data may be delayed."
            )

        features = make_features(raw)

        # ====================================================
        # CHART
        # ====================================================

        latest_day = latest_ts.date()

        chart_data = features[
            features.index.date == latest_day
        ]

        window = CHART_WINDOWS[chart_window]

        if window is not None:
            chart_data = chart_data.tail(window)

        fig = go.Figure()

        fig.add_trace(
            go.Candlestick(
                x=chart_data.index,
                open=chart_data["open"],
                high=chart_data["high"],
                low=chart_data["low"],
                close=chart_data["close"],
                name="QQQ"
            )
        )

        for name, color in [
            ("ema9", "orange"),
            ("ema20", "blue"),
            ("vwap", "purple")
        ]:
            fig.add_trace(
                go.Scatter(
                    x=chart_data.index,
                    y=chart_data[name],
                    mode="lines",
                    name=name.upper(),
                    line=dict(
                        color=color,
                        width=1.5
                    )
                )
            )
            # ============================================================
# ADD PRICE ACTION PATTERNS TO QQQ CHART
# ============================================================

patterns = detect_price_patterns(raw)

if not patterns.empty:
    # Only show patterns from the chart's trading day.
    patterns = patterns[
        patterns.index.date == latest_day
    ].copy()

    # The latest five-minute candle must be complete.
    patterns = patterns[
        patterns["detected_at"] <= latest_ts
    ]

    # Use event time, not the start of the candle.
    if show_patterns:
        detected = patterns[
            patterns["pattern"] != ""
        ].copy()

        if not chart_data.empty:
            detected = detected[
                detected["detected_at"] >= chart_data.index.min()
            ]

        bullish = detected[
            detected["direction"] == "BULLISH"
        ]

        bearish = detected[
            detected["direction"] == "BEARISH"
        ]

        fig.add_trace(
            go.Scatter(
                x=bullish["detected_at"],
                y=bullish["low"] * 0.9995,
                mode="markers",
                name="Bullish pattern",
                marker=dict(
                    symbol="triangle-up",
                    size=12,
                    color="green"
                ),
                text=bullish["pattern"],
                hovertemplate=(
                    "%{text}<br>"
                    "%{x}<br>"
                    "Price: %{y:.2f}"
                    "<extra></extra>"
                )
            )
        )

        fig.add_trace(
            go.Scatter(
                x=bearish["detected_at"],
                y=bearish["high"] * 1.0005,
                mode="markers",
                name="Bearish pattern",
                marker=dict(
                    symbol="triangle-down",
                    size=12,
                    color="red"
                ),
                text=bearish["pattern"],
                hovertemplate=(
                    "%{text}<br>"
                    "%{x}<br>"
                    "Price: %{y:.2f}"
                    "<extra></extra>"
                )
            )
        )

    if show_levels and not chart_data.empty:
        latest_pattern = patterns.iloc[-1] if len(patterns) else None

        if latest_pattern is not None:
            for column, color, label in [
                ("support", "green", "Support"),
                ("resistance", "red", "Resistance")
            ]:
                level = latest_pattern[column]

                if pd.notna(level):
                    fig.add_hline(
                        y=float(level),
                        line_color=color,
                        line_dash="dash",
                        annotation_text=(
                            f"{label}: ${level:.2f}"
                        ),
                        annotation_position="top right"
                    )

        fig.update_layout(
            height=520,
            xaxis_rangeslider_visible=False,
            title=f"QQQ - {chart_window}",
            xaxis_title="Eastern Time",
            yaxis_title="Price ($)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )
        if show_pattern_table and not patterns.empty:
    st.subheader("QQQ Price Action Patterns")

    events = patterns[
        patterns["pattern"] != ""
    ].copy()

    if events.empty:
        st.info(
            "No qualifying price-action patterns "
            "detected in the available session."
        )
    else:
        events = events[[
            "detected_at",
            "close",
            "pattern",
            "direction",
            "support",
            "resistance",
            "volume_ratio"
        ]].tail(20)

        st.dataframe(
            events,
            use_container_width=True
        )

        latest_event = events.iloc[-1]

        st.caption(
            "Most recent pattern: "
            f"{latest_event['pattern']} "
            f"({latest_event['direction']})"
        )

        # ====================================================
        # MODEL TRAINING - COMPLETED SESSIONS ONLY
        # ====================================================

        days = sorted(
            features["session"].unique()
        )

        if len(days) < 17:
            st.warning(
                "At least 17 trading sessions "
                "are required for model training."
            )
            return

        training_days = days[:-1]

        training_df = features[
            features["session"].isin(
                training_days
            )
        ]

        # Create a stable model cache key.
        # Changes when completed training data changes.
        training_signature = (
            f"{feed}|{training_days[-1]}|"
            f"{len(training_df)}|"
            f"{training_df.close.iloc[-1]:.6f}|"
            f"{training_df.volume.sum():.0f}"
        )

        training_key = hashlib.sha256(
            training_signature.encode()
        ).hexdigest()

        models = cached_models(
            training_key,
            training_df
        )

        # ====================================================
        # FORECAST
        # ====================================================

        forecast_df = forecast_latest(
            features,
            models,
            min_confidence
        )

        if forecast_df.empty:
            st.warning(
                "No valid forecasts available."
            )
            return

        st.subheader("QQQ Intraday Predictions")

        st.dataframe(
            forecast_df.style.format({
                "Confidence": "{:.1%}",
                "P(DOWN)": "{:.1%}",
                "P(SIDEWAYS)": "{:.1%}",
                "P(UP)": "{:.1%}"
            }),
            hide_index=True,
            use_container_width=True
        )

        st.caption(
            "UP/DOWN predictions below the "
            "selected confidence threshold "
            "are labeled WAIT. SIDEWAYS remains "
            "a separate predicted class. "
            "Probabilities are uncalibrated."
        )

        # ====================================================
        # INDICATOR SNAPSHOT
        # ====================================================

        latest = features.iloc[-1]

        st.subheader("Current Indicator Alignment")

        indicators = {
            "Price": latest["close"],
            "VWAP": latest["vwap"],
            "EMA 9": latest["ema9"],
            "EMA 20": latest["ema20"],
            "RSI": latest["rsi"],
            "MACD histogram": latest["macd_hist"],
            "ADX": latest["adx"],
            "Relative volume": latest["rvol"]
        }

        indicator_df = pd.DataFrame(
            [
                {"Indicator": k, "Value": v}
                for k, v in indicators.items()
            ]
        )

        st.dataframe(
            indicator_df,
            hide_index=True,
            use_container_width=True
        )

    except requests.HTTPError as exc:
        st.error(
            f"Alpaca API error: {exc}"
        )

    except Exception as exc:
        st.error(
            f"Live refresh failed: {exc}"
        )


live_dashboard()
