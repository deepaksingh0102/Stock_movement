"""Core logic for the QQQ intraday walk-forward study.

Kept free of Streamlit so it can be tested and run from a script or cron job.

Design rules
------------
* Every regular session is reindexed to a full 390-minute grid (09:30-15:59 ET).
  Horizons are therefore exact clock offsets, never "N rows ahead".
* A missing bar is never invented as a price. Indicator state is carried through
  gaps with flat bars so EMAs keep running, but a gap bar is never a prediction
  time, a training row, or a label target.
* Labels are volatility-scaled: UP / DOWN means the forward return beats
  +/- k * sigma_t * sqrt(horizon), where sigma_t is a causal EWMA of 1-minute
  return volatility known at time t.
* Baseline is the class frequency of the training set the model was fit on.
  A model only has skill if it beats that on log loss.
* Every metric that matters is also reported per session, because minute-level
  forecasts overlap heavily and the honest sample size is the number of days.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import requests
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TZ = "America/New_York"
HORIZONS = [5, 15, 30, 60]
SESSION_MINUTES = 390
CLASSES = ["DOWN", "SIDEWAYS", "UP"]
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}

FEATURES = [
    "ret1", "ret5", "ret15", "ema_spread", "ema50_gap", "vwap_gap", "rsi",
    "macd_hist", "adx", "atr_pct", "rvol", "range_pct", "body_pct", "wick_up",
    "wick_down", "break20", "sin_time", "cos_time",
]
MIN_FEATURES = 12          # a row needs at least this many non-NaN features
DEFAULT_K = 0.5            # label threshold in units of sigma * sqrt(horizon)
MIN_THR_BPS = 0.5          # floor so a quiet stretch cannot make the threshold ~0
VOL_HALFLIFE = 30          # minutes, EWMA half-life for sigma_t
WARMUP_MINUTES = 15        # skip the opening minutes (return features not yet defined)
MODEL_C = 1.0              # inverse L2 strength for the logistic regression
MIN_COVERAGE = 0.6         # sessions with fewer real bars are dropped (early closes etc.)


# --------------------------------------------------------------------------- data
def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw bars -> regular-session OHLCV indexed by tz-aware ET timestamps."""
    x = raw.copy()
    x.columns = [str(c).lower() for c in x.columns]
    if "timestamp" in x.columns:
        x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
        x = x.set_index("timestamp")
    else:
        x.index = pd.to_datetime(x.index, utc=True)  # naive index is treated as UTC
    x.index = x.index.tz_convert(TZ)
    x.index.name = "timestamp"
    x = x[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce").dropna()
    x = x[~x.index.duplicated(keep="last")].sort_index()
    x = x.between_time("09:30", "15:59")
    x = x[x.index.dayofweek < 5]
    x = x[(x.close > 0) & (x.high >= x.low)]
    return x


def load_csv(path_or_buffer) -> pd.DataFrame:
    return clean(pd.read_csv(path_or_buffer))


def fetch_alpaca(start, end, feed, key, secret, symbol="QQQ") -> pd.DataFrame:
    """Paginated 1-minute bars from Alpaca. `end=None` means "up to now"."""
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    params = {"start": f"{start}T00:00:00Z", "timeframe": "1Min", "feed": feed,
              "limit": 10000, "adjustment": "split", "sort": "asc"}
    if end is not None:
        params["end"] = f"{end}T00:00:00Z"
    rows = []
    with requests.Session() as s:
        for _ in range(500):
            r = s.get(f"https://data.alpaca.markets/v2/stocks/{symbol}/bars",
                      params=params, headers=headers, timeout=45)
            r.raise_for_status()
            payload = r.json()
            rows.extend(payload.get("bars") or [])
            token = payload.get("next_page_token")
            if not token:
                break
            params["page_token"] = token
        else:
            raise RuntimeError("Too many pages")
    if not rows:
        raise ValueError("Alpaca returned no data")
    df = pd.DataFrame(rows).rename(columns={"t": "timestamp", "o": "open", "h": "high",
                                            "l": "low", "c": "close", "v": "volume"})
    return clean(df)


def to_grid(rth: pd.DataFrame, live: bool = False, min_coverage: float = MIN_COVERAGE):
    """Reindex each session to the full minute grid. Missing bars stay NaN.

    Sessions with fewer than `min_coverage` of their minutes are dropped (half days,
    partial downloads). With live=True the final session is kept as-is and its grid
    stops at the last real bar, because that session is still in progress.
    """
    if rth.empty:
        raise ValueError("No regular-session bars in the data")
    last_day = rth.index[-1].date()
    parts, dropped = [], []
    for d, sub in rth.groupby(rth.index.date):
        idx = pd.date_range(f"{d} 09:30", periods=SESSION_MINUTES, freq="min", tz=TZ)
        if live and d == last_day:
            idx = idx[idx <= sub.index.max()]
        elif len(sub) / SESSION_MINUTES < min_coverage:
            dropped.append((d, len(sub)))
            continue
        g = sub.reindex(idx)
        g["session"] = d
        g["minute_idx"] = np.arange(len(idx))
        parts.append(g)
    if not parts:
        raise ValueError("No usable sessions after coverage filter")
    grid = pd.concat(parts)
    grid.index.name = "timestamp"
    return grid, dropped


def prepare(raw: pd.DataFrame, live: bool = False, min_coverage: float = MIN_COVERAGE):
    """raw bars -> (grid, info). `info` reports what was dropped and how many gaps exist."""
    rth = clean(raw)
    grid, dropped = to_grid(rth, live=live, min_coverage=min_coverage)
    info = {
        "sessions": int(grid.session.nunique()),
        "dropped_sessions": dropped,
        "missing_bars": int(grid.close.isna().sum()),
        "bars": int(grid.close.notna().sum()),
    }
    return grid, info


# ----------------------------------------------------------------------- features
def _rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def make_features(grid: pd.DataFrame, k: float = DEFAULT_K, horizons=HORIZONS) -> pd.DataFrame:
    """Causal features, volatility estimate, forward returns and label thresholds.

    Everything at row t uses data up to and including bar t's close. `future_{h}` and
    `thr_{h}` are the only columns that look ahead (the first is the outcome, the
    second is known at t).
    """
    x = grid.copy()
    sess = x.session
    x["is_bar"] = x.close.notna()
    cr = x.close.copy()                    # real closes (NaN on missing bars)
    vr = x.volume.copy()
    c = cr.ffill()                         # flat carry through gaps and overnight
    o = x.open.fillna(c)
    h = x.high.fillna(c)
    l = x.low.fillna(c)
    v = x.volume.fillna(0.0)

    # Returns use real closes, so a gap yields NaN instead of a stale price.
    g_cr = cr.groupby(sess)
    for n in (1, 5, 15):
        x[f"ret{n}"] = (cr / g_cr.shift(n) - 1) * 1e4

    e9 = c.ewm(span=9, adjust=False).mean()
    e20 = c.ewm(span=20, adjust=False).mean()
    e50 = c.ewm(span=50, adjust=False).mean()
    x["ema_spread"] = (e9 / e20 - 1) * 1e4
    x["ema50_gap"] = (c / e50 - 1) * 1e4

    tp = (h + l + c) / 3
    cum_pv = (tp * v).groupby(sess).cumsum()
    cum_v = v.groupby(sess).cumsum().replace(0, np.nan)
    x["vwap_gap"] = (c / (cum_pv / cum_v) - 1) * 1e4

    # Momentum / trend terms are reset at the session open so the overnight gap does
    # not leak into RSI, ATR or ADX for the first few bars.
    delta = c.groupby(sess).diff()
    gain, loss = _rma(delta.clip(lower=0), 14), _rma((-delta).clip(lower=0), 14)
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    rsi[(loss == 0) & (gain > 0)] = 100
    rsi[(loss == 0) & (gain == 0)] = 50
    x["rsi"] = rsi

    m = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    x["macd_hist"] = (m - m.ewm(span=9, adjust=False).mean()) / c * 1e4

    prev = c.groupby(sess).shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    atr = _rma(tr, 14)
    x["atr_pct"] = atr / c * 1e4
    up, down = h.groupby(sess).diff(), -l.groupby(sess).diff()
    pdm = up.where((up > down) & (up > 0), 0.0)
    mdm = down.where((down > up) & (down > 0), 0.0)
    pdi = 100 * _rma(pdm, 14) / atr.replace(0, np.nan)
    mdi = 100 * _rma(mdm, 14) / atr.replace(0, np.nan)
    x["adx"] = _rma(100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan), 14)

    # Relative volume: same minute, mean of the previous 10 sessions (current day excluded).
    vol_mat = pd.DataFrame({"s": sess.to_numpy(), "m": x.minute_idx.to_numpy(), "v": vr.to_numpy()}) \
        .pivot(index="s", columns="m", values="v")
    base = vol_mat.shift(1).rolling(10, min_periods=3).mean()
    rv = (vol_mat / base.where(base > 0)).clip(0, 20).stack(future_stack=True)
    x["rvol"] = rv.reindex(pd.MultiIndex.from_arrays([sess.to_numpy(), x.minute_idx.to_numpy()])).to_numpy()

    x["range_pct"] = (h - l) / c * 1e4
    x["body_pct"] = (c - o) / c * 1e4
    x["wick_up"] = (h - np.maximum(c, o)) / c * 1e4
    x["wick_down"] = (np.minimum(c, o) - l) / c * 1e4
    hi20 = h.groupby(sess).transform(lambda s: s.rolling(20, min_periods=20).max()).groupby(sess).shift(1)
    lo20 = l.groupby(sess).transform(lambda s: s.rolling(20, min_periods=20).min()).groupby(sess).shift(1)
    brk = np.where(c > hi20, 1.0, np.where(c < lo20, -1.0, 0.0))
    x["break20"] = np.where(hi20.isna() | lo20.isna(), np.nan, brk)
    x["sin_time"] = np.sin(2 * np.pi * x.minute_idx / SESSION_MINUTES)
    x["cos_time"] = np.cos(2 * np.pi * x.minute_idx / SESSION_MINUTES)

    # Causal volatility (bps per minute). The first bar of a session has no return, so
    # the EWMA simply carries the prior session's last estimate into the open.
    r = np.log(c / c.groupby(sess).shift(1)) * 1e4
    x["vol_bps"] = np.sqrt((r ** 2).ewm(halflife=VOL_HALFLIFE, min_periods=20, ignore_na=True).mean())

    for hz in horizons:
        fut = cr.groupby(sess).shift(-hz)          # exact hz minutes ahead; NaN if no bar
        x[f"future_{hz}"] = (fut / cr - 1) * 1e4
        x[f"thr_{hz}"] = np.maximum(k * x.vol_bps * np.sqrt(hz), MIN_THR_BPS)
    return x


# ------------------------------------------------------------------------ labels
def make_labels(future_bps, thr_bps) -> np.ndarray:
    """0 = DOWN, 1 = SIDEWAYS, 2 = UP, NaN where the outcome or threshold is unknown."""
    f = np.asarray(future_bps, dtype=float)
    t = np.asarray(thr_bps, dtype=float)
    y = np.where(f > t, 2.0, np.where(f < -t, 0.0, 1.0))
    y[np.isnan(f) | np.isnan(t)] = np.nan
    return y


def _usable(x: pd.DataFrame) -> pd.Series:
    """Rows that may be used to train or predict: a real bar, past the opening warm-up,
    with enough features present."""
    return x.is_bar & (x.minute_idx >= WARMUP_MINUTES) & (x[FEATURES].notna().sum(axis=1) >= MIN_FEATURES)


def _new_model() -> object:
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=MODEL_C, max_iter=500))


def _probs(model, X) -> np.ndarray:
    p = model.predict_proba(X)
    out = np.zeros((len(X), 3))
    for j, cl in enumerate(model.classes_):
        out[:, int(cl)] = p[:, j]
    return out


# ------------------------------------------------------------------- walk-forward
def fit_predict(x: pd.DataFrame, horizon: int, min_train_days: int = 15, refit_every: int = 5) -> pd.DataFrame:
    """Expanding-window walk-forward. The model that scores session i is fit only on
    sessions before i, and refit every `refit_every` sessions."""
    tgt, thr = f"future_{horizon}", f"thr_{horizon}"
    codes, days = pd.factorize(x.session, sort=True)
    if len(days) < min_train_days + 2:
        raise ValueError(f"Need at least {min_train_days + 2} sessions; found {len(days)}")
    ok = _usable(x).to_numpy()
    y_all = make_labels(x[tgt], x[thr])
    out, model, prior, train_through = [], None, None, None

    for i in range(min_train_days, len(days)):
        if model is None or (i - min_train_days) % refit_every == 0:
            tr = (codes < i) & ok & ~np.isnan(y_all)
            y = y_all[tr].astype(int)
            if len(np.unique(y)) < 3:
                model = None
                continue
            model = _new_model().fit(x.loc[tr, FEATURES], y)
            prior = np.bincount(y, minlength=3) / len(y)
            train_through = days[i - 1]
        te = (codes == i) & ok
        if model is None or not te.any():
            continue
        t = x.loc[te]
        P = _probs(model, t[FEATURES])
        z = pd.DataFrame(index=t.index)
        z["close"] = t.close
        z["session"] = t.session
        z["minute_idx"] = t.minute_idx
        z["horizon"] = horizon
        z["prediction"] = np.array(CLASSES)[P.argmax(1)]
        z["confidence"] = P.max(1)
        z["p_down"], z["p_sideways"], z["p_up"] = P.T
        z["prior_down"], z["prior_sideways"], z["prior_up"] = prior
        yt = y_all[te]
        z["actual"] = pd.Series(np.where(np.isnan(yt), None, np.array(CLASSES, dtype=object)[np.nan_to_num(yt).astype(int)]),
                                index=t.index, dtype=object)
        z["future_bps"] = t[tgt]
        z["thr_bps"] = t[thr]
        z["train_through"] = train_through
        out.append(z)
    if not out:
        raise ValueError("Insufficient training examples")
    return pd.concat(out)


def run_walk_forward(x: pd.DataFrame, horizons=HORIZONS, min_train_days: int = 15, refit_every: int = 5) -> dict:
    return {h: fit_predict(x, h, min_train_days, refit_every) for h in horizons}


# ---------------------------------------------------------------------- metrics
def score_rows(pred: pd.DataFrame) -> pd.DataFrame:
    """Row-level scores for every prediction whose outcome is known."""
    z = pred.dropna(subset=["actual"]).copy()
    if z.empty:
        return z
    yi = z.actual.map(CLASS_IDX).to_numpy(int)
    P = z[["p_down", "p_sideways", "p_up"]].to_numpy()
    B = z[["prior_down", "prior_sideways", "prior_up"]].to_numpy()
    n = np.arange(len(z))
    onehot = np.eye(3)[yi]
    z["ll_model"] = -np.log(np.clip(P[n, yi], 1e-15, 1))
    z["ll_prior"] = -np.log(np.clip(B[n, yi], 1e-15, 1))
    z["brier_model"] = ((P - onehot) ** 2).sum(1)
    z["brier_prior"] = ((B - onehot) ** 2).sum(1)
    z["correct"] = (P.argmax(1) == yi).astype(float)
    z["prior_correct"] = (B.argmax(1) == yi).astype(float)
    z["signal"] = np.select([P.argmax(1) == 2, P.argmax(1) == 0], [1, -1], 0)
    z["gross_bps"] = z.signal * z.future_bps
    return z


def pooled_metrics(pred: pd.DataFrame) -> dict | None:
    """All evaluated rows pooled. Rows overlap in time, so treat these as descriptive and
    use the per-day numbers for any statement about skill."""
    z = score_rows(pred)
    if z.empty:
        return None
    ll_m, ll_p = z.ll_model.mean(), z.ll_prior.mean()
    shares = z.actual.value_counts(normalize=True)
    return {
        "Evaluated": len(z),
        "Accuracy": z.correct.mean(),
        "Prior-class accuracy": z.prior_correct.mean(),
        "Log loss": ll_m,
        "Prior log loss": ll_p,
        "Log-loss skill": 1 - ll_m / ll_p,
        "Brier skill": 1 - z.brier_model.mean() / z.brier_prior.mean(),
        "Actionable %": (z.signal != 0).mean(),
        "% DOWN": shares.get("DOWN", 0.0),
        "% SIDEWAYS": shares.get("SIDEWAYS", 0.0),
        "% UP": shares.get("UP", 0.0),
    }


def _spearman(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 3 or a.nunique() < 2 or b.nunique() < 2:
        return np.nan
    return float(a.rank().corr(b.rank()))


def daily_metrics(pred: pd.DataFrame, cost_bps: float = 1.0) -> pd.DataFrame:
    """One row per session. `ll_gain` > 0 means the model beat the prior-class baseline
    that day. `edge_*` is the mean signed forward return of the UP/DOWN calls (SIDEWAYS =
    no trade); the net figure subtracts an assumed round-trip cost."""
    z = score_rows(pred)
    rows = {}
    for s, d in z.groupby("session"):
        act = d[d.signal != 0]
        gross = act.gross_bps.mean() if len(act) else np.nan
        rows[s] = {
            "n": len(d),
            "accuracy": d.correct.mean(),
            "prior_accuracy": d.prior_correct.mean(),
            "logloss": d.ll_model.mean(),
            "prior_logloss": d.ll_prior.mean(),
            "ll_gain": (d.ll_prior - d.ll_model).mean(),
            "brier_gain": (d.brier_prior - d.brier_model).mean(),
            "ic": _spearman(d.p_up - d.p_down, d.future_bps),
            "n_signals": len(act),
            "edge_gross_bps": gross,
            "edge_net_bps": gross - cost_bps if len(act) else np.nan,
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "session"
    return out


def summarize_days(daily: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Mean across sessions with a bootstrap-over-days 95% interval. Days are treated as
    independent, which is generous for a market with regime persistence."""
    d = daily.copy()
    d["acc_gain"] = d.accuracy - d.prior_accuracy
    cols = {"ll_gain": "Log-loss gain vs prior", "brier_gain": "Brier gain vs prior",
            "acc_gain": "Accuracy gain vs prior", "ic": "Rank IC (p_up - p_down vs return)",
            "edge_gross_bps": "Edge per signal, gross (bps)", "edge_net_bps": "Edge per signal, net (bps)"}
    rng = np.random.default_rng(seed)
    rows = []
    for col, label in cols.items():
        v = d[col].to_numpy(float)
        v = v[~np.isnan(v)]
        if len(v) == 0:
            continue
        idx = rng.integers(0, len(v), size=(n_boot, len(v)))
        means = v[idx].mean(1)
        rows.append({"Metric": label, "Mean": v.mean(), "CI low": np.percentile(means, 2.5),
                     "CI high": np.percentile(means, 97.5), "Days > 0": (v > 0).mean(), "Days": len(v)})
    return pd.DataFrame(rows)


def calibration_table(pred: pd.DataFrame, edges=None) -> pd.DataFrame:
    """Predicted vs observed frequency for the UP and DOWN probabilities."""
    z = score_rows(pred)
    edges = np.arange(0, 1.0001, 0.1) if edges is None else edges
    rows = []
    for cls, col in (("UP", "p_up"), ("DOWN", "p_down")):
        b = pd.cut(z[col], edges, include_lowest=True)
        for iv, d in z.groupby(b, observed=True):
            rows.append({"class": cls, "bin": str(iv), "n": len(d), "mean_pred": d[col].mean(),
                         "observed": (d.actual == cls).mean()})
    return pd.DataFrame(rows)


# -------------------------------------------------------------------- live logger
LOG_COLUMNS = [
    "logged_at_utc", "bar_time", "horizon", "close", "prediction", "confidence",
    "p_down", "p_sideways", "p_up", "prior_down", "prior_sideways", "prior_up",
    "thr_bps", "k", "train_sessions", "train_rows", "train_through", "resolve_time",
    "future_close", "future_bps", "actual", "correct", "resolved_at_utc",
]


def predict_live(x: pd.DataFrame, k: float = DEFAULT_K, horizons=HORIZONS, min_train_days: int = 15):
    """Predict from the latest real bar. The model is fit on sessions strictly before
    that bar's session, exactly as in the walk-forward test. Returns (rows, skipped)
    where `skipped` explains every horizon that was not predicted."""
    bars = x[x.is_bar]
    if bars.empty:
        raise ValueError("No bars")
    t = bars.index.max()
    last = x.loc[t]
    day = last.session
    codes, days = pd.factorize(x.session, sort=True)
    cur = int(np.where(days == day)[0][0])
    rows, skipped = [], []
    if cur < min_train_days:
        return rows, [(h, f"only {cur} prior sessions, need {min_train_days}") for h in horizons]
    ok = _usable(x).to_numpy()
    now = pd.Timestamp.now(tz="UTC").isoformat()
    for h in horizons:
        if last.minute_idx < WARMUP_MINUTES:
            skipped.append((h, "inside the opening warm-up")); continue
        if last.minute_idx + h > SESSION_MINUTES - 1:
            skipped.append((h, "outcome would fall after the close")); continue
        if not bool(ok[x.index.get_loc(t)]):
            skipped.append((h, "too few features on the latest bar")); continue
        y_all = make_labels(x[f"future_{h}"], x[f"thr_{h}"])
        tr = (codes < cur) & ok & ~np.isnan(y_all)
        y = y_all[tr].astype(int)
        if len(np.unique(y)) < 3:
            skipped.append((h, "training data lacks a class")); continue
        model = _new_model().fit(x.loc[tr, FEATURES], y)
        P = _probs(model, x.loc[[t], FEATURES])[0]
        prior = np.bincount(y, minlength=3) / len(y)
        rows.append({
            "logged_at_utc": now, "bar_time": t.isoformat(), "horizon": h, "close": float(last.close),
            "prediction": CLASSES[int(P.argmax())], "confidence": float(P.max()),
            "p_down": P[0], "p_sideways": P[1], "p_up": P[2],
            "prior_down": prior[0], "prior_sideways": prior[1], "prior_up": prior[2],
            "thr_bps": float(last[f"thr_{h}"]), "k": k,
            "train_sessions": int(cur), "train_rows": int(tr.sum()), "train_through": str(days[cur - 1]),
            "resolve_time": (t + pd.Timedelta(minutes=h)).isoformat(),
        })
    return rows, skipped


def _read_log(path: str) -> pd.DataFrame:
    text = ["logged_at_utc", "bar_time", "prediction", "train_through", "resolve_time", "actual", "resolved_at_utc"]
    log = pd.read_csv(path, dtype={c: object for c in text})
    log = log.reindex(columns=LOG_COLUMNS)
    for c in text:                       # all-empty columns must stay writable as text
        log[c] = log[c].astype(object)
    return log


def _write_atomic(df: pd.DataFrame, path: str) -> None:
    folder = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=folder, suffix=".tmp")
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def append_log(rows: list, path: str) -> int:
    """Append predictions; a (bar_time, horizon) pair already in the log is never duplicated."""
    new = pd.DataFrame(rows).reindex(columns=LOG_COLUMNS)
    if new.empty:
        return 0
    if os.path.exists(path):
        old = _read_log(path)
        seen = set(zip(old.bar_time, old.horizon.astype(int)))
        new = new[[(b, int(h)) not in seen for b, h in zip(new.bar_time, new.horizon)]]
        if new.empty:
            return 0
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    _write_atomic(out, path)
    return len(new)


def resolve_log(path: str, grid: pd.DataFrame) -> dict:
    """Fill in outcomes for logged predictions whose resolution minute is now in the data.
    A prediction whose exact minute is absent although later data exists is marked NO_BAR
    and excluded from scoring (never approximated from a neighbouring bar)."""
    if not os.path.exists(path):
        return {"resolved": 0, "no_bar": 0, "pending": 0}
    log = _read_log(path)
    log["actual"] = log["actual"].astype(object)
    closes = grid.close.dropna()
    closes.index = closes.index.tz_convert("UTC")
    last_utc = closes.index.max()
    now = pd.Timestamp.now(tz="UTC").isoformat()
    res = nob = 0
    for i in log.index[log.actual.isna()]:
        rt = pd.Timestamp(log.at[i, "resolve_time"]).tz_convert("UTC")
        if rt > last_utc:
            continue
        if rt not in closes.index:
            log.at[i, "actual"], log.at[i, "resolved_at_utc"] = "NO_BAR", now
            nob += 1
            continue
        fc = float(closes.loc[rt])
        bps = (fc / float(log.at[i, "close"]) - 1) * 1e4
        thr = float(log.at[i, "thr_bps"])
        act = "UP" if bps > thr else "DOWN" if bps < -thr else "SIDEWAYS"
        log.at[i, "future_close"], log.at[i, "future_bps"], log.at[i, "actual"] = fc, bps, act
        log.at[i, "correct"] = float(act == log.at[i, "prediction"])
        log.at[i, "resolved_at_utc"] = now
        res += 1
    _write_atomic(log, path)
    return {"resolved": res, "no_bar": nob, "pending": int(log.actual.isna().sum())}


def summarize_log(path: str) -> pd.DataFrame:
    """Forward-test scorecard per horizon from resolved log rows, against the prior baseline."""
    if not os.path.exists(path):
        return pd.DataFrame()
    log = _read_log(path)
    log = log[log.actual.isin(CLASSES)]
    rows = []
    for h, d in log.groupby("horizon"):
        d = d.copy()
        d["session"] = pd.to_datetime(d.bar_time, utc=True).dt.tz_convert(TZ).dt.date
        m = pooled_metrics(d)
        if m:
            rows.append({"Horizon": f"{int(h)} min", "Sessions": d.session.nunique(), **m})
    return pd.DataFrame(rows)
