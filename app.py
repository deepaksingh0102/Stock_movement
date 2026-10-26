import datetime as dt
import os
import warnings

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import qqq_core as q

warnings.filterwarnings("ignore")
st.set_page_config(page_title="QQQ Intraday Walk-forward", layout="wide")

LOG_PATH = os.environ.get("QQQ_LOG_PATH", "live_predictions.csv")
FMT_SUM = {c: "{:.4f}" for c in ["Mean", "CI low", "CI high"]} | {"Days > 0": "{:.0%}"}


def secret(name):
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


@st.cache_data(ttl=900, show_spinner=False)
def cached_alpaca(start, end, feed, key, sec):
    return q.fetch_alpaca(start, end, feed, key, sec)


@st.cache_data(show_spinner=False)
def cached_csv(buf):
    import io
    return q.load_csv(io.BytesIO(buf))


st.title("QQQ | Intraday walk-forward validation")
st.caption("Research tool. Not financial advice, and no orders are placed. The question this app answers is "
           "whether the model beats a naive baseline out of sample, not what to trade.")

with st.sidebar:
    st.header("Data")
    source = st.radio("Source", ["Upload CSV", "Alpaca API"])
    uploaded = key = sec = None
    if source == "Upload CSV":
        uploaded = st.file_uploader("QQQ 1-minute CSV (timestamp,open,high,low,close,volume)", type=["csv"])
    else:
        start = st.date_input("Start", dt.date.today() - dt.timedelta(days=120))
        feed = st.selectbox("Feed", ["iex", "sip"])
        key = st.text_input("Alpaca key", value=secret("ALPACA_API_KEY"), type="password")
        sec = st.text_input("Alpaca secret", value=secret("ALPACA_SECRET_KEY"), type="password")
    st.header("Model")
    k = st.slider("Label threshold k (sigma units)", 0.1, 1.5, q.DEFAULT_K, 0.05,
                  help="UP/DOWN = forward return beyond ±k·σ·√horizon, with σ a causal EWMA of 1-minute volatility.")
    min_train = st.slider("Initial training sessions", 10, 35, 15)
    refit = st.slider("Refit every N sessions", 1, 10, 5)
    q.MODEL_C = st.select_slider("Regularization C (smaller = stronger)", [0.0005, 0.001, 0.01, 0.1, 1.0], 1.0)
    cost = st.number_input("Assumed round-trip cost (bps)", 0.0, 10.0, 1.0, 0.1)
    run = st.button("Run walk-forward test", type="primary")


def load_raw():
    if source == "Upload CSV":
        if uploaded is None:
            raise ValueError("Upload a CSV first.")
        return cached_csv(uploaded.getvalue())
    if not key or not sec:
        raise ValueError("Provide Alpaca credentials.")
    return cached_alpaca(str(start), None, feed, key, sec)


if run:
    try:
        raw = load_raw()
        with st.spinner("Building grid, features and walk-forward forecasts..."):
            grid, info = q.prepare(raw.reset_index())
            x = q.make_features(grid, k=k)
            st.session_state.update(grid=grid, info=info, x=x, cost=cost,
                                    results=q.run_walk_forward(x, min_train_days=min_train, refit_every=refit))
    except Exception as e:
        st.error(f"Unable to complete test: {e}")

tab_val, tab_live = st.tabs(["Walk-forward validation", "Live forecast & log"])

with tab_val:
    if "results" not in st.session_state:
        st.info("Load data and click **Run walk-forward test**.")
    else:
        res, info, grid, cost_bps = (st.session_state[n] for n in ("results", "info", "grid", "cost"))
        st.write(f"**Data:** {grid.index.min():%Y-%m-%d} to {grid.index.max():%Y-%m-%d} · "
                 f"{info['sessions']} sessions · {info['bars']:,} bars · "
                 f"{info['missing_bars']:,} missing minutes (never filled with invented prices)")
        if info["dropped_sessions"]:
            st.warning("Dropped sessions with too few bars (half days or partial downloads): " +
                       ", ".join(f"{d} ({n} bars)" for d, n in info["dropped_sessions"]))

        st.subheader("Verdict by horizon: does the model beat the baseline?")
        st.caption("Baseline = class frequencies of the training set the model was fit on. "
                   "Per-day numbers are means across test sessions with a bootstrap-over-days 95% interval; "
                   "an interval that excludes zero is the minimum bar for claiming skill.")
        verdict = []
        for h, z in res.items():
            s = q.summarize_days(q.daily_metrics(z, cost_bps)).set_index("Metric")
            ll = s.loc["Log-loss gain vs prior"]
            status = ("Beats baseline" if ll["CI low"] > 0 else
                      "Worse than baseline" if ll["CI high"] < 0 else "Indistinguishable from baseline")
            verdict.append({"Horizon": f"{h} min", "Test days": int(ll["Days"]), "Log-loss gain": ll["Mean"],
                            "95% CI low": ll["CI low"], "95% CI high": ll["CI high"],
                            "Days beating baseline": ll["Days > 0"], "Verdict": status})
        st.dataframe(pd.DataFrame(verdict).style.format(
            {"Log-loss gain": "{:.4f}", "95% CI low": "{:.4f}", "95% CI high": "{:.4f}", "Days beating baseline": "{:.0%}"}),
            hide_index=True, use_container_width=True)

        st.subheader("Pooled metrics (overlapping rows, descriptive only)")
        pooled = pd.DataFrame([{"Horizon": f"{h} min", **m} for h, z in res.items() if (m := q.pooled_metrics(z))])
        st.dataframe(pooled.style.format({c: "{:.1%}" for c in pooled.columns if c.startswith("%") or "ccuracy" in c or c == "Actionable %"}
                                         | {"Log loss": "{:.3f}", "Prior log loss": "{:.3f}",
                                            "Log-loss skill": "{:+.3f}", "Brier skill": "{:+.3f}"}),
                     hide_index=True, use_container_width=True)

        h = st.selectbox("Horizon for detail", q.HORIZONS, index=1)
        z = res[h]
        daily = q.daily_metrics(z, cost_bps)
        st.subheader(f"{h}-minute horizon: per-day detail")
        st.dataframe(q.summarize_days(daily).style.format(FMT_SUM | {"Days": "{:.0f}"}), hide_index=True, use_container_width=True)
        fig = go.Figure(go.Bar(x=daily.index.astype(str), y=daily.ll_gain,
                               marker_color=["#2a9d8f" if v > 0 else "#e76f51" for v in daily.ll_gain]))
        fig.update_layout(height=300, title="Log-loss gain vs prior baseline, by session (above 0 = model better)",
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Per-day table"):
            st.dataframe(daily.style.format("{:.4f}"), use_container_width=True)

        st.subheader("Calibration")
        cal = q.calibration_table(z)
        fig = go.Figure()
        for cls, colr in (("UP", "#2a9d8f"), ("DOWN", "#e76f51")):
            c = cal[cal["class"] == cls]
            fig.add_trace(go.Scatter(x=c.mean_pred, y=c.observed, mode="lines+markers", name=cls,
                                     line_color=colr, text=c.n, hovertemplate="pred %{x:.2f}, observed %{y:.2f}, n=%{text}"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect", line=dict(dash="dash", color="gray")))
        fig.update_layout(height=340, xaxis_title="Predicted probability", yaxis_title="Observed frequency",
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        scored = q.score_rows(z)
        fwd = scored.groupby("prediction").agg(n=("future_bps", "size"), mean_fwd_bps=("future_bps", "mean"),
                                               median_fwd_bps=("future_bps", "median"))
        st.write("**Average forward return by predicted class (bps)**")
        st.dataframe(fwd.style.format({"mean_fwd_bps": "{:.2f}", "median_fwd_bps": "{:.2f}"}), use_container_width=True)

        st.subheader("Replay a session")
        sessions = sorted(z.session.unique())
        chosen = st.selectbox("Session", sessions, index=len(sessions) - 1)
        day = z[z.session == chosen]
        valid = day.dropna(subset=["actual"])
        if len(valid):
            st.write("**Confusion matrix** (actual rows, predicted columns)")
            st.dataframe(pd.crosstab(valid.actual, valid.prediction, dropna=False))
        st.dataframe(day.reset_index()[["timestamp", "close", "prediction", "confidence", "p_up", "p_down", "p_sideways",
                                        "thr_bps", "future_bps", "actual"]].tail(60), hide_index=True, use_container_width=True)
        cd = grid[(grid.session == chosen) & grid.close.notna()]
        fig = go.Figure(go.Candlestick(x=cd.index, open=cd.open, high=cd.high, low=cd.low, close=cd.close))
        fig.update_layout(height=420, xaxis_rangeslider_visible=False, title=f"QQQ {chosen} (ET)",
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("Download all predictions CSV", pd.concat(res.values()).reset_index().to_csv(index=False).encode(),
                           "QQQ_walkforward_predictions.csv", "text/csv")

with tab_live:
    st.write("Forward testing is the only evidence that is not contaminated by the choices made while building the model. "
             "Each click trains on all **completed** sessions, predicts from the latest bar, and appends the forecast to the "
             "log with its resolution time. Outcomes are filled in from later data, using the exact resolution minute only.")
    if "x" not in st.session_state:
        st.info("Run the walk-forward test first so data is loaded.")
    else:
        st.caption("Needs data that includes the latest completed bar (Alpaca source, or a freshly exported CSV). "
                   "The live session is treated as in progress, and the walk-forward grid is rebuilt for it.")
        c1, c2 = st.columns(2)
        if c1.button("Log live forecast"):
            try:
                raw = load_raw()
                g, _ = q.prepare(raw.reset_index(), live=True)
                xl = q.make_features(g, k=k)
                rows, skipped = q.predict_live(xl, k=k, min_train_days=min_train)
                n = q.append_log(rows, LOG_PATH)
                st.success(f"Latest bar {g.close.dropna().index.max():%Y-%m-%d %H:%M} ET · logged {n} new forecast(s).")
                if rows:
                    st.dataframe(pd.DataFrame(rows)[["bar_time", "horizon", "prediction", "confidence", "p_down", "p_sideways",
                                                     "p_up", "thr_bps", "resolve_time"]], hide_index=True, use_container_width=True)
                for hz, why in skipped:
                    st.caption(f"{hz} min not predicted: {why}")
            except Exception as e:
                st.error(f"Live forecast failed: {e}")
        if c2.button("Resolve outcomes"):
            try:
                g, _ = q.prepare(load_raw(), live=True)
                st.write(q.resolve_log(LOG_PATH, g))
            except Exception as e:
                st.error(f"Could not resolve: {e}")
    sc = q.summarize_log(LOG_PATH)
    if not sc.empty:
        st.subheader("Forward-test scorecard (resolved forecasts only)")
        st.dataframe(sc.style.format({c: "{:.1%}" for c in sc.columns if c.startswith("%") or "ccuracy" in c or c == "Actionable %"}
                                     | {"Log loss": "{:.3f}", "Prior log loss": "{:.3f}",
                                        "Log-loss skill": "{:+.3f}", "Brier skill": "{:+.3f}"}),
                     hide_index=True, use_container_width=True)
    if os.path.exists(LOG_PATH):
        st.download_button("Download prediction log", open(LOG_PATH, "rb").read(), "live_predictions.csv", "text/csv")
        st.caption("Streamlit Community Cloud storage is ephemeral. Download the log regularly, or run `python live_log.py` "
                   "on a machine you control to keep a permanent record.")
