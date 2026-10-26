import datetime as dt
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import requests
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

st.set_page_config(page_title='QQQ Intraday Predictor', layout='wide')
HORIZONS = [5, 15, 30, 60]
TZ = 'America/New_York'
FEATURES = ['ret1','ret5','ret15','ema_spread','ema50_gap','vwap_gap','rsi','macd_hist','adx','atr_pct','rvol','range_pct','body_pct','wick_up','wick_down','break20','sin_time','cos_time']

def clean(raw):
    x = raw.copy()

    # Normalize column names
    x.columns = [str(c).lower() for c in x.columns]

    # Convert timestamp to New York timezone
    if "timestamp" in x.columns:
        parsed = pd.to_datetime(
            x["timestamp"],
            errors="coerce"
        )

        if parsed.dt.tz is None:
            # Assume timezone-naive CSV timestamps are Eastern
            x["timestamp"] = parsed.dt.tz_localize(
                "America/New_York",
                ambiguous="NaT",
                nonexistent="NaT"
            )
        else:
            x["timestamp"] = parsed.dt.tz_convert(
                "America/New_York"
            )

        x = x.set_index("timestamp")

    # Handle existing datetime index
    if not isinstance(x.index, pd.DatetimeIndex):
        raise ValueError("Missing valid timestamp column")

    if x.index.tz is None:
        x.index = x.index.tz_localize(
            "America/New_York",
            ambiguous="NaT",
            nonexistent="NaT"
        )
    else:
        x.index = x.index.tz_convert(
            "America/New_York"
        )

    # Keep required price and volume columns
    required = ["open", "high", "low", "close", "volume"]

    x = x[required].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()

    # Remove invalid and duplicate timestamps
    x = x[~x.index.isna()]
    x = x[~x.index.duplicated(keep="last")]
    x = x.sort_index()
        # Keep regular US trading hours
    x = x.between_time("09:30", "15:59")
    x = x[x.index.dayofweek < 5]

    return x

@st.cache_data(ttl=60, show_spinner=False)
def alpaca(start, end, feed, key, secret):
    headers = {'APCA-API-KEY-ID':key, 'APCA-API-SECRET-KEY':secret}
    params = {'start':start,'end':end, 'timeframe':'1Min','feed':feed,'limit':10000,'adjustment':'split','sort':'asc'}
    rows=[]
    with requests.Session() as s:
        for _ in range(200):
            r=s.get('https://data.alpaca.markets/v2/stocks/QQQ/bars',params=params,headers=headers,timeout=45)
            r.raise_for_status()
            payload=r.json()
            rows.extend(payload.get('bars',[]))
            token=payload.get('next_page_token')
            if not token: break
            params['page_token']=token
        else: raise RuntimeError('Too many pages')
    if not rows: raise ValueError('Alpaca returned no data')
    return clean(pd.DataFrame(rows).rename(columns={'t':'timestamp','o':'open','h':'high','l':'low','c':'close','v':'volume'}))

def rma(s,n):
    return s.ewm(alpha=1/n,adjust=False,min_periods=n).mean()

def make_features(df):
    x=df.copy()
    c,h,l,o,v=x.close,x.high,x.low,x.open,x.volume
    # Do not carry daily returns across sessions.
    day=pd.Series(x.index.date,index=x.index)
    def day_shift(series,n): return series.groupby(day).shift(n)
    x['ret1']=(c/day_shift(c,1)-1)*10000
    x['ret5']=(c/day_shift(c,5)-1)*10000
    x['ret15']=(c/day_shift(c,15)-1)*10000
    e9=c.ewm(span=9,adjust=False).mean(); e20=c.ewm(span=20,adjust=False).mean(); e50=c.ewm(span=50,adjust=False).mean()
    x['ema_spread']=(e9/e20-1)*10000
    x['ema50_gap']=(c/e50-1)*10000
    tp=(h+l+c)/3
    x['vwap_gap']=(c/((tp*v).groupby(day).cumsum()/v.groupby(day).cumsum().replace(0,np.nan))-1)*10000
    delta=c.diff(); gain=rma(delta.clip(lower=0),14); loss=rma((-delta).clip(lower=0),14)
    x['rsi']=100-100/(1+gain/loss.replace(0,np.nan))
    x.loc[(loss==0)&(gain>0),'rsi']=100
    x.loc[(loss==0)&(gain==0),'rsi']=50
    m=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean()
    x['macd_hist']=(m-m.ewm(span=9,adjust=False).mean())/c*10000
    prev=c.shift(); tr=pd.concat([h-l,(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1)
    atr=rma(tr,14); x['atr_pct']=atr/c*10000
    up=h.diff(); down=-l.diff(); pdm=up.where((up>down)&(up>0),0); mdm=down.where((down>up)&(down>0),0)
    pdi=100*rma(pdm,14)/atr.replace(0,np.nan); mdi=100*rma(mdm,14)/atr.replace(0,np.nan)
    x['adx']=rma(100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan),14)
    # Same-minute prior-day volume baseline, with no current/future day volume.
    minute=x.index.hour*60+x.index.minute
    vol=x.assign(day=x.index.date,minute=minute).pivot(index='day',columns='minute',values='volume')
    base=vol.shift(1).rolling(10,min_periods=3).mean()
    x['rvol']=[v/base.loc[d,m] if pd.notna(base.loc[d,m]) and base.loc[d,m]>0 else np.nan for d,m,v in zip(x.index.date,minute,x.volume)]
    x['rvol']=x.rvol.clip(0,20)
    x['range_pct']=(h-l)/c*10000; x['body_pct']=(c-o)/c*10000
    x['wick_up']=(h-np.maximum(c,o))/c*10000; x['wick_down']=(np.minimum(c,o)-l)/c*10000
    high20=day_shift(h.groupby(day).rolling(20,min_periods=20).max().droplevel(0),1)
    low20=day_shift(l.groupby(day).rolling(20,min_periods=20).min().droplevel(0),1)
    x['break20']=np.where(c>high20,1,np.where(c<low20,-1,0))
    minute_in_day=minute-(9*60+30)
    x['sin_time']=np.sin(2*np.pi*minute_in_day/390)
    x['cos_time']=np.cos(2*np.pi*minute_in_day/390)
    x['session']=x.index.date
    # Session-close outcomes, NEVER shift across days.
    for horizon in HORIZONS:
        future=c.groupby(day).shift(-horizon)
        x[f'future_{horizon}']=(future/c-1)*10000
    return x

def fit_predict(x,horizon,min_train_days=15,refit_every=5):
    target=f'future_{horizon}'
    days=np.array(sorted(x.session.unique()))
    if len(days)<min_train_days+2:
        raise ValueError(f'Need at least {min_train_days+2} sessions; found {len(days)}')
    output=[]
    model=None
    for i in range(min_train_days,len(days)):
        # Model for day i trained on prior days only. Exclude last h bars of prior day automatically via NaN labels.
        if model is None or (i-min_train_days)%refit_every==0:
            train=x[x.session.isin(days[:i])].dropna(subset=[target]).copy()
            train=train[train[FEATURES].notna().sum(axis=1)>=12]
            # 3 classes: down (< -threshold), sideways, up (> threshold).
            # Threshold is predeclared, not tuned on the test set.
            threshold={5:5,15:8,30:12,60:18}[horizon]
            y=np.where(train[target]>threshold,2,np.where(train[target]<-threshold,0,1))
            if len(np.unique(y))<3: continue
            model=make_pipeline(SimpleImputer(strategy='median',add_indicator=False),StandardScaler(),LogisticRegression(max_iter=350,class_weight=None))
            model.fit(train[FEATURES],y)
        test=x[x.session==days[i]].copy()
        test=test[test[FEATURES].notna().sum(axis=1)>=12]
        if test.empty or model is None: continue
        p=model.predict_proba(test[FEATURES]); prob=np.zeros((len(test),3))
        for j,cl in enumerate(model.classes_): prob[:,int(cl)]=p[:,j]
        test['p_down'],test['p_sideways'],test['p_up']=prob.T
        test['prediction']=np.array(['DOWN','SIDEWAYS','UP'])[np.argmax(prob,axis=1)]
        test['confidence']=prob.max(axis=1)
        test['actual']=np.where(test[target]>threshold,'UP',np.where(test[target]<-threshold,'DOWN','SIDEWAYS'))
        test.loc[test[target].isna(),'actual']=None
        test['horizon']=horizon
        output.append(test[['close','session','horizon','prediction','confidence','p_down','p_sideways','p_up','actual',target]])
    if not output: raise ValueError('Insufficient training examples')
    return pd.concat(output)

def metrics(pred):
    z=pred.dropna(subset=['actual']).copy()
    if z.empty: return None
    classes=['DOWN','SIDEWAYS','UP']
    probs=z[['p_down','p_sideways','p_up']].to_numpy()
    yi=z.actual.map({k:i for i,k in enumerate(classes)}).to_numpy()
    base=max(z.actual.value_counts(normalize=True))
    return {'Evaluated':len(z),'Accuracy':accuracy_score(z.actual,z.prediction),'Majority baseline':base,
            'Log loss':log_loss(yi,probs,labels=[0,1,2]),'Multiclass Brier':np.mean(np.sum((probs-np.eye(3)[yi])**2,axis=1)),
            'Actionable %':(z.prediction!='SIDEWAYS').mean()}

st.title('QQQ | Intraday Prediction & Walk-forward Validation')
st.caption('Research tool only. Predictions are statistical estimates, not guarantees. No order execution.')
# Read credentials automatically from Streamlit Secrets
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
with st.sidebar.expander("Alpaca Connection Status"):
    st.write(
        "API Key:",
        "Configured" if API_KEY else "Missing"
    )

    st.write(
        "Secret Key:",
        "Configured" if API_SECRET else "Missing"
    )

    if API_KEY and API_SECRET:
        st.success("Credentials loaded from Secrets")
    else:
        st.error("Credentials not found")
        
CHART_WINDOWS = {
    "5 minutes": 5,
    "15 minutes": 15,
    "1 hour": 60,
    "2 hours": 120,
    "3 hours": 180,
    "4 hours": 240,
    "Since market open": None,
}

with st.sidebar:
    st.header("QQQ Live Chart")

    chart_window = st.selectbox(
        "Chart timeframe",
        list(CHART_WINDOWS.keys()),
        index=6,
    )

    feed = st.selectbox(
        "Feed",
        ["iex", "sip"],
        index=0 if DEFAULT_FEED == "iex" else 1,
    )

    st.caption("Auto-refresh: every 30 seconds")

    st.divider()
    st.header("Historical Validation")

    uploaded = st.file_uploader(
        "Optional: QQQ 1-minute CSV",
        type=["csv"],
    )

    min_train = st.slider(
        "Initial training sessions", 10, 35, 15
    )
    refit = st.slider(
        "Refit every N sessions", 1, 10, 5
    )

    run = st.button(
        "Run walk-forward test", type="primary"
    )

if not API_KEY or not API_SECRET:
    st.warning(
        "Missing Alpaca credentials. "
        "Configure them in Streamlit Settings > Secrets."
    )

if run:
    try:
        if uploaded is not None:
            raw = clean(pd.read_csv(uploaded))
        else:
            if not API_KEY or not API_SECRET:
                raise ValueError(
                    "Alpaca credentials missing from Secrets."
                )

            now = (
                dt.datetime.now(dt.timezone.utc)
                - dt.timedelta(minutes=2)
            )

            raw = alpaca(
                (now - dt.timedelta(days=75)).isoformat(),
                now.isoformat(),
                feed,
                API_KEY,
                API_SECRET,
            )

        if raw.empty:
            raise ValueError(
                "No regular-session QQQ bars returned."
            )

        st.session_state["raw"] = raw

        with st.spinner("Running walk-forward test..."):
            feat = make_features(raw)
            results = {
                h: fit_predict(feat, h, min_train, refit)
                for h in HORIZONS
            }

        st.session_state["results"] = results

    except Exception as exc:
        st.error(f"Unable to complete test: {exc}")
if 'results' in st.session_state:
    results=st.session_state['results']; raw=st.session_state['raw']
    st.write(f'**QQQ data:** {raw.index.min()} through {raw.index.max()} | {len(raw):,} minute bars | {len(set(raw.index.date))} sessions')
    records=[]
    for h,z in results.items():
        m=metrics(z)
        if m: records.append({'Horizon':f'{h} min',**m})
    summary=pd.DataFrame(records)
    st.subheader('Out-of-sample performance (chronological walk-forward)')
    st.dataframe(summary.style.format({'Accuracy':'{:.1%}','Majority baseline':'{:.1%}','Log loss':'{:.3f}','Multiclass Brier':'{:.3f}','Actionable %':'{:.1%}'}),hide_index=True,use_container_width=True)
    st.caption('Majority baseline is the most common realized class over the test period; it is a descriptive reference, not a tradable strategy. Results are correlated because forecasts overlap. No spread, slippage or fill simulation is included.')
    h=st.selectbox('Prediction horizon (minutes)',HORIZONS,index=1)
    z=results[h]
    days=sorted(z.session.unique())
    chosen=st.selectbox('Historical replay day',days,index=len(days)-1)
    day=z[z.session==chosen]
    valid=day.dropna(subset=['actual'])
    if len(valid):
        cm=pd.crosstab(valid.actual,valid.prediction,dropna=False)
        st.write('**Confusion matrix** (actual rows, predicted columns)')
        st.dataframe(cm)
    view=day.copy()
    view['timestamp']=view.index
    st.dataframe(view[['timestamp','close','prediction','confidence','p_up','p_down','p_sideways','actual']].tail(60),hide_index=True,use_container_width=True)
    chart=raw[raw.index.date==chosen]
    fig=go.Figure(go.Candlestick(x=chart.index,open=chart.open,high=chart.high,low=chart.low,close=chart.close,name='QQQ'))
    fig.update_layout(height=480,xaxis_rangeslider_visible=False,title=f'QQQ {chosen} (ET)')
    st.plotly_chart(fig,use_container_width=True)
    csv=pd.concat(results.values()).reset_index().rename(columns={'index':'timestamp'}).to_csv(index=False).encode()
    st.download_button('Download all predictions CSV',csv,'QQQ_walkforward_predictions.csv','text/csv')
    st.info('Latest test-day probabilities are historical replay estimates. To generate a live forecast, download the latest complete QQQ candles and rerun; this version deliberately does not extrapolate an unvalidated model into live trading.')
else:
    st.info('Upload the existing QQQ 1-minute CSV or connect Alpaca, then click Run walk-forward test. This upgrades your existing Streamlit app; no second app is needed.')


st.divider()
st.subheader("QQQ Live Chart & Forecast")

@st.fragment(run_every="30s")
def live_forecast():
    if not API_KEY or not API_SECRET:
        st.info("Configure Alpaca keys in Streamlit Secrets.")
        return

    try:
        now = dt.datetime.now(dt.timezone.utc)
        end_dt = now - dt.timedelta(minutes=2)

        raw = alpaca(
            (end_dt - dt.timedelta(days=45)).isoformat(),
            end_dt.isoformat(),
            feed,
            API_KEY,
            API_SECRET,
        )

        latest = raw.iloc[-1]
        last_ts = raw.index[-1]

        st.metric(
            "QQQ Latest Completed Close",
            f"${latest.close:.2f}",
        )

        st.caption(
            f"Feed: {feed.upper()} | "
            f"Last candle: {last_ts}"
        )

        # Select the most recent regular trading session
        session_data = raw[
            raw.index.date == last_ts.date()
        ]

        window_minutes = CHART_WINDOWS[chart_window]

        if window_minutes is None:
            chart_data = session_data
        else:
            chart_data = session_data.tail(window_minutes)

        fig = go.Figure(
            go.Candlestick(
                x=chart_data.index,
                open=chart_data.open,
                high=chart_data.high,
                low=chart_data.low,
                close=chart_data.close,
                name="QQQ",
            )
        )

        fig.update_layout(
            height=480,
            xaxis_rangeslider_visible=False,
            title=f"QQQ - {chart_window} (Eastern)",
        )

        st.plotly_chart(fig, use_container_width=True)

        # Generate forecasts using the original V2 model
        if len(set(raw.index.date)) < 17:
            st.warning(
                "At least 17 historical sessions are needed."
            )
            return

        feat = make_features(raw)

        latest_day = feat.session.iloc[-1]
        train = feat[feat.session != latest_day]
        latest_features = feat.iloc[[-1]]

        if (
            latest_features[FEATURES]
            .notna()
            .sum(axis=1)
            .iloc[0] < 12
        ):
            st.warning("Insufficient features.")
            return

        forecasts = []

        for h in HORIZONS:
            target = f"future_{h}"

            tr = train.dropna(subset=[target]).copy()
            tr = tr[
                tr[FEATURES].notna().sum(axis=1) >= 12
            ]

            threshold = {
                5: 5,
                15: 8,
                30: 12,
                60: 18,
            }[h]

            y = np.where(
                tr[target] > threshold,
                2,
                np.where(tr[target] < -threshold, 0, 1),
            )

            if len(np.unique(y)) < 3:
                continue

            model = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(max_iter=350),
            )

            model.fit(tr[FEATURES], y)

            p = model.predict_proba(
                latest_features[FEATURES]
            )[0]

            probs = np.zeros(3)

            for j, cls in enumerate(model.classes_):
                probs[int(cls)] = p[j]

            direction = ["DOWN", "SIDEWAYS", "UP"][
                int(np.argmax(probs))
            ]

            forecasts.append({
                "Horizon": f"{h} min",
                "Forecast": direction,
                "P(DOWN)": probs[0],
                "P(SIDEWAYS)": probs[1],
                "P(UP)": probs[2],
            })

        if forecasts:
            st.dataframe(
                pd.DataFrame(forecasts).style.format({
                    "P(DOWN)": "{:.1%}",
                    "P(SIDEWAYS)": "{:.1%}",
                    "P(UP)": "{:.1%}",
                }),
                hide_index=True,
                use_container_width=True,
            )

            st.caption(
                "Uncalibrated Version 2 probabilities. "
                "Market data refreshes every 30 seconds."
            )

    except requests.HTTPError as exc:
        st.error(f"Alpaca API error: {exc}")

    except Exception as exc:
        st.error(f"Live refresh failed: {exc}")


live_forecast()
