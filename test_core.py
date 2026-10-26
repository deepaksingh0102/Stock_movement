import os, tempfile, unittest, warnings
import numpy as np, pandas as pd
import qqq_core as q
warnings.filterwarnings("ignore")
CSV = os.environ.get("QQQ_TEST_CSV", "QQQ_1min_60d.csv")   # needs >= 30 full sessions
RAW = pd.read_csv(CSV)
SMALL = dict(min_train_days=15, refit_every=1)

def build(raw=RAW, live=False, k=q.DEFAULT_K):
    g, info = q.prepare(raw, live=live)
    return q.make_features(g, k=k), g, info

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.g, cls.info = build()

    def test_full_grid(self):
        self.assertEqual(self.info["missing_bars"], 0)
        self.assertTrue((self.g.groupby("session").size() == 390).all())

    def test_future_is_exact_clock_offset(self):
        x = self.x
        for h in (5, 15, 60):
            t = x.index[1000]
            tgt = t + pd.Timedelta(minutes=h)
            self.assertAlmostEqual(x.loc[t, f"future_{h}"], (x.loc[tgt, "close"] / x.loc[t, "close"] - 1) * 1e4, places=9)

    def test_gaps_give_nan_not_neighbour(self):
        rng = np.random.default_rng(1)
        drop = rng.random(len(RAW)) < 0.04
        raw = RAW[~drop]
        x, g, info = build(raw)
        self.assertGreater(info["missing_bars"], 100)
        ref = self.x  # complete data
        for h in (5, 15):
            f = x[f"future_{h}"]
            # target minute missing -> NaN
            tgt_missing = x.close.shift(-h).isna()  # grid is contiguous per session; cross-day rows handled below
            same_day = x.session == x.session.shift(-h)
            bad = (tgt_missing & same_day & x.is_bar) & f.notna()
            self.assertEqual(int(bad.sum()), 0)
            # where defined, it must equal the complete-data value at the same timestamp (same prices)
            both = f.notna() & ref[f"future_{h}"].reindex(x.index).notna()
            np.testing.assert_allclose(f[both].to_numpy(), ref[f"future_{h}"].reindex(x.index)[both].to_numpy(), atol=1e-9)
        # gap bars are never used as prediction rows
        self.assertFalse(q._usable(x)[~x.is_bar].any())

    def test_features_are_causal(self):
        # features at times <= T must not change when everything after T is removed or scrambled
        T = self.x.index[self.x.session == sorted(self.x.session.unique())[30]][200]
        raw_ts = pd.to_datetime(RAW.timestamp, utc=True)
        trunc = RAW[raw_ts <= T.tz_convert("UTC")]
        x2, _, _ = build(trunc, live=True)
        a = self.x.loc[:T, q.FEATURES + ["vol_bps"]]
        b = x2.loc[:T, q.FEATURES + ["vol_bps"]]
        pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-9, rtol=1e-9)

    def test_walk_forward_ignores_future_data(self):
        # corrupt every session after day 25; predictions for sessions <= 24 must be identical
        days = sorted(self.x.session.unique())
        cut = days[25]
        raw = RAW.copy()
        ts = pd.to_datetime(raw.timestamp, utc=True).dt.tz_convert(q.TZ)
        late = ts.dt.date >= cut
        rng = np.random.default_rng(0)
        for c in ("open", "high", "low", "close"):
            raw.loc[late, c] = raw.loc[late, c] * (1 + rng.normal(0, 0.01, late.sum()))
        raw["high"] = raw[["open", "high", "low", "close"]].max(axis=1); raw["low"] = raw[["open", "high", "low", "close"]].min(axis=1)
        x2, _, _ = build(raw)
        p1 = q.fit_predict(self.x, 15, 15, 1); p2 = q.fit_predict(x2, 15, 15, 1)
        # a session's label needs only that same session, so sessions strictly before `cut` are unaffected
        a = p1[p1.session < cut]; b = p2[p2.session < cut]
        pd.testing.assert_frame_equal(a[["p_down", "p_sideways", "p_up", "prediction"]], b[["p_down", "p_sideways", "p_up", "prediction"]])

    def test_train_sessions_strictly_before_test(self):
        p = q.fit_predict(self.x, 15, 15, 5)
        self.assertTrue((pd.to_datetime(p.train_through) < pd.to_datetime(p.session)).all())

    def test_labels(self):
        y = q.make_labels([5, -5, 0.1, np.nan, 3], [2, 2, 2, 2, np.nan])
        np.testing.assert_array_equal(y[:3], [2, 0, 1]); self.assertTrue(np.isnan(y[3]) and np.isnan(y[4]))

    def test_threshold_scales_with_vol(self):
        x = self.x.dropna(subset=["vol_bps"])
        r = x["thr_60"] / x["thr_15"]
        self.assertAlmostEqual(float(r.median()), 2.0, places=6)   # sqrt(60/15)
        self.assertGreater(x.thr_15.max() / x.thr_15.min(), 2)      # varies across regimes

    def test_live_matches_walk_forward_and_logger(self):
        days = sorted(self.x.session.unique())
        d = days[28]
        T = self.x.index[self.x.session == d][120]
        ts = pd.to_datetime(RAW.timestamp, utc=True)
        part = RAW[ts <= T.tz_convert("UTC")]
        xl, gl, _ = build(part, live=True)
        rows, skipped = q.predict_live(xl, horizons=[5, 15])
        self.assertEqual([r["horizon"] for r in rows], [5, 15])
        wf = {h: q.fit_predict(self.x, h, 15, 1) for h in (5, 15)}
        for r in rows:
            w = wf[r["horizon"]].loc[T]
            for c in ("p_down", "p_sideways", "p_up"):
                self.assertAlmostEqual(r[c], float(w[c]), places=7)
            self.assertAlmostEqual(r["thr_bps"], float(w["thr_bps"]), places=9)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "log.csv")
            self.assertEqual(q.append_log(rows, path), 2)
            self.assertEqual(q.append_log(rows, path), 0)            # no duplicates
            r0 = q.resolve_log(path, gl)                               # not yet resolvable
            self.assertEqual((r0["resolved"], r0["pending"]), (0, 2))
            # later data arrives: resolve against the full grid
            r1 = q.resolve_log(path, self.g)
            self.assertEqual(r1["resolved"], 2)
            log = pd.read_csv(path)
            for _, row in log.iterrows():
                h = int(row.horizon)
                self.assertAlmostEqual(row.future_bps, float(self.x.loc[T, f"future_{h}"]), places=7)
                exp = wf[h].loc[T, "actual"]
                self.assertEqual(row.actual, exp)
            s = q.summarize_log(path)
            self.assertEqual(len(s), 2)

    def test_logger_marks_missing_resolution_bar(self):
        d = sorted(self.x.session.unique())[28]
        T = self.x.index[self.x.session == d][120]
        ts = pd.to_datetime(RAW.timestamp, utc=True)
        xl, _, _ = build(RAW[ts <= T.tz_convert("UTC")], live=True)
        rows, _ = q.predict_live(xl, horizons=[5])
        miss_ts = (T + pd.Timedelta(minutes=5)).tz_convert("UTC")
        holed = RAW[ts != miss_ts]
        _, gh, _ = build(holed)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "log.csv"); q.append_log(rows, path)
            r = q.resolve_log(path, gh)
            self.assertEqual((r["resolved"], r["no_bar"]), (0, 1))
            self.assertEqual(pd.read_csv(path).actual.iloc[0], "NO_BAR")
            self.assertTrue(q.summarize_log(path).empty)

    def test_live_skips_near_close_and_warmup(self):
        d = sorted(self.x.session.unique())[28]
        idx = self.x.index[self.x.session == d]
        ts = pd.to_datetime(RAW.timestamp, utc=True)
        for pos, expect_h in ((5, []), (375, [])):   # warm-up; 15 min before close
            xl, _, _ = build(RAW[ts <= idx[pos].tz_convert("UTC")], live=True)
            rows, sk = q.predict_live(xl, horizons=[30, 60])
            self.assertEqual([r["horizon"] for r in rows], expect_h); self.assertEqual(len(sk), 2)

    def test_half_day_dropped(self):
        d = sorted(self.x.session.unique())[10]
        ts = pd.to_datetime(RAW.timestamp, utc=True).dt.tz_convert(q.TZ)
        raw = RAW[~((ts.dt.date == d) & (ts.dt.time > pd.Timestamp("12:59").time()))]   # keeps 210/390 bars
        _, info = q.prepare(raw)
        self.assertEqual([x[0] for x in info["dropped_sessions"]], [d])
        self.assertEqual(info["sessions"], 40)

if __name__ == "__main__":
    unittest.main(verbosity=2)
