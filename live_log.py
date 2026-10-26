"""Headless live-forecast logger. Run once per bar (or every N minutes) during market hours, e.g. via cron:

    */5 9-16 * * 1-5  cd /path/to/repo && ALPACA_API_KEY=... ALPACA_SECRET_KEY=... python live_log.py

Each run: (1) fetches recent 1-minute bars, (2) resolves any logged forecasts whose resolution minute has
arrived, (3) trains on completed sessions and appends forecasts for the latest bar, (4) prints the scorecard.
A (bar_time, horizon) pair is never logged twice, so overlapping runs are safe.
"""
import argparse, datetime as dt, os, sys
import pandas as pd
import qqq_core as q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="live_predictions.csv")
    ap.add_argument("--csv", help="Use this CSV instead of Alpaca (must include the latest bar)")
    ap.add_argument("--feed", default="iex", choices=["iex", "sip"])
    ap.add_argument("--history-days", type=int, default=120, help="Calendar days of history to fetch")
    ap.add_argument("--k", type=float, default=q.DEFAULT_K)
    ap.add_argument("--min-train", type=int, default=15)
    ap.add_argument("--dry-run", action="store_true", help="Print forecasts without writing the log")
    a = ap.parse_args()

    if a.csv:
        raw = q.load_csv(a.csv)
    else:
        key, sec = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
        if not key or not sec:
            sys.exit("Set ALPACA_API_KEY and ALPACA_SECRET_KEY")
        raw = q.fetch_alpaca(dt.date.today() - dt.timedelta(days=a.history_days), None, a.feed, key, sec)
    grid, info = q.prepare(raw.reset_index(), live=True)
    print(f"latest bar {grid.close.dropna().index.max()}  sessions={info['sessions']}  missing={info['missing_bars']}")
    if not a.dry_run:
        print("resolved:", q.resolve_log(a.log, grid))
    x = q.make_features(grid, k=a.k)
    rows, skipped = q.predict_live(x, k=a.k, min_train_days=a.min_train)
    for r in rows:
        print(f"  {r['horizon']:>2}m  {r['prediction']:<8} conf={r['confidence']:.2f}  "
              f"p(dn/side/up)={r['p_down']:.2f}/{r['p_sideways']:.2f}/{r['p_up']:.2f}  thr=±{r['thr_bps']:.1f}bps")
    for h, why in skipped:
        print(f"  {h:>2}m  skipped: {why}")
    if not a.dry_run:
        print("logged:", q.append_log(rows, a.log))
        sc = q.summarize_log(a.log)
        if not sc.empty:
            print(sc.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
