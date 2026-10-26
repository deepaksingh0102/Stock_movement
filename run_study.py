"""Run the walk-forward study on a CSV and print the results.  Usage: python run_study.py data.csv [k]"""
import sys, warnings
import pandas as pd
import qqq_core as q
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)

def main(path, k=q.DEFAULT_K, min_train=15, refit=5):
    grid, info = q.prepare(pd.read_csv(path))
    print("DATA", info)
    x = q.make_features(grid, k=k)
    res = q.run_walk_forward(x, min_train_days=min_train, refit_every=refit)
    pooled = pd.DataFrame([{"Horizon": f"{h} min", **q.pooled_metrics(z)} for h, z in res.items()])
    print("\nPOOLED (rows overlap; descriptive only)")
    print(pooled.round(4).to_string(index=False))
    for h, z in res.items():
        d = q.daily_metrics(z)
        print(f"\nPER-DAY SUMMARY  horizon={h}  ({len(d)} test sessions)")
        print(q.summarize_days(d).round(4).to_string(index=False))
    return x, res

if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else q.DEFAULT_K)
