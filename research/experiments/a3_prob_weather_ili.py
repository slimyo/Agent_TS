"""A3 · Weather + ILI 的 Chronos-2 prob metrics。"""
from __future__ import annotations
import json, time
from pathlib import Path
from research.baseline.chronos2 import predict as c2_predict
from research.utils.data_loader import load_series
from research.utils.metrics import mae
from research.utils.prob_metrics import prob_metrics_from_quantiles
from research.utils.splitter import few_shot_split

CFG = [("Weather", 96), ("ILI", 24)]
NS = [10, 20, 50, 100]
SEEDS = [1, 42, 123]

def main():
    out = Path("research/results/a3_prob_metrics_weather_ili.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as fh:
        for ds, H in CFG:
            series, meta = load_series(ds)
            for N in NS:
                for seed in SEEDS:
                    sp = few_shot_split(series, N=N, H=H, seed=seed)
                    t0 = time.time()
                    y = c2_predict(train=sp.train, val=sp.val, H=H, seed=seed, season_m=meta.season_m)
                    q = c2_predict.last_quantiles
                    pm = prob_metrics_from_quantiles(sp.test, q)
                    pm["mae"] = mae(sp.test, y)
                    wall = time.time() - t0
                    row = {"dataset": ds, "N": N, "seed": seed, "H": H, "method": "chronos2",
                           "wall_time": round(wall, 2), **{k: round(v, 6) for k, v in pm.items()}}
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    print(f"{ds:9} N={N:3} seed={seed:3}  MAE={pm['mae']:.4f} CRPS={pm['crps']:.4f}")

if __name__ == "__main__":
    main()
