"""task #67 · Gated Residual experiment.

1) Generate cells: 5 datasets × 3 N × 2 seeds, run C2, save (features, C2_pred, y_true)
2) Train gated residual via LODO CV
3) Report mean MAE vs C2-alone, count helped/hurt cells
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np

from research.baseline.chronos2 import predict as c2_predict
from research.agent.gated_residual import featurize_history, evaluate_lodo
from research.utils.data_loader import load_series

DATASETS = ["ETTh1", "ETTh2", "ECL", "Exchange", "Weather", "ILI"]
N_VALS = [10, 20, 40]
H = 96
SEEDS = [1, 42]
CACHE = Path("research/results/gated_residual_cells.jsonl")


def build_cells():
    """Generate C2 prediction cells. Cache to JSONL."""
    cached = {}
    if CACHE.exists():
        for l in CACHE.open():
            r = json.loads(l)
            cached[(r["dataset"], r["N"], r["seed"])] = r
    fh = CACHE.open("a")
    cells = list(cached.values())
    for ds in DATASETS:
        try:
            series, meta = load_series(ds)
        except Exception as e:
            print(f"  skip {ds}: {e!r}"); continue
        L = len(series)
        # Use last 50% for test, sample distinct windows
        start_min = L // 2
        for N in N_VALS:
            for seed in SEEDS:
                key = (ds, N, seed)
                if key in cached: continue
                rng = np.random.default_rng(seed)
                # start_idx: history len = max(N*24, 256), need start_idx + H <= L
                ctx_len = max(N * 24, 256)
                max_start = L - H - 1
                min_start = max(start_min, ctx_len + 1)
                if max_start <= min_start:
                    print(f"  skip {ds} N={N} s={seed}: not enough length")
                    continue
                start_idx = int(rng.integers(min_start, max_start))
                history = series[start_idx - ctx_len: start_idx]
                y_true = series[start_idx: start_idx + H]
                t0 = time.time()
                try:
                    c2_pred = c2_predict(history, np.array([]), H=H, seed=seed)
                except Exception as e:
                    print(f"  FAIL {ds} N={N} s={seed}: {e!r}"); continue
                feats = featurize_history(history, c2_pred)
                row = {"dataset": ds, "N": N, "seed": seed,
                       "history": history.tolist()[-200:],
                       "c2_pred": c2_pred.tolist(),
                       "y_true": y_true.tolist(),
                       "features": feats.tolist(),
                       "wall_time": round(time.time() - t0, 2)}
                fh.write(json.dumps(row) + "\n"); fh.flush()
                mae = float(np.mean(np.abs(y_true - c2_pred)))
                print(f"  {ds:10} N={N:3} s={seed:2}: C2 MAE={mae:.4f} ({row['wall_time']}s)")
                cells.append(row)
    fh.close()
    return cells


def main():
    cells = build_cells()
    print(f"\n=== Total cells: {len(cells)} ===")
    if len(cells) < 12:
        print("Not enough cells for LODO. Aborting."); return
    # Run LODO at multiple tau
    for tau in [0.3, 0.5, 0.7, 0.9]:
        print(f"\n--- tau={tau} ---")
        results = evaluate_lodo([{**c, "features": np.array(c["features"]),
                                  "c2_pred": np.array(c["c2_pred"]),
                                  "y_true": np.array(c["y_true"]),
                                  "history": np.array(c.get("history", []))} for c in cells],
                                 tau=tau)
        print(f"  C2 mean MAE:  {results['mean_c2_mae']:.4f}")
        print(f"  GR mean MAE:  {results['mean_gr_mae']:.4f}  (Δ={results['mean_gr_mae']-results['mean_c2_mae']:+.4f})")
        print(f"  helped/hurt/tied: {results['n_helped']}/{results['n_hurt']}/{results['n_tied']} of {results['n_total']}")
        # Save
        out = Path(f"research/results/gated_residual_lodo_tau{int(tau*10)}.json")
        with out.open("w") as fh: json.dump(results, fh, default=float, indent=2)


if __name__ == "__main__":
    main()
