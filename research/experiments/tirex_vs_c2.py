"""task #69 · TiRex vs Chronos-2 head-to-head on existing forecast cells.

Uses cached ground-truth from gated_residual_cells.jsonl (34 cells, 6 datasets).
For each cell, run TiRex on the same history → compare MAE vs cached C2 MAE.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from research.baseline.tirex import predict as tirex_predict


def main():
    cells_path = Path("research/results/gated_residual_cells.jsonl")
    if not cells_path.exists():
        print("Need gated_residual_cells.jsonl first."); return
    out = Path("research/results/tirex_vs_c2.jsonl")
    cells = [json.loads(l) for l in cells_path.open()]
    print(f"Running TiRex on {len(cells)} cached cells...\n")
    fh = out.open("w")
    for c in cells:
        history = np.array(c["history"])  # capped at 200 in cache
        y_true = np.array(c["y_true"])
        c2_pred = np.array(c["c2_pred"])
        H = len(y_true)
        try:
            tirex_pred = tirex_predict(history, np.array([]), H=H, seed=c["seed"])
        except Exception as e:
            print(f"  FAIL {c['dataset']} N={c['N']} s={c['seed']}: {type(e).__name__}: {e}")
            continue
        mae_c2 = float(np.mean(np.abs(y_true - c2_pred)))
        mae_tirex = float(np.mean(np.abs(y_true - tirex_pred)))
        row = {"dataset": c["dataset"], "N": c["N"], "seed": c["seed"],
               "mae_c2": round(mae_c2, 6), "mae_tirex": round(mae_tirex, 6),
               "delta": round(mae_tirex - mae_c2, 6),
               "winner": "tirex" if mae_tirex < mae_c2 else ("c2" if mae_c2 < mae_tirex else "tie")}
        fh.write(json.dumps(row) + "\n"); fh.flush()
        print(f"  {c['dataset']:10} N={c['N']:3} s={c['seed']:2}: "
              f"C2={mae_c2:.4f} TiRex={mae_tirex:.4f} Δ={row['delta']:+.4f} {row['winner']}")
    fh.close()
    # Aggregate
    rows = [json.loads(l) for l in out.open()]
    from collections import Counter
    print(f"\n=== Aggregate ({len(rows)} cells) ===")
    print(f"  C2 mean MAE: {np.mean([r['mae_c2'] for r in rows]):.4f}")
    print(f"  TiRex mean MAE: {np.mean([r['mae_tirex'] for r in rows]):.4f}")
    print(f"  Winners: {dict(Counter(r['winner'] for r in rows))}")
    # Per-dataset
    from collections import defaultdict
    by = defaultdict(list)
    for r in rows: by[r["dataset"]].append(r)
    print(f"\n{'dataset':10} {'C2':>10} {'TiRex':>10} {'Δ rel':>8} {'TiRex W/T/L':>14}")
    for ds, rs in by.items():
        c2 = np.mean([r['mae_c2'] for r in rs])
        tx = np.mean([r['mae_tirex'] for r in rs])
        w = sum(r['winner']=='tirex' for r in rs)
        l = sum(r['winner']=='c2' for r in rs)
        t = len(rs) - w - l
        print(f"{ds:10} {c2:>10.4f} {tx:>10.4f} {(tx-c2)/c2*100:>+7.2f}% {w}/{t}/{l}")


if __name__ == "__main__":
    main()
