"""task #69 follow-up · Toto/TiRex/C2 three-way head-to-head on cached cells.

Uses gated_residual_cells.jsonl (34 cells, 6 datasets). Toto needs longer context
than the cached 200 points (it expects ≥patch_size). We use the historical 200 if
sufficient; else pad / skip.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np

from research.baseline.toto import predict as toto_predict


def main():
    cells = [json.loads(l) for l in open("research/results/gated_residual_cells.jsonl")]
    print(f"Toto on {len(cells)} cells (Toto Open Base 1.0)...\n")
    out = Path("research/results/toto_vs_c2.jsonl")
    fh = out.open("w")
    for c in cells:
        hist = np.array(c["history"])
        y_true = np.array(c["y_true"])
        c2_pred = np.array(c["c2_pred"])
        H = len(y_true)
        # Toto needs reasonably long context (it pads short). Use cached 200 directly.
        try:
            t0 = time.time()
            toto_pred = toto_predict(hist, np.array([]), H=H, seed=c["seed"])
            dt = time.time() - t0
        except Exception as e:
            print(f"  FAIL {c['dataset']} N={c['N']} s={c['seed']}: {type(e).__name__}")
            continue
        mae_c2 = float(np.mean(np.abs(y_true - c2_pred)))
        mae_to = float(np.mean(np.abs(y_true - toto_pred)))
        row = {"dataset": c["dataset"], "N": c["N"], "seed": c["seed"],
               "mae_c2": round(mae_c2, 6), "mae_toto": round(mae_to, 6),
               "delta": round(mae_to - mae_c2, 6), "wall": round(dt, 2)}
        fh.write(json.dumps(row) + "\n"); fh.flush()
        win = "toto" if mae_to < mae_c2 else "c2"
        print(f"  {c['dataset']:10} N={c['N']:3} s={c['seed']:2}: "
              f"C2={mae_c2:.4f} Toto={mae_to:.4f} Δ={mae_to-mae_c2:+.4f} {win}")
    fh.close()

    # Combine with TiRex results
    print(f"\n=== 3-way analysis ===")
    rows = [json.loads(l) for l in out.open()]
    tirex = {(r["dataset"], r["N"], r["seed"]): r["mae_tirex"]
             for r in (json.loads(l) for l in open("research/results/tirex_vs_c2.jsonl"))}
    c2 = np.mean([r['mae_c2'] for r in rows])
    to = np.mean([r['mae_toto'] for r in rows])
    print(f"  C2 alone:    {c2:.4f}")
    print(f"  Toto alone:  {to:.4f}")
    # Oracle Toto vs C2
    oracle_2 = np.mean([min(r['mae_c2'], r['mae_toto']) for r in rows])
    # Oracle C2/TiRex/Toto
    triplet = []
    for r in rows:
        k = (r["dataset"], r["N"], r["seed"])
        tx = tirex.get(k, r["mae_c2"])
        triplet.append(min(r["mae_c2"], r["mae_toto"], tx))
    oracle_3 = np.mean(triplet)
    print(f"  Oracle C2/Toto: {oracle_2:.4f} (gain vs C2: {(c2-oracle_2)/c2*100:+.2f}%)")
    print(f"  Oracle C2/Toto/TiRex: {oracle_3:.4f} (gain vs C2: {(c2-oracle_3)/c2*100:+.2f}%)")
    # Per-dataset
    from collections import defaultdict
    by = defaultdict(list)
    for r in rows: by[r['dataset']].append(r)
    print(f"\n{'dataset':10} {'C2':>10} {'TiRex':>10} {'Toto':>10} {'Best':>10}")
    for ds, rs in by.items():
        c2_ = np.mean([r['mae_c2'] for r in rs])
        to_ = np.mean([r['mae_toto'] for r in rs])
        tx_ = np.mean([tirex.get((r['dataset'],r['N'],r['seed']), r['mae_c2']) for r in rs])
        b = min(c2_, to_, tx_)
        print(f"{ds:10} {c2_:>10.4f} {tx_:>10.4f} {to_:>10.4f} {b:>10.4f}")


if __name__ == "__main__":
    main()
