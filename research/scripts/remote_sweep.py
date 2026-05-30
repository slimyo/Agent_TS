"""Remote sweep: run one of {timer, time_moe, sundial} on cached gated_residual cells.

Usage:
  python remote_sweep.py timer
  python remote_sweep.py time_moe
  python remote_sweep.py sundial

Reads research/results/gated_residual_cells.jsonl (history + y_true + c2_pred).
Writes research/results/<model>_vs_c2.jsonl.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np


MODEL_MAP = {
    "timer":    ("research.baseline.timer",    "Timer-S1"),
    "time_moe": ("research.baseline.time_moe", "Time-MoE-50M"),
    "sundial":  ("research.baseline.sundial",  "Sundial-128m"),
}


def main(model_key: str):
    mod_path, model_name = MODEL_MAP[model_key]
    mod = __import__(mod_path, fromlist=["predict"])
    cells = [json.loads(l) for l in open("research/results/gated_residual_cells.jsonl")]
    out = Path(f"research/results/{model_key}_vs_c2.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    # resume support
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            try:
                r = json.loads(l)
                done.add((r["dataset"], r["N"], r["seed"]))
            except Exception: pass

    fh = out.open("a")
    print(f"=== {model_name} on {len(cells)} cells (skipping {len(done)} done) ===")
    t_total = time.time()
    for c in cells:
        key = (c["dataset"], c["N"], c["seed"])
        if key in done: continue
        hist = np.array(c["history"], dtype=np.float32)
        y_true = np.array(c["y_true"], dtype=np.float32)
        c2_pred = np.array(c["c2_pred"], dtype=np.float32)
        H = len(y_true)
        try:
            t0 = time.time()
            y_p = mod.predict(hist, np.array([], dtype=np.float32), H=H, seed=c["seed"])
            dt = time.time() - t0
        except Exception as e:
            print(f"  FAIL {c['dataset']} N={c['N']} s={c['seed']}: {type(e).__name__}: {e}")
            continue
        mae_c2 = float(np.mean(np.abs(y_true - c2_pred)))
        mae_md = float(np.mean(np.abs(y_true - y_p)))
        row = {"dataset": c["dataset"], "N": c["N"], "seed": c["seed"],
               "mae_c2": round(mae_c2, 6),
               f"mae_{model_key}": round(mae_md, 6),
               "delta": round(mae_md - mae_c2, 6),
               "wall": round(dt, 2)}
        fh.write(json.dumps(row) + "\n"); fh.flush()
        win = model_key if mae_md < mae_c2 else "c2"
        print(f"  {c['dataset']:10} N={c['N']:3} s={c['seed']:2}: "
              f"C2={mae_c2:.4f} {model_key}={mae_md:.4f} Δ={mae_md-mae_c2:+.4f} -> {win}  ({dt:.1f}s)")
    fh.close()
    print(f"=== done in {time.time()-t_total:.1f}s ===")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in MODEL_MAP:
        print(f"usage: {sys.argv[0]} {{{'|'.join(MODEL_MAP)}}}")
        sys.exit(1)
    main(sys.argv[1])
