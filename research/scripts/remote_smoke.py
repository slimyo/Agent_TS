"""Smoke test Timer-S1 / Time-MoE / Sundial on remote GPU.

Each model: load + 1 forecast on synthetic sine; print wall time + output shape.
"""
from __future__ import annotations
import json, time, traceback
from pathlib import Path
import numpy as np

OUT = Path("research/results/remote_smoke.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(0)
SIG = (np.sin(np.arange(400) * 0.1) + 0.1 * rng.standard_normal(400)).astype(np.float32)
H = 24

def run(name, mod_path):
    print(f"\n=== {name} ===")
    t0 = time.time()
    rec = {"model": name, "ok": False}
    try:
        from importlib import import_module
        mod = import_module(mod_path)
        t1 = time.time()
        out = mod.predict(SIG, np.array([], dtype=np.float32), H=H, seed=1)
        t2 = time.time()
        rec.update(ok=True, load_s=round(t1-t0, 2), infer_s=round(t2-t1, 2),
                   out_shape=list(out.shape), out_tail=[float(x) for x in out[-3:]])
        print(f"  load {t1-t0:.1f}s | infer {t2-t1:.2f}s | shape {out.shape} | tail {out[-3:]}")
    except Exception as e:
        rec.update(error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
        print(f"  FAIL: {type(e).__name__}: {e}")
    with OUT.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


if __name__ == "__main__":
    OUT.write_text("")  # reset
    for name, path in [
        ("Time-MoE-50M", "research.baseline.time_moe"),
        ("Sundial-128m", "research.baseline.sundial"),
        ("Timer-S1",     "research.baseline.timer"),
    ]:
        run(name, path)
    print("\nDone. Results in", OUT)
