"""task #69 · Expanded library sweep — 8 classifiers on industrial + UCR-5.

Evaluate the 3 new baselines (minirocket / weasel / catch22) alongside existing
5 (b1-b4) across both saturated UCR-5 and industrial datasets. Identify niches
where new classifiers win that B7v3's 5-classifier router couldn't capture.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np

from research.baseline.tsc_classical import (b1_knn_dtw, b2_knn_euclid, b3_rocket,
                                              b5_minirocket, b6_weasel, b7_catch22)
from research.baseline.moment_classifier import classify_1nn as b4a, classify_logreg as b4b
from research.utils.ucr_loader import load_ucr_fewshot

DATASETS = [
    # UCR-5 saturated
    "Coffee", "BeetleFly", "BirdChicken", "ECG200", "TwoLeadECG",
    # Industrial
    "Wafer", "ECG5000", "FordA", "FordB", "Strawberry",
]
N_VALS = [5, 10]
SEEDS = [1, 42]
MAX_TEST = 200
OUT = Path("research/results/expanded_lib_sweep.jsonl")
CLASSIFIERS = {
    "rocket": b3_rocket,
    "minirocket": b5_minirocket,
    "weasel": b6_weasel,
    "catch22": b7_catch22,
    "euclid_1nn": b2_knn_euclid,
    "moment_1nn": b4a,
    "moment_logreg": b4b,
    "dtw_1nn": b1_knn_dtw,
}


def main():
    OUT.parent.mkdir(exist_ok=True, parents=True)
    done = set()
    if OUT.exists():
        for l in OUT.read_text().splitlines():
            try:
                r = json.loads(l); done.add((r["dataset"], r["N"], r["seed"], r["method"]))
            except: pass
    fh = OUT.open("a")
    for ds in DATASETS:
        for n in N_VALS:
            for seed in SEEDS:
                try:
                    X_tr, y_tr, X_te, y_te = load_ucr_fewshot(ds, n_per_class=n, seed=seed)
                except Exception as e:
                    print(f"  skip {ds} N={n} s={seed}: {e!r}"); continue
                if len(X_te) > MAX_TEST:
                    rng = np.random.default_rng(seed)
                    idx = rng.choice(len(X_te), MAX_TEST, replace=False)
                    X_te, y_te = X_te[idx], y_te[idx]
                # Skip DTW on very long industrial (>500 length) due to cost
                cands = list(CLASSIFIERS.items())
                if X_tr.shape[1] > 500:
                    cands = [(n, f) for n, f in cands if n != "dtw_1nn"]
                for name, fn in cands:
                    if (ds, n, seed, name) in done: continue
                    t0 = time.time()
                    try:
                        yp = fn(X_tr, y_tr, X_te)
                        acc = float((yp == y_te).mean())
                        wt = round(time.time() - t0, 2)
                        row = {"dataset": ds, "N": n, "seed": seed, "method": name,
                               "acc": round(acc, 4), "wall_time": wt}
                        fh.write(json.dumps(row) + "\n"); fh.flush()
                        print(f"  {ds:14} N={n:2} s={seed:2} {name:13}: acc={acc:.3f} ({wt}s)")
                    except Exception as e:
                        print(f"  FAIL {ds} N={n} s={seed} {name}: {type(e).__name__}")
    fh.close()


if __name__ == "__main__":
    main()
