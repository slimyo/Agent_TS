"""task #63 / P1-I · Industrial case study on UCR industrial / medical / fault subsets.

数据集（已下载）：
  - Wafer (semiconductor manufacturing, 1000 train, binary fault)
  - ECG5000 (medical, 500 train, 5-class)
  - FordA (engine fault diagnostics, 3601 train, binary)
  - FordB (engine fault, 3636 train, binary)
  - Strawberry (spectroscopy, 613 train, binary)

跑 B1/B2/B3/B4 baselines + B7v3 Router on N-shot subsets，build industrial deployment table.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from research.baseline.tsc_classical import b1_knn_dtw, b2_knn_euclid, b3_rocket
from research.baseline.moment_classifier import classify_1nn as b4a, classify_logreg as b4b
from research.agent.clf_planner import classification_planner
from research.utils.ucr_loader import load_ucr_fewshot

INDUSTRIAL_DATASETS = ["Wafer", "ECG5000", "FordA", "FordB", "Strawberry"]
N_PER_CLASS = [5, 10]   # N=3 太少
SEEDS = [1, 42]
MAX_TEST = 200

MEMORY_PATH = "/tmp/clf_memory_v2.jsonl"


def subsample(X, y, n_max, seed=0):
    if len(X) <= n_max: return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), n_max, replace=False)
    return X[idx], y[idx]


def main():
    out = Path("research/results/industrial_case.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            try:
                r = json.loads(l)
                done.add((r["dataset"], r["N_per_class"], r["seed"], r["method"]))
            except Exception: pass
    fh = out.open("a")
    for ds in INDUSTRIAL_DATASETS:
        for n in N_PER_CLASS:
            for seed in SEEDS:
                try:
                    X_tr, y_tr, X_te, y_te = load_ucr_fewshot(ds, n_per_class=n, seed=seed)
                    X_te, y_te = subsample(X_te, y_te, MAX_TEST, seed)
                except Exception as e:
                    print(f"  skip {ds} {n} {seed}: {e!r}")
                    continue
                if X_tr.shape[1] > 1000:
                    methods = {"B2_euclid": b2_knn_euclid, "B3_rocket": b3_rocket,
                                "B4a_moment_1nn": b4a, "B4b_moment_lr": b4b}
                else:
                    methods = {"B1_dtw": b1_knn_dtw, "B2_euclid": b2_knn_euclid,
                                "B3_rocket": b3_rocket,
                                "B4a_moment_1nn": b4a, "B4b_moment_lr": b4b}
                for name, fn in methods.items():
                    key = (ds, n, seed, name)
                    if key in done: continue
                    t0 = time.time()
                    try:
                        yp = fn(X_tr, y_tr, X_te)
                        acc = float((yp == y_te).mean())
                        wt = time.time() - t0
                        row = {"dataset": ds, "N_per_class": n, "seed": seed,
                                "method": name, "acc": round(acc, 4),
                                "wall_time": round(wt, 2)}
                        fh.write(json.dumps(row) + "\n"); fh.flush()
                        print(f"  {ds:14} n={n} seed={seed} {name:18}: acc={acc:.3f} ({wt:.1f}s)")
                    except Exception as e:
                        print(f"  FAIL {ds} {n} {seed} {name}: {type(e).__name__}")
                # B7v3 router
                key = (ds, n, seed, "B7v3_router")
                if key not in done:
                    t0 = time.time()
                    try:
                        _, yp, trace = classification_planner(
                            X_tr, y_tr, X_te, season_m=1,
                            use_cv=True, cv_method="loo",
                            margin=0.10, default_classifier="rocket",
                            n_min_for_routing=7,
                            use_memory=True, memory_path=MEMORY_PATH,
                            use_enhanced_features=True, weighted_vote_min_ratio=0.55,
                        )
                        acc = float((yp == y_te).mean())
                        wt = time.time() - t0
                        row = {"dataset": ds, "N_per_class": n, "seed": seed,
                                "method": "B7v3_router", "acc": round(acc, 4),
                                "chosen": trace.chosen, "wall_time": round(wt, 2)}
                        fh.write(json.dumps(row) + "\n"); fh.flush()
                        print(f"  {ds:14} n={n} seed={seed} B7v3 chose '{trace.chosen}': acc={acc:.3f} ({wt:.1f}s)")
                    except Exception as e:
                        print(f"  FAIL B7v3 {ds} {n} {seed}: {type(e).__name__}")
    fh.close()


if __name__ == "__main__":
    main()
