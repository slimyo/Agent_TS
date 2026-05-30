"""task #66 · B7v4 industrial sweep with industrial-signature override.

Same 5 industrial datasets × 2 N × 2 seeds. Compare:
  - B7v3 (baseline, 25-dim features, no industrial signature)
  - B7v4 (30-dim features + industrial signature → Euclid prior)
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np

from research.agent.clf_planner import classification_planner
from research.utils.ucr_loader import load_ucr_fewshot

INDUSTRIAL_DATASETS = ["Wafer", "ECG5000", "FordA", "FordB", "Strawberry"]
N_PER_CLASS = [5, 10]
SEEDS = [1, 42]
MAX_TEST = 200
MEMORY_PATH_V4 = "/tmp/clf_memory_v4_industrial.jsonl"


def subsample(X, y, n_max, seed=0):
    if len(X) <= n_max: return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), n_max, replace=False)
    return X[idx], y[idx]


def main():
    out = Path("research/results/industrial_b7v4.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Fresh memory file
    if Path(MEMORY_PATH_V4).exists(): Path(MEMORY_PATH_V4).unlink()
    fh = out.open("w")
    for ds in INDUSTRIAL_DATASETS:
        for n in N_PER_CLASS:
            for seed in SEEDS:
                try:
                    X_tr, y_tr, X_te, y_te = load_ucr_fewshot(ds, n_per_class=n, seed=seed)
                    X_te, y_te = subsample(X_te, y_te, MAX_TEST, seed)
                except Exception as e:
                    print(f"  skip {ds} {n} {seed}: {e!r}")
                    continue
                t0 = time.time()
                _, yp, trace = classification_planner(
                    X_tr, y_tr, X_te, season_m=1,
                    use_cv=True, cv_method="loo",
                    margin=0.15, default_classifier="rocket",
                    n_min_for_routing=7,
                    use_memory=True, memory_path=MEMORY_PATH_V4,
                    use_enhanced_features=True, weighted_vote_min_ratio=0.55,
                    use_industrial_signature=True,
                )
                acc = float((yp == y_te).mean())
                wt = time.time() - t0
                row = {"dataset": ds, "N_per_class": n, "seed": seed,
                       "method": "B7v4_router", "acc": round(acc, 4),
                       "chosen": trace.chosen, "reason": trace.chosen_reason,
                       "wall_time": round(wt, 2)}
                fh.write(json.dumps(row) + "\n"); fh.flush()
                print(f"  {ds:11} n={n} s={seed} chose '{trace.chosen}': acc={acc:.3f} ({wt:.1f}s)")
    fh.close()


if __name__ == "__main__":
    main()
