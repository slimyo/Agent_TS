"""TaskB · UCR few-shot 分类完整 sweep。

跑 6 baselines × 多个 UCR 数据集 × N-shot:
  B1 1-NN DTW       (经典强 baseline)
  B2 1-NN Euclid    (sanity)
  B3 Rocket         (kernel SOTA)
  B4a MOMENT 1-NN   (TSFM embedding + 1-NN)
  B4b MOMENT LogReg (TSFM embedding + linear probe)
  B5 LLM-direct     (raw series → LLM)
  B6 Agent          (Curator diagnosis → LLM ICL)

Output: research/results/taskb_ucr.jsonl
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from research.baseline.tsc_classical import b1_knn_dtw, b2_knn_euclid, b3_rocket
from research.baseline.moment_classifier import classify_1nn as b4a_moment_1nn
from research.baseline.moment_classifier import classify_logreg as b4b_moment_lr
from research.agent.tsc_classifier import b5_llm_direct, b6_agent
from research.utils.ucr_loader import load_ucr_fewshot, UCR_DATASETS_RECOMMENDED

DATASETS = ["Coffee", "ECG200", "TwoLeadECG", "BeetleFly", "BirdChicken"]
# Multi-class (skipped in this initial sweep due to LLM cost):
# "ECG5000" (5-class), "Crop" (24-class)

N_PER_CLASS = [3, 5, 10]
SEEDS = [1, 42]

# 限制每数据集的 test 数（LLM B5/B6 太贵）
MAX_TEST_LLM = 20
MAX_TEST_FAST = 200  # B1-B4 跑得快，可多点


def subsample_test(X_te, y_te, n_max, seed=0):
    if len(X_te) <= n_max:
        return X_te, y_te
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_te), size=n_max, replace=False)
    return X_te[idx], y_te[idx]


def season_m_for(name: str) -> int:
    """启发式：UCR 大多无季节，给 1。"""
    return 1


def main():
    out = Path("research/results/taskb_ucr.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            try:
                r = json.loads(l)
                done.add((r["dataset"], r["N_per_class"], r["seed"], r["method"]))
            except Exception:
                pass
    print(f"resuming: {len(done)} cells done")

    fh = out.open("a")
    for ds in DATASETS:
        for n in N_PER_CLASS:
            for seed in SEEDS:
                # Load few-shot
                X_tr, y_tr, X_te, y_te = load_ucr_fewshot(ds, n_per_class=n, seed=seed)
                X_te_fast, y_te_fast = subsample_test(X_te, y_te, MAX_TEST_FAST, seed)
                X_te_llm, y_te_llm = subsample_test(X_te, y_te, MAX_TEST_LLM, seed)
                sm = season_m_for(ds)

                methods = {
                    "B1_dtw":     (b1_knn_dtw, X_te_fast, y_te_fast),
                    "B2_euclid":  (b2_knn_euclid, X_te_fast, y_te_fast),
                    "B3_rocket":  (b3_rocket, X_te_fast, y_te_fast),
                    "B4a_moment_1nn":  (b4a_moment_1nn, X_te_fast, y_te_fast),
                    "B4b_moment_lr":   (b4b_moment_lr, X_te_fast, y_te_fast),
                    "B5_llm_direct":   (b5_llm_direct, X_te_llm, y_te_llm),
                    "B6_agent":        (lambda *a, **k: b6_agent(*a, season_m=sm, **k),
                                        X_te_llm, y_te_llm),
                }

                for name, (fn, Xt, yt) in methods.items():
                    key = (ds, n, seed, name)
                    if key in done:
                        print(f"  skip {key}")
                        continue
                    t0 = time.time()
                    try:
                        y_p = fn(X_tr, y_tr, Xt)
                        acc = float((y_p == yt).mean())
                        # macro F1
                        from sklearn.metrics import f1_score
                        try:
                            f1 = float(f1_score(yt, y_p, average="macro"))
                        except Exception:
                            f1 = 0.0
                        wall = time.time() - t0
                        row = {"dataset": ds, "N_per_class": n, "seed": seed,
                               "method": name, "n_test": len(yt),
                               "acc": round(acc, 4), "macro_f1": round(f1, 4),
                               "wall_time": round(wall, 2)}
                        fh.write(json.dumps(row) + "\n")
                        fh.flush()
                        print(f"  {ds:13} n={n} seed={seed} {name:18}  "
                              f"acc={acc:.3f} f1={f1:.3f} ({wall:.1f}s, n_test={len(yt)})")
                    except Exception as e:
                        print(f"  FAIL {key}: {e!r}")

    fh.close()


if __name__ == "__main__":
    main()
