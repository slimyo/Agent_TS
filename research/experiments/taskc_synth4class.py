"""P6.4 / task #27 · 合成 4-class fault 分类 — boundary 论证 cross-check。

关键预测：当 class labels = 诊断概念（normal/trend_break/seasonal_break/outlier_burst）时，
Agent (B6) 应该击败 Rocket (B3)，反向印证 §5 boundary characterization。

实验设计:
  - base: ETTh1 / ECL series → 切 N_per_class=15 train + 30 test per class × 4 classes = 60+120
  - 注入 4 种 fault (utils/inject_fault.py)
  - 跑 B1-B6 (复用 task #26 baseline 实现)
  - 2 seeds，2 datasets = 4 settings
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
from research.utils.data_loader import load_series
from research.utils.inject_fault import build_synthetic_dataset, FAULT_LABELS

DATASETS = ["ETTh1", "ECL"]
N_PER_CLASS_TRAIN = [3, 5, 10]
N_PER_CLASS_TEST = 20  # test 集每 class 20 个
SEEDS = [1, 42]
WINDOW_LEN = 96
MAX_TEST_LLM = 20  # LLM B5/B6 总测试 cap


def main():
    out = Path("research/results/taskc_synth4class.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            try:
                r = json.loads(l)
                done.add((r["dataset"], r["N_per_class"], r["seed"], r["method"]))
            except Exception:
                pass
    print(f"resuming: {len(done)} done")

    fh = out.open("a")
    for ds in DATASETS:
        series, meta = load_series(ds)
        for n in N_PER_CLASS_TRAIN:
            for seed in SEEDS:
                # Build train + test on same base, but different rng splits
                rng_seed_train = seed * 1000
                rng_seed_test = seed * 1000 + 1
                X_tr, y_tr = build_synthetic_dataset(
                    series, window_len=WINDOW_LEN, n_per_class=n,
                    seed=rng_seed_train, season_m=meta.season_m,
                )
                X_te, y_te = build_synthetic_dataset(
                    series, window_len=WINDOW_LEN, n_per_class=N_PER_CLASS_TEST,
                    seed=rng_seed_test, season_m=meta.season_m,
                )
                # Subsample for LLM
                rng = np.random.default_rng(seed)
                idx_llm = rng.choice(len(X_te), size=min(MAX_TEST_LLM, len(X_te)),
                                     replace=False)
                X_te_llm = X_te[idx_llm]
                y_te_llm = y_te[idx_llm]
                print(f"\n=== {ds} N_per_class={n} seed={seed} | train={X_tr.shape}, test={X_te.shape} ===")
                methods = {
                    "B1_dtw":           (b1_knn_dtw, X_te, y_te),
                    "B2_euclid":        (b2_knn_euclid, X_te, y_te),
                    "B3_rocket":        (b3_rocket, X_te, y_te),
                    "B4a_moment_1nn":   (b4a_moment_1nn, X_te, y_te),
                    "B4b_moment_lr":    (b4b_moment_lr, X_te, y_te),
                    "B5_llm_direct":    (b5_llm_direct, X_te_llm, y_te_llm),
                    "B6_agent":         (lambda *a, **k: b6_agent(*a, season_m=meta.season_m, **k),
                                          X_te_llm, y_te_llm),
                }
                for name, (fn, Xt, yt) in methods.items():
                    key = (ds, n, seed, name)
                    if key in done:
                        continue
                    t0 = time.time()
                    try:
                        y_p = fn(X_tr, y_tr, Xt)
                        acc = float((y_p == yt).mean())
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
                        print(f"  {name:18}: acc={acc:.3f} f1={f1:.3f} ({wall:.1f}s)")
                    except Exception as e:
                        print(f"  FAIL {name}: {e!r}")
    fh.close()


if __name__ == "__main__":
    main()
