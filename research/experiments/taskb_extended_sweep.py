"""task #42 / P6.4c · Less-saturated UCR 5 datasets 扩展 sweep。

验证 B7v3 在 multi-class + industrial + spectroscopy + 较大 train pool 数据上是否仍领先 Rocket。

数据集（按 saturation 评估）：
  - ECG5000      500×4500   5-class    医学，多类（少饱和）
  - Crop         7200×16800 24-class   遥感，超多类（极少饱和）
  - Wafer        1000×6164  2-class    工业故障（极不平衡）
  - Strawberry   613×370    2-class    光谱（TSFM 训练外）
  - GunPoint     50×150     2-class    motion（已下载未跑）

跑 B1-B6 + B7v3，N_per_class ∈ {3, 5, 10}，2 seeds。
24-class Crop 需 N_per_class ≥ 5 才有意义，故 Crop 跳过 N=3。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from research.agent.clf_planner import classification_planner
from research.agent.tsc_classifier import b5_llm_direct
from research.baseline.moment_classifier import classify_1nn as b4a_moment_1nn
from research.baseline.moment_classifier import classify_logreg as b4b_moment_lr
from research.baseline.tsc_classical import b1_knn_dtw, b2_knn_euclid, b3_rocket
from research.utils.ucr_loader import load_ucr_fewshot

DATASETS = ["GunPoint", "Strawberry", "Wafer", "ECG5000", "Crop"]
N_PER_CLASS = {
    "GunPoint": [3, 5, 10],
    "Strawberry": [3, 5, 10],
    "Wafer": [3, 5, 10],
    "ECG5000": [5, 10],         # 5-class，太少不可行
    "Crop": [5, 10],            # 24-class，N=3 不可行
}
SEEDS = [1, 42]
MAX_TEST_FAST = 200   # B1-B4 + B7v3（包含 Rocket）
MAX_TEST_LLM = 20     # B5 LLM 太贵

MEMORY_PATH = "/tmp/clf_memory_v2.jsonl"  # 复用 task #41 增强 memory


def subsample_test(X, y, n_max, seed=0):
    if len(X) <= n_max: return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=n_max, replace=False)
    return X[idx], y[idx]


def main():
    out = Path("research/results/taskb_extended_ucr.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            try:
                r = json.loads(l)
                done.add((r["dataset"], r["N_per_class"], r["seed"], r["method"]))
            except Exception: pass

    fh = out.open("a")
    for ds in DATASETS:
        for n in N_PER_CLASS[ds]:
            for seed in SEEDS:
                try:
                    X_tr, y_tr, X_te, y_te = load_ucr_fewshot(ds, n_per_class=n, seed=seed)
                except Exception as e:
                    print(f"  skip {ds} {n} {seed}: load failed {e!r}")
                    continue
                X_te_fast, y_te_fast = subsample_test(X_te, y_te, MAX_TEST_FAST, seed)
                X_te_llm, y_te_llm = subsample_test(X_te, y_te, MAX_TEST_LLM, seed)

                methods = {
                    "B1_dtw":          (b1_knn_dtw, X_te_fast, y_te_fast),
                    "B2_euclid":       (b2_knn_euclid, X_te_fast, y_te_fast),
                    "B3_rocket":       (b3_rocket, X_te_fast, y_te_fast),
                    "B4a_moment_1nn":  (b4a_moment_1nn, X_te_fast, y_te_fast),
                    "B4b_moment_lr":   (b4b_moment_lr, X_te_fast, y_te_fast),
                    "B5_llm_direct":   (b5_llm_direct, X_te_llm, y_te_llm),
                    "B7v3_router":     ("router", X_te_fast, y_te_fast),
                }
                for name, (fn_or_tag, Xt, yt) in methods.items():
                    key = (ds, n, seed, name)
                    if key in done: continue
                    t0 = time.time()
                    try:
                        if fn_or_tag == "router":
                            _, y_p, trace = classification_planner(
                                X_tr, y_tr, Xt, season_m=1,
                                use_cv=True, cv_method="loo",
                                margin=0.10, default_classifier="rocket",
                                n_min_for_routing=7,
                                use_memory=True,
                                memory_path=MEMORY_PATH,
                                use_enhanced_features=True,
                                weighted_vote_min_ratio=0.55,
                            )
                            chosen = trace.chosen
                        else:
                            y_p = fn_or_tag(X_tr, y_tr, Xt)
                            chosen = name
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
                               "chosen_classifier": chosen, "wall_time": round(wall, 2)}
                        fh.write(json.dumps(row) + "\n"); fh.flush()
                        print(f"  {ds:12} n={n} seed={seed} {name:18}  "
                              f"acc={acc:.3f} f1={f1:.3f} ({wall:.1f}s)")
                    except Exception as e:
                        print(f"  FAIL {key}: {e!r}")
    fh.close()


if __name__ == "__main__":
    main()
