"""task #38/#41 · 从 taskb_ucr.jsonl 构建增强版 ClfMemory bank。

⚠ 数据泄漏修复（feedback 问题 6）：记忆里**可投票**的 per-classifier acc 一律用
  训练集内 CV 估计（部署可得），best_classifier = CV-winner。原 sweep 里的
  test-set acc 仅作为 `test_acc` / `all_clf_accs` 写入，标注为 AUDIT ONLY，
  决策代码不再读取。

用 25-dim feature + z-score normalization。输出：
  /tmp/clf_memory_v2.jsonl       (memory bank)
  /tmp/clf_memory_v2_norm.npz    (z-score mean/std, sidecar)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from research.agent.clf_memory import ClfCase, ClfMemory
from research.agent.clf_planner import loo_cv_acc, kfold_cv_acc
from research.utils.series_features import FEATURE_ORDER, featurize_cell
from research.utils.ucr_loader import load_ucr_fewshot


METHOD_TO_CLF = {
    "B1_dtw": "dtw_1nn",
    "B2_euclid": "euclid_1nn",
    "B3_rocket": "rocket",
    "B4a_moment_1nn": "moment_1nn",
    "B4b_moment_lr": "moment_logreg",
}

CV_METHOD = "kfold"   # k-fold (3 fits/cell) — LOO×MOMENT 太慢；memory 只需
                      # deployment-available 的 CV 估计，不必与 planner 决策期 CV 同法


def _cv_accs_for_cell(X_tr, y_tr) -> dict:
    """训练集内 CV 估计每个候选 classifier 的 acc（部署可得，无 test 泄漏）。"""
    out = {}
    for clf in METHOD_TO_CLF.values():
        try:
            if CV_METHOD == "kfold":
                a, _ = kfold_cv_acc(X_tr, y_tr, clf, k=3, seed=0)
            else:
                a, _ = loo_cv_acc(X_tr, y_tr, clf)
            if np.isfinite(a):
                out[clf] = float(a)
        except Exception as e:
            print(f"    cv fail {clf}: {e!r}")
    return out


def main():
    sweep_path = "research/results/taskb_ucr.jsonl"
    rows = [json.loads(l) for l in open(sweep_path)]
    by_cell = defaultdict(dict)
    for r in rows:
        if r["method"] in METHOD_TO_CLF:
            by_cell[(r["dataset"], r["N_per_class"], r["seed"])][r["method"]] = r["acc"]

    # 收集 raw 25-dim features + CV-winner per cell
    raw_feats = []
    cases_info = []
    for (ds, n, seed), accs in by_cell.items():
        try:
            X_tr, y_tr, _, _ = load_ucr_fewshot(ds, n_per_class=n, seed=seed)
            feat = featurize_cell(X_tr, y_tr)
            test_accs = {METHOD_TO_CLF[m]: float(a) for m, a in accs.items()}
            cv_accs = _cv_accs_for_cell(X_tr, y_tr)          # deployment-safe
            if not cv_accs:
                print(f"skip {ds} {n} {seed}: no finite CV acc")
                continue
            best = max(cv_accs.items(), key=lambda kv: kv[1])  # CV-winner
            raw_feats.append(feat)
            cases_info.append({
                "ds": ds, "n": n, "seed": seed,
                "best_classifier": best[0],          # CV-winner (deployment-safe)
                "cv_accs": cv_accs,                  # votable
                "test_acc": float(test_accs.get(best[0], float("nan"))),  # AUDIT
                "all_clf_accs": test_accs,           # AUDIT ONLY
            })
            print(f"  {ds:13} n={n} seed={seed}: CV-winner={best[0]} "
                  f"(cv={best[1]:.3f}, test={test_accs.get(best[0], float('nan')):.3f})")
        except Exception as e:
            print(f"skip {ds} {n} {seed}: {e!r}")

    raw_feats = np.stack(raw_feats, axis=0)  # [N_cases, D]
    print(f"Raw features: {raw_feats.shape}")

    # z-score normalization
    mean = raw_feats.mean(axis=0)
    std = raw_feats.std(axis=0) + 1e-9
    norm_feats = (raw_feats - mean) / std
    # L2 normalize for cosine similarity
    norms = np.linalg.norm(norm_feats, axis=1, keepdims=True) + 1e-9
    norm_feats = norm_feats / norms
    print(f"After z-score + L2-norm: shape={norm_feats.shape}, "
          f"sim(self) avg={(norm_feats @ norm_feats.T).diagonal().mean():.4f}")

    # 验证 — 看 BeetleFly N=3 query 后的 sim 分布是否打散
    bf_idx = next((i for i, c in enumerate(cases_info)
                   if c["ds"] == "BeetleFly" and c["n"] == 3 and c["seed"] == 1), None)
    if bf_idx is not None:
        sims = norm_feats @ norm_feats[bf_idx]
        top5_idx = np.argsort(-sims)[:5]
        print(f"\nBeetleFly N=3 seed=1 top-5 neighbors (after enhancement):")
        for j in top5_idx:
            ci = cases_info[j]
            print(f"  sim={sims[j]:.4f} {ci['ds']:13} n={ci['n']} seed={ci['seed']:3} "
                  f"best={ci['best_classifier']}")

    # 保存 norm stats
    out_dir = Path("/tmp")
    np.savez(out_dir / "clf_memory_v2_norm.npz",
             mean=mean.astype(np.float32),
             std=std.astype(np.float32))
    print(f"\nSaved norm stats → /tmp/clf_memory_v2_norm.npz")

    # 写 memory bank
    mem_path = out_dir / "clf_memory_v2.jsonl"
    if mem_path.exists():
        mem_path.unlink()
    mem = ClfMemory(mem_path, dim=norm_feats.shape[1])
    for vec, ci in zip(norm_feats, cases_info):
        mem.add(ClfCase(
            diag_feature=vec.tolist(),
            best_classifier=ci["best_classifier"],   # CV-winner
            test_acc=ci["test_acc"],                  # AUDIT ONLY
            meta={"dataset": ci["ds"], "N_per_class": ci["n"], "seed": ci["seed"]},
            all_clf_accs=ci["all_clf_accs"],          # AUDIT ONLY
            cv_accs=ci["cv_accs"],                    # deployment-safe votable
        ))
    print(f"Built memory bank → {mem_path} ({len(mem)} cases)")


if __name__ == "__main__":
    main()
