"""task #50 / P0-2 Level 1 · Learned Margin · 替换 margin=0.10 常量。

设计：
  - 对每个 (dataset, N, seed) cell：
    - rocket_acc = test acc of rocket
    - best_acc = max test acc over all classifiers
    - optimal_margin* = best_acc - rocket_acc （正值 = 应偏离；负/零 = 应保 rocket）
  - 用 Curator 25-dim features 训回归头预测 optimal_margin
  - 集成 LOO CV：若 best_other_cv_acc - rocket_cv_acc > predicted_margin → 偏离

对比 B7v3 固定 margin=0.10。
"""
from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from research.utils.series_features import featurize_cell
from research.utils.ucr_loader import load_ucr_fewshot

METHOD_TO_CLF = {
    "B1_dtw": "dtw_1nn", "B2_euclid": "euclid_1nn",
    "B3_rocket": "rocket", "B4a_moment_1nn": "moment_1nn",
    "B4b_moment_lr": "moment_logreg",
}


def build_margin_training_set():
    """(features, optimal_margin) 对。"""
    sweep_files = [
        "research/results/taskb_ucr.jsonl",
        "research/results/taskb_extended_ucr.jsonl",
    ]
    by_cell = defaultdict(dict)
    for fp in sweep_files:
        if not Path(fp).exists(): continue
        for line in open(fp):
            r = json.loads(line)
            if r["method"] not in METHOD_TO_CLF: continue
            by_cell[(r["dataset"], r["N_per_class"], r["seed"])][METHOD_TO_CLF[r["method"]]] = r["acc"]

    X, y, info = [], [], []
    for (ds, n, seed), accs in by_cell.items():
        if len(accs) < 3 or "rocket" not in accs: continue
        try:
            X_tr, y_tr, _, _ = load_ucr_fewshot(ds, n_per_class=n, seed=seed)
        except Exception: continue
        feat = featurize_cell(X_tr, y_tr)
        rocket = accs["rocket"]
        best_other = max((a for c, a in accs.items() if c != "rocket"), default=rocket)
        margin = max(0.0, best_other - rocket)  # 应偏离的"门槛"
        X.append(feat); y.append(margin)
        info.append({"ds": ds, "N": n, "seed": seed,
                      "optimal_margin": margin, "rocket_acc": rocket,
                      "best_other_acc": best_other, "all_accs": accs})
    return np.stack(X), np.array(y, dtype=np.float32), info


def train_margin_head():
    X, y, info = build_margin_training_set()
    print(f"Training set: X={X.shape}, y={y.shape}")
    print(f"  optimal_margin distribution: mean={y.mean():.3f}, std={y.std():.3f}, "
          f"max={y.max():.3f}, frac_zero={(y==0).mean():.2%}")

    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor

    scaler = StandardScaler().fit(X)
    X_z = scaler.transform(X)

    # LODO CV
    datasets = sorted(set(i["ds"] for i in info))
    print(f"\nLODO CV across {len(datasets)} datasets:")
    learned_sum = 0.0
    fixed_sum = 0.0  # B7v3 margin=0.10
    rocket_sum = 0.0
    oracle_sum = 0.0
    for held in datasets:
        train_mask = np.array([i["ds"] != held for i in info])
        test_mask = ~train_mask
        if test_mask.sum() == 0: continue
        reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=0)
        reg.fit(X_z[train_mask], y[train_mask])
        margin_pred = reg.predict(X_z[test_mask])
        test_info = [info[i] for i, m in enumerate(test_mask) if m]
        for ci_info, m_pred in zip(test_info, margin_pred):
            rocket_acc = ci_info["rocket_acc"]
            best_other_acc = ci_info["best_other_acc"]
            actual_gap = best_other_acc - rocket_acc

            # Learned margin: 若 actual_gap > predicted_margin → 偏离 (picked best_other)
            # 注：现实中 actual_gap 不可知（test acc 未知），需用 CV 估计代替
            # 这里用 oracle (actual_gap) 测 head 的 sensitivity
            picked = best_other_acc if actual_gap > m_pred else rocket_acc
            learned_sum += picked

            # B7v3 fixed margin=0.10
            picked_fixed = best_other_acc if actual_gap > 0.10 else rocket_acc
            fixed_sum += picked_fixed

            rocket_sum += rocket_acc
            oracle_sum += max(ci_info["all_accs"].values())

    n_total = len(info)
    print(f"\n=== LODO CV aggregate ({n_total} cells, oracle-aware margin) ===")
    print(f"  Learned margin (predict gap):  {learned_sum/n_total:.4f}")
    print(f"  Fixed margin=0.10 (B7v3):      {fixed_sum/n_total:.4f}")
    print(f"  Rocket-alone:                  {rocket_sum/n_total:.4f}")
    print(f"  Oracle:                        {oracle_sum/n_total:.4f}")
    print(f"  Learned vs Fixed:              {(learned_sum-fixed_sum)/n_total*100:+.2f}pp")
    print(f"  Learned vs Rocket:             {(learned_sum-rocket_sum)/n_total*100:+.2f}pp")

    # Train final on all
    reg_final = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=0)
    reg_final.fit(X_z, y)
    with open("research/results/learned_margin.pkl", "wb") as f:
        pickle.dump({"reg": reg_final, "scaler": scaler}, f)
    print(f"\nSaved → research/results/learned_margin.pkl")

    # Sample predictions
    print(f"\nSample predictions:")
    print(f'  {"ds":13} N seed | true_gap  pred_margin  ok?')
    for i in range(min(15, n_total)):
        true_gap = info[i]["best_other_acc"] - info[i]["rocket_acc"]
        m_pred = float(reg_final.predict(X_z[i:i+1])[0])
        ok = "✓" if (true_gap > m_pred) == (true_gap > 0.10) else "Δ"
        print(f'  {info[i]["ds"]:13} {info[i]["N"]} {info[i]["seed"]:4} | '
              f'{true_gap:>+8.3f}  {m_pred:>+10.3f}  {ok}')


if __name__ == "__main__":
    train_margin_head()
