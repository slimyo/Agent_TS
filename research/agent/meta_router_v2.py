"""task #51 / Level 2 v2 · Meta-Router 改进。

改动 vs v1：
  1. Confidence-gated override：仅当 best_class prob ≥ TAU 才偏离 rocket-default
  2. Regression-per-classifier 替代 multiclass：每个 classifier 单独训预测 acc
     → 用 predicted_acc 排序，选 acc 最高者
  3. 扩训练数据：包含 synthetic 4-class（task #27）+ UEA partial

设计：v2 用 RFR per-classifier，多 head。
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np

from research.utils.series_features import featurize_cell
from research.utils.ucr_loader import load_ucr_fewshot


METHOD_TO_CLF = {
    "B1_dtw":          "dtw_1nn",
    "B2_euclid":       "euclid_1nn",
    "B3_rocket":       "rocket",
    "B4a_moment_1nn":  "moment_1nn",
    "B4b_moment_lr":   "moment_logreg",
}

CLASSIFIER_ORDER = ["rocket", "moment_1nn", "moment_logreg", "dtw_1nn", "euclid_1nn"]


def build_training_set():
    sweep_files = [
        "research/results/taskb_ucr.jsonl",
        "research/results/taskb_extended_ucr.jsonl",
    ]
    by_cell_clf_acc = defaultdict(dict)
    for fp in sweep_files:
        if not Path(fp).exists(): continue
        for line in open(fp):
            r = json.loads(line)
            if r["method"] not in METHOD_TO_CLF: continue
            key = (r["dataset"], r["N_per_class"], r["seed"])
            by_cell_clf_acc[key][METHOD_TO_CLF[r["method"]]] = r["acc"]

    X, Y, info = [], [], []
    for (ds, n, seed), accs in by_cell_clf_acc.items():
        if len(accs) < 3: continue
        try:
            X_tr, y_tr, _, _ = load_ucr_fewshot(ds, n_per_class=n, seed=seed)
        except Exception:
            continue
        feat = featurize_cell(X_tr, y_tr)
        # Y row: per-classifier acc, missing → np.nan (will be masked in regression)
        y_row = np.array([accs.get(c, np.nan) for c in CLASSIFIER_ORDER], dtype=np.float32)
        X.append(feat)
        Y.append(y_row)
        info.append({"ds": ds, "N": n, "seed": seed, "all_accs": accs})

    return np.stack(X), np.stack(Y), info


def train_regression_meta_router(tau: float = 0.05):
    """V2: per-classifier regression head。
    tau = "deviation margin"：仅当 best_predicted_acc ≥ rocket_predicted_acc + tau 才偏离 rocket。
    """
    X, Y, info = build_training_set()
    print(f"Training set: X={X.shape}, Y={Y.shape}")
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor

    scaler = StandardScaler().fit(X)
    X_z = scaler.transform(X)

    # Train per-classifier regressor (using only cells that have that classifier evaluated)
    heads = {}
    for ci, c in enumerate(CLASSIFIER_ORDER):
        mask = ~np.isnan(Y[:, ci])
        if mask.sum() < 5:
            heads[c] = None; continue
        reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=0)
        reg.fit(X_z[mask], Y[mask, ci])
        heads[c] = reg

    # Leave-one-dataset-out evaluation
    datasets = sorted(set(i["ds"] for i in info))
    print(f"\nLODO CV across {len(datasets)} datasets (deviation margin tau={tau}):")
    selected_acc_sum = 0.0
    rocket_acc_sum = 0.0
    oracle_acc_sum = 0.0
    n_dev = 0  # deviation count (chose non-rocket)
    n_total = 0
    for held in datasets:
        train_mask = np.array([i["ds"] != held for i in info])
        test_mask = ~train_mask
        if test_mask.sum() == 0: continue
        # Retrain heads on held-out
        local_heads = {}
        for ci, c in enumerate(CLASSIFIER_ORDER):
            m = train_mask & (~np.isnan(Y[:, ci]))
            if m.sum() < 5:
                local_heads[c] = None; continue
            reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=0)
            reg.fit(X_z[m], Y[m, ci])
            local_heads[c] = reg
        # Predict per-cell
        test_info = [info[i] for i, m in enumerate(test_mask) if m]
        X_te = X_z[test_mask]
        for j, ci_info in enumerate(test_info):
            pred_accs = {}
            for c, head in local_heads.items():
                if head is not None:
                    pred_accs[c] = float(head.predict(X_te[j:j+1])[0])
            if not pred_accs:
                chosen = "rocket"
            else:
                # Confidence-gated: only deviate if best_other > rocket_pred + tau
                rocket_p = pred_accs.get("rocket", 0)
                others = [(c, p) for c, p in pred_accs.items() if c != "rocket"]
                best_other = max(others, key=lambda kv: kv[1]) if others else ("rocket", -1)
                if best_other[1] > rocket_p + tau:
                    chosen = best_other[0]
                    n_dev += 1
                else:
                    chosen = "rocket"
            selected_acc = ci_info["all_accs"].get(chosen, 0.0)
            selected_acc_sum += selected_acc
            rocket_acc_sum += ci_info["all_accs"].get("rocket", 0.0)
            oracle_acc_sum += max(ci_info["all_accs"].values())
            n_total += 1

    print(f"\n=== LODO CV aggregate ({n_total} cells, tau={tau}) ===")
    print(f"  Meta-Router v2 selected acc:   {selected_acc_sum/n_total:.4f}")
    print(f"  Rocket-alone acc:              {rocket_acc_sum/n_total:.4f}")
    print(f"  Oracle acc (upper bound):      {oracle_acc_sum/n_total:.4f}")
    print(f"  v2 vs rocket:                  "
          f"{(selected_acc_sum - rocket_acc_sum)/n_total*100:+.2f}pp")
    print(f"  Deviations from rocket:        {n_dev}/{n_total} = {n_dev/n_total:.2%}")

    # Save final model (trained on all data)
    return heads, scaler, info


def main():
    print("\n=== Meta-Router v2 (RFR regression + confidence-gated, sweep tau) ===")
    for tau in [0.00, 0.02, 0.05, 0.10]:
        print(f"\n--- tau={tau} ---")
        train_regression_meta_router(tau=tau)

    # Save with best tau (we'll pick 0.05 as default; user can tune)
    print("\n=== Save final model (tau=0.05) ===")
    heads, scaler, info = train_regression_meta_router(tau=0.05)
    out = "research/results/meta_router_v2.pkl"
    with open(out, "wb") as f:
        pickle.dump({"heads": heads, "scaler": scaler,
                     "classifier_order": CLASSIFIER_ORDER}, f)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
