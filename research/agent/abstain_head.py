"""task #46 / P0-1 · Abstain head 修 Specialist Bias。

设计：训一个轻量二分类头 (in-taxonomy vs out-of-taxonomy) on 25-dim handcrafted features.
预测 LLM Agent 输出后 hard-override：if abstain head says OOT → force primary_fault = out_of_taxonomy.

训练集（100 labeled cells）:
  - 50 in-taxonomy (task #25: trend_break/seasonal_flip/variance_explode/outlier_burst/stationarity_flip)
  - 50 OOT (task #43: missing_data_gap/heavy_noise/mode_collapse/freq_mod/quantization)

评估：abstain on/off 在 50 OOT cells 上的 OOT-recall + R1 on in-tax cells (确认未损害 in-tax 性能)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research.utils.data_loader import load_series
from research.utils.fault_taxonomy import detect_faults
from research.utils.inject_fault import (
    OOT_INJECTORS, RCA_INJECTORS, build_oot_rca_dataset,
    build_rca_synthetic_dataset,
)
from research.utils.series_features import featurize_cell


def extract_abstain_features(train: np.ndarray) -> np.ndarray:
    """从 train 序列提取 abstain head 的特征。
    用 25-dim series_features + 5 个 in-taxonomy fault scores（让 head 知道每个 in-tax 类匹配多深）。
    """
    # series_features 需要 [N, L] 输入；这里 train 是单序列
    X_pseudo = train.reshape(1, -1).astype(np.float32)
    y_pseudo = np.array([0])
    feat_25 = featurize_cell(X_pseudo, y_pseudo)
    # 加 5 个 in-tax fault scores（强信号 → 不需 abstain，弱信号 → 应 abstain）
    in_tax_scores = detect_faults(train, season_m=1)
    score_vec = np.array([
        in_tax_scores["trend_break"],
        in_tax_scores["seasonal_flip"],
        in_tax_scores["variance_explode"],
        in_tax_scores["outlier_burst"],
        in_tax_scores["stationarity_flip"],
    ], dtype=np.float32)
    # max score 也是强信号
    max_score = score_vec.max()
    score_vec = np.concatenate([score_vec, [max_score]])
    return np.concatenate([feat_25, score_vec]).astype(np.float32)  # 25 + 6 = 31 dim


def build_training_set(window_len: int = 96, n_per_class: int = 5, seed: int = 1):
    """50 in-tax (label=0) + 50 OOT (label=1) labeled cells."""
    X = []
    y = []
    cells_info = []
    for ds in ["ETTh1", "ECL"]:
        series, meta = load_series(ds)
        # in-taxonomy
        in_tax_cells = build_rca_synthetic_dataset(
            series, window_len=window_len, n_per_class=n_per_class,
            seed=seed, season_m=meta.season_m,
        )
        for c in in_tax_cells:
            X.append(extract_abstain_features(c["train"]))
            y.append(0)  # 0 = in-taxonomy
            cells_info.append({"ds": ds, "kind": "in_tax", "fault": c["fault_label"]})
        # out-of-taxonomy
        oot_cells = build_oot_rca_dataset(
            series, window_len=window_len, n_per_class=n_per_class,
            seed=seed, season_m=meta.season_m,
        )
        for c in oot_cells:
            X.append(extract_abstain_features(c["train"]))
            y.append(1)  # 1 = OOT
            cells_info.append({"ds": ds, "kind": "oot", "fault": c["fault_label"]})
    return np.stack(X), np.array(y, dtype=int), cells_info


def train_abstain_head(X: np.ndarray, y: np.ndarray, method: str = "logreg"):
    """训练 + 5-fold CV 评估。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X)
    X_z = scaler.transform(X)

    if method == "logreg":
        clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
    elif method == "rf":
        clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0,
                                      class_weight="balanced")
    else:
        raise ValueError(method)

    # 5-fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    y_pred = cross_val_predict(clf, X_z, y, cv=cv)
    y_proba = cross_val_predict(clf, X_z, y, cv=cv, method="predict_proba")[:, 1]

    # Fit final model on all data
    clf.fit(X_z, y)
    return clf, scaler, y_pred, y_proba


def main():
    print("Building training set ...")
    X, y, info = build_training_set(window_len=96, n_per_class=5, seed=1)
    print(f"  X={X.shape}, y={y.shape} (in_tax={int((y==0).sum())}, oot={int((y==1).sum())})")

    print("\n=== Training abstain head ===")
    for method in ["logreg", "rf"]:
        clf, scaler, y_pred, y_proba = train_abstain_head(X, y, method=method)
        acc = float((y_pred == y).mean())
        # 分别看 in-tax / OOT 的预测正确率
        in_tax_acc = float((y_pred[y == 0] == 0).mean())
        oot_recall = float((y_pred[y == 1] == 1).mean())
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y, y_proba)
        print(f"\n  {method:8}:")
        print(f"    accuracy (5-fold CV)         = {acc:.3f}")
        print(f"    in-tax precision (predict 0) = {in_tax_acc:.3f}")
        print(f"    OOT recall (predict 1)       = {oot_recall:.3f}")
        print(f"    AUC                          = {auc:.3f}")

    # Save best model (rf usually wins for tabular)
    print("\n=== Save logreg model ===")
    clf, scaler, _, _ = train_abstain_head(X, y, method="logreg")
    import pickle
    out_dir = Path("research/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "abstain_head.pkl", "wb") as f:
        pickle.dump({"clf": clf, "scaler": scaler, "feature_dim": X.shape[1]}, f)
    print("  saved → research/results/abstain_head.pkl")

    # Confusion per fault type
    print("\n=== Per-fault breakdown (LogReg CV) ===")
    clf_lr, scaler_lr, y_pred_lr, _ = train_abstain_head(X, y, method="logreg")
    from collections import defaultdict
    by_fault = defaultdict(lambda: [0, 0])  # [correct, total]
    for i, ci in enumerate(info):
        key = f"{ci['kind']}_{ci['fault']}"
        target = 0 if ci['kind'] == 'in_tax' else 1
        by_fault[key][0] += int(y_pred_lr[i] == target)
        by_fault[key][1] += 1
    for k in sorted(by_fault):
        c, t = by_fault[k]
        print(f"  {k:35} {c}/{t} = {c/t:.2f}")


if __name__ == "__main__":
    main()
