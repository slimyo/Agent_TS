"""task #49 / P0-2 · Level 2 Meta-Router · 替代启发式 routing。

输入：25-dim handcrafted features (series_features.featurize_cell)
输出：5-class softmax over {rocket, moment_1nn, moment_logreg, dtw_1nn, euclid_1nn}

训练数据：复用历史 sweep results
  - taskb_ucr.jsonl (UCR-5, 30 unique cells × 7 methods → 30 labeled)
  - taskb_extended_ucr.jsonl (UCR-extended ~25 cells)
  - taskc_synth4class.jsonl (synthetic 4-class, 12 cells)
Total ≈ 70 labeled cells × 25-dim features.

替换路径：
  classification_planner 加 use_meta_router: bool 参数
  当 True 时绕过 LOO CV + margin + memory consensus 整个决策栈，直接用 head 预测。
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
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
    """从历史 sweep results 构造 (features, best_classifier_idx) 标签对。"""
    from collections import defaultdict

    sweep_files = [
        "research/results/taskb_ucr.jsonl",
        "research/results/taskb_extended_ucr.jsonl",
    ]
    by_cell_clf_acc = defaultdict(dict)
    for fp in sweep_files:
        if not Path(fp).exists():
            continue
        for line in open(fp):
            r = json.loads(line)
            if r["method"] not in METHOD_TO_CLF:
                continue
            key = (r["dataset"], r["N_per_class"], r["seed"])
            by_cell_clf_acc[key][METHOD_TO_CLF[r["method"]]] = r["acc"]

    X, y, info = [], [], []
    for (ds, n, seed), accs in by_cell_clf_acc.items():
        # 至少要 ≥3 个 classifier 的结果才有意义
        if len(accs) < 3:
            continue
        # 跳过 UCR 大数据集（Crop 24-class 之类不在 ucr_loader 里）
        try:
            X_tr, y_tr, _, _ = load_ucr_fewshot(ds, n_per_class=n, seed=seed)
        except Exception:
            continue
        feat = featurize_cell(X_tr, y_tr)
        # Label = arg max over classifiers
        best = max(accs.items(), key=lambda kv: kv[1])
        try:
            label_idx = CLASSIFIER_ORDER.index(best[0])
        except ValueError:
            continue
        X.append(feat)
        y.append(label_idx)
        info.append({"ds": ds, "N": n, "seed": seed,
                     "best_classifier": best[0], "best_acc": best[1],
                     "all_accs": accs})

    return np.stack(X), np.array(y), info


def train_meta_router(method: str = "logreg"):
    """5-fold leave-one-dataset-out CV + train final on all."""
    X, y, info = build_training_set()
    print(f"Training set: X={X.shape} y={y.shape} (n_classes={len(np.unique(y))})")

    # Class distribution
    from collections import Counter
    counts = Counter([CLASSIFIER_ORDER[i] for i in y])
    print(f"Label distribution: {dict(counts)}")

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X)
    X_z = scaler.transform(X)

    # Leave-one-dataset-out CV
    datasets = sorted(set(i["ds"] for i in info))
    print(f"\nLeave-one-dataset-out CV across {len(datasets)} datasets:")
    correct = 0
    oracle_acc_sum = 0.0
    selected_acc_sum = 0.0
    for held_ds in datasets:
        train_mask = np.array([i["ds"] != held_ds for i in info])
        test_mask = ~train_mask
        if test_mask.sum() == 0: continue
        if method == "logreg":
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
        else:
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=100, max_depth=5,
                                          class_weight="balanced", random_state=0)
        clf.fit(X_z[train_mask], y[train_mask])
        y_pred = clf.predict(X_z[test_mask])
        y_true = y[test_mask]
        # Accuracy of label match
        acc = float((y_pred == y_true).mean())
        # Accuracy if we pick predicted classifier (real test acc on each cell)
        held_info = [info[i] for i, m in enumerate(test_mask) if m]
        for ci, pred_idx in zip(held_info, y_pred):
            picked_clf = CLASSIFIER_ORDER[pred_idx]
            selected_acc_sum += ci["all_accs"].get(picked_clf, 0.0)
            oracle_acc_sum += ci["best_acc"]
        correct += int((y_pred == y_true).sum())
        print(f"  held={held_ds:13} n_test={test_mask.sum()}  "
              f"label_match={acc:.3f}")

    n_total = len(info)
    print(f"\n=== LODO CV aggregate ({n_total} cells) ===")
    print(f"  Meta-Router label match (pick best classifier): {correct}/{n_total} = {correct/n_total:.3f}")
    print(f"  Mean selected-classifier acc:  {selected_acc_sum/n_total:.4f}")
    print(f"  Mean oracle acc (upper bound): {oracle_acc_sum/n_total:.4f}")
    print(f"  Gap to oracle: {(oracle_acc_sum - selected_acc_sum)/n_total*100:+.2f}pp")

    # Compare to "always rocket" baseline
    rocket_acc_sum = sum(ci["all_accs"].get("rocket", 0.0) for ci in info)
    print(f"  Mean rocket-alone acc:         {rocket_acc_sum/n_total:.4f}")
    print(f"  Meta-Router vs rocket:         "
          f"{(selected_acc_sum - rocket_acc_sum)/n_total*100:+.2f}pp")

    # Train final model on all data
    if method == "logreg":
        from sklearn.linear_model import LogisticRegression
        final = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    else:
        from sklearn.ensemble import RandomForestClassifier
        final = RandomForestClassifier(n_estimators=100, max_depth=5,
                                        class_weight="balanced", random_state=0)
    final.fit(X_z, y)

    return final, scaler, info


def save_meta_router(method: str = "logreg",
                     out_path: str = "research/results/meta_router.pkl"):
    clf, scaler, info = train_meta_router(method=method)
    with open(out_path, "wb") as f:
        pickle.dump({"clf": clf, "scaler": scaler,
                     "classifier_order": CLASSIFIER_ORDER,
                     "method": method}, f)
    print(f"\nSaved → {out_path}")
    return clf, scaler


# ---------- Inference API（给 clf_planner 用） ---------- #

_CACHE = None


def _load_meta_router(path: str = "research/results/meta_router.pkl"):
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    if not Path(path).exists():
        _CACHE = None
        return None
    with open(path, "rb") as f:
        _CACHE = pickle.load(f)
    return _CACHE


def predict_classifier(X_train: np.ndarray, y_train: np.ndarray,
                       confidence_threshold: float = 0.0) -> tuple[str, float, dict]:
    """task #49 公共接口：返回 (chosen_classifier_name, confidence, full_probs)。
    若所有 prob < confidence_threshold → 返回 'rocket' default。
    """
    head = _load_meta_router()
    if head is None:
        return "rocket", 0.0, {}
    feat = featurize_cell(X_train, y_train).reshape(1, -1)
    feat_z = head["scaler"].transform(feat)
    proba = head["clf"].predict_proba(feat_z)[0]
    classes = head["classifier_order"]
    # Map sklearn class_ indices to classifier_order
    sk_classes = head["clf"].classes_
    full_probs = {classes[int(c)]: float(proba[i]) for i, c in enumerate(sk_classes)}
    best_idx = int(np.argmax(proba))
    chosen = classes[int(sk_classes[best_idx])]
    confidence = float(proba[best_idx])
    if confidence < confidence_threshold:
        chosen = "rocket"
    return chosen, confidence, full_probs


if __name__ == "__main__":
    print("\n=== Training Meta-Router (logreg) ===")
    save_meta_router(method="logreg")
    print("\n=== Training Meta-Router (rf) ===")
    save_meta_router(method="rf", out_path="research/results/meta_router_rf.pkl")
