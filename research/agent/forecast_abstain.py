"""task #22 / C10 · Forecasting Abstain Head — RCA #46 思想移植到 forecasting wrapper。

设计：训二分类头预测"wrapper deviation 是否会帮上 MAE"。
  - Label = 1: v10_mae < c2_mae - margin → 应保持 v10 deviation
  - Label = 0: v10_mae >= c2_mae - margin → 应 strict fallback to Chronos-2

预测时若 P(helps) < threshold → 强制 fallback to Chronos-2，避免 v10 catastrophic 偏离失误。

训练数据来源：复用 v10 + Chronos-2 sweeps（已有）。
"""
from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from research.agent.curator_uq import diagnose
from research.utils.data_loader import load_series
from research.utils.splitter import few_shot_split


def build_forecast_abstain_data(margin: float = 0.02):
    """从历史 v10/v9/v11 + Chronos-2 sweeps 构造 (features, helped?) 标签。

    Label：
      1 = v10_mae < c2_mae * (1 - margin)  (wrapper 显著优于 C2)
      0 = otherwise (wrapper 无显著优势或恶化)
    """
    # Load c2 baselines
    c2_files = [
        "research/results/f4_chronos2.jsonl",
        "research/results/f4_bolt_c2_ecl_exchange.jsonl",
        "research/results/f4_bolt_c2_weather.jsonl",
        "research/results/f4_bolt_c2_ili.jsonl",
    ]
    c2_lookup = {}
    for fp in c2_files:
        if not Path(fp).exists(): continue
        for line in open(fp):
            r = json.loads(line)
            if r.get("method") not in ("chronos2", None): continue
            key = (r["dataset"], r["N"], r["seed"])
            c2_lookup[key] = r["mae"]

    # Load v10 / v9 (forecasting wrapper) results
    v10_files = [
        "research/results/p10_adapt_v10_n10.jsonl",
        "research/results/p9_adapt_v9.jsonl",
    ]
    v10_results = []
    for fp in v10_files:
        if not Path(fp).exists(): continue
        for line in open(fp):
            r = json.loads(line)
            v10_results.append(r)

    # Build (features, label)
    X, y, info = [], [], []
    for r in v10_results:
        key = (r["dataset"], r["N"], r["seed"])
        if key not in c2_lookup: continue
        c2_mae = c2_lookup[key]
        v10_mae = r["mae"]
        # Label
        helped = 1 if v10_mae < c2_mae * (1 - margin) else 0

        # Features: Curator on the split's train series
        try:
            series, meta = load_series(r["dataset"])
            sp = few_shot_split(series, N=r["N"], H=r["H"], seed=r["seed"])
            d = diagnose(sp.train, season_m=meta.season_m)
            cmap = {"high": 1.0, "mid": 0.5, "low": 0.0}
            feat = np.array([
                np.log1p(d.n),
                np.tanh(d.trend_tstat / 5.0),
                d.adf_pvalue,
                d.acf_peak_value,
                np.log1p(d.acf_peak_lag),
                cmap[d.trend_conf_xc],
                cmap[d.season_conf_xc],
                cmap[d.stat_conf_xc],
                d.std / (abs(d.mean) + 1e-6),
                np.tanh(d.trend_slope),
                np.log1p(getattr(d, "outlier_count_z3", 0)),
                np.tanh(getattr(d, "variance_ratio", 1.0) - 1.0),
                np.log1p(r["N"]),  # cell size
                v10_mae / (c2_mae + 1e-9),  # ratio as feature (oracle)
            ], dtype=np.float32)
        except Exception as e:
            continue
        X.append(feat)
        y.append(helped)
        info.append({"ds": r["dataset"], "N": r["N"], "seed": r["seed"],
                      "v10_mae": v10_mae, "c2_mae": c2_mae, "helped": helped})

    return np.stack(X), np.array(y, dtype=int), info


def train_forecast_abstain():
    X, y, info = build_forecast_abstain_data(margin=0.02)
    print(f"Training set: X={X.shape}, y={y.shape} (helped={int(y.sum())}, hurt/tie={int((y==0).sum())})")

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score

    # Exclude the v10/c2 ratio feature for honest evaluation (it's oracle)
    X_honest = X[:, :-1]  # 13-dim

    print(f"\nHonest features (no oracle ratio): {X_honest.shape}")
    scaler = StandardScaler().fit(X_honest)
    X_z = scaler.transform(X_honest)

    for method in ["logreg", "rf"]:
        if method == "logreg":
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        else:
            clf = RandomForestClassifier(n_estimators=100, max_depth=4,
                                          class_weight="balanced", random_state=0)
        try:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            y_pred = cross_val_predict(clf, X_z, y, cv=cv)
            y_proba = cross_val_predict(clf, X_z, y, cv=cv, method="predict_proba")[:, 1]
            acc = float((y_pred == y).mean())
            auc = roc_auc_score(y, y_proba)
            help_recall = float((y_pred[y==1] == 1).mean()) if (y==1).sum()>0 else 0
            hurt_precision = float((y_pred[y==0] == 0).mean()) if (y==0).sum()>0 else 0
            print(f"\n  {method:8}: 5-fold CV")
            print(f"    accuracy={acc:.3f}  AUC={auc:.3f}")
            print(f"    helped recall (predict 1 when label 1)= {help_recall:.3f}")
            print(f"    hurt precision (predict 0 when label 0)= {hurt_precision:.3f}")
        except Exception as e:
            print(f"  {method}: CV failed {e!r}")

    # Eval realistic MAE improvement: if predict 0 → use c2_mae, else v10_mae
    print("\n=== Realistic MAE improvement if abstain head used ===")
    clf_final = RandomForestClassifier(n_estimators=100, max_depth=4,
                                         class_weight="balanced", random_state=0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    y_proba = cross_val_predict(clf_final, X_z, y, cv=cv, method="predict_proba")[:, 1]
    threshold = 0.5
    abstain_mae_sum = 0.0
    v10_mae_sum = 0.0
    c2_mae_sum = 0.0
    for i, ci in enumerate(info):
        if y_proba[i] >= threshold:  # predicts helped → keep v10
            chosen_mae = ci["v10_mae"]
        else:  # predicts hurt/tie → abstain to c2
            chosen_mae = ci["c2_mae"]
        abstain_mae_sum += chosen_mae
        v10_mae_sum += ci["v10_mae"]
        c2_mae_sum += ci["c2_mae"]
    n = len(info)
    print(f"  Abstain-decision mean MAE: {abstain_mae_sum/n:.4f}")
    print(f"  v10 mean MAE:               {v10_mae_sum/n:.4f}")
    print(f"  Chronos-2 alone mean MAE:   {c2_mae_sum/n:.4f}")
    print(f"  Abstain vs v10:  {(abstain_mae_sum - v10_mae_sum)/n:+.4f}  (negative is improvement)")
    print(f"  Abstain vs C2:   {(abstain_mae_sum - c2_mae_sum)/n:+.4f}")

    # Save final
    clf_final.fit(X_z, y)
    out = "research/results/forecast_abstain_head.pkl"
    with open(out, "wb") as f:
        pickle.dump({"clf": clf_final, "scaler": scaler}, f)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    train_forecast_abstain()
