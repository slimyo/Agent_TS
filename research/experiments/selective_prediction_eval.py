"""task #64 / P1-J · Selective Prediction (f,g) 形式化实证。

直接对接 §3.0.3：(f, g) where f=base predictor, g=abstain gate.
量化 coverage-risk curve 在两个场景：

(1) RCA: f=Agent 5-class predict, g=abstain head (P(OOT))
    coverage(g) = fraction of test cells where g=1 (predict in-tax)
    selective_risk = error rate on in-tax cells where g=1

(2) Forecasting: f=v10 wrapper, g=memory consensus revert
    coverage(g) = fraction where wrapper used (g=1)
    selective_risk = MAE on those cells

输出 coverage-risk 曲线点 + 选定 threshold 的 operating points。
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np


def load_jsonl(p):
    if not Path(p).exists(): return []
    return [json.loads(l) for l in open(p)]


# ========================================
# 1) RCA Selective Prediction
# ========================================
def rca_selective():
    """RCA: Agent + abstain head as (f, g).
    Use task #46 abstain_eval data: per-cell P(OOT) + agent prediction + GT.
    """
    abst = load_jsonl("research/results/taska_abstain_eval.jsonl")
    # We have agent_no_abstain (= f) and binary abstain decision implied by p > 0.5
    # We need raw P(OOT) per cell to build coverage-risk curve
    # Reload from abstain head + recompute proba

    print("\n=== RCA Selective Prediction (f, g) ===")
    print("f = Agent 5-class predict; g = abstain head P(in-tax) > τ\n")

    if not Path("research/results/abstain_head.pkl").exists():
        print("abstain head not found, skipping")
        return

    with open("research/results/abstain_head.pkl", "rb") as fh:
        head = pickle.load(fh)
    # Build per-cell features + apply head
    from research.agent.abstain_head import build_training_set
    X, y, info = build_training_set(window_len=96, n_per_class=5, seed=1)
    # y=0 means in-taxonomy, y=1 means OOT
    X_z = head["scaler"].transform(X)
    proba_oot = head["clf"].predict_proba(X_z)[:, 1]

    # For each cell, mark:
    # - GT class label (in-tax or OOT)
    # - abstain head P(OOT)
    # - Agent prediction (from abstain_eval data) - cached only for cells in both sets

    # Coverage-risk over threshold τ
    print(f"{'τ':>6} {'cov(g)':>8} {'sel_acc':>9} {'OOT_recall':>11} {'in-tax_FN':>11}")
    for tau in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        # g=1 means predict in-tax (P(OOT) < tau)
        g_predict = proba_oot < tau   # boolean
        # Coverage = how many cells we predict (don't abstain)
        coverage = float(g_predict.mean())
        # Selective accuracy on g=1: among predicted, how many actually in-tax (y=0)?
        if g_predict.sum() > 0:
            sel_acc = float((y[g_predict] == 0).mean())
        else: sel_acc = float("nan")
        # OOT recall: among true OOT (y=1), how many correctly abstained (g=0)
        if (y == 1).sum() > 0:
            oot_recall = float((g_predict[y == 1] == False).mean())
        else: oot_recall = float("nan")
        # in-tax false negative: among true in-tax (y=0), how many wrongly abstained (g=0)
        if (y == 0).sum() > 0:
            in_fn = float((g_predict[y == 0] == False).mean())
        else: in_fn = float("nan")
        print(f"{tau:>6.2f} {coverage:>8.3f} {sel_acc:>9.3f} {oot_recall:>11.3f} {in_fn:>11.3f}")

    # AUC
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, proba_oot)
    print(f"\nAUC (abstain head detecting OOT): {auc:.4f}")


# ========================================
# 2) Forecasting Selective Prediction
# ========================================
def forecast_selective():
    """Forecasting: f=v10 wrapper, g=should_we_use_wrapper?
    Use forecast_abstain_head.pkl trained in task #22.
    """
    print("\n=== Forecasting Selective Prediction (f, g) ===")
    print("f = v10 wrapper (deviation from Chronos-2); g = abstain head P(wrapper helps)\n")

    if not Path("research/results/forecast_abstain_head.pkl").exists():
        print("forecast abstain head not found, skipping")
        return

    with open("research/results/forecast_abstain_head.pkl", "rb") as fh:
        head = pickle.load(fh)

    from research.agent.forecast_abstain import build_forecast_abstain_data
    X, y, info = build_forecast_abstain_data(margin=0.02)
    # y=1 = wrapper helped; y=0 = wrapper hurt/tied
    X_honest = X[:, :-1]  # drop oracle ratio
    X_z = head["scaler"].transform(X_honest)
    proba_helps = head["clf"].predict_proba(X_z)[:, 1]

    # Coverage-risk over threshold
    print(f"{'τ':>6} {'cov(g)':>8} {'mean_mae':>10} {'help_recall':>12} {'over_use':>10}")
    for tau in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        g_use_wrapper = proba_helps > tau
        coverage = float(g_use_wrapper.mean())
        # Total MAE: use wrapper if g=1, else Chronos-2
        total_mae = 0.0
        for i, ci in enumerate(info):
            chosen_mae = ci["v10_mae"] if g_use_wrapper[i] else ci["c2_mae"]
            total_mae += chosen_mae
        mean_mae = total_mae / len(info)
        # Help-recall: among truly helpful cells (y=1), did we use wrapper?
        if (y == 1).sum() > 0:
            help_recall = float(g_use_wrapper[y == 1].mean())
        else: help_recall = float("nan")
        # Over-use: among hurt cells (y=0), did we wrongly use wrapper?
        if (y == 0).sum() > 0:
            over_use = float(g_use_wrapper[y == 0].mean())
        else: over_use = float("nan")
        print(f"{tau:>6.2f} {coverage:>8.3f} {mean_mae:>10.4f} {help_recall:>12.3f} {over_use:>10.3f}")

    # Reference points
    c2_mae = np.mean([ci["c2_mae"] for ci in info])
    v10_mae = np.mean([ci["v10_mae"] for ci in info])
    print(f"\nReference:")
    print(f"  Chronos-2 always (cov=0): mae={c2_mae:.4f}")
    print(f"  v10 always (cov=1):       mae={v10_mae:.4f}")


def main():
    rca_selective()
    forecast_selective()


if __name__ == "__main__":
    main()
