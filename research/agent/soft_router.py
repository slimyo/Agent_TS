"""task #68 · Soft Router for TSC (feedback C架构#1).

Replace hard argmax routing with softmax-weighted ensemble:

  pred_accs[k] = MetaRouterHead_k(features)
  weights = softmax(beta * pred_accs)
  pred[i] = argmax_c sum_k weights[k] * I[classifier_k(x_i) == c]

Hyperparameters:
  - beta: temperature (low=uniform, high=sharp/argmax)
  - candidates: classifier list

Hard router (existing B7v3) is the limit β→∞.
"""
from __future__ import annotations
import json, pickle
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

from research.utils.series_features import featurize_cell
from research.utils.ucr_loader import load_ucr_fewshot
from research.baseline.tsc_classical import b1_knn_dtw, b2_knn_euclid, b3_rocket
from research.baseline.moment_classifier import classify_1nn as b4a, classify_logreg as b4b


CLF_FUNCS = {
    "rocket": b3_rocket,
    "moment_1nn": b4a,
    "moment_logreg": b4b,
    "dtw_1nn": b1_knn_dtw,
    "euclid_1nn": b2_knn_euclid,
}
CLASSIFIER_ORDER = ["rocket", "moment_1nn", "moment_logreg", "dtw_1nn", "euclid_1nn"]

METHOD_TO_CLF = {
    "B1_dtw":          "dtw_1nn",
    "B2_euclid":       "euclid_1nn",
    "B3_rocket":       "rocket",
    "B4a_moment_1nn":  "moment_1nn",
    "B4b_moment_lr":   "moment_logreg",
}


def build_training_set(include_industrial: bool = True):
    """Collect per-cell features + per-classifier accs across all sweeps."""
    sweep_files = [
        "research/results/taskb_ucr.jsonl",
        "research/results/taskb_extended_ucr.jsonl",
    ]
    if include_industrial:
        sweep_files.append("research/results/industrial_case.jsonl")
    by_cell = defaultdict(dict)
    for fp in sweep_files:
        if not Path(fp).exists(): continue
        for line in open(fp):
            r = json.loads(line)
            if r["method"] not in METHOD_TO_CLF: continue
            key = (r["dataset"], r["N_per_class"], r["seed"])
            by_cell[key][METHOD_TO_CLF[r["method"]]] = r["acc"]

    X, Y, info = [], [], []
    for (ds, n, seed), accs in by_cell.items():
        if len(accs) < 3: continue
        try:
            X_tr, y_tr, _, _ = load_ucr_fewshot(ds, n_per_class=n, seed=seed)
        except Exception:
            continue
        feat = featurize_cell(X_tr, y_tr)
        y_row = np.array([accs.get(c, np.nan) for c in CLASSIFIER_ORDER], dtype=np.float32)
        X.append(feat); Y.append(y_row)
        info.append({"ds": ds, "N": n, "seed": seed, "all_accs": accs})
    return np.stack(X), np.stack(Y), info


def train_heads(X_z, Y, train_mask):
    """Train per-classifier RFR on cells in train_mask."""
    from sklearn.ensemble import RandomForestRegressor
    heads = {}
    for ci, c in enumerate(CLASSIFIER_ORDER):
        mask = train_mask & (~np.isnan(Y[:, ci]))
        if mask.sum() < 5:
            heads[c] = None; continue
        reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=0)
        reg.fit(X_z[mask], Y[mask, ci])
        heads[c] = reg
    return heads


def softmax_weights(pred_accs: dict, beta: float, candidates: list[str]) -> dict:
    """Return softmax(beta * pred_accs) over candidates."""
    accs = np.array([pred_accs.get(c, 0.0) for c in candidates], dtype=float)
    z = beta * (accs - accs.max())   # numerical stability
    w = np.exp(z); w /= w.sum()
    return dict(zip(candidates, w.tolist()))


def soft_vote(per_clf_preds: dict, weights: dict, n_test: int) -> np.ndarray:
    """Weighted vote: for each test sample, sum weights of classifiers per class."""
    classes = sorted({c for p in per_clf_preds.values() for c in np.unique(p)})
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    score = np.zeros((n_test, len(classes)))
    for clf, preds in per_clf_preds.items():
        w = weights.get(clf, 0.0)
        if w == 0: continue
        for i, p in enumerate(preds):
            score[i, cls_to_idx[p]] += w
    return np.array([classes[s.argmax()] for s in score])


def lodo_evaluate(beta: float, candidates: list[str], verbose: bool = False):
    """Leave-one-dataset-out CV: train heads on N-1, soft-vote on held-out."""
    X, Y, info = build_training_set()
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X)
    X_z = scaler.transform(X)
    datasets = sorted(set(i["ds"] for i in info))
    rows = []
    for held in datasets:
        train_mask = np.array([i["ds"] != held for i in info])
        test_idxs = [i for i, m in enumerate(train_mask) if not m]
        heads = train_heads(X_z, Y, train_mask)
        if all(h is None for h in heads.values()): continue
        for idx in test_idxs:
            ci_info = info[idx]
            # Predict per-classifier acc
            pred_accs = {c: float(h.predict(X_z[idx:idx+1])[0])
                         for c, h in heads.items() if h is not None}
            # Soft weights
            w_soft = softmax_weights(pred_accs, beta, candidates)
            # Hard weights: argmax (B7v3-like)
            chosen_hard = max(((c, pred_accs.get(c, -1)) for c in candidates),
                              key=lambda kv: kv[1])[0]
            # Load test data + get per-classifier predictions
            ds, n, seed = ci_info["ds"], ci_info["N"], ci_info["seed"]
            try:
                X_tr, y_tr, X_te, y_te = load_ucr_fewshot(ds, n_per_class=n, seed=seed)
            except Exception:
                continue
            if len(X_te) > 200:
                rng = np.random.default_rng(seed)
                sub = rng.choice(len(X_te), 200, replace=False)
                X_te, y_te = X_te[sub], y_te[sub]
            per_clf_preds = {}
            for c in candidates:
                if c not in CLF_FUNCS: continue
                try:
                    yp = CLF_FUNCS[c](X_tr, y_tr, X_te)
                    per_clf_preds[c] = yp
                except Exception: pass
            if not per_clf_preds: continue
            # Soft router result
            yp_soft = soft_vote(per_clf_preds, w_soft, len(X_te))
            soft_acc = float((yp_soft == y_te).mean())
            # Hard router result
            hard_acc = float((per_clf_preds.get(chosen_hard, y_te) == y_te).mean()) if chosen_hard in per_clf_preds else 0
            # Rocket baseline
            rocket_acc = float((per_clf_preds.get("rocket", y_te) == y_te).mean()) if "rocket" in per_clf_preds else 0
            # Oracle
            oracle_acc = max(float((p == y_te).mean()) for p in per_clf_preds.values())
            row = {"dataset": ds, "N": n, "seed": seed, "beta": beta,
                   "rocket_acc": rocket_acc, "hard_acc": hard_acc,
                   "soft_acc": soft_acc, "oracle_acc": oracle_acc,
                   "weights": w_soft, "chosen_hard": chosen_hard}
            rows.append(row)
            if verbose:
                print(f"  {ds:14} N={n:2} s={seed}: rocket={rocket_acc:.3f} hard={hard_acc:.3f} soft={soft_acc:.3f} oracle={oracle_acc:.3f}")
    return rows


def main():
    candidates = ["rocket", "moment_1nn", "dtw_1nn", "euclid_1nn"]
    print(f"\n=== Soft Router LODO sweep over beta ===")
    print(f"Candidates: {candidates}\n")
    summary = []
    for beta in [1, 3, 5, 10, 20, 50, 100]:
        print(f"--- beta={beta} ---")
        rows = lodo_evaluate(beta, candidates, verbose=False)
        if not rows: print("  no cells"); continue
        r_mean = np.mean([r["rocket_acc"] for r in rows])
        h_mean = np.mean([r["hard_acc"] for r in rows])
        s_mean = np.mean([r["soft_acc"] for r in rows])
        o_mean = np.mean([r["oracle_acc"] for r in rows])
        n_h_win = sum(r["soft_acc"] > r["rocket_acc"] for r in rows)
        n_h_lose = sum(r["soft_acc"] < r["rocket_acc"] for r in rows)
        print(f"  n={len(rows)} cells, rocket={r_mean:.4f} hard={h_mean:.4f} soft={s_mean:.4f} oracle={o_mean:.4f}")
        print(f"  soft vs rocket: {(s_mean-r_mean)*100:+.2f}pp  (W/L/T: {n_h_win}/{n_h_lose}/{len(rows)-n_h_win-n_h_lose})")
        print(f"  soft vs hard:   {(s_mean-h_mean)*100:+.2f}pp")
        summary.append({"beta": beta, "rocket": r_mean, "hard": h_mean,
                        "soft": s_mean, "oracle": o_mean, "n_cells": len(rows),
                        "win": n_h_win, "lose": n_h_lose})
    out = Path("research/results/soft_router_lodo.json")
    with out.open("w") as fh: json.dump(summary, fh, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
