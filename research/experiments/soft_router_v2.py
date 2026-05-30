"""task #68 v2 — Soft Router with cached per-classifier predictions.

Fix 22h-hang of v1: precompute classifier predictions ONCE, then apply
softmax(beta · pred_accs) over the same predictions for all β values.

Outputs incremental JSONL (one row per (cell, beta)) to allow partial reads.
"""
from __future__ import annotations
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from research.agent.soft_router import (
    CLASSIFIER_ORDER, CLF_FUNCS, build_training_set, train_heads,
    softmax_weights, soft_vote,
)
from research.utils.ucr_loader import load_ucr_fewshot

CACHE_PATH = Path("research/results/soft_router_pred_cache.jsonl")
RESULT_PATH = Path("research/results/soft_router_lodo_v2.jsonl")
CANDIDATES = ["rocket", "moment_1nn", "euclid_1nn"]    # drop dtw_1nn — too slow on industrial
BETAS = [1, 3, 5, 10, 20, 50, 100]


def build_or_load_cache():
    """For each cell × candidate, compute test predictions once and cache as JSONL.

    cache row keys: dataset, N, seed, classifier, test_y (truncated to 200),
                    pred_y (matching), wall_time, X_test_sub_indices
    """
    done = set()
    if CACHE_PATH.exists():
        for l in CACHE_PATH.read_text().splitlines():
            try:
                r = json.loads(l)
                done.add((r["dataset"], r["N"], r["seed"], r["clf"]))
            except: pass
    print(f"Cache: {len(done)} entries already done.")

    X, Y, info = build_training_set()
    fh = CACHE_PATH.open("a")
    for idx, ci in enumerate(info):
        ds, n, seed = ci["ds"], ci["N"], ci["seed"]
        try:
            X_tr, y_tr, X_te, y_te = load_ucr_fewshot(ds, n_per_class=n, seed=seed)
        except Exception as e:
            print(f"  skip {ds} N={n} s={seed}: load fail")
            continue
        if len(X_te) > 200:
            rng = np.random.default_rng(seed)
            sub = rng.choice(len(X_te), 200, replace=False)
            X_te, y_te = X_te[sub], y_te[sub]
        # Skip DTW for length > 500
        L = X_tr.shape[1]
        for clf in CANDIDATES:
            key = (ds, n, seed, clf)
            if key in done: continue
            if clf not in CLF_FUNCS: continue
            t0 = time.time()
            try:
                yp = CLF_FUNCS[clf](X_tr, y_tr, X_te)
                dt = time.time() - t0
                row = {"dataset": ds, "N": n, "seed": seed, "clf": clf, "L": int(L),
                       "y_true": y_te.tolist(), "y_pred": yp.tolist(),
                       "acc": float((yp == y_te).mean()), "wall_time": round(dt, 2)}
                fh.write(json.dumps(row) + "\n"); fh.flush()
                done.add(key)
                print(f"  [{idx+1}/{len(info)}] {ds:14} n={n:2} s={seed:2} {clf:12}: acc={row['acc']:.3f} ({dt:.1f}s)")
            except Exception as e:
                print(f"  FAIL {ds} {clf}: {type(e).__name__}: {str(e)[:80]}")
    fh.close()


def load_cache():
    cache = defaultdict(dict)
    for l in CACHE_PATH.open():
        r = json.loads(l)
        cache[(r["dataset"], r["N"], r["seed"])][r["clf"]] = {
            "y_pred": np.array(r["y_pred"]), "y_true": np.array(r["y_true"]),
            "acc": r["acc"]
        }
    return cache


def lodo_with_cache():
    cache = load_cache()
    X, Y, info = build_training_set()
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X)
    X_z = scaler.transform(X)
    datasets = sorted(set(i["ds"] for i in info))

    fh = RESULT_PATH.open("w")
    summary = []
    for beta in BETAS:
        print(f"\n--- β = {beta} ---")
        rows = []
        for held in datasets:
            train_mask = np.array([i["ds"] != held for i in info])
            heads = train_heads(X_z, Y, train_mask)
            if all(h is None for h in heads.values()): continue
            for idx, ci in enumerate(info):
                if ci["ds"] != held: continue
                ds, n, seed = ci["ds"], ci["N"], ci["seed"]
                cell_preds = cache.get((ds, n, seed))
                if not cell_preds: continue
                # Meta-router pred per classifier
                pred_accs = {c: float(h.predict(X_z[idx:idx+1])[0])
                             for c, h in heads.items() if h is not None and c in CANDIDATES}
                if not pred_accs: continue
                # Soft weights
                w_soft = softmax_weights(pred_accs, beta, CANDIDATES)
                # Hard winner
                chosen_hard = max(((c, pred_accs.get(c, -1)) for c in CANDIDATES),
                                  key=lambda kv: kv[1])[0]
                # Build per_clf_preds for soft_vote
                per_clf_preds = {c: cell_preds[c]["y_pred"] for c in CANDIDATES
                                 if c in cell_preds}
                if not per_clf_preds: continue
                y_te = cell_preds[next(iter(per_clf_preds))]["y_true"]
                yp_soft = soft_vote(per_clf_preds, w_soft, len(y_te))
                soft_acc = float((yp_soft == y_te).mean())
                hard_acc = cell_preds[chosen_hard]["acc"] if chosen_hard in cell_preds else 0
                rocket_acc = cell_preds.get("rocket", {}).get("acc", 0)
                oracle_acc = max(c["acc"] for c in cell_preds.values())
                row = {"dataset": ds, "N": n, "seed": seed, "beta": beta,
                       "rocket_acc": rocket_acc, "hard_acc": hard_acc,
                       "soft_acc": soft_acc, "oracle_acc": oracle_acc,
                       "chosen_hard": chosen_hard, "weights": w_soft}
                fh.write(json.dumps(row) + "\n"); fh.flush()
                rows.append(row)
        if rows:
            r_m = np.mean([r["rocket_acc"] for r in rows])
            h_m = np.mean([r["hard_acc"] for r in rows])
            s_m = np.mean([r["soft_acc"] for r in rows])
            o_m = np.mean([r["oracle_acc"] for r in rows])
            print(f"  n={len(rows)} rocket={r_m:.4f} hard={h_m:.4f} soft={s_m:.4f} oracle={o_m:.4f}")
            print(f"  soft vs rocket: {(s_m-r_m)*100:+.2f}pp")
            summary.append({"beta": beta, "rocket": r_m, "hard": h_m,
                            "soft": s_m, "oracle": o_m, "n": len(rows)})
    fh.close()
    with open("research/results/soft_router_v2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    print("=== Phase 1: build prediction cache (skip if already done) ===")
    build_or_load_cache()
    print("\n=== Phase 2: β sweep over cached predictions ===")
    lodo_with_cache()
    print("\n✓ done.")
