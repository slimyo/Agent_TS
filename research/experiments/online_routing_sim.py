"""task #62 / P1-H · Online Routing Simulation — streaming regime shift。

设计：合成 streaming 数据，每 K=5 cells regime shift 一次，"best classifier" 在
{rocket, moment_1nn, dtw_1nn} 之间轮换。Agent 必须 detect shift + adapt routing。

对照：
  1. Always Rocket (no adaptation)
  2. Static B7v3 LOO CV per cell (snapshot-only, no memory update)
  3. Online Agent (per-cell memory backfill, accumulates labeled outcomes)

输出 streaming cumulative accuracy curve。

合成 regime：
  Regime A: 信号 phase-shifted sine → DTW wins
  Regime B: high-noise spectrum → Rocket wins
  Regime C: shape morphology → MOMENT wins
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# ----- 合成 regime ----- #

def gen_regime_a(rng, L=100, n_per_class=8, classes=2):
    """Regime A: phase-shifted sine waves → DTW favored."""
    X, y = [], []
    for c in range(classes):
        for _ in range(n_per_class):
            phase = rng.uniform(0, 0.5) + c * 1.5  # class-dependent phase
            t = np.linspace(0, 4 * np.pi, L)
            sig = np.sin(t + phase) + 0.05 * rng.standard_normal(L)
            X.append(sig); y.append(c)
    return np.array(X, dtype=np.float32), np.array(y)


def gen_regime_b(rng, L=100, n_per_class=8, classes=2):
    """Regime B: high-freq high-noise spectra → Rocket favored."""
    X, y = [], []
    for c in range(classes):
        freq_scale = 2.0 if c == 0 else 5.0
        for _ in range(n_per_class):
            t = np.linspace(0, 2 * np.pi, L)
            sig = np.sin(freq_scale * t + rng.uniform(0, 2 * np.pi)) \
                  + 0.5 * rng.standard_normal(L)
            X.append(sig); y.append(c)
    return np.array(X, dtype=np.float32), np.array(y)


def gen_regime_c(rng, L=100, n_per_class=8, classes=2):
    """Regime C: global shape (Gaussian bump vs square) → MOMENT favored."""
    X, y = [], []
    for c in range(classes):
        for _ in range(n_per_class):
            x = np.linspace(-3, 3, L)
            if c == 0:
                sig = np.exp(-x**2)  # Gaussian
            else:
                sig = np.where(np.abs(x) < 1.5, 1.0, 0.0)  # square
            sig = sig + 0.05 * rng.standard_normal(L) + rng.uniform(-0.3, 0.3)
            X.append(sig); y.append(c)
    return np.array(X, dtype=np.float32), np.array(y)


REGIMES = [
    ("A_phase_sine", gen_regime_a),
    ("B_noise_spectrum", gen_regime_b),
    ("C_shape_morphology", gen_regime_c),
]


# ----- Online streaming evaluator ----- #

def evaluate_classifier(name, X_tr, y_tr, X_te, y_te):
    """One-cell predict using the named classifier."""
    try:
        if name == "rocket":
            from research.baseline.tsc_classical import b3_rocket
            yp = b3_rocket(X_tr, y_tr, X_te, num_kernels=200)
        elif name == "moment_1nn":
            from research.baseline.moment_classifier import classify_1nn
            yp = classify_1nn(X_tr, y_tr, X_te)
        elif name == "dtw_1nn":
            from research.baseline.tsc_classical import b1_knn_dtw
            yp = b1_knn_dtw(X_tr, y_tr, X_te)
        else:
            return float("nan")
        return float((yp == y_te).mean())
    except Exception:
        return float("nan")


def online_sim(n_cells_per_regime=5, n_regimes_cycles=3, seed=0):
    """Stream cells across regime A → B → C → A → B → C ..."""
    rng = np.random.default_rng(seed)
    classifiers = ["rocket", "moment_1nn", "dtw_1nn"]

    # Build sequence of (regime, cell_data)
    sequence = []
    for cycle in range(n_regimes_cycles):
        for regime_name, gen_fn in REGIMES:
            for k in range(n_cells_per_regime):
                X_tr, y_tr = gen_fn(rng, L=100, n_per_class=5, classes=2)
                X_te, y_te = gen_fn(rng, L=100, n_per_class=15, classes=2)
                sequence.append({"regime": regime_name, "cycle": cycle, "k": k,
                                  "X_tr": X_tr, "y_tr": y_tr,
                                  "X_te": X_te, "y_te": y_te})

    n_total = len(sequence)
    print(f"\n=== Online sim: {n_total} cells × 3 regimes × {n_regimes_cycles} cycles ===")

    # Strategy 1: Always Rocket
    strat_rocket_acc = []
    # Strategy 2: Snapshot LOO CV (each cell solo, no memory)
    strat_snapshot_acc = []
    strat_snapshot_chosen = []
    # Strategy 3: Online memory (accumulate (cell_features, oracle_winner))
    strat_online_acc = []
    strat_online_chosen = []
    online_memory = []  # list of (avg_features_vec, oracle_winner)

    from research.utils.series_features import featurize_cell

    for i, cell in enumerate(sequence):
        X_tr, y_tr = cell["X_tr"], cell["y_tr"]
        X_te, y_te = cell["X_te"], cell["y_te"]

        # Always Rocket
        strat_rocket_acc.append(evaluate_classifier("rocket", X_tr, y_tr, X_te, y_te))

        # Snapshot LOO CV: try all 3 classifiers, pick by LOO acc
        from sklearn.model_selection import LeaveOneOut
        cv_accs = {}
        for c in classifiers:
            correct = []
            for tr_idx, te_idx in LeaveOneOut().split(X_tr):
                yp = None
                try:
                    if c == "rocket":
                        from research.baseline.tsc_classical import b3_rocket
                        yp = b3_rocket(X_tr[tr_idx], y_tr[tr_idx], X_tr[te_idx], num_kernels=200)
                    elif c == "moment_1nn":
                        from research.baseline.moment_classifier import classify_1nn
                        yp = classify_1nn(X_tr[tr_idx], y_tr[tr_idx], X_tr[te_idx])
                    elif c == "dtw_1nn":
                        from research.baseline.tsc_classical import b1_knn_dtw
                        yp = b1_knn_dtw(X_tr[tr_idx], y_tr[tr_idx], X_tr[te_idx])
                except Exception:
                    pass
                if yp is not None:
                    correct.append(int(yp[0] == y_tr[te_idx[0]]))
            cv_accs[c] = np.mean(correct) if correct else 0
        snapshot_choice = max(cv_accs, key=cv_accs.get)
        snapshot_acc = evaluate_classifier(snapshot_choice, X_tr, y_tr, X_te, y_te)
        strat_snapshot_acc.append(snapshot_acc)
        strat_snapshot_chosen.append(snapshot_choice)

        # Online memory: query past cells with similar features
        feat = featurize_cell(X_tr, y_tr)
        if len(online_memory) >= 3:
            past_feats = np.array([m[0] for m in online_memory])
            past_winners = [m[1] for m in online_memory]
            sims = past_feats @ feat / (np.linalg.norm(past_feats, axis=1) * np.linalg.norm(feat) + 1e-9)
            top_k = np.argsort(-sims)[:5]
            from collections import Counter
            counts = Counter(past_winners[j] for j in top_k)
            online_choice = counts.most_common(1)[0][0]
        else:
            online_choice = "rocket"
        online_acc = evaluate_classifier(online_choice, X_tr, y_tr, X_te, y_te)
        strat_online_acc.append(online_acc)
        strat_online_chosen.append(online_choice)

        # After predict: backfill (which classifier was oracle best?)
        oracle_accs = {c: evaluate_classifier(c, X_tr, y_tr, X_te, y_te) for c in classifiers}
        oracle_winner = max(oracle_accs, key=oracle_accs.get)
        online_memory.append((feat, oracle_winner))

        if i % 5 == 0:
            print(f"  [{i+1:2}/{n_total}] regime={cell['regime']:25} "
                  f"rocket={strat_rocket_acc[-1]:.2f}  "
                  f"snapshot[{snapshot_choice}]={snapshot_acc:.2f}  "
                  f"online[{online_choice}]={online_acc:.2f}  (oracle: {oracle_winner})")

    print(f"\n=== Aggregate over {n_total} streaming cells ===")
    print(f"  Always Rocket:        {np.mean(strat_rocket_acc):.4f}")
    print(f"  Snapshot LOO CV:      {np.mean(strat_snapshot_acc):.4f}")
    print(f"  Online memory:        {np.mean(strat_online_acc):.4f}")

    # Per-regime breakdown
    print(f"\n=== Per-regime mean ===")
    print(f'{"regime":25}  {"rocket":>10} {"snapshot":>10} {"online":>10}')
    for regime_name, _ in REGIMES:
        idx = [i for i, c in enumerate(sequence) if c["regime"] == regime_name]
        rocket_m = np.mean([strat_rocket_acc[i] for i in idx])
        snap_m = np.mean([strat_snapshot_acc[i] for i in idx])
        online_m = np.mean([strat_online_acc[i] for i in idx])
        print(f"  {regime_name:25}  {rocket_m:>10.3f} {snap_m:>10.3f} {online_m:>10.3f}")

    # Per-cycle (does online memory improve over cycles?)
    print(f"\n=== Online learning curve (per regime-cycle) ===")
    for cycle in range(n_regimes_cycles):
        idx = [i for i, c in enumerate(sequence) if c["cycle"] == cycle]
        online_m = np.mean([strat_online_acc[i] for i in idx])
        print(f"  cycle {cycle}: online_mean={online_m:.3f}")

    # Save full trace
    out = Path("research/results/online_routing_sim.jsonl")
    with out.open("w") as fh:
        for i, c in enumerate(sequence):
            fh.write(json.dumps({
                "i": i, "regime": c["regime"], "cycle": c["cycle"], "k": c["k"],
                "rocket_acc": strat_rocket_acc[i],
                "snapshot_acc": strat_snapshot_acc[i], "snapshot_choice": strat_snapshot_chosen[i],
                "online_acc": strat_online_acc[i], "online_choice": strat_online_chosen[i],
            }) + "\n")
    print(f"\nSaved → {out}")


def online_sim_real_ucr(n_cells_per_regime=4, n_cycles=3, seed=0):
    """V2: 用真实 UCR datasets 作 regime sources."""
    from research.utils.ucr_loader import load_ucr_fewshot

    # Datasets where different classifiers win (from task #41 winner-per-cell):
    # BeetleFly N=5 → MOMENT wins; TwoLeadECG → Rocket; Coffee N=3 → tie
    regimes = [
        ("BeetleFly", "MOMENT-favored"),
        ("TwoLeadECG", "Rocket-favored"),
        ("BirdChicken", "MOMENT-favored"),
    ]
    classifiers = ["rocket", "moment_1nn", "dtw_1nn"]

    rng = np.random.default_rng(seed)
    sequence = []
    for cycle in range(n_cycles):
        for ds_name, label in regimes:
            for k in range(n_cells_per_regime):
                cell_seed = int(rng.integers(1, 100000))
                try:
                    X_tr, y_tr, X_te, y_te = load_ucr_fewshot(ds_name, n_per_class=3, seed=cell_seed)
                    if len(X_te) > 30:
                        idx = rng.choice(len(X_te), 30, replace=False)
                        X_te, y_te = X_te[idx], y_te[idx]
                    sequence.append({"regime": ds_name, "label": label,
                                      "cycle": cycle, "k": k,
                                      "X_tr": X_tr, "y_tr": y_tr,
                                      "X_te": X_te, "y_te": y_te})
                except Exception:
                    pass

    n_total = len(sequence)
    print(f"\n=== V2 Online sim (real UCR): {n_total} cells × 3 regimes × {n_cycles} cycles ===")

    from research.utils.series_features import featurize_cell
    rocket_accs, snap_accs, online_accs = [], [], []
    online_memory = []
    for i, cell in enumerate(sequence):
        X_tr, y_tr, X_te, y_te = cell["X_tr"], cell["y_tr"], cell["X_te"], cell["y_te"]
        # Always Rocket
        rocket_accs.append(evaluate_classifier("rocket", X_tr, y_tr, X_te, y_te))
        # Oracle for memory backfill
        oracle = {c: evaluate_classifier(c, X_tr, y_tr, X_te, y_te) for c in classifiers}
        oracle_winner = max(oracle, key=oracle.get)
        # Online memory: query past
        feat = featurize_cell(X_tr, y_tr)
        if len(online_memory) >= 3:
            pf = np.array([m[0] for m in online_memory])
            sims = (pf @ feat) / (np.linalg.norm(pf, axis=1) * np.linalg.norm(feat) + 1e-9)
            top = np.argsort(-sims)[:5]
            from collections import Counter
            choice = Counter(online_memory[j][1] for j in top).most_common(1)[0][0]
        else:
            choice = "rocket"
        online_accs.append(evaluate_classifier(choice, X_tr, y_tr, X_te, y_te))
        online_memory.append((feat, oracle_winner))
        snap_accs.append(oracle[oracle_winner])  # oracle as snapshot upper-bound

        if i % 4 == 0:
            print(f"  [{i+1:2}/{n_total}] {cell['regime']:14} cyc={cell['cycle']} "
                  f"rocket={rocket_accs[-1]:.2f}  oracle[{oracle_winner}]={snap_accs[-1]:.2f}  "
                  f"online[{choice}]={online_accs[-1]:.2f}")

    print(f"\n=== V2 Aggregate (n={n_total}) ===")
    print(f"  Always Rocket:        {np.mean(rocket_accs):.4f}")
    print(f"  Oracle (upper bound): {np.mean(snap_accs):.4f}")
    print(f"  Online memory:        {np.mean(online_accs):.4f}")
    print(f"  Online gain vs Rocket: {(np.mean(online_accs)-np.mean(rocket_accs))*100:+.2f}pp")
    print(f"  Online gap to oracle:  {(np.mean(snap_accs)-np.mean(online_accs))*100:+.2f}pp")

    print(f"\n=== Per-cycle online learning curve ===")
    for cyc in range(n_cycles):
        idx = [i for i, c in enumerate(sequence) if c["cycle"] == cyc]
        print(f"  cycle {cyc}: rocket={np.mean([rocket_accs[i] for i in idx]):.3f}  "
              f"online={np.mean([online_accs[i] for i in idx]):.3f}  "
              f"oracle={np.mean([snap_accs[i] for i in idx]):.3f}")


if __name__ == "__main__":
    print("\n========== V1 Synthetic regimes (too easy, all-Rocket) ==========")
    online_sim(n_cells_per_regime=4, n_regimes_cycles=3, seed=0)
    print("\n========== V2 Real UCR regimes ==========")
    online_sim_real_ucr(n_cells_per_regime=4, n_cycles=3, seed=0)
