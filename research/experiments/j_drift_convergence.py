"""Round 6 R6-J · Long walk-forward drift-convergence test.

Demos showed drift_engine *signals are silent* on short runs because the
default window (recent=30, history=90 ⇒ ≥120 obs) is far above demo length.
This script walks ETTh1 long enough (n_steps ≥ 140) for drift to fire,
optionally injecting a fault midway so the new `pred_residual_z` signal
(F-R6.1 fix) actually crosses threshold and triggers remediation.

Two modes:
    clean    · canonical ETTh1, see whether natural variation crosses thresholds
    injected · variance_explode at mid-walk → expect drift signals to fire

Each step records:
    state.memory_trust / explore_scale / regime_stale
    drift_history length, signals, actions

Pipeline reuses `g_real_demo.run(record_state_trace=True)`.

Run:
    mamba run -n tsci python -m research.experiments.j_drift_convergence
    mamba run -n tsci python -m research.experiments.j_drift_convergence --mode injected
"""
from __future__ import annotations
import argparse
import time
import numpy as np

from research.utils.data_loader import load_series
from research.utils.inject_fault import inject_variance_explode
from research.experiments.g_real_demo import run as run_pipeline
from research.agent.router_state import reset_state


def _inject_midwalk(series: np.ndarray, warmup: int, n_steps: int, step: int,
                    ratio: float = 4.0, seed: int = 0) -> np.ndarray:
    """Apply variance_explode to a window that overlaps the *latter half*
    of the walk-forward path. The injector itself splits the input at its
    midpoint, so we pass the slice covering [warmup, warmup+n_steps*step]
    and graft the modified portion back onto the canonical series."""
    rng = np.random.default_rng(seed)
    out = series.copy()
    walk_lo = warmup
    walk_hi = min(len(series), warmup + n_steps * step + 50)
    chunk = out[walk_lo:walk_hi]
    if len(chunk) < 40:
        return out
    modded = inject_variance_explode(chunk, rng, ratio=ratio)
    out[walk_lo:walk_hi] = modded
    return out


def run_convergence(mode: str = "clean",
                    dataset: str = "ETTh1",
                    n_steps: int = 140,
                    warmup: int = 100,
                    step: int = 30,
                    H: int = 24,
                    drift_check_every: int = 10,
                    drift_min_observations: int = 30,
                    seed: int = 0,
                    verbose: bool = True) -> dict:
    series, meta = load_series(dataset)
    series = series.astype(np.float64)
    target_len = warmup + n_steps * step + H + 50
    series = series[:target_len]
    threshold = float(np.quantile(series[:warmup], 0.95))

    if mode == "injected":
        series = _inject_midwalk(series, warmup, n_steps, step,
                                  ratio=4.0, seed=seed)
        label = f"injected variance_explode×4"
    else:
        label = "clean"

    reset_state()
    state_path = f"research/results/router_state_j_{mode}.jsonl"
    # Use cheap candidates so 140 steps complete in < 2 min.
    cheap_candidates = ["naive_drift", "arima_ets", "chronos_bolt"]

    t0 = time.time()
    res = run_pipeline(
        dataset=dataset, H=H, n_steps=n_steps,
        warmup=warmup, step=step,
        upper_threshold=threshold,
        state_path=state_path,
        series_override=series,
        label=label,
        drift_check_every=drift_check_every,
        drift_min_observations=drift_min_observations,
        candidate_models=cheap_candidates,
        verbose=False,
        record_state_trace=True,
    )
    elapsed = time.time() - t0

    tl = res["timeline"]
    state = res["state"]

    if verbose:
        print("=" * 100)
        print(f"R6-J drift convergence · {dataset} [{label}]  "
              f"n_steps={len(tl)}  every={drift_check_every}  "
              f"threshold={threshold:.2f}")
        print("=" * 100)

        # 1. Drift event detail (these are the *interesting* rows)
        events = [r for r in tl if r["drift_fired"]]
        print(f"\nDrift events fired: {len(events)} / {len(tl)} steps")
        if events:
            print(f"\n{'t':>5} {'n_obs':>5} {'feat':>5} {'res_ks':>6} "
                  f"{'rout':>5} {'mem_mm':>6} {'pred_z':>6}  "
                  f"detected → actions")
            print("-" * 100)
            for r in events:
                sig = r.get("drift_signals", {})
                det = r.get("drift_detected", {})
                acts = r.get("drift_actions", [])
                fired = [k for k, v in det.items() if v]
                print(f"{r['t']:>5} {r['state_drift_history_len']:>5} "
                      f"{sig.get('feature_kl', 0):>5.2f} "
                      f"{sig.get('residual_ks', 0):>6.2f} "
                      f"{sig.get('routing_kl', 0):>5.2f} "
                      f"{sig.get('memory_mismatch', 0):>6.2f} "
                      f"{sig.get('pred_residual_z', 0):>6.2f}  "
                      f"{','.join(fired):<30} → {','.join(acts)}")

        # 2. State trace at sampled points
        print(f"\nState trace (sampled every {max(1, len(tl)//10)} steps):")
        print(f"{'step':>5} {'t':>5} {'trust':>6} {'explore':>8} "
              f"{'stale':>5} {'drift_n':>7}  {'interv':>10}")
        sample_ix = list(range(0, len(tl), max(1, len(tl) // 10)))
        if len(tl) - 1 not in sample_ix:
            sample_ix.append(len(tl) - 1)
        for ix in sample_ix:
            r = tl[ix]
            print(f"{ix:>5} {r['t']:>5} "
                  f"{r['state_memory_trust']:>6.2f} "
                  f"{r['state_explore_scale']:>8.2f} "
                  f"{str(r['state_regime_stale']):>5} "
                  f"{r['state_drift_history_len']:>7}  "
                  f"{r['intervention']:>10}")

        # 3. Aggregates
        from collections import Counter
        cnt = Counter(r["intervention"] for r in tl)
        n_fb = sum(1 for r in tl if r["fallback_used"])
        mae = float(np.mean([r["outcome_mae"] for r in tl]))
        print(f"\nIntervention counts: {dict(cnt)}")
        print(f"Mean h=1 MAE: {mae:.3f}")
        print(f"safe_predict fallbacks: {n_fb} steps")
        print(f"Total drift events logged: {len(state.drift_history)}")
        print(f"Final state · memory_trust={getattr(state, 'memory_trust', 1.0):.2f}  "
              f"explore_scale={getattr(state, 'bandit_explore_scale', 1.0):.2f}  "
              f"regime_stale={getattr(state, 'regime_stale', False)}")
        print(f"\n(elapsed: {elapsed:.1f}s)")

    reset_state()
    return {"timeline": tl, "n_drift_events": len(state.drift_history),
            "final_memory_trust": float(getattr(state, "memory_trust", 1.0)),
            "final_explore_scale": float(getattr(state, "bandit_explore_scale", 1.0)),
            "elapsed_s": elapsed}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["clean", "injected", "both"],
                    default="both")
    ap.add_argument("--dataset", default="ETTh1")
    ap.add_argument("--n_steps", type=int, default=140)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--step", type=int, default=30)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.mode == "both":
        results = {}
        for m in ("clean", "injected"):
            print()
            results[m] = run_convergence(
                mode=m, dataset=args.dataset, n_steps=args.n_steps,
                warmup=args.warmup, step=args.step,
                drift_check_every=args.every, seed=args.seed)
        print("\n" + "=" * 100)
        print("CLEAN vs INJECTED · summary")
        print("=" * 100)
        for m, r in results.items():
            print(f"  {m:>9}: n_drift_events={r['n_drift_events']:>3}  "
                  f"trust={r['final_memory_trust']:.2f}  "
                  f"explore={r['final_explore_scale']:.2f}  "
                  f"elapsed={r['elapsed_s']:.1f}s")
    else:
        run_convergence(mode=args.mode, dataset=args.dataset,
                         n_steps=args.n_steps, warmup=args.warmup,
                         step=args.step, drift_check_every=args.every,
                         seed=args.seed)
