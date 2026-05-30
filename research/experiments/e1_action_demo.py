"""Round 6 E1 demo · End-to-end forecast → action → observe loop.

Walks a synthetic series forward in time and at each step exercises the
*full* Round 6 pipeline:

    adaptive_decide  (route, log telemetry, attach drift-aware factors)
        ↓
    cheap forecast   (mean + std from naive_drift / EMA — task-agnostic)
        ↓
    decide_from_router (raw conf → calibrated → risk → cost-min → intervention)
        ↓
    adaptive_observe (bandit + memory update, auto drift_step every N)
        ↓
    optional drift refit + memory_trust / explore_scale propagation

The series has two phases:
    [0, T_drift)         stable around setpoint=50  → expect MONITOR
    [T_drift, T_end)     drifts upward toward 110   → expect rising
                                                       INSPECT → THROTTLE → SHUTDOWN

Run:
    mamba run -n tsci python -m research.experiments.e1_action_demo
"""
from __future__ import annotations
import argparse
import time
import numpy as np

from research.agent.router_state import RouterState
from research.agent.bayesian_router import RouterConfig
from research.agent.adaptive_planner import adaptive_decide, adaptive_observe
from research.agent.action_layer import (
    ForecastDist, ActionContext, ActionConfig, decide_from_router)


# ─── synthetic data ──────────────────────────────────────────────────────────


def make_series(T_total: int = 240, T_drift: int = 150,
                seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty(T_total, dtype=np.float64)
    out[:T_drift] = 50.0 + 2.0 * rng.standard_normal(T_drift)
    # Linear ramp 50 → 110 over the drift window
    ramp = np.linspace(50.0, 110.0, T_total - T_drift)
    out[T_drift:] = ramp + 3.0 * rng.standard_normal(T_total - T_drift)
    return out


# ─── lightweight forecast (no TSFM dep) ─────────────────────────────────────


def cheap_forecast(history: np.ndarray, h: int = 1) -> ForecastDist:
    """EMA + residual-std → (mean, std) — stand-in for an actual model."""
    if len(history) < 5:
        return ForecastDist(mean=float(history[-1]), std=1.0, horizon_steps=h)
    # exponential moving average
    alpha = 0.3
    s = float(history[0])
    for x in history[1:]:
        s = alpha * x + (1 - alpha) * s
    resid = history - np.array([s] * len(history))
    sigma = float(np.std(resid[-30:]) + 0.5)
    return ForecastDist(mean=s, std=sigma, horizon_steps=h)


# ─── demo loop ───────────────────────────────────────────────────────────────


def run(T_total: int = 240, T_drift: int = 150, warmup: int = 40,
        decision_every: int = 2, upper_threshold: float = 100.0,
        seed: int = 0, verbose: bool = True) -> dict:
    series = make_series(T_total, T_drift, seed=seed)
    state = RouterState()

    cfg_router = RouterConfig(
        decide_mode="argmax", embedding_name="hand25",
        enable_bandit=True,
        drift_check_every=10, drift_min_observations=20,
    )
    cfg_action = ActionConfig()
    candidates = ["chronos2", "naive_drift", "arima_ets"]

    # Counters
    counts = {"MONITOR": 0, "INSPECT": 0, "THROTTLE": 0,
              "SHUTDOWN": 0, "ESCALATE": 0}
    timeline = []

    for t in range(warmup, T_total - 1, decision_every):
        history = series[:t]
        truth_next = float(series[t])    # outcome to observe at this step

        # 1. Route (logs telemetry; takes drift state into account)
        plan = adaptive_decide(
            "forecast", history, candidates, cfg_router, state,
            dataset="demo_pump", N=t, H=1,
        )

        # 2. Cheap forecast distribution
        fcst = cheap_forecast(history, h=1)

        # 3. Action decision (uses calibrated confidence pulled from state)
        ctx = ActionContext(upper_threshold=upper_threshold,
                            current_value=float(history[-1]),
                            asset_id="demo_pump", horizon_steps=1)
        decision = decide_from_router(fcst, ctx, state=state,
                                       config=cfg_action,
                                       refit_every=25,
                                       min_obs_for_calibration=20)
        counts[decision.intervention] += 1

        # 4. Observe outcome (closes bandit + memory loop, may trigger drift_step)
        outcome = float(abs(truth_next - fcst.mean))   # simple MAE proxy
        observe_res = adaptive_observe(state, plan, outcome)

        timeline.append({
            "t": t, "true": truth_next, "fcst_mean": fcst.mean,
            "fcst_std": fcst.std, "p_breach": decision.risk.p_breach,
            "conf": decision.confidence,
            "intervention": decision.intervention,
            "chose_model": plan.chosen,
            "memory_trust": getattr(state, "memory_trust", 1.0),
            "explore_scale": getattr(state, "bandit_explore_scale", 1.0),
            "drift_fired": "drift" in observe_res,
        })

    if verbose:
        print("=" * 78)
        print(f"Round 6 E1 demo · T_total={T_total} T_drift={T_drift} "
              f"warmup={warmup} every={decision_every}")
        print("=" * 78)
        print(f"{'t':>4} {'true':>7} {'mean':>7} {'std':>5} "
              f"{'p_br':>5} {'conf':>5} {'mem':>4} {'exp':>4} "
              f"{'model':>12}  decision")
        print("-" * 78)
        # show first 5 + last 10 + drift events
        flagged = [r for r in timeline if r["drift_fired"]]
        sample = (timeline[:5] +
                  [{"t": "...", **{k: "" for k in timeline[0] if k != "t"}}] +
                  flagged +
                  timeline[-10:])
        seen = set()
        for r in sample:
            key = r.get("t", "?")
            if key in seen: continue
            seen.add(key)
            if isinstance(key, str):
                print(f"{key:>4}")
                continue
            mark = "  *drift*" if r["drift_fired"] else ""
            print(f"{r['t']:>4} {r['true']:>7.2f} {r['fcst_mean']:>7.2f} "
                  f"{r['fcst_std']:>5.2f} {r['p_breach']:>5.2f} "
                  f"{r['conf']:>5.2f} {r['memory_trust']:>4.2f} "
                  f"{r['explore_scale']:>4.2f} {r['chose_model']:>12}  "
                  f"{r['intervention']}{mark}")

        print("\n" + "=" * 78)
        print(f"Intervention counts: {counts}")
        print(f"Drift events fired: {sum(1 for r in timeline if r['drift_fired'])}")
        print(f"State at end: n_decisions={state.n_decisions}  "
              f"n_observations={state.n_observations}")
        print(f"  memory_trust={getattr(state, 'memory_trust', 1.0):.2f}  "
              f"explore_scale={getattr(state, 'bandit_explore_scale', 1.0):.2f}  "
              f"regime_stale={getattr(state, 'regime_stale', False)}")
        print(f"  drift_history len: {len(state.drift_history)}")

    return {"counts": counts, "timeline": timeline, "state": state}


# ─── entry ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--T_total", type=int, default=240)
    ap.add_argument("--T_drift", type=int, default=150)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--every", type=int, default=2,
                    help="decision step interval")
    ap.add_argument("--threshold", type=float, default=100.0,
                    help="upper threshold for breach probability")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run(T_total=args.T_total, T_drift=args.T_drift, warmup=args.warmup,
        decision_every=args.every, upper_threshold=args.threshold,
        seed=args.seed)
