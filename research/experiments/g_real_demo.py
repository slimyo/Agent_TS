"""Round 6 R6-G demo · End-to-end real-data run wiring B2 + B3 + E1 + R6-E.

Pipeline executed at each walk-forward step:

    adaptive_decide          (R6-A)  · posterior over models, telemetry log
        ↓
    schedule_from_state      (R6-E)  · pick a subset to actually run,
                                       respect latency / vram / remote budgets
        ↓
    safe_predict each in run_list   · circuit-breaker + naive fallback
        ↓
    ensemble → ForecastDist (mean, std)
        ↓
    decide_from_router       (E1+B2) · risk → cost-min → intervention,
                                       calibrated confidence pulled from state
        ↓
    adaptive_observe         (R6-A)  · close bandit/memory loop
                                       (auto drift_step every N obs)

Data: ETTh1 OT column, ~17 k hourly samples. We walk forward from N=200 in
steps of 100, predicting H=24 each time, for ≤ `n_steps` decisions.

Run:
    mamba run -n tsci python -m research.experiments.g_real_demo
    mamba run -n tsci python -m research.experiments.g_real_demo --n_steps 8 --H 24
"""
from __future__ import annotations
import argparse
import time
import numpy as np

from research.utils.data_loader import load_series
from research.agent.router_state import RouterState, get_state, persist_state
from research.agent.bayesian_router import (
    RouterConfig, AvailabilityPrior, NPrior, TypePrior, CRPSPrior)
from research.agent.adaptive_planner import adaptive_decide, adaptive_observe
from research.agent.inference_scheduler import (
    schedule_from_state, default_profiles, SchedulerConfig)
from research.agent.action_layer import (
    ForecastDist, ActionContext, ActionConfig, decide_from_router)
from research.agent.forecaster_reflect import STRATEGY_FN
from research.agent.safe_predict import safe_predict


def _ensemble_pred(plan, run_list: list[str],
                   train: np.ndarray, val: np.ndarray, H: int, season_m: int
                   ) -> tuple[float, float, dict[str, np.ndarray], dict[str, bool]]:
    """Run each model in `run_list` safely; ensemble via posterior weights.

    Returns (mean[0], std_across_models, per_model_preds, fallback_flags).
    """
    preds: dict[str, np.ndarray] = {}
    fallback: dict[str, bool] = {}

    def _raw(m: str) -> np.ndarray:
        return STRATEGY_FN[m](train, val, H, season_m)

    for m in run_list:
        if m not in STRATEGY_FN:
            continue
        res = safe_predict(
            model_name=m, predict_fn=_raw,
            H=H, history=train,
            fallback_model="naive_drift", fallback_predict_fn=_raw,
            register_outcome=False,
        )
        preds[m] = res.pred
        fallback[m] = bool(res.fallback_used)

    # Residual-std fallback: rolling first-difference of recent train tail.
    # Always computed so single-model case still has a meaningful uncertainty.
    if len(train) >= 25:
        resid_std = float(np.std(np.diff(train[-50:])))
    else:
        resid_std = float(np.std(train) + 1e-6)

    if not preds:
        return float(train[-1]), resid_std, {}, {}

    # Posterior-weighted ensemble at h=1 (first step ahead)
    post = dict(plan.posterior)
    weights = np.array([post.get(m, 1e-6) for m in preds])
    weights = weights / weights.sum()
    stack = np.stack([preds[m] for m in preds], axis=0)   # [M, H]
    ens   = (weights[:, None] * stack).sum(axis=0)        # [H]
    # Combine cross-model disagreement with residual std (RSS)
    cross_std = float(np.std(stack[:, 0]) + 1e-6)
    eff_std = float(np.sqrt(cross_std**2 + resid_std**2))
    return float(ens[0]), eff_std, preds, fallback


def run(dataset: str = "ETTh1", H: int = 24, n_steps: int = 8,
        warmup: int = 200, step: int = 100,
        upper_threshold: float | None = None,
        state_path: str = "research/results/router_state_g_demo.jsonl",
        series_override: np.ndarray | None = None,
        label: str = "",
        drift_check_every: int = 3,
        drift_min_observations: int = 3,
        candidate_models: list[str] | None = None,
        verbose: bool = True,
        record_state_trace: bool = False) -> dict:
    series, meta = load_series(dataset)
    series = series.astype(np.float64)
    if series_override is not None:
        # Caller-provided series (e.g. fault-injected / truncated variant).
        # Must contain at least warmup + n_steps*step + H samples to walk.
        if len(series_override) < warmup + H + 1:
            raise ValueError(f"series_override length {len(series_override)} "
                             f"too short for warmup={warmup} + H={H}")
        series = series_override.astype(np.float64)
    if upper_threshold is None:
        # 95th percentile of warm-up window as threshold
        upper_threshold = float(np.quantile(series[:warmup], 0.95))

    # Fresh state for the demo so we don't pollute a long-running router_state
    import os
    if os.path.exists(state_path): os.remove(state_path)
    state = get_state(state_path)

    candidates = candidate_models or [m for m in
                  ["naive_drift", "naive_seasonal", "arima_ets",
                   "chronos2", "chronos_bolt"]
                  if m in STRATEGY_FN]
    cfg_router = RouterConfig(
        priors=[
            AvailabilityPrior(local_models=tuple(candidates), remote_models=()),
            CRPSPrior(),
            NPrior(default_model="chronos2", N_threshold=15, strength=2.0),
            TypePrior(),
        ],
        decide_mode="argmax", embedding_name="hand25",
        enable_bandit=True,
        drift_check_every=drift_check_every,
        drift_min_observations=drift_min_observations,
    )
    cfg_action = ActionConfig()
    cfg_sched  = SchedulerConfig(latency_budget_s=3.0, vram_budget_gb=8.0,
                                  max_models=3)
    profiles = default_profiles()
    season_m = meta.season_m

    timeline = []
    t = warmup
    for k in range(n_steps):
        if t + H > len(series): break
        train = series[:t]
        truth_window = series[t:t + H]

        # 1. Route
        plan = adaptive_decide(
            "forecast", train.astype(np.float32),
            candidates=candidates, config=cfg_router, state=state,
            dataset=dataset, N=t, H=H,
        )

        # 2. Scheduler (uses B2 calibrator via state)
        exec_plan = schedule_from_state(plan, state, cfg_sched, profiles)
        run_list = exec_plan.run_list()

        # 3. Run + ensemble
        val_proxy = np.array([])    # forecast_reflect strategies accept empty val
        mean, std, preds, fbk = _ensemble_pred(
            plan, run_list, train, val_proxy, H, season_m)

        # 4. Action layer
        ctx = ActionContext(upper_threshold=upper_threshold,
                            current_value=float(train[-1]),
                            asset_id=dataset, horizon_steps=H)
        fcst = ForecastDist(mean=mean, std=std, horizon_steps=H)
        decision = decide_from_router(fcst, ctx, state=state,
                                       config=cfg_action,
                                       refit_every=5,
                                       min_obs_for_calibration=5)

        # 5. Observe
        outcome = float(np.mean(np.abs(truth_window[:1] - mean)))
        obs_res = adaptive_observe(state, plan, outcome)

        rec = {
            "t": t, "truth_t": float(truth_window[0]),
            "ensemble_mean": mean, "ensemble_std": std,
            "p_breach": decision.risk.p_breach,
            "conf": decision.confidence,
            "intervention": decision.intervention,
            "chosen_top1": plan.chosen,
            "scheduler_run": run_list,
            "scheduler_skip": [s.model for s in exec_plan.steps
                                if s.action == "skip"],
            "fallback_used": [m for m, v in fbk.items() if v],
            "outcome_mae": outcome,
            "drift_fired": "drift" in obs_res,
        }
        if record_state_trace:
            rec["state_memory_trust"]  = float(getattr(state, "memory_trust", 1.0))
            rec["state_explore_scale"] = float(getattr(state, "bandit_explore_scale", 1.0))
            rec["state_regime_stale"]  = bool(getattr(state, "regime_stale", False))
            rec["state_drift_history_len"] = len(state.drift_history)
            if "drift" in obs_res:
                sig = obs_res["drift"].get("signals", {})
                rec["drift_signals"] = {
                    "feature_kl":      sig.get("feature_kl"),
                    "residual_ks":     sig.get("residual_ks"),
                    "routing_kl":      sig.get("routing_kl"),
                    "memory_mismatch": sig.get("memory_mismatch"),
                    "pred_residual_z": sig.get("pred_residual_z"),
                }
                rec["drift_detected"] = sig.get("detected", {})
                rec["drift_actions"]  = [a["kind"] for a in obs_res["drift"].get("actions", [])]
        timeline.append(rec)
        t += step

    persist_state()

    if verbose:
        print("=" * 96)
        suffix = f" [{label}]" if label else ""
        print(f"R6-G real-data demo · dataset={dataset}{suffix}  H={H}  threshold={upper_threshold:.2f}")
        print(f"  walking from t={warmup} step={step} for {len(timeline)} decisions")
        print("=" * 96)
        print(f"{'t':>5} {'truth':>7} {'pred':>7} {'std':>5} "
              f"{'p_br':>5} {'conf':>5} {'top1':>14} "
              f"{'scheduled':>30}  decision")
        print("-" * 96)
        for r in timeline:
            mark = "  *drift*" if r["drift_fired"] else ""
            fb = f" [fbk:{','.join(r['fallback_used'])}]" if r["fallback_used"] else ""
            run_str = ",".join(r["scheduler_run"])
            print(f"{r['t']:>5} {r['truth_t']:>7.2f} {r['ensemble_mean']:>7.2f} "
                  f"{r['ensemble_std']:>5.2f} {r['p_breach']:>5.2f} "
                  f"{r['conf']:>5.2f} {r['chosen_top1']:>14} "
                  f"{run_str:>30}  {r['intervention']}{mark}{fb}")

        print("\n" + "=" * 96)
        from collections import Counter
        cnt = Counter(r["intervention"] for r in timeline)
        sched_cnt = Counter(tuple(r["scheduler_run"]) for r in timeline)
        print(f"Intervention counts: {dict(cnt)}")
        print(f"Scheduler run-list distribution (top-5):")
        for cmb, n in sched_cnt.most_common(5):
            print(f"  {list(cmb)}: {n}")
        n_fb = sum(1 for r in timeline if r["fallback_used"])
        print(f"Safe-predict fallbacks: {n_fb} steps")
        print(f"Drift events: {sum(1 for r in timeline if r['drift_fired'])}")
        mae = np.mean([r["outcome_mae"] for r in timeline])
        print(f"Mean h=1 MAE: {mae:.3f}")
        print(f"State at end: n_decisions={state.n_decisions}  "
              f"n_obs={state.n_observations}  "
              f"memory_trust={getattr(state, 'memory_trust', 1.0):.2f}  "
              f"explore_scale={getattr(state, 'bandit_explore_scale', 1.0):.2f}")
        print(f"  drift_history len: {len(state.drift_history)}")

    return {"timeline": timeline, "state": state}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ETTh1")
    ap.add_argument("--H", type=int, default=24)
    ap.add_argument("--n_steps", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--step", type=int, default=100)
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()
    t0 = time.time()
    run(dataset=args.dataset, H=args.H, n_steps=args.n_steps,
        warmup=args.warmup, step=args.step,
        upper_threshold=args.threshold)
    print(f"\n(total elapsed: {time.time() - t0:.1f}s)")
