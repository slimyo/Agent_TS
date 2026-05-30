"""Round 6 R6-E · Inference Scheduler (feedback 后§4 "Runtime Orchestrator").

Turns a posterior over candidate models into an ordered execution plan that
respects latency / VRAM / remote budgets and stops escalating once the
*marginal utility* of running the next model goes negative.

Utility for adding model M as the next inference:
    U(M) = accuracy_gain(M) · (1 − current_confidence)
         − w_lat   · latency(M)
         − w_vram  · vram(M)
         − (remote_penalty if M is remote else 0)

Reading:
  - `(1 − current_confidence)` is the "headroom": if you're already 90 %
    sure, an extra model can claim at most 10 % of its accuracy_gain
  - latency / VRAM enter linearly; weights have units of "accuracy per s"
    and "accuracy per GB" so the scheduler is task-agnostic once weights
    are calibrated for the deployment

Schedule loop:
  1. rank candidates by posterior (descending)
  2. add top-1 unconditionally  (you have to run *something*)
  3. for each subsequent candidate compute U
     – if U > 0 AND budget remaining → mark "run"
     – else → mark "skip" and stop escalating
  4. After each accepted model, the planner re-estimates confidence
     using a simple posterior_mass + agreement bonus model

Integration:
  - `schedule_from_state(state, plan, ...)` pulls posterior from plan.top_k
    and confidence from `state._calibrator` (Round 6 B2)
  - `default_profiles()` returns sensible (latency_s, vram_gb) numbers for
    the existing forecast model library; override per deployment
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional


# ─── data classes ────────────────────────────────────────────────────────────


@dataclass
class ModelProfile:
    """Per-model runtime characteristics. Override defaults for your env."""
    name:             str
    est_latency_s:    float = 0.5     # wall-clock per inference
    est_vram_gb:      float = 1.0
    est_accuracy_gain: float = 0.05   # expected ΔP(correct) vs current best
    is_remote:        bool = False


@dataclass
class SchedulerConfig:
    w_latency:         float = 0.02    # utility per second cost
    w_vram:            float = 0.01    # utility per GB cost
    remote_penalty:    float = 0.10    # flat penalty on any remote model
    latency_budget_s:  float = 5.0     # total wall-clock budget for whole plan
    vram_budget_gb:    float = 16.0    # peak VRAM budget (max across plan)
    max_models:        int = 4         # upper bound on cascade depth
    agreement_bonus:   float = 0.05    # confidence bump per agreeing model
    min_confidence_stop: float = 0.95  # early-stop if confidence already high


@dataclass
class ExecutionStep:
    model:       str
    action:      str           # "run" | "skip"
    utility:     float
    posterior:   float         # posterior mass at scheduling time
    est_latency_s: float
    est_vram_gb:   float
    is_remote:   bool
    reason:      str


@dataclass
class ExecutionPlan:
    steps: list[ExecutionStep] = field(default_factory=list)
    total_latency_s: float = 0.0
    peak_vram_gb:    float = 0.0
    final_confidence: float = 0.0
    early_stop:      bool = False

    def run_list(self) -> list[str]:
        return [s.model for s in self.steps if s.action == "run"]

    def summary(self) -> dict:
        return {
            "run":   self.run_list(),
            "skip":  [s.model for s in self.steps if s.action == "skip"],
            "total_latency_s": round(self.total_latency_s, 3),
            "peak_vram_gb":    round(self.peak_vram_gb, 3),
            "final_confidence": round(self.final_confidence, 3),
            "early_stop":      self.early_stop,
        }


# ─── core scheduler ──────────────────────────────────────────────────────────


def _utility(profile: ModelProfile, current_conf: float,
             config: SchedulerConfig) -> float:
    headroom = max(0.0, 1.0 - current_conf)
    u = profile.est_accuracy_gain * headroom \
        - config.w_latency * profile.est_latency_s \
        - config.w_vram    * profile.est_vram_gb
    if profile.is_remote:
        u -= config.remote_penalty
    return u


def schedule(posterior: dict[str, float],
             profiles: dict[str, ModelProfile],
             config: Optional[SchedulerConfig] = None,
             initial_confidence: float = 0.0) -> ExecutionPlan:
    """Build an ExecutionPlan from posterior + per-model profiles.

    Args:
        posterior: {model: posterior_mass} from BayesianRouter
        profiles:  per-candidate ModelProfile (missing names get defaults)
        config:    SchedulerConfig (defaults are illustrative)
        initial_confidence: prior confidence before any model runs
                           (e.g. from B2 calibrator)
    """
    if config is None: config = SchedulerConfig()
    plan = ExecutionPlan(final_confidence=initial_confidence)
    if not posterior:
        return plan

    ranked = sorted(posterior.items(), key=lambda kv: -kv[1])
    current_conf = float(initial_confidence)
    elapsed_lat = 0.0
    peak_vram = 0.0

    for i, (name, mass) in enumerate(ranked[:config.max_models]):
        prof = profiles.get(name, ModelProfile(name=name))
        u = _utility(prof, current_conf, config)

        # First slot: always run — we need at least one inference.
        if i == 0:
            step = ExecutionStep(
                model=name, action="run", utility=u, posterior=float(mass),
                est_latency_s=prof.est_latency_s, est_vram_gb=prof.est_vram_gb,
                is_remote=prof.is_remote, reason="top-1 mandatory",
            )
            plan.steps.append(step)
            elapsed_lat += prof.est_latency_s
            peak_vram = max(peak_vram, prof.est_vram_gb)
            # bump confidence by posterior mass of top-1 (raw) + small
            current_conf = max(current_conf, float(mass))
            continue

        # Subsequent slots: gated on utility + budgets + confidence
        reason_skip = None
        if current_conf >= config.min_confidence_stop:
            reason_skip = f"confidence already {current_conf:.2f} ≥ {config.min_confidence_stop}"
        elif elapsed_lat + prof.est_latency_s > config.latency_budget_s:
            reason_skip = (f"latency budget exceeded "
                            f"({elapsed_lat + prof.est_latency_s:.2f}s "
                            f"> {config.latency_budget_s}s)")
        elif max(peak_vram, prof.est_vram_gb) > config.vram_budget_gb:
            reason_skip = (f"vram budget exceeded "
                            f"({prof.est_vram_gb} > {config.vram_budget_gb}GB)")
        elif u <= 0:
            reason_skip = f"utility {u:.3f} ≤ 0 (escalation not worth cost)"

        if reason_skip is not None:
            plan.steps.append(ExecutionStep(
                model=name, action="skip", utility=u, posterior=float(mass),
                est_latency_s=prof.est_latency_s, est_vram_gb=prof.est_vram_gb,
                is_remote=prof.is_remote, reason=reason_skip,
            ))
            plan.early_stop = True
            break

        plan.steps.append(ExecutionStep(
            model=name, action="run", utility=u, posterior=float(mass),
            est_latency_s=prof.est_latency_s, est_vram_gb=prof.est_vram_gb,
            is_remote=prof.is_remote,
            reason=f"utility {u:.3f} > 0, headroom={1-current_conf:.2f}",
        ))
        elapsed_lat += prof.est_latency_s
        peak_vram = max(peak_vram, prof.est_vram_gb)
        # Each agreeing run bumps confidence by `agreement_bonus`
        current_conf = min(1.0, current_conf + config.agreement_bonus)

    plan.total_latency_s = elapsed_lat
    plan.peak_vram_gb    = peak_vram
    plan.final_confidence = current_conf
    return plan


# ─── defaults for the existing forecast library ──────────────────────────────


def default_profiles() -> dict[str, ModelProfile]:
    """Illustrative per-model defaults — override per deployment.

    Numbers are educated guesses; tune via offline measurement.
    """
    P = ModelProfile
    return {
        # Local cheap baselines
        "naive_drift":    P("naive_drift",    0.01,  0.0, 0.00, False),
        "naive_seasonal": P("naive_seasonal", 0.01,  0.0, 0.00, False),
        "arima_ets":      P("arima_ets",      0.30,  0.0, 0.02, False),
        # Local TSFMs (mid-tier)
        "chronos":        P("chronos",        0.40,  2.0, 0.04, False),
        "chronos_small":  P("chronos_small",  0.30,  1.5, 0.03, False),
        # Local heavy TSFMs
        "chronos2":       P("chronos2",       0.80,  4.0, 0.12, False),
        "tirex":          P("tirex",          0.70,  3.5, 0.10, False),
        "toto":           P("toto",           1.20,  6.0, 0.10, False),
        "timesfm2":       P("timesfm2",       1.50,  8.0, 0.09, False),
        "moirai":         P("moirai",         1.30,  6.5, 0.08, False),
        "moirai2":        P("moirai2",        1.40,  7.0, 0.08, False),
        "time_moe":       P("time_moe",       2.00, 10.0, 0.08, False),
        # Remote
        "llmtime":        P("llmtime",        4.00,  0.0, 0.03, True),
        "sundial":        P("sundial",        3.00,  0.0, 0.05, True),
        "timer":          P("timer",          3.00,  0.0, 0.05, True),
    }


# ─── integration with RouterState + AdaptivePlan ─────────────────────────────


def schedule_from_state(plan, state, config: Optional[SchedulerConfig] = None,
                        profiles: Optional[dict[str, ModelProfile]] = None
                        ) -> ExecutionPlan:
    """Convenience: build execution plan from an AdaptivePlan + RouterState.

    Confidence prior comes from B2 calibrator if available; otherwise the raw
    posterior_max is used.
    """
    profiles = profiles or default_profiles()
    posterior = dict(plan.posterior) if hasattr(plan, "posterior") else {}
    raw_conf = max(posterior.values()) if posterior else 0.0
    init_conf = raw_conf
    cal = getattr(state, "_calibrator", None) if state is not None else None
    if cal is not None and cal.fit_info is not None:
        try:
            init_conf = float(cal.calibrate(raw_conf))
        except Exception:
            pass
    return schedule(posterior, profiles, config=config,
                    initial_confidence=init_conf)


# ─── smoke ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Round 6 R6-E · Inference Scheduler smoke")
    print("=" * 60)

    profiles = default_profiles()
    cfg = SchedulerConfig()

    # Case 1: confident posterior (top-1 dominates) → expect early stop
    print("\n[Case 1: confident posterior — top-1 dominates]")
    post = {"chronos2": 0.92, "tirex": 0.05, "toto": 0.02, "naive_drift": 0.01}
    p = schedule(post, profiles, cfg, initial_confidence=0.0)
    for s in p.steps:
        print(f"  {s.action:>4} {s.model:<14} U={s.utility:+.3f} "
              f"post={s.posterior:.2f} lat={s.est_latency_s:.2f}s "
              f"vram={s.est_vram_gb:.1f}GB  · {s.reason}")
    print(f"  summary: {p.summary()}")

    # Case 2: diffuse posterior (low confidence) → escalate further
    print("\n[Case 2: diffuse posterior — low confidence]")
    post = {"chronos2": 0.30, "tirex": 0.25, "toto": 0.20,
            "moirai": 0.15, "naive_drift": 0.10}
    p = schedule(post, profiles, cfg, initial_confidence=0.0)
    for s in p.steps:
        print(f"  {s.action:>4} {s.model:<14} U={s.utility:+.3f} "
              f"post={s.posterior:.2f} lat={s.est_latency_s:.2f}s "
              f"vram={s.est_vram_gb:.1f}GB  · {s.reason}")
    print(f"  summary: {p.summary()}")

    # Case 3: tight latency budget → forces early stop
    print("\n[Case 3: tight latency budget 1.5s]")
    cfg_tight = SchedulerConfig(latency_budget_s=1.5)
    p = schedule(post, profiles, cfg_tight, initial_confidence=0.0)
    for s in p.steps:
        print(f"  {s.action:>4} {s.model:<14} U={s.utility:+.3f} "
              f"· {s.reason}")
    print(f"  summary: {p.summary()}")

    # Case 4: VRAM constrained (edge device)
    print("\n[Case 4: edge VRAM=3GB cap]")
    cfg_edge = SchedulerConfig(vram_budget_gb=3.0)
    p = schedule(post, profiles, cfg_edge, initial_confidence=0.0)
    for s in p.steps:
        print(f"  {s.action:>4} {s.model:<14} U={s.utility:+.3f} "
              f"vram={s.est_vram_gb:.1f}GB · {s.reason}")
    print(f"  summary: {p.summary()}")

    # Case 5: remote model rank-2 → remote_penalty should kill it
    print("\n[Case 5: remote in top-2]")
    post = {"chronos2": 0.5, "sundial": 0.4, "naive_drift": 0.1}
    p = schedule(post, profiles, cfg, initial_confidence=0.0)
    for s in p.steps:
        print(f"  {s.action:>4} {s.model:<14} U={s.utility:+.3f} "
              f"remote={s.is_remote} · {s.reason}")
    print(f"  summary: {p.summary()}")
