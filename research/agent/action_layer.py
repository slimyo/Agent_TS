"""Round 6 E1 · Action Layer (feedback 后§9 "你当前 architecture 最缺：Action Layer").

Bridges `predict` → `decide intervention`:

    forecast      ─┐
    uncertainty   ─┼─► assess_risk → choose_intervention → simulate_costs ─► ActionDecision
    context       ─┘

Task-agnostic by design — works for any forecast that yields (mean, std) and
a Context carrying threshold info. The policy is driven by two scalars:
risk (probability of threshold breach) and confidence (calibrated P(correct)).

Intervention vocabulary (one of):
    MONITOR     · default; continue observing
    INSPECT     · uncertainty high but risk moderate → manual check
    THROTTLE    · moderate risk + decent confidence → soft action
    SHUTDOWN    · high risk + high confidence → hard intervention
    ESCALATE    · high risk AND low confidence → defer to human

Cost matrix encodes loss of (action, true_outcome) pairs and is minimized via
expected cost. Defaults are illustrative; production callers should override.

Inputs:
    forecast   = ForecastDist(mean, std, quantiles?)
    context    = ActionContext(upper_threshold, ...)
    confidence = float ∈ [0,1] from calibration.py
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, Literal
import math


# ─── data classes ────────────────────────────────────────────────────────────


Intervention = Literal["MONITOR", "INSPECT", "THROTTLE", "SHUTDOWN", "ESCALATE"]


@dataclass
class ForecastDist:
    """Minimal predictive distribution interface."""
    mean: float
    std: float
    quantiles: Optional[dict[float, float]] = None   # e.g. {0.05: ..., 0.95: ...}
    horizon_steps: int = 1                            # steps ahead this represents
    lead_time_seconds: float = 0.0                    # absolute lead time if known


@dataclass
class ActionContext:
    """Task-agnostic context for an action decision."""
    upper_threshold: Optional[float] = None    # breach when prediction > this
    lower_threshold: Optional[float] = None    # breach when prediction < this
    current_value: Optional[float] = None      # latest observation, optional
    asset_id: Optional[str] = None             # for logging / audit
    horizon_steps: int = 1


@dataclass
class ActionConfig:
    """Policy + cost thresholds."""
    risk_high:       float = 0.70    # breach prob threshold for HIGH risk
    risk_moderate:   float = 0.40
    confidence_high: float = 0.70    # calibrated P(correct)
    confidence_low:  float = 0.30
    # Cost matrix (action, breach? bool) → cost. Lower = better.
    # Action choices: monitor / inspect / throttle / shutdown / escalate.
    cost_matrix: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        # action       (no_breach, breach)
        "MONITOR":     (0.0,   100.0),    # cheap if safe, catastrophic if breach
        "INSPECT":     (2.0,    20.0),    # cheap manual cost, mitigates damage
        "THROTTLE":    (5.0,    15.0),    # productivity loss, reduces damage
        "SHUTDOWN":    (30.0,    5.0),    # big productivity loss, prevents most damage
        "ESCALATE":    (3.0,    30.0),    # manual review delay, partial mitigation
    })


@dataclass
class RiskAssessment:
    """Risk of breaching a threshold within the forecast horizon."""
    p_breach: float                       # ∈ [0, 1]
    breach_quantile: Optional[float]      # which q of forecast crosses threshold
    direction: Optional[str]              # "upper" | "lower" | None
    margin: float                         # signed distance: mean − threshold


@dataclass
class ActionDecision:
    """Final intervention recommendation with audit trail."""
    intervention: Intervention
    risk: RiskAssessment
    confidence: float
    expected_costs: dict[str, float]
    reason: str


# ─── 1. Risk evaluation ──────────────────────────────────────────────────────


def _phi(x: float) -> float:
    """Standard normal CDF (no scipy dep)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def assess_risk(forecast: ForecastDist, ctx: ActionContext) -> RiskAssessment:
    """P(threshold breach) under Gaussian forecast.

    If both thresholds given, returns max breach prob across the two.
    If no threshold, returns p_breach=0 (caller should treat as MONITOR).
    """
    if ctx.upper_threshold is None and ctx.lower_threshold is None:
        return RiskAssessment(p_breach=0.0, breach_quantile=None,
                              direction=None, margin=0.0)

    sigma = max(forecast.std, 1e-9)
    p_up = p_down = 0.0
    if ctx.upper_threshold is not None:
        p_up = 1.0 - _phi((ctx.upper_threshold - forecast.mean) / sigma)
    if ctx.lower_threshold is not None:
        p_down = _phi((ctx.lower_threshold - forecast.mean) / sigma)

    if p_up >= p_down:
        direction = "upper" if ctx.upper_threshold is not None else None
        margin = forecast.mean - (ctx.upper_threshold or 0.0)
        p_breach = p_up
    else:
        direction = "lower"
        margin = forecast.mean - (ctx.lower_threshold or 0.0)
        p_breach = p_down

    # Which quantile crosses the threshold (informational)
    breach_q = None
    if forecast.quantiles and direction:
        target = ctx.upper_threshold if direction == "upper" else ctx.lower_threshold
        if direction == "upper":
            crossing = [q for q, v in forecast.quantiles.items() if v >= target]
            breach_q = min(crossing) if crossing else None
        else:
            crossing = [q for q, v in forecast.quantiles.items() if v <= target]
            breach_q = max(crossing) if crossing else None

    return RiskAssessment(p_breach=float(p_breach),
                          breach_quantile=breach_q,
                          direction=direction, margin=float(margin))


# ─── 2. Cost simulation ──────────────────────────────────────────────────────


def expected_costs(risk: RiskAssessment, config: ActionConfig) -> dict[str, float]:
    """Expected cost per intervention = p_breach * C(a, breach) + (1-p) * C(a, no_breach)."""
    p = risk.p_breach
    out = {}
    for action, (c_safe, c_breach) in config.cost_matrix.items():
        out[action] = (1 - p) * c_safe + p * c_breach
    return out


# ─── 3. Intervention policy ──────────────────────────────────────────────────


def choose_intervention(risk: RiskAssessment, confidence: float,
                        config: ActionConfig) -> tuple[Intervention, str]:
    """Rule-augmented cost-minimizer.

    Hard overrides (interpretability > pure cost-min):
        risk high  + conf high  → SHUTDOWN
        risk high  + conf low   → ESCALATE  (don't act blindly on uncertain alarm)
        conf low   + risk mod   → INSPECT   (gather info before acting)

    Otherwise: argmin expected cost across matrix.
    """
    p = risk.p_breach
    if p >= config.risk_high and confidence >= config.confidence_high:
        return "SHUTDOWN", (f"p_breach={p:.2f}≥{config.risk_high} "
                             f"and conf={confidence:.2f}≥{config.confidence_high}")
    if p >= config.risk_high and confidence < config.confidence_low:
        return "ESCALATE", (f"high risk p={p:.2f} with low conf={confidence:.2f} "
                             f"→ defer to human")
    if confidence < config.confidence_low and p >= config.risk_moderate:
        return "INSPECT", (f"low conf={confidence:.2f}, moderate risk p={p:.2f} "
                            f"→ gather info first")

    # Cost-min path
    costs = expected_costs(risk, config)
    best = min(costs, key=costs.get)
    return best, f"argmin expected cost (costs={ {k: round(v,2) for k,v in costs.items()} })"


# ─── 4. Orchestrator ─────────────────────────────────────────────────────────


def decide_action(forecast: ForecastDist,
                  ctx: ActionContext,
                  confidence: float = 0.5,
                  config: Optional[ActionConfig] = None) -> ActionDecision:
    """End-to-end: forecast → risk → intervention.

    Args:
        forecast: predictive distribution (mean, std, optional quantiles)
        ctx: threshold + asset context
        confidence: calibrated P(correct) from calibration.py
        config: thresholds and cost matrix
    """
    if config is None: config = ActionConfig()
    risk = assess_risk(forecast, ctx)
    costs = expected_costs(risk, config)
    intervention, reason = choose_intervention(risk, confidence, config)
    return ActionDecision(
        intervention=intervention, risk=risk, confidence=confidence,
        expected_costs=costs, reason=reason,
    )


# ─── 5. Convenience: integrate with router output ────────────────────────────


def _get_or_fit_calibrator(state,
                            refit_every: int = 50,
                            min_obs: int = 20,
                            metric: str = "posterior_max"):
    """Lazy-fit / cache a ConfidenceCalibrator on `state`.

    Refit conditions:
      - first call (no cached calibrator), or
      - n_observations grew by ≥ refit_every since last fit
    Returns None when too few outcomes are available.
    """
    n_obs = int(getattr(state, "n_observations", 0))
    if n_obs < min_obs:
        return None
    cal = getattr(state, "_calibrator", None)
    last_fit_n = int(getattr(state, "_calibrator_fit_n", -1))
    if cal is not None and (n_obs - last_fit_n) < refit_every:
        return cal
    try:
        from research.agent.calibration import fit_from_state
        cal = fit_from_state(state, metric=metric)
        state._calibrator = cal
        state._calibrator_fit_n = n_obs
        return cal
    except Exception:
        return cal   # keep previous if refit fails


def decide_from_router(forecast: ForecastDist,
                       ctx: ActionContext,
                       state=None,                       # RouterState
                       config: Optional[ActionConfig] = None,
                       metric: str = "posterior_max",
                       refit_every: int = 50,
                       min_obs_for_calibration: int = 20,
                       ) -> ActionDecision:
    """End-to-end action with calibration-aware confidence.

    Confidence pipeline:
      1. raw = max(last_posterior) (or top1-top2 gap if metric="posterior_gap")
      2. calibrated = ConfidenceCalibrator.calibrate(raw)
         · calibrator is lazily fit from state.telemetry (Round 6 B2)
         · refit every `refit_every` new observations
         · fall back to raw if not enough data (n_obs < min_obs_for_calibration)
      3. pass calibrated to decide_action

    The chosen behavior_tier (B2 4-tier table) is appended to the decision
    reason for auditing.
    """
    raw_conf = 0.5
    if state is not None and getattr(state, "telemetry", None):
        last = state.telemetry[-1] if state.telemetry else None
        if last and last.posterior:
            try:
                vals = sorted(last.posterior.values(), reverse=True)
                if metric == "posterior_gap":
                    raw_conf = float(vals[0] - (vals[1] if len(vals) > 1 else 0.0))
                else:
                    raw_conf = float(vals[0])
            except Exception:
                raw_conf = 0.5

    calibrated_conf = raw_conf
    calibrator_used = False
    if state is not None:
        cal = _get_or_fit_calibrator(state,
                                      refit_every=refit_every,
                                      min_obs=min_obs_for_calibration,
                                      metric=metric)
        if cal is not None and cal.fit_info is not None \
                and cal.fit_info.n_total >= min_obs_for_calibration:
            calibrated_conf = float(cal.calibrate(raw_conf))
            calibrator_used = True

    decision = decide_action(forecast, ctx,
                              confidence=calibrated_conf, config=config)

    # Audit: annotate reason with calibration provenance + tier
    from research.agent.calibration import ConfidenceCalibrator
    tier = ConfidenceCalibrator.behavior_tier(calibrated_conf)
    provenance = (f"calibrated:{raw_conf:.2f}→{calibrated_conf:.2f} "
                   f"[tier={tier}]") if calibrator_used \
                  else f"raw:{raw_conf:.2f} (calibrator not ready) [tier={tier}]"
    decision.reason = f"{provenance} | {decision.reason}"
    return decision


# ─── smoke ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Round 6 E1 · Action Layer smoke")
    print("=" * 60)

    cfg = ActionConfig()

    # Case 1: Safe — mean far below upper threshold, low risk
    f1 = ForecastDist(mean=50.0, std=5.0)
    ctx1 = ActionContext(upper_threshold=100.0, asset_id="pump_1")
    d1 = decide_action(f1, ctx1, confidence=0.85, config=cfg)
    print(f"\n[Case 1: safe]  p_breach={d1.risk.p_breach:.4f}")
    print(f"  → {d1.intervention}  ({d1.reason})")

    # Case 2: Moderate risk + high confidence → cost-min likely THROTTLE
    f2 = ForecastDist(mean=95.0, std=4.0)
    d2 = decide_action(f2, ctx1, confidence=0.80, config=cfg)
    print(f"\n[Case 2: mod risk]  p_breach={d2.risk.p_breach:.4f}")
    print(f"  → {d2.intervention}  ({d2.reason})")

    # Case 3: High risk + high confidence → SHUTDOWN override
    f3 = ForecastDist(mean=115.0, std=5.0)
    d3 = decide_action(f3, ctx1, confidence=0.85, config=cfg)
    print(f"\n[Case 3: high risk + conf]  p_breach={d3.risk.p_breach:.4f}")
    print(f"  → {d3.intervention}  ({d3.reason})")

    # Case 4: High risk + low confidence → ESCALATE override
    d4 = decide_action(f3, ctx1, confidence=0.20, config=cfg)
    print(f"\n[Case 4: high risk + low conf]  p_breach={d4.risk.p_breach:.4f}")
    print(f"  → {d4.intervention}  ({d4.reason})")

    # Case 5: Low conf, moderate risk → INSPECT
    f5 = ForecastDist(mean=92.0, std=6.0)
    d5 = decide_action(f5, ctx1, confidence=0.20, config=cfg)
    print(f"\n[Case 5: low conf, mod risk]  p_breach={d5.risk.p_breach:.4f}")
    print(f"  → {d5.intervention}  ({d5.reason})")

    # Case 6: Lower threshold breach (e.g. battery voltage drop)
    f6 = ForecastDist(mean=3.2, std=0.4)
    ctx6 = ActionContext(lower_threshold=3.0, asset_id="battery_7")
    d6 = decide_action(f6, ctx6, confidence=0.80, config=cfg)
    print(f"\n[Case 6: lower threshold]  p_breach={d6.risk.p_breach:.4f}  "
          f"dir={d6.risk.direction}")
    print(f"  → {d6.intervention}  ({d6.reason})")

    # Case 7: quantile crossing (informational field)
    f7 = ForecastDist(mean=90.0, std=8.0,
                       quantiles={0.5: 90.0, 0.75: 96.0, 0.9: 103.0, 0.95: 107.0})
    d7 = decide_action(f7, ctx1, confidence=0.65, config=cfg)
    print(f"\n[Case 7: quantile info]  p_breach={d7.risk.p_breach:.4f}  "
          f"breach starts at q={d7.risk.breach_quantile}")
    print(f"  → {d7.intervention}  ({d7.reason})")
