"""feedback Items 一/二/三 综合 · Prior-aware hierarchical forecasting planner.

Replaces `planner_adaptive.py` (hand-coded weights, 4-model库) with a
prior-driven L0/L1/L2 router over the full 12-model library.

Pipeline per request:

  L0 (free, ~1ms): cheap features → decide if Chronos-2 is "trusted"
       → if yes, return single-model plan (no further computation)

  L1 (Chronos-2 alone): the trusted path. Most requests stop here.

  L2 (small ensemble, ~3 models): when L0 flags uncertainty,
       posterior = BMA(prior · exp(-CV_loss / σ²)).
       Top-K by posterior; renormalized weights.

Priors layered (feedback §二):
  1. Static π_k from `prior_crps.get_prior(dataset=...)`
     — 1/MAE ratio across historical cells (Item 2 from this round).
  2. N-conditional override (feedback §二.3): N<15 → boost Chronos-2 to 0.9
     while down-weighting other TSFMs (insufficient data for reliable CV).
  3. Type prior (feedback §二.2): point predictors (naive/arima/llmtime)
     multiplied by 0.3 — they degrade CRPS even when MAE is competitive.
  4. Availability mask: models without local/remote wiring → drop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

from research.agent.prior_crps import bma_posterior, get_prior

# Type taxonomy — feedback §二.2 / Item 类型先验
POINT_PREDICTORS = {"naive_drift", "naive_seasonal", "arima_ets", "llmtime"}
PROBABILISTIC = {"chronos", "chronos2", "chronos_bolt", "timesfm2", "moirai", "moirai2",
                 "tirex", "toto", "toto2", "time_moe", "sundial", "timer"}

# Availability map (local vs remote vs blocked); aligns with finish-1 §4.1
LOCAL_MODELS = {"naive_drift", "naive_seasonal", "arima_ets", "llmtime",
                "chronos", "chronos2", "chronos_bolt", "timesfm2", "moirai",
                "moirai2", "tirex", "toto", "toto2"}
REMOTE_MODELS = {"time_moe", "sundial", "timer"}
DEFAULT_MODEL = "chronos2"

# L0 trust thresholds (cheap proxy: π_k of default in current ds context)
L0_TRUST_THRESHOLD = 0.45   # if static prior already favors C2 by ≥45%, skip L2
L0_N_MAX_FAST_PATH = 500    # for very long N, CV is cheap enough; revert to L2

Combine = Literal["single", "ensemble", "safe"]


@dataclass
class PriorPlan:
    level: Literal["L1", "L2"]
    strategies: list[str]
    weights: list[float]
    combine: Combine
    reason: str
    posterior: dict[str, float] = field(default_factory=dict)


def _apply_type_prior(prior: dict[str, float], factor: float = 0.3) -> dict[str, float]:
    """Down-weight point predictors (feedback §二.2)."""
    out = {k: (v * factor if k in POINT_PREDICTORS else v) for k, v in prior.items()}
    Z = sum(out.values())
    return {k: v / Z for k, v in out.items()} if Z > 0 else prior


def _apply_n_prior(prior: dict[str, float], N: int) -> dict[str, float]:
    """N<15 → Chronos-2 to 0.9, rest shares 0.1 proportionally (feedback §二.3)."""
    if N >= 15 or DEFAULT_MODEL not in prior:
        return prior
    rest = {k: v for k, v in prior.items() if k != DEFAULT_MODEL}
    Z_rest = sum(rest.values())
    if Z_rest == 0:
        return {DEFAULT_MODEL: 1.0}
    out = {DEFAULT_MODEL: 0.9}
    for k, v in rest.items():
        out[k] = 0.1 * (v / Z_rest)
    return out


def _apply_availability(prior: dict[str, float],
                        allow_remote: bool) -> dict[str, float]:
    """Drop models not available in current deployment."""
    allowed = LOCAL_MODELS | (REMOTE_MODELS if allow_remote else set())
    out = {k: v for k, v in prior.items() if k in allowed}
    Z = sum(out.values())
    return {k: v / Z for k, v in out.items()} if Z > 0 else prior


def compose_prior(dataset: Optional[str], N: int,
                  allow_remote: bool = False,
                  type_prior_factor: float = 0.3) -> dict[str, float]:
    """Stack all priors: static π_k → type → N → availability."""
    p = get_prior(dataset=dataset)
    if not p:
        # Cold-start fallback: uniform over locally-available probabilistic models
        candidates = (PROBABILISTIC & LOCAL_MODELS) | ({DEFAULT_MODEL} if allow_remote or DEFAULT_MODEL in LOCAL_MODELS else set())
        p = {k: 1.0 / len(candidates) for k in candidates}
    p = _apply_type_prior(p, factor=type_prior_factor)
    p = _apply_n_prior(p, N)
    p = _apply_availability(p, allow_remote)
    return p


def make_prior_plan(dataset: Optional[str], N: int, H: int,
                    cv_losses: Optional[dict[str, float]] = None,
                    sigma_sq: float = 0.5,
                    top_k: int = 3,
                    allow_remote: bool = False,
                    force_level: Optional[Literal["L1", "L2"]] = None,
                    ) -> PriorPlan:
    """Build a hierarchical plan.

    Args:
        dataset: dataset name for prior conditioning; None → unconditional prior
        N: training size; <15 triggers safe N-prior
        H: forecast horizon (currently unused, kept for API symmetry)
        cv_losses: optional per-model CV loss (CRPS or MAE); enables BMA posterior.
                   If None → falls back to pure prior π_k.
        sigma_sq: BMA temperature; smaller → sharper
        top_k: ensemble size cap at L2
        allow_remote: include remote-only models (timer/time_moe/sundial)
        force_level: bypass L0 decision (debugging / ablation)

    Returns: PriorPlan with .level, .strategies, .weights, .posterior
    """
    prior = compose_prior(dataset, N, allow_remote=allow_remote)

    # Decide L0 → L1 vs L2
    default_p = prior.get(DEFAULT_MODEL, 0.0)
    if force_level == "L1" or (force_level is None
                               and default_p >= L0_TRUST_THRESHOLD
                               and N <= L0_N_MAX_FAST_PATH
                               and cv_losses is None):
        return PriorPlan(
            level="L1",
            strategies=[DEFAULT_MODEL],
            weights=[1.0],
            combine="single",
            reason=(f"L0→L1 fast path: π({DEFAULT_MODEL}|ds={dataset})={default_p:.2f} "
                    f"≥ {L0_TRUST_THRESHOLD}, N={N} ≤ {L0_N_MAX_FAST_PATH}, no CV signal"),
            posterior={DEFAULT_MODEL: 1.0},
        )

    # L2 path: BMA if CV losses given, else pure prior
    if cv_losses:
        common = {k: cv_losses[k] for k in cv_losses if k in prior}
        if common:
            posterior = bma_posterior(common, sigma_sq=sigma_sq,
                                      prior={k: prior[k] for k in common})
        else:
            posterior = prior
    else:
        posterior = prior

    # Top-K by posterior, renormalize
    ranked = sorted(posterior.items(), key=lambda kv: -kv[1])[:top_k]
    Z = sum(p for _, p in ranked)
    weights = [(p / Z) for _, p in ranked] if Z > 0 else [1.0 / len(ranked)] * len(ranked)
    strategies = [k for k, _ in ranked]

    return PriorPlan(
        level="L2",
        strategies=strategies,
        weights=weights,
        combine="ensemble" if len(strategies) > 1 else "single",
        reason=(f"L2 ensemble: top-{top_k} by "
                f"{'BMA posterior (σ²={:.2f})'.format(sigma_sq) if cv_losses else 'static π_k'}; "
                f"prior favored {DEFAULT_MODEL}={default_p:.2f} < {L0_TRUST_THRESHOLD}"),
        posterior=dict(ranked),
    )


def epsilon_greedy_perturb(plan: PriorPlan, eps: float,
                            rng: Optional[np.random.Generator] = None,
                            min_eps_strategies: int = 2,
                            ) -> tuple[PriorPlan, bool]:
    """feedback §三.4 · ε-greedy exploration for counterfactual data collection.

    With probability ε, replace top-1 strategy with a sample drawn from the
    remaining posterior (renormalized). This forces the wrapper to occasionally
    pick a non-greedy candidate, generating data that the memory layer would
    otherwise never see (counterfactual coverage for Item 4 diversity vote).

    Returns (perturbed_plan, was_explored). If was_explored=False, the original
    plan is returned unchanged.

    Args:
        plan: PriorPlan to potentially perturb
        eps: exploration rate ∈ [0, 1]
        rng: numpy Generator; default uses fresh default_rng()
        min_eps_strategies: skip exploration if plan has < N strategies
            (no alternatives to explore — e.g. L1 single-model plans
            with no posterior over alternatives)
    """
    if rng is None:
        rng = np.random.default_rng()
    posterior = plan.posterior or {s: w for s, w in zip(plan.strategies, plan.weights)}
    if len(posterior) < min_eps_strategies or eps <= 0:
        return plan, False
    if rng.random() >= eps:
        return plan, False
    # explore: sample from non-greedy posterior
    items = sorted(posterior.items(), key=lambda kv: -kv[1])
    alt_items = items[1:]
    Z = sum(p for _, p in alt_items)
    if Z <= 0:
        return plan, False
    probs = np.array([p / Z for _, p in alt_items])
    pick_idx = int(rng.choice(len(alt_items), p=probs))
    chosen, _ = alt_items[pick_idx]
    new_plan = PriorPlan(
        level=plan.level,
        strategies=[chosen],
        weights=[1.0],
        combine="single",
        reason=plan.reason + f" | ε-explore ({eps:.2f}): override top-1 with {chosen}",
        posterior=plan.posterior,
    )
    return new_plan, True


if __name__ == "__main__":
    print("=== Test 1: Weather (C2 dominant per Item 2 finding) ===")
    p = make_prior_plan(dataset="Weather", N=100, H=96)
    print(f"  level={p.level} combine={p.combine}")
    print(f"  strategies={p.strategies}  weights={[f'{w:.2f}' for w in p.weights]}")
    print(f"  reason: {p.reason}\n")

    print("=== Test 2: ECL (Toto dominant per Item 2 finding) ===")
    p = make_prior_plan(dataset="ECL", N=100, H=96, allow_remote=False)
    print(f"  level={p.level} combine={p.combine}")
    print(f"  strategies={p.strategies}  weights={[f'{w:.2f}' for w in p.weights]}")
    print(f"  posterior: {p.posterior}")
    print(f"  reason: {p.reason}\n")

    print("=== Test 3: N=10 cold-start (N<15 prior should kick in) ===")
    p = make_prior_plan(dataset="ETTh1", N=10, H=96)
    print(f"  level={p.level} combine={p.combine}")
    print(f"  strategies={p.strategies}  weights={[f'{w:.2f}' for w in p.weights]}")
    print(f"  reason: {p.reason}\n")

    print("=== Test 4: BMA with CV losses (TiRex stronger this cell) ===")
    p = make_prior_plan(dataset="Exchange", N=50, H=96,
                        cv_losses={"chronos2": 0.025, "tirex": 0.013, "toto": 0.020},
                        sigma_sq=0.001, top_k=3)
    print(f"  level={p.level} combine={p.combine}")
    print(f"  strategies={p.strategies}  weights={[f'{w:.2f}' for w in p.weights]}")
    print(f"  posterior: {p.posterior}")
    print(f"  reason: {p.reason}\n")

    print("=== Test 5: allow_remote=True (Timer-S1 in pool) ===")
    p = make_prior_plan(dataset=None, N=100, H=96, allow_remote=True)
    print(f"  level={p.level} combine={p.combine} strategies={p.strategies}")
    print(f"  weights={[f'{w:.2f}' for w in p.weights]}")

    print("\n=== Test 6: ε-greedy on ECL L2 plan (ε=0.4, 20 trials) ===")
    p_ecl = make_prior_plan(dataset="ECL", N=100, H=96)
    rng = np.random.default_rng(0)
    counts: dict[str, int] = {}
    n_explored = 0
    for _ in range(20):
        p_new, was_exp = epsilon_greedy_perturb(p_ecl, eps=0.4, rng=rng)
        if was_exp: n_explored += 1
        s = p_new.strategies[0]
        counts[s] = counts.get(s, 0) + 1
    print(f"  base top-1: {p_ecl.strategies[0]}  posterior: {p_ecl.posterior}")
    print(f"  explored {n_explored}/20 trials; pick distribution: {counts}")
