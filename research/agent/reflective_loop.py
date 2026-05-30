"""Round 6 B1 · Reflective Execution Loop (feedback 后§2 closed-loop).

Replaces single-shot `decide → predict → return` with:

    Layer 0: fast hypothesis (top-1 prediction + confidence)
    Layer 1: if posterior_gap < τ_gap → escalate to top-K ensemble
    Layer 2: if still uncertain → memory-neighbor specialist run
    Layer 3: post-hoc critique (residual analysis, agreement check)
    Layer 4: store counterfactual outcome for memory

The loop is *progressive*: cheap path first, escalate only when needed.

API:
    result = reflective_predict(plan, predict_fn, series, ...)
    # result.final_pred + result.layers_used + result.confidence

Where predict_fn is a callable `(model_name) → y_hat`. Loop hides this from
the caller by trying the top-1 first and only invoking predict_fn for
escalated candidates if confidence is low.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from research.agent.adaptive_planner import AdaptivePlan


@dataclass
class ReflectionResult:
    final_pred: np.ndarray
    final_chosen: str
    layers_used: list[str]            # ["L0", "L1", ...]
    per_layer_preds: dict[str, dict]   # {layer: {chosen: ndarray}}
    confidence: float                  # posterior gap or calibrated, ∈ [0,1]
    confidence_source: str             # "posterior_gap" | "calibrated"
    critique: dict                     # disagreement / residual stats
    reasoning_chain: list[str]


def _posterior_gap(plan: AdaptivePlan) -> float:
    """top1 − top2 posterior gap. ∈ [0, 1]."""
    if not plan.top_k or len(plan.top_k) < 2:
        return 1.0
    return float(plan.top_k[0][1] - plan.top_k[1][1])


def _ensemble_predict(preds: dict[str, np.ndarray],
                      weights: dict[str, float] | None = None) -> np.ndarray:
    """Weighted mean ensemble. Uniform if weights None."""
    keys = list(preds.keys())
    if weights is None:
        w = np.ones(len(keys)) / len(keys)
    else:
        w = np.array([weights.get(k, 0.0) for k in keys])
        w = w / max(w.sum(), 1e-9)
    stacked = np.stack([preds[k] for k in keys])
    return (stacked * w[:, None]).sum(axis=0)


def _residual_critique(preds: dict[str, np.ndarray], history: np.ndarray
                        ) -> dict:
    """Cross-model agreement + residual scale checks."""
    if len(preds) < 2:
        return {"max_pairwise_dist": 0.0, "agreement": 1.0}
    vals = list(preds.values())
    diffs = []
    for i in range(len(vals)):
        for j in range(i+1, len(vals)):
            diffs.append(float(np.mean(np.abs(vals[i] - vals[j]))))
    max_d = max(diffs)
    hist_std = float(np.std(history)) + 1e-6
    # agreement = 1 - (max disagreement / history std), clipped to [0,1]
    agreement = max(0.0, min(1.0, 1.0 - max_d / hist_std))
    return {"max_pairwise_dist": max_d, "history_std": hist_std,
            "agreement": agreement}


def reflective_predict(
    plan: AdaptivePlan,
    predict_fn: Callable[[str], np.ndarray],
    history: np.ndarray,
    tau_gap: float = 0.10,
    tau_agreement: float = 0.5,
    enable_l2: bool = True,
    enable_l3: bool = True,
    max_escalation: int = 3,
) -> ReflectionResult:
    """Progressive inference: L0 cheap → escalate on low confidence.

    Args:
        plan: result of adaptive_decide()
        predict_fn: callable (model_name) → y_hat ndarray; should raise on failure
        history: train series (for residual critique)
        tau_gap: minimum posterior_gap to trust L0 only
        tau_agreement: minimum cross-model agreement to skip L3 critique
        enable_l2: if False, skip memory-neighbor escalation
        enable_l3: if False, skip post-hoc critique
        max_escalation: cap candidates tried (cost guard)
    """
    layers = ["L0"]
    chain = []
    per_layer_preds: dict[str, dict] = {}
    conf_source = "posterior_gap"

    # ─── L0 · top-1 cheap inference ────────────────────────────────────
    top1 = plan.top_k[0][0]
    chain.append(f"L0: top-1 = {top1} (π={plan.top_k[0][1]:.3f})")
    pred_top1 = predict_fn(top1)
    per_layer_preds["L0"] = {top1: pred_top1}
    gap = _posterior_gap(plan)
    chain.append(f"L0: posterior_gap = {gap:.3f}")

    if gap >= tau_gap:
        # L0 confident, return immediately
        chain.append(f"L0 confident (gap ≥ {tau_gap}), skip escalation")
        return ReflectionResult(
            final_pred=pred_top1, final_chosen=top1,
            layers_used=layers, per_layer_preds=per_layer_preds,
            confidence=min(1.0, gap / max(tau_gap, 1e-6)),
            confidence_source=conf_source,
            critique={}, reasoning_chain=chain,
        )

    # ─── L1 · small ensemble (top-K) ──────────────────────────────────
    layers.append("L1")
    top_k_models = [k for k, _ in plan.top_k[:max_escalation]]
    chain.append(f"L1: low gap, ensemble top-{len(top_k_models)} = {top_k_models}")
    l1_preds = {top1: pred_top1}
    for m in top_k_models[1:]:
        try:
            l1_preds[m] = predict_fn(m)
        except Exception as e:
            chain.append(f"L1: skip {m} ({type(e).__name__})")
    per_layer_preds["L1"] = dict(l1_preds)
    weights = {k: plan.posterior.get(k, 0.0) for k in l1_preds}
    ens_pred = _ensemble_predict(l1_preds, weights)
    chain.append(f"L1: posterior-weighted ensemble over {len(l1_preds)} models")

    final_pred = ens_pred
    final_chosen = top1 + "+ens"

    # ─── L2 · memory-neighbor specialist (optional) ───────────────────
    if enable_l2 and len(layers) < 4:
        # If state has memory cases that match plan.regime, retrieve their best model
        # (this hook would consult RouterState — left as future expansion;
        # for now we skip if not provided)
        chain.append("L2: memory specialist lookup (skipped; needs RouterState)")
        layers.append("L2-skip")

    # ─── L3 · critique (residual / agreement) ─────────────────────────
    critique = {}
    if enable_l3 and len(l1_preds) >= 2:
        layers.append("L3")
        critique = _residual_critique(l1_preds, history)
        chain.append(f"L3: agreement={critique['agreement']:.3f}, "
                     f"max_pair_dist={critique['max_pairwise_dist']:.3f}")
        if critique["agreement"] < tau_agreement:
            chain.append(f"L3 WARNING: low agreement (<{tau_agreement}), "
                         "downstream should mark high-uncertainty")

    confidence = min(1.0, gap / max(tau_gap, 1e-6))
    if critique:
        # blend agreement into confidence
        confidence = 0.5 * confidence + 0.5 * critique.get("agreement", 0.5)

    return ReflectionResult(
        final_pred=final_pred, final_chosen=final_chosen,
        layers_used=layers, per_layer_preds=per_layer_preds,
        confidence=confidence, confidence_source=conf_source,
        critique=critique, reasoning_chain=chain,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Round 6 B1 · Reflective loop smoke")
    print("=" * 60)
    from research.agent.adaptive_planner import AdaptivePlan
    import numpy as np

    history = np.sin(np.arange(100) * 0.1).astype(np.float32)

    # Mock predict_fn: returns different vectors per model
    def predict_fn(m):
        base = np.linspace(0.0, 1.0, 24)
        if m == "chronos2":  return base * 0.9
        if m == "tirex":     return base * 1.1
        if m == "toto":      return base * 0.95
        if m == "naive":     return base * 0.0
        raise RuntimeError("unknown")

    # ─── Case 1: high posterior gap → L0 only
    plan_clear = AdaptivePlan(
        task="forecast", chosen="chronos2", regime=0, z=None,
        decide_mode="argmax", config_snapshot={},
        prior_contribs={}, lik_contribs={},
        posterior={"chronos2": 0.7, "tirex": 0.2, "toto": 0.1},
        top_k=[("chronos2", 0.7), ("tirex", 0.2), ("toto", 0.1)],
    )
    res = reflective_predict(plan_clear, predict_fn, history, tau_gap=0.10)
    print(f"\n[Case 1] CLEAR posterior:")
    print(f"  layers_used: {res.layers_used}")
    print(f"  chosen: {res.final_chosen}  confidence: {res.confidence:.3f}")
    print(f"  chain:")
    for c in res.reasoning_chain: print(f"    {c}")

    # ─── Case 2: low gap → L1 ensemble + L3 critique
    plan_uncertain = AdaptivePlan(
        task="forecast", chosen="chronos2", regime=0, z=None,
        decide_mode="argmax", config_snapshot={},
        prior_contribs={}, lik_contribs={},
        posterior={"chronos2": 0.40, "tirex": 0.35, "toto": 0.25},
        top_k=[("chronos2", 0.40), ("tirex", 0.35), ("toto", 0.25)],
    )
    res2 = reflective_predict(plan_uncertain, predict_fn, history, tau_gap=0.10)
    print(f"\n[Case 2] UNCERTAIN posterior (gap=0.05):")
    print(f"  layers_used: {res2.layers_used}")
    print(f"  chosen: {res2.final_chosen}  confidence: {res2.confidence:.3f}")
    print(f"  critique: {res2.critique}")
    print(f"  chain:")
    for c in res2.reasoning_chain: print(f"    {c}")
