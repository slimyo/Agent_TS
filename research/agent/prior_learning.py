"""Round 7 M3 · Empirical Bayes Prior Strength (feedback 前§2.B).

Learns the `.strength` knob of each PriorFactor by correlating its
contribution log_prior_F(chosen) with the observed outcome.

    r = Pearson(log_prior_F(chosen),  -outcome)

    r > 0:  factor 把高 log_prior 给了实际表现好的 model → 有用 → 加强
    r < 0:  factor 反向相关 → 没用 → 削弱 (strength → 0 = prune)

Update rule:
    strength_new = clip(strength · (1 + lr · r),  0,  max_strength)

State persistence:
    Learned strengths are written to `state.learned_prior_strengths`:
        dict[factor_name, float]
    Future `adaptive_decide` invocations should consult this dict when
    constructing PriorFactor instances (default behavior unchanged if
    state has no learned values).

Safety:
    - Only factors with a `.strength` attribute are touched
    - Need ≥ `min_samples` records with outcome AND factor in prior_contribs
    - Winsorize outcomes to [q05, q95] before correlation
    - lr default 0.05 keeps updates gentle
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import math

import numpy as np


# ─── Config ──────────────────────────────────────────────────────────────────


@dataclass
class EBConfig:
    lr: float = 0.05
    max_strength: float = 5.0
    min_strength: float = 0.0
    min_samples: int = 30
    winsorize_q: float = 0.05


# ─── Core ────────────────────────────────────────────────────────────────────


def _pearson(xs: np.ndarray, ys: np.ndarray) -> float:
    if xs.size < 2: return 0.0
    sx, sy = float(xs.std()), float(ys.std())
    if sx < 1e-9 or sy < 1e-9: return 0.0
    return float(np.corrcoef(xs, ys)[0, 1])


def learn_prior_strengths(state, router,
                           config: Optional[EBConfig] = None) -> dict:
    """Update `factor.strength` for each PriorFactor with that attribute.

    Walks `state.telemetry` to extract (log_prior_F(chosen), -outcome) pairs;
    correlation drives a multiplicative update. Also stashes the new strength
    in `state.learned_prior_strengths[factor.name]` for cross-session use.

    Returns a per-factor summary dict.
    """
    if config is None: config = EBConfig()
    tel = list(state.telemetry)
    out = {"status": "ok", "factors": {}}

    if not hasattr(state, "learned_prior_strengths") or \
            state.learned_prior_strengths is None:
        state.learned_prior_strengths = {}

    # Pre-collect outcome winsorization bounds
    outcomes_all = np.array([r.outcome for r in tel if r.outcome is not None],
                             dtype=np.float64)
    if outcomes_all.size < config.min_samples:
        return {"status": "skipped",
                "reason": f"only {outcomes_all.size} outcomes < {config.min_samples}"}
    lo, hi = np.quantile(outcomes_all, [config.winsorize_q,
                                          1 - config.winsorize_q])

    for f in getattr(router, "priors", []):
        fname = getattr(f, "name", type(f).__name__)
        if not hasattr(f, "strength"):
            out["factors"][fname] = {"skipped": "no .strength attribute"}
            continue

        # Collect aligned pairs
        xs, ys = [], []
        for r in tel:
            if r.outcome is None: continue
            contribs = r.prior_contribs.get(fname)
            if not contribs: continue
            lp = contribs.get(r.chosen)
            if lp is None or not math.isfinite(float(lp)): continue
            xs.append(float(lp))
            y = float(np.clip(r.outcome, lo, hi))
            ys.append(-y)              # higher = better

        if len(xs) < config.min_samples:
            out["factors"][fname] = {
                "skipped": f"only {len(xs)} valid records < {config.min_samples}"}
            continue

        xs_arr, ys_arr = np.asarray(xs), np.asarray(ys)
        r_corr = _pearson(xs_arr, ys_arr)
        old = float(f.strength)
        new = float(np.clip(old * (1.0 + config.lr * r_corr),
                              config.min_strength, config.max_strength))
        f.strength = new
        state.learned_prior_strengths[fname] = new

        out["factors"][fname] = {
            "n_samples": len(xs),
            "pearson_r": round(r_corr, 4),
            "strength_old": round(old, 4),
            "strength_new": round(new, 4),
            "delta_pct": round(100 * (new - old) / max(old, 1e-9), 2),
        }
    return out


# ─── Convenience: read learned strength back ─────────────────────────────────


def apply_learned_strengths(router, state) -> dict:
    """Push learned strengths from state into router.priors in-place.

    Useful when caller rebuilds priors fresh per call (factories) and wants
    to seed each new factor with its previously-learned strength.
    """
    learned = getattr(state, "learned_prior_strengths", {}) or {}
    applied = {}
    for f in getattr(router, "priors", []):
        fname = getattr(f, "name", type(f).__name__)
        if not hasattr(f, "strength"): continue
        if fname in learned:
            f.strength = float(learned[fname])
            applied[fname] = f.strength
    return applied


# ─── Smoke ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Round 7 M3 · Empirical Bayes Prior strength smoke")
    print("=" * 60)
    import time
    from research.agent.router_state import RouterState, TelemetryRecord
    from research.agent.bayesian_router import BayesianRouter, NPrior, TypePrior

    rng = np.random.default_rng(0)

    # Build synthetic state where NPrior actually helps (positive correlation)
    # and TypePrior actively hurts (negative correlation) — verify learning
    # adjusts both.
    state = RouterState()
    cands = ["chronos2", "tirex", "naive_drift"]
    # Hand-built records:
    #   When NPrior log says "chronos2 highest" AND chronos2 is chosen,
    #     outcome should be LOW (good).
    #   When TypePrior log says "chronos2 highest" AND chronos2 is chosen,
    #     outcome should be HIGH (bad).
    for i in range(120):
        chosen = "chronos2" if rng.random() < 0.6 else \
                 ("tirex" if rng.random() < 0.5 else "naive_drift")
        # NPrior helpful: high lp_n → low outcome
        lp_n = 1.0 + 0.5 * rng.standard_normal()
        outcome_part_n = -0.30 * lp_n          # higher lp_n → lower outcome ✓
        # TypePrior harmful: high lp_t → high outcome
        lp_t = 0.5 * rng.standard_normal()
        outcome_part_t = +0.40 * lp_t          # higher lp_t → higher outcome ✗
        noise = 0.10 * rng.standard_normal()
        outcome = max(0.0, 0.6 + outcome_part_n + outcome_part_t + noise)

        rec = TelemetryRecord(
            t=time.time(),
            ctx_summary={"regime": 0, "dataset": "X", "N": 50, "H": 24},
            chosen=chosen,
            posterior={chosen: 0.7, **{c: 0.15 for c in cands if c != chosen}},
            prior_contribs={
                "N_prior": {chosen: lp_n,
                            **{c: 0.0 for c in cands if c != chosen}},
                "type":    {chosen: lp_t,
                            **{c: 0.0 for c in cands if c != chosen}},
            },
            lik_contribs={}, decide_mode="argmax",
            outcome=float(outcome),
        )
        state.telemetry.append(rec)

    router = BayesianRouter(
        candidates=cands,
        priors=[NPrior(default_model="chronos2", strength=2.0),
                TypePrior()],
        likelihoods=[],
    )

    print(f"\nBefore learning:")
    for f in router.priors:
        s = getattr(f, "strength", None)
        print(f"  {f.name:<14} strength = {s}")

    res = learn_prior_strengths(state, router, EBConfig(lr=0.2))

    print(f"\nLearning result (status={res['status']}):")
    for fname, info in res["factors"].items():
        if "skipped" in info:
            print(f"  {fname:<14} {info['skipped']}")
            continue
        sign = "↑" if info["delta_pct"] > 0 else "↓"
        print(f"  {fname:<14} n={info['n_samples']:<4} r={info['pearson_r']:+.3f}  "
              f"strength {info['strength_old']:.3f} → {info['strength_new']:.3f}  "
              f"({sign}{abs(info['delta_pct']):.1f}%)")

    print(f"\nstate.learned_prior_strengths = {state.learned_prior_strengths}")

    # Test apply_learned_strengths: rebuild factors fresh, verify they pick up
    # the learned values.
    print(f"\n[apply_learned_strengths round-trip]")
    fresh_router = BayesianRouter(
        candidates=cands,
        priors=[NPrior(default_model="chronos2", strength=2.0),   # default
                TypePrior()],
        likelihoods=[],
    )
    applied = apply_learned_strengths(fresh_router, state)
    print(f"  applied: {applied}")
    for f in fresh_router.priors:
        if hasattr(f, "strength"):
            print(f"  {f.name:<14} after apply: strength = {f.strength:.3f}")
