"""feedback §三.1 软路由 (概率版) · L2 quantile ensemble for PriorPlan.

Given a PriorPlan with strategies + weights, fetch each strategy's quantile
forecast and combine via weighted linear pool on a common quantile grid.

Linear pool (not isotonic) is chosen because:
  1. It is what feedback §三.1 's pseudo-code implies (`weights[k] * quantile_pred_k`).
  2. CRPS is consistent under linear pooling.
  3. No isotonic correction needed for monotone weights.

Model quantile API tabulated below (verified against current files, 2026-05-27):

  chronos2      21 levels  via `predict_with_uncertainty(...) -> (median, entropy, [21, H])`
  tirex          9 levels  via `predict_with_uncertainty(...) -> (median, entropy, [9, H])`
  timer (S1)     9 levels  via `predict_with_uncertainty(...) -> {"quantiles":[9,H],...}`
  chronos_bolt   9 levels  (Bolt is C2 alias, 21 levels — handled as chronos2)
  others        point-only  → degenerate quantile = median replicated

Common target grid: 9 levels [0.1..0.9] — minimal common denominator across TSFMs.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np

from research.agent.planner_prior_aware import PriorPlan

TARGET_LEVELS = np.linspace(0.1, 0.9, 9)


# Quantile grids known per model (NATIVE levels — _align_to_target interpolates if needed)
KNOWN_GRIDS: dict[str, np.ndarray] = {
    "chronos2":     np.linspace(0.0, 1.0, 21),   # native 21-grid; interp to TARGET on use
    "chronos_bolt": np.linspace(0.0, 1.0, 21),
    "tirex":        TARGET_LEVELS,
    "timer":        TARGET_LEVELS,
    "time_moe":     TARGET_LEVELS,   # 9 by spec
    "sundial":      TARGET_LEVELS,
    "toto":         TARGET_LEVELS,
    "toto2":        TARGET_LEVELS,
    "timesfm2":     TARGET_LEVELS,
    "moirai":       TARGET_LEVELS,
    "moirai2":      TARGET_LEVELS,
}

POINT_ONLY = {"naive_drift", "naive_seasonal", "arima_ets", "llmtime", "chronos"}


@dataclass
class EnsembleResult:
    median: np.ndarray              # [H]
    quantiles: np.ndarray           # [9, H] on TARGET_LEVELS
    levels: np.ndarray              # [9] = TARGET_LEVELS
    per_model_quantiles: dict[str, np.ndarray]   # debug
    per_model_weights: dict[str, float]


def _align_to_target(q: np.ndarray, source_levels: np.ndarray) -> np.ndarray:
    """Interpolate quantile matrix [|Q|, H] from source_levels to TARGET_LEVELS.

    If shapes already match TARGET_LEVELS exactly, no-op (preserves model native precision).
    """
    if (q.shape[0] == len(TARGET_LEVELS) and len(source_levels) == len(TARGET_LEVELS)
            and np.allclose(source_levels, TARGET_LEVELS)):
        return q
    H = q.shape[1]
    out = np.empty((len(TARGET_LEVELS), H), dtype=q.dtype)
    for t in range(H):
        out[:, t] = np.interp(TARGET_LEVELS, source_levels, q[:, t])
    return out


def _point_to_degenerate_quantiles(point_pred: np.ndarray) -> np.ndarray:
    """Replicate a point forecast across all TARGET_LEVELS (Dirac at median)."""
    return np.tile(point_pred[None, :], (len(TARGET_LEVELS), 1))


def fetch_quantiles(strategy: str, train: np.ndarray, val: np.ndarray,
                    H: int, seed: int, season_m: int = 1,
                    predict_fns: Optional[dict[str, Callable]] = None,
                    ) -> np.ndarray:
    """Returns quantile matrix [9, H] on TARGET_LEVELS for any registered strategy.

    Args:
        predict_fns: optional override map {strategy: predict_with_uncertainty fn}
                     — used for testing or for routing to remote-subprocess wrappers.
    """
    if predict_fns and strategy in predict_fns:
        fn = predict_fns[strategy]
        out = fn(train, val, H, seed=seed, season_m=season_m)
    else:
        # Import-on-demand — keeps the planner module light
        from importlib import import_module
        mod = import_module(f"research.baseline.{strategy}")
        if strategy in POINT_ONLY:
            point = mod.predict(train, val, H, seed=seed, season_m=season_m)
            return _point_to_degenerate_quantiles(np.asarray(point))
        if not hasattr(mod, "predict_with_uncertainty"):
            point = mod.predict(train, val, H, seed=seed, season_m=season_m)
            return _point_to_degenerate_quantiles(np.asarray(point))
        out = mod.predict_with_uncertainty(train, H, seed=seed, season_m=season_m)

    # Normalize output shape: support tuple (med, ent, [Q,H]) or dict {quantiles:[Q,H],...}
    if isinstance(out, dict):
        q = np.asarray(out["quantiles"])
        levels = np.asarray(out.get("quantile_levels", TARGET_LEVELS))
    elif isinstance(out, tuple) and len(out) >= 3:
        q = np.asarray(out[2])
        # Trust q.shape[0] to determine native grid; fall back to KNOWN_GRIDS by length
        if q.shape[0] == len(TARGET_LEVELS):
            levels = TARGET_LEVELS
        elif q.shape[0] == 21:
            levels = np.linspace(0.0, 1.0, 21)
        else:
            levels = KNOWN_GRIDS.get(strategy, np.linspace(0.0, 1.0, q.shape[0]))
    else:
        # Fell back to point — replicate
        return _point_to_degenerate_quantiles(np.asarray(out))

    # Make sure shape is [Q, H]; if [H, Q] transpose
    if q.shape[0] != len(levels) and q.shape[1] == len(levels):
        q = q.T
    return _align_to_target(q, levels)


def ensemble_predict(plan: PriorPlan, train: np.ndarray, val: np.ndarray,
                     H: int, seed: int = 42, season_m: int = 1,
                     predict_fns: Optional[dict[str, Callable]] = None,
                     ) -> EnsembleResult:
    """Run each strategy, aggregate quantiles via linear pool with plan.weights."""
    if len(plan.strategies) != len(plan.weights):
        raise ValueError("plan.strategies and plan.weights length mismatch")

    per_model: dict[str, np.ndarray] = {}
    weights: dict[str, float] = {}
    for s, w in zip(plan.strategies, plan.weights):
        per_model[s] = fetch_quantiles(s, train, val, H, seed=seed, season_m=season_m,
                                       predict_fns=predict_fns)
        weights[s] = w

    # Linear pool on aligned [9, H] tensors
    stacked = np.stack([per_model[s] for s in plan.strategies], axis=0)   # [K, 9, H]
    w_arr = np.array(plan.weights).reshape(-1, 1, 1)
    ens_q = (stacked * w_arr).sum(axis=0)
    median = ens_q[len(TARGET_LEVELS) // 2]   # q=0.5 row (index 4 of 9)

    return EnsembleResult(median=median, quantiles=ens_q, levels=TARGET_LEVELS,
                          per_model_quantiles=per_model, per_model_weights=weights)


if __name__ == "__main__":
    # Synthetic unit test — avoid remote model loads.
    from research.agent.planner_prior_aware import PriorPlan
    rng = np.random.default_rng(0)
    train = (np.sin(np.arange(100) * 0.1) + 0.1 * rng.standard_normal(100)).astype(np.float32)
    H = 12

    # Fake 3 model predict_fns with distinct quantile spreads
    def fake_a(tr, va, H, **kw):
        med = np.linspace(0.0, 0.5, H)
        spread = np.linspace(0.1, 0.3, H)
        q = np.stack([med + (lvl - 0.5) * spread * 2 for lvl in TARGET_LEVELS])
        return med, 0.0, q
    def fake_b(tr, va, H, **kw):
        med = np.linspace(0.0, 0.5, H) + 0.1
        spread = np.linspace(0.05, 0.2, H)
        q = np.stack([med + (lvl - 0.5) * spread * 2 for lvl in TARGET_LEVELS])
        return med, 0.0, q
    def fake_c(tr, va, H, **kw):
        med = np.linspace(0.0, 0.5, H) - 0.05
        spread = np.linspace(0.2, 0.5, H)
        q = np.stack([med + (lvl - 0.5) * spread * 2 for lvl in TARGET_LEVELS])
        return med, 0.0, q

    plan = PriorPlan(level="L2", strategies=["chronos2", "tirex", "toto"],
                     weights=[0.5, 0.3, 0.2], combine="ensemble",
                     reason="synthetic")
    fns = {"chronos2": fake_a, "tirex": fake_b, "toto": fake_c}
    result = ensemble_predict(plan, train, np.array([], dtype=np.float32), H=H,
                              predict_fns=fns)
    print(f"  ensemble median shape: {result.median.shape}")
    print(f"  ensemble quantiles shape: {result.quantiles.shape}  levels={result.levels}")
    print(f"  median[0:3]: {result.median[:3]}")
    print(f"  q10[0:3]: {result.quantiles[0, :3]}  q90[0:3]: {result.quantiles[-1, :3]}")
    print(f"  per-model weights used: {result.per_model_weights}")

    # Sanity: ensemble median ≈ Σ w_k · model_k_median
    expected = 0.5 * fake_a(train, None, H)[0] + 0.3 * fake_b(train, None, H)[0] + 0.2 * fake_c(train, None, H)[0]
    err = float(np.max(np.abs(result.median - expected)))
    print(f"  median linear-pool sanity: max|err| = {err:.6f}  {'OK' if err < 1e-6 else 'FAIL'}")
