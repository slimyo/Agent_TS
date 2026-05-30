"""Round 7 M2 · Model 自动淘汰 (feedback 前§4 表 "Model 淘汰机制").

按 per-(regime, model) bandit belief 把系统性差的模型从候选集屏蔽掉，
防止 model library 永久污染。被淘汰的 (regime, model) 对在后续 router
决策中通过 `EliminationPrior` 加上 -∞-等价的 log_prior。

API:
    cull_models(state, candidates, config) -> dict (summary)
    EliminationPrior(state_ref=state)

Safety constraints:
    - min_keep_per_regime ≥ 2: 每 regime 至少留 2 个候选
    - protect: tuple[str] — 永不淘汰（默认 naive_drift + chronos2 兜底）
    - min_observations: 该 (regime, model) 至少观察过 N 次才能被淘汰
    - resurrect_on_drift: drift_engine 触发 boost_exploration 时自动清空 culled

State mutation:
    state.culled: dict[int, set[str]]   # regime → culled models
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math


# ─── Config ──────────────────────────────────────────────────────────────────


@dataclass
class CullingConfig:
    fraction: float = 0.15
    min_keep: int = 2
    min_observations: int = 5
    protect: tuple = ("naive_drift", "chronos2")
    resurrect_on_drift: bool = True


# ─── Core ────────────────────────────────────────────────────────────────────


def _belief(state, regime: int, model: str) -> tuple[float, int]:
    """Return (μ, n) for (regime, model). Returns (+inf, 0) if not seen."""
    if not hasattr(state, "bandit") or not hasattr(state.bandit, "_state"):
        return float("inf"), 0
    rec = state.bandit._state.get((regime, model))
    if rec is None:
        return float("inf"), 0
    n, s, _ = rec
    if n < 1e-6:
        return float("inf"), 0
    return s / n, int(n)


def cull_models(state, candidates: list[str],
                config: Optional[CullingConfig] = None) -> dict:
    """Update `state.culled` (dict[regime, set[str]]) based on bandit beliefs.

    For each regime seen in state.bandit._state:
      1. Compute μ for every candidate model with ≥ min_observations obs
      2. Drop models in `protect` from elimination consideration
      3. Sort by μ ascending (lower=better); cull bottom `fraction` of the rest
      4. Refuse to cull if it would drop the keep count below `min_keep`

    Returns a per-regime summary dict.
    """
    if config is None:
        config = CullingConfig()
    if not hasattr(state, "bandit") or not hasattr(state.bandit, "_state"):
        return {"status": "skipped", "reason": "no bandit state"}

    # Discover regimes from bandit
    regimes = sorted({r for (r, _) in state.bandit._state.keys()})
    if not hasattr(state, "culled") or state.culled is None:
        state.culled = {}

    summary = {"status": "ok", "regimes": {}, "newly_culled": [],
                "total_culled": 0}
    for r in regimes:
        # Per-regime ranking
        scored: list[tuple[str, float, int]] = []
        for m in candidates:
            mu, n = _belief(state, r, m)
            if math.isinf(mu): continue
            scored.append((m, mu, n))
        if not scored:
            summary["regimes"][r] = {"reason": "no observations"}
            continue

        # Eligibility: enough obs, not protected, not already culled
        already = state.culled.get(r, set())
        eligible = [t for t in scored
                    if t[2] >= config.min_observations
                    and t[0] not in config.protect
                    and t[0] not in already]
        if not eligible:
            summary["regimes"][r] = {
                "reason": "no eligible candidates",
                "scored": [(m, round(mu, 3), n) for m, mu, n in scored],
            }
            continue

        # Already-protected + already-culled don't count toward keep_count
        # (they are not "live" candidates anyway)
        total_live = len(eligible) + sum(
            1 for t in scored if t[0] in config.protect)
        if total_live <= config.min_keep:
            summary["regimes"][r] = {
                "reason": f"would breach min_keep={config.min_keep}",
                "live_count": total_live,
            }
            continue

        # Cull bottom `fraction`
        eligible.sort(key=lambda t: t[1], reverse=True)   # WORST first
        n_to_cull = max(1, int(math.ceil(config.fraction * len(eligible))))
        n_to_cull = min(n_to_cull, total_live - config.min_keep)
        cull_now = eligible[:n_to_cull]

        if cull_now:
            state.culled.setdefault(r, set()).update(t[0] for t in cull_now)
            for (m, mu, n) in cull_now:
                summary["newly_culled"].append(
                    {"regime": int(r), "model": m, "mu": round(mu, 3), "n": int(n)})
            summary["total_culled"] += len(cull_now)
        summary["regimes"][r] = {
            "ranked": [(m, round(mu, 3), n) for m, mu, n in scored],
            "culled_now":   [t[0] for t in cull_now],
            "already_culled": sorted(state.culled.get(r, set())),
        }
    return summary


def resurrect(state, regime: Optional[int] = None) -> dict:
    """Clear culled set (per-regime or globally) — used on drift response."""
    if not hasattr(state, "culled") or state.culled is None:
        return {"cleared": []}
    if regime is None:
        cleared = {r: sorted(ms) for r, ms in state.culled.items()}
        state.culled = {}
        return {"cleared": cleared}
    ms = state.culled.pop(regime, None)
    return {"cleared": {regime: sorted(ms) if ms else []}}


# ─── EliminationPrior (BayesianRouter factor) ────────────────────────────────


@dataclass
class EliminationPrior:
    """log_prior = log_factor (≈ -∞) for (regime, model) in state.culled.

    Reads regime from ctx.features["regime"] (set by adaptive_planner).
    Falls back to 0.0 mask when state_ref or features missing — safe no-op.
    """
    name: str = "elimination"
    state_ref: object = None
    log_factor: float = -50.0    # exp(-50) ≈ 1e-22, effective mask

    def log_prior(self, candidates, ctx):
        if self.state_ref is None or ctx.features is None:
            return {m: 0.0 for m in candidates}
        regime = ctx.features.get("regime")
        if regime is None:
            return {m: 0.0 for m in candidates}
        culled = getattr(self.state_ref, "culled", {}).get(regime, set())
        return {m: (self.log_factor if m in culled else 0.0) for m in candidates}

    def __call__(self, candidates, ctx):
        return self.log_prior(candidates, ctx)


# ─── Smoke ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Round 7 M2 · Model culling smoke")
    print("=" * 60)
    from research.agent.router_state import RouterState

    state = RouterState()
    cands = ["chronos2", "tirex", "toto", "moirai", "naive_drift"]

    # Regime 0: chronos2 best (0.30), tirex ok (0.40), toto bad (0.85),
    #           moirai catastrophic (1.20), naive_drift fallback (0.50)
    obs = [
        (0, "chronos2",   0.28), (0, "chronos2",   0.30), (0, "chronos2",   0.32),
        (0, "chronos2",   0.31), (0, "chronos2",   0.29),
        (0, "tirex",      0.38), (0, "tirex",      0.40), (0, "tirex",      0.42),
        (0, "tirex",      0.39), (0, "tirex",      0.41),
        (0, "toto",       0.80), (0, "toto",       0.85), (0, "toto",       0.90),
        (0, "toto",       0.82), (0, "toto",       0.88),
        (0, "moirai",     1.15), (0, "moirai",     1.20), (0, "moirai",     1.25),
        (0, "moirai",     1.18), (0, "moirai",     1.22),
        (0, "naive_drift", 0.48), (0, "naive_drift", 0.50), (0, "naive_drift", 0.52),
    ]
    for r, m, l in obs:
        state.bandit.observe(r, m, l)

    cfg = CullingConfig(fraction=0.30, min_keep=2, min_observations=3)
    summary = cull_models(state, cands, cfg)
    print(f"\nstatus: {summary['status']}  total_culled: {summary['total_culled']}")
    print(f"newly culled:")
    for c in summary["newly_culled"]:
        print(f"  regime={c['regime']:<2} {c['model']:<12} μ={c['mu']:.3f} n={c['n']}")
    print(f"\nstate.culled = {state.culled}")
    print(f"\nper-regime detail:")
    for r, info in summary["regimes"].items():
        print(f"  regime {r}: {info}")

    # ─── EliminationPrior factor test ──────────────────────────────────
    print(f"\n[EliminationPrior factor]")
    from research.agent.bayesian_router import Context
    elim = EliminationPrior(state_ref=state)
    ctx = Context(dataset="X", N=50, H=24,
                  features={"z": None, "regime": 0})
    lp = elim.log_prior(cands, ctx)
    for m in cands:
        mark = " ← masked" if lp[m] < -10 else ""
        print(f"  {m:<14} log_prior = {lp[m]:>+8.2f}{mark}")

    # Resurrect test
    print(f"\n[resurrect on drift]")
    res = resurrect(state)
    print(f"  cleared: {res['cleared']}")
    print(f"  state.culled after resurrect: {state.culled}")
