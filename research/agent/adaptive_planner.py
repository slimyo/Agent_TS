"""Round 6 A3 · Unified Adaptive Planner — single entry point for all tasks.

Replaces the per-task entry points (forecaster_reflect ADAPTTS_PLANNER=bandit
branch, clf_planner use_bayesian flag) with a uniform interface:

    plan = adaptive_decide(task, series, candidates, config, state)
    # run plan.chosen ...
    adaptive_observe(state, plan, outcome)

Where:
    task: "forecast" | "classification" | "anomaly"
    series: np.ndarray [L] (univariate) or [C, L] (multi-channel)
    candidates: list of allowed model names
    config: RouterConfig
    state: RouterState

The planner internally:
    1. Builds embedding z = f_φ(series)
    2. Assigns regime r via regime_fn from state
    3. Constructs Context + Evidence
    4. Calls BayesianRouter.decide(...)
    5. Logs telemetry to state
    6. Returns Plan dataclass

This decouples task-specific wrappers from routing logic.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np

from research.agent.bayesian_router import (
    BayesianRouter, RouterConfig, Context, Evidence,
    AvailabilityPrior, NPrior, TypePrior, EntropyPrior, IndustrialPrior,
    BanditLikelihoodFactor, CVLikelihood, MemoryLikelihood,
)


@dataclass
class AdaptivePlan:
    """Result of adaptive_decide()."""
    task: str
    chosen: str
    posterior: dict[str, float]
    top_k: list[tuple[str, float]]   # ranked candidates with posterior
    regime: Optional[int]
    z: Optional[np.ndarray]
    decide_mode: str
    config_snapshot: dict
    prior_contribs: dict
    lik_contribs: dict


# ─── Embedding routing (A3 sub-component) ─────────────────────────────────────

def _get_embedding(name: str):
    """Lazy load embedding by name."""
    from research.agent.representation import (
        HandFeatureEmbedding, MomentEmbedding, Chronos2Embedding)
    if name == "hand25":
        return HandFeatureEmbedding()
    elif name == "moment":
        return MomentEmbedding()
    elif name == "chronos2":
        return Chronos2Embedding()
    else:
        raise ValueError(f"unknown embedding: {name}")


def _get_regime_fn(state, embedding):
    """Build regime_fn: z → int from current state's centroids."""
    if state.regime_centroids is None:
        return lambda z: 0
    centroids = state.regime_centroids
    def regime_fn(z):
        z = np.asarray(z)
        if z.ndim == 1:
            z = z[None, :]
        sims = z @ centroids.T   # cosine since L2-normalized
        return int(np.argmax(sims[0]))
    return regime_fn


# ─── Public API ───────────────────────────────────────────────────────────────

def adaptive_decide(
    task: str,
    series: np.ndarray,
    candidates: list[str],
    config: RouterConfig,
    state,                              # research.agent.router_state.RouterState
    cv_losses: Optional[dict[str, float]] = None,
    cv_std: Optional[dict[str, float]] = None,
    dataset: Optional[str] = None,
    N: Optional[int] = None,
    H: Optional[int] = None,
    industrial_signal: Optional[float] = None,
) -> AdaptivePlan:
    """Single decision entry point.

    Builds embedding → regime → Bayesian posterior → decide.
    Logs to state.telemetry.
    """
    # 1. embed
    emb = _get_embedding(config.embedding_name)
    z = emb.embed(series.astype(np.float32))

    # 2. regime
    regime_fn = _get_regime_fn(state, emb)
    regime = regime_fn(z)

    # 3. priors / likelihoods — assemble from config + auto bandit factor
    priors = list(config.priors) if config.priors else [
        AvailabilityPrior(local_models=tuple(candidates), remote_models=()),
        NPrior(default_model=candidates[0], N_threshold=15, strength=2.0),
        TypePrior(),
    ]
    likelihoods = list(config.likelihoods) if config.likelihoods else []

    bandit_lf = None
    if config.enable_bandit:
        bandit_lf = BanditLikelihoodFactor(
            bandit=state.bandit, regime_fn=regime_fn, scale=1.0,
            thompson_noise=(config.decide_mode == "thompson"),
            state_ref=state,   # Round 6 B3 · consume bandit_explore_scale
        )
        # ensure bandit factor in likelihoods (not duplicate)
        if not any(isinstance(l, BanditLikelihoodFactor) for l in likelihoods):
            likelihoods.append(bandit_lf)
    if cv_losses and not any(isinstance(l, CVLikelihood) for l in likelihoods):
        likelihoods.append(CVLikelihood(sigma_sq=0.5))

    # Round 6 B3 · auto-attach state_ref to any drift-aware factor that came
    # in via config.likelihoods (MemoryLikelihood, BanditLikelihoodFactor).
    for lf in likelihoods:
        if hasattr(lf, "state_ref") and getattr(lf, "state_ref", None) is None:
            lf.state_ref = state

    # Round 7 M2 · auto-attach EliminationPrior (consumes state.culled)
    from research.agent.model_culling import EliminationPrior
    if not any(isinstance(p, EliminationPrior) for p in priors):
        priors.append(EliminationPrior(state_ref=state))
    for p in priors:
        if isinstance(p, EliminationPrior) and p.state_ref is None:
            p.state_ref = state

    # Round 7 M3 · seed each prior with previously-learned strength
    from research.agent.prior_learning import apply_learned_strengths

    # 4. Build router
    router = BayesianRouter(
        candidates=list(candidates),
        priors=priors, likelihoods=likelihoods,
        bandit_factor=bandit_lf, regime_fn=regime_fn,
    )
    apply_learned_strengths(router, state)

    # 5. Build Context + Evidence
    ctx = Context(
        dataset=dataset, N=N, H=H, industrial=industrial_signal,
        features={"z": z, "regime": int(regime)},   # Round 7 M2: expose regime
        allow_remote=config.enable_remote,
    )
    ev = Evidence(cv_losses=cv_losses, cv_std=cv_std)

    # 6. Decide + capture prior/likelihood contributions for telemetry
    prior_contribs = {pf.name if hasattr(pf, "name") else type(pf).__name__:
                       pf(candidates, ctx) for pf in priors}
    lik_contribs = {lf.name if hasattr(lf, "name") else type(lf).__name__:
                     lf(candidates, ctx, ev) for lf in likelihoods}
    # Round 8 M1 · Meta-bandit selects decide_mode when enabled / set to "auto"
    decide_mode_used = config.decide_mode
    meta_info = None
    if config.meta_bandit_enable or config.decide_mode == "auto":
        from research.agent.meta_bandit import (
            MetaBanditState, MetaBanditConfig, select_mode, from_dict)
        mb_state = from_dict(state.meta_bandit_dict) \
                   if state.meta_bandit_dict else MetaBanditState(
                       decay=config.meta_bandit_decay)
        mb_cfg = MetaBanditConfig(
            cold_start_K=config.meta_bandit_cold_K,
            selection=config.meta_bandit_selection,
        )
        decide_mode_used, meta_info = select_mode(mb_state, mb_cfg)
        # stash back so observe sees latest serialized form
        from research.agent.meta_bandit import to_dict
        state.meta_bandit_dict = to_dict(mb_state)

    chosen, post = router.decide(ctx, ev, mode=decide_mode_used,
                                  lam=config.risk_lambda)

    # 7. Build plan
    top_k = sorted(post.items(), key=lambda kv: -kv[1])[:5]
    plan = AdaptivePlan(
        task=task,
        chosen=chosen,
        posterior=dict(post),
        top_k=top_k,
        regime=regime,
        z=z,
        decide_mode=decide_mode_used,
        config_snapshot={
            "embedding_name": config.embedding_name,
            "meta_bandit_enable": config.meta_bandit_enable
                                   or config.decide_mode == "auto",
            "meta_info": meta_info,
            "enable_remote": config.enable_remote,
            "enable_bandit": config.enable_bandit,
            "risk_lambda": config.risk_lambda,
            "drift_check_every": config.drift_check_every,
            "drift_min_observations": config.drift_min_observations,
            "drift_apply": config.drift_apply,
            # Round 7 M2
            "candidates": list(candidates),
            "cull_every": config.cull_every,
            "cull_fraction": config.cull_fraction,
            "cull_min_keep": config.cull_min_keep,
            "cull_min_observations": config.cull_min_observations,
            "cull_protect": tuple(config.cull_protect),
            "cull_resurrect_on_drift": config.cull_resurrect_on_drift,
            # Round 7 M3 — stash live priors so EB learner can mutate in place
            "priors_handle": priors,
            "eb_learn_every": config.eb_learn_every,
            "eb_lr": config.eb_lr,
            "eb_max_strength": config.eb_max_strength,
            "eb_min_samples": config.eb_min_samples,
        },
        prior_contribs={k: dict(v) for k, v in prior_contribs.items()},
        lik_contribs={k: dict(v) for k, v in lik_contribs.items()},
    )

    # 8. Telemetry
    state.log_decision(
        ctx=ctx, chosen=chosen, posterior=post,
        prior_contribs=plan.prior_contribs,
        lik_contribs=plan.lik_contribs,
        decide_mode=config.decide_mode,
    )

    return plan


def adaptive_observe(state, plan: AdaptivePlan, outcome: float) -> dict:
    """Close the loop after observing actual outcome.

    Updates BanditState + RouterState memory + tags last telemetry record.
    """
    # 1. bandit update (per-regime, per-chosen)
    if plan.regime is not None:
        state.bandit.observe(plan.regime, plan.chosen, float(outcome))

    # Round 8 M1 · update meta-bandit on actual decide_mode used
    if plan.config_snapshot.get("meta_bandit_enable") and \
            getattr(plan, "decide_mode", None) is not None:
        try:
            from research.agent.meta_bandit import (
                MetaBanditState, from_dict, to_dict)
            mb_state = from_dict(state.meta_bandit_dict) \
                       if state.meta_bandit_dict else MetaBanditState()
            mb_state.observe(plan.decide_mode, float(outcome))
            state.meta_bandit_dict = to_dict(mb_state)
        except Exception as e:
            pass  # silent: meta-bandit is non-critical
    # 2. memory storage (z + outcome)
    fake_ctx = type("Ctx", (), {
        "dataset": plan.config_snapshot.get("dataset"),
        "N": None, "H": None,
    })()
    state.add_observation(fake_ctx, z=plan.z, chosen=plan.chosen,
                          outcome=outcome, regime=plan.regime)

    result = {"regime": plan.regime, "chosen": plan.chosen,
              "outcome": float(outcome),
              "updated_belief": state.bandit.belief(plan.regime, plan.chosen)
                                if plan.regime is not None else None}

    # Round 6 B3 · auto-drift trigger
    snap = plan.config_snapshot
    every = int(snap.get("drift_check_every", 0) or 0)
    min_obs = int(snap.get("drift_min_observations", 100) or 100)
    if every > 0 and state.n_observations >= min_obs \
            and (state.n_observations % every == 0):
        try:
            from research.agent.drift_engine import run_drift_step
            drift_out = run_drift_step(state, config=None,
                                        apply=bool(snap.get("drift_apply", True)))
            state.last_drift_check = drift_out
            result["drift"] = drift_out
            # Round 7 M2 · resurrect culled models on drift (boost_exploration)
            if snap.get("cull_resurrect_on_drift", True) and \
                    drift_out.get("actions"):
                kinds = {a["kind"] for a in drift_out["actions"]}
                if "boost_exploration" in kinds and getattr(state, "culled", None):
                    from research.agent.model_culling import resurrect
                    result["culled_reset"] = resurrect(state)
        except Exception as e:
            result["drift_err"] = f"{type(e).__name__}: {e}"

    # Round 7 M2 · model culling trigger
    cull_every = int(snap.get("cull_every", 0) or 0)
    if cull_every > 0 and state.n_observations >= cull_every \
            and (state.n_observations % cull_every == 0):
        try:
            from research.agent.model_culling import cull_models, CullingConfig
            cfg_cull = CullingConfig(
                fraction=float(snap.get("cull_fraction", 0.15)),
                min_keep=int(snap.get("cull_min_keep", 2)),
                min_observations=int(snap.get("cull_min_observations", 5)),
                protect=tuple(snap.get("cull_protect",
                                        ("naive_drift", "chronos2"))),
            )
            cands = snap.get("candidates", [])
            if cands:
                result["culling"] = cull_models(state, list(cands), cfg_cull)
        except Exception as e:
            result["culling_err"] = f"{type(e).__name__}: {e}"

    # Round 7 M3 · Empirical Bayes prior strength learning
    eb_every = int(snap.get("eb_learn_every", 0) or 0)
    if eb_every > 0 and state.n_observations >= eb_every \
            and (state.n_observations % eb_every == 0):
        try:
            from research.agent.prior_learning import (
                learn_prior_strengths, EBConfig)
            from research.agent.bayesian_router import BayesianRouter
            # Reconstruct a thin router view over config.priors that holds the
            # *current* prior instances (which adaptive_decide attached). We use
            # the stashed router_handle when present; else use config.priors.
            router_view = type("RouterView", (), {
                "priors": snap.get("priors_handle", []),
            })()
            if router_view.priors:
                cfg_eb = EBConfig(
                    lr=float(snap.get("eb_lr", 0.05)),
                    max_strength=float(snap.get("eb_max_strength", 5.0)),
                    min_samples=int(snap.get("eb_min_samples", 30)),
                )
                result["eb_learning"] = learn_prior_strengths(
                    state, router_view, cfg_eb)
        except Exception as e:
            result["eb_err"] = f"{type(e).__name__}: {e}"

    return result


# ─── Demo ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Round 6 A3 · Unified adaptive planner smoke")
    print("=" * 60)
    from research.agent.router_state import RouterState

    rng = np.random.default_rng(0)
    series = (np.sin(np.arange(100) * 0.1) + 0.1 * rng.standard_normal(100)).astype(np.float32)

    candidates = ["chronos2", "tirex", "toto", "naive_drift", "arima_ets"]
    cfg = RouterConfig(
        decide_mode="argmax", enable_bandit=True, embedding_name="hand25",
    )
    state = RouterState()

    plan = adaptive_decide("forecast", series, candidates, cfg, state,
                            dataset="ETTh1", N=50, H=24)
    print(f"task={plan.task}  chosen={plan.chosen}  regime={plan.regime}")
    print(f"top_k: {plan.top_k[:3]}")
    print(f"prior factors: {list(plan.prior_contribs)}")
    print(f"lik factors:   {list(plan.lik_contribs)}")

    # simulate outcome (chosen=naive_drift bad)
    res = adaptive_observe(state, plan, outcome=0.42)
    print(f"\nobserve result: {res}")
    print(f"state.summary: {state.summary()}")

    # second decide: same series, expect bandit factor influenced
    plan2 = adaptive_decide("forecast", series, candidates, cfg, state,
                             dataset="ETTh1", N=50, H=24)
    print(f"\n2nd call chose: {plan2.chosen}  posterior: "
          + " ".join(f"{k}={v:.3f}" for k,v in plan2.top_k[:3]))
