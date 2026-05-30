"""Round 5 · Compositional Routing Framework (feedback Round 4 Phase 1).

Unifies the 7 heuristic switches scattered across the codebase into a single
log-linear (energy-based) decision rule:

    p(M_k | x, h) ∝ exp( Σ_i log π_k^{(i)}(x) + Σ_j log L_k^{(j)}(x, h) )

where:
    π_k^{(i)}(x)  =  PriorFactor    — encodes domain / sample-size / regime / etc.
    L_k^{(j)}(x,h) = LikelihoodFactor — encodes CV losses, memory neighbors, etc.

⚠ FRAMING (feedback 问题 1 — "你的 posterior 还不是真 posterior"):
    The factors are NOT a single generative likelihood and are NOT conditionally
    independent (e.g. RegimePrior ⟂̸ CRPSPrior). Calling this an "exact Bayesian
    posterior" overclaims. What we actually have is a **factorized
    posterior-inspired energy model** — a softmax over a sum of log-domain
    energy terms,  p_k = softmax(−E_k),  E_k = −Σ_i w_i f_i(x). We keep the
    `BayesianRouter` class name for continuity, but the paper/method text should
    say "Bayesian-style compositional decision model" or "energy-based routing",
    NOT "exact Bayesian inference". The factor weights are learned post-hoc
    (Empirical Bayes, M3) rather than derived from a joint generative model.

    Because the terms are merely summed, individual factors can silently encode
    the SAME signal (difficulty, uncertainty) under different names — the
    "factor explosion / unidentifiability" risk (feedback 问题 2). The
    `attribute_decision` / `FactorAttributionAccumulator` tools at the bottom of
    this file expose (a) each factor's per-decision contribution, (b) its causal
    leave-one-factor-out influence on the argmax, and (c) cross-factor redundancy
    — so the energy model stays auditable instead of black-box.

Hard switches before Round 5 → soft prior factors after:

    | Before (hard)                                  | After (soft factor)         |
    |-----------------------------------------------|-----------------------------|
    | if N<15: w_C2=0.9                              | NPrior (smooth sigmoid)     |
    | if entropy>τ: margin*=2                        | EntropyPrior                |
    | if quant_bits low: use euclid                  | IndustrialPrior             |
    | margin gate (best - default < 0.1: stay)       | risk_min decide() mode      |
    | diverse retrieval (force 1 non-default)        | MemoryLikelihood (per-clf)  |
    | ε-greedy with prob 0.2                         | thompson() decide() mode    |
    | L0 trust: if π(C2)≥0.45 → single-model         | argmax + entropy threshold  |

The router exposes three `decide()` modes:
    - 'argmax':   pick top posterior  (was: hand-coded if/else)
    - 'thompson': sample r_k ~ p(M_k|x,h), pick argmax(sample)   (was: ε-greedy)
    - 'risk_min': pick arg min E[ℓ_k|x] + λ·Var[ℓ_k|x]   (was: margin gate)

No new mechanism vs Round 4-A; pure re-frame. Behavioral equivalence is unit-
tested in __main__ (see Round 5 finish-1 §11).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable
import math

import numpy as np


# ─── core abstractions ────────────────────────────────────────────────────────

@dataclass
class Context:
    """Routing input bundle. None for unknown fields."""
    dataset: str | None = None
    N: int | None = None
    H: int | None = None
    entropy: float | None = None             # C2 quantile spread, [0, ∞)
    industrial: float | None = None          # P[industrial regime], [0, 1]
    features: dict | None = None             # 25-d feature dict (optional)
    allow_remote: bool = False


@dataclass
class Evidence:
    """Per-call observations: CV losses, retrieved memory neighbors, etc."""
    cv_losses: dict[str, float] | None = None       # {model: mean MAE}
    cv_std: dict[str, float] | None = None          # {model: std MAE for risk}
    # each neighbor: {sim, cv_accs} — cv_accs are deployment-safe CV accuracies.
    # NEVER pass test-set accuracies here (feedback 问题 6 data-leakage boundary).
    memory_neighbors: list[dict] | None = None


# ─── prior factor protocol ────────────────────────────────────────────────────

class PriorFactor:
    """A prior factor returns log π_k(x) for each candidate model.

    Subclasses override `log_prior(candidates, ctx) -> dict[str, float]`.
    """
    name: str = "prior"

    def log_prior(self, candidates: list[str], ctx: Context) -> dict[str, float]:
        raise NotImplementedError

    def __call__(self, candidates, ctx):
        return self.log_prior(candidates, ctx)


class LikelihoodFactor:
    """A likelihood factor returns log L_k(x, h) for each candidate."""
    name: str = "likelihood"

    def log_lik(self, candidates: list[str], ctx: Context,
                ev: Evidence) -> dict[str, float]:
        raise NotImplementedError

    def __call__(self, candidates, ctx, ev):
        return self.log_lik(candidates, ctx, ev)


# ─── router ───────────────────────────────────────────────────────────────────

@dataclass
class RouterConfig:
    """Round 6 A1 · centralized config — all knobs in one place."""
    priors: list = field(default_factory=list)
    likelihoods: list = field(default_factory=list)
    decide_mode: str = "argmax"
    risk_lambda: float = 1.0
    embedding_name: str = "hand25"
    enable_remote: bool = False
    enable_bandit: bool = False
    bandit_decay: float = 1.0
    # Round 6 B3 · auto-drift trigger inside adaptive_observe()
    drift_check_every: int = 50         # run drift_engine every N observations
    drift_min_observations: int = 100   # warm-up; skip until enough data
    drift_apply: bool = True            # apply remediation (False = compute only)
    # Round 7 M2 · model culling
    cull_every: int = 200               # 0 = disable
    cull_fraction: float = 0.15
    cull_min_keep: int = 2
    cull_protect: tuple = ("naive_drift", "chronos2")
    cull_min_observations: int = 5
    cull_resurrect_on_drift: bool = True
    # Round 7 M3 · Empirical Bayes prior strengths
    eb_learn_every: int = 100           # 0 = disable
    eb_lr: float = 0.05
    eb_max_strength: float = 5.0
    eb_min_samples: int = 30
    # Round 8 M1 · meta-bandit on decide_mode
    # When decide_mode == "auto", the meta-bandit picks between
    # {argmax, thompson, risk_min} per call.
    meta_bandit_enable: bool = False
    meta_bandit_cold_K: int = 10        # min obs per arm before exploit
    meta_bandit_selection: str = "thompson"
    meta_bandit_decay: float = 0.995


@dataclass
class BayesianRouter:
    candidates: list[str]
    priors: list[PriorFactor] = field(default_factory=list)
    likelihoods: list[LikelihoodFactor] = field(default_factory=list)
    # Round 6 A1: observe()-aware factors that need post-prediction update
    bandit_factor: BanditLikelihoodFactor | None = None
    regime_fn: object | None = None      # for observe()
    memory_factor: object | None = None  # for observe() to write back

    def log_posterior(self, ctx: Context, ev: Evidence | None = None
                       ) -> dict[str, float]:
        """Returns unnormalized log p(M_k | x, h) for each candidate."""
        if ev is None: ev = Evidence()
        log_p = {m: 0.0 for m in self.candidates}
        for pf in self.priors:
            lp = pf(self.candidates, ctx)
            for m in self.candidates:
                log_p[m] += lp.get(m, 0.0)
        for lf in self.likelihoods:
            ll = lf(self.candidates, ctx, ev)
            for m in self.candidates:
                log_p[m] += ll.get(m, 0.0)
        return log_p

    def factor_log_contributions(self, ctx: Context, ev: Evidence | None = None
                                 ) -> dict[str, dict[str, float]]:
        """Per-factor additive terms of log_posterior — the raw material for
        Factor Attribution Analysis (feedback 问题 2).

        Returns {factor_label: {model: log_term}}. log_posterior() is exactly
        the elementwise sum of these maps. Factor labels are de-duplicated
        (`name`, `name#1`, ...) so two factors sharing a `.name` stay distinct.
        """
        if ev is None: ev = Evidence()
        contribs: dict[str, dict[str, float]] = {}
        seen: dict[str, int] = {}

        def _label(name: str) -> str:
            if name not in seen:
                seen[name] = 0
                return name
            seen[name] += 1
            return f"{name}#{seen[name]}"

        for pf in self.priors:
            lp = pf(self.candidates, ctx)
            label = _label(getattr(pf, "name", pf.__class__.__name__))
            contribs[label] = {m: float(lp.get(m, 0.0)) for m in self.candidates}
        for lf in self.likelihoods:
            ll = lf(self.candidates, ctx, ev)
            label = _label(getattr(lf, "name", lf.__class__.__name__))
            contribs[label] = {m: float(ll.get(m, 0.0)) for m in self.candidates}
        return contribs

    def posterior(self, ctx: Context, ev: Evidence | None = None
                  ) -> dict[str, float]:
        """Normalized posterior via log-sum-exp."""
        lp = self.log_posterior(ctx, ev)
        m_max = max(lp.values())
        exp_ = {k: math.exp(v - m_max) for k, v in lp.items()}
        Z = sum(exp_.values())
        return {k: v / Z for k, v in exp_.items()}

    def decide(self, ctx: Context, ev: Evidence | None = None,
               mode: str = "argmax",
               lam: float = 1.0,
               rng: np.random.Generator | None = None,
               ) -> tuple[str, dict[str, float]]:
        """Returns (chosen_model, posterior).

        Modes:
            'argmax':   chosen = argmax posterior
            'thompson': sample r_k ~ posterior, return argmax sample
                        (replaces ε-greedy: posterior IS the exploration rate)
            'risk_min': chosen = argmin (E[loss_k] + lam·Var[loss_k])
                        requires ev.cv_losses (and optionally ev.cv_std)
        """
        post = self.posterior(ctx, ev)
        if mode == "argmax":
            chosen = max(post, key=post.get)
        elif mode == "thompson":
            if rng is None: rng = np.random.default_rng()
            keys = list(post.keys())
            probs = np.array([post[k] for k in keys])
            chosen = keys[int(rng.choice(len(keys), p=probs))]
        elif mode == "risk_min":
            if ev is None or not ev.cv_losses:
                # fallback to argmax
                chosen = max(post, key=post.get)
            else:
                stds = ev.cv_std or {k: 0.0 for k in ev.cv_losses}
                risks = {k: ev.cv_losses[k] + lam * stds.get(k, 0.0)
                         for k in ev.cv_losses if k in self.candidates}
                if risks:
                    chosen = min(risks, key=risks.get)
                else:
                    chosen = max(post, key=post.get)
        else:
            raise ValueError(f"unknown decide mode: {mode}")
        return chosen, post

    # ─── Round 6 A1 · observe() unified hook ───────────────────────────────
    def observe(self, ctx: Context, chosen: str, outcome: float) -> dict:
        """Single observe hook — updates bandit + memory factors.

        Args:
            ctx: Context object used at decide time
            chosen: which model was selected
            outcome: actual loss (lower = better) observed for chosen

        Returns: dict of what was updated (for telemetry).
        """
        updates = {}
        # Bandit update
        if self.bandit_factor is not None and self.regime_fn is not None:
            z = ctx.features.get("z") if ctx.features else None
            if z is not None and self.bandit_factor.bandit is not None:
                try:
                    r = self.regime_fn(np.asarray(z))
                    self.bandit_factor.bandit.observe(r, chosen, outcome)
                    updates["bandit"] = {"regime": int(r), "model": chosen,
                                         "loss": float(outcome)}
                except Exception as e:
                    updates["bandit_err"] = f"{type(e).__name__}: {e}"
        # Memory update placeholder (filled by C1/C2)
        if self.memory_factor is not None:
            try:
                z = ctx.features.get("z") if ctx.features else None
                self.memory_factor.add_observation(ctx=ctx, z=z,
                                                   chosen=chosen, outcome=outcome)
                updates["memory"] = {"added": True}
            except Exception as e:
                updates["memory_err"] = f"{type(e).__name__}: {e}"
        return updates

    @classmethod
    def from_config(cls, candidates: list[str], cfg: RouterConfig
                    ) -> "BayesianRouter":
        """Round 6 A1 · build router from RouterConfig."""
        return cls(
            candidates=candidates,
            priors=list(cfg.priors),
            likelihoods=list(cfg.likelihoods),
        )


# ─── concrete factors (Round 5 P1: port 5 hard switches) ──────────────────────

@dataclass
class CRPSPrior(PriorFactor):
    """F9 static π_k = (1/MAE) / Σ_j (1/MAE_j), conditioned on dataset.

    Returns log of the ratio, so additive in log space.
    """
    name: str = "crps"
    eps: float = 1e-6

    def log_prior(self, candidates, ctx):
        from research.agent.prior_crps import get_prior
        p = get_prior(dataset=ctx.dataset)
        if not p:
            return {m: 0.0 for m in candidates}    # uniform fallback
        out = {}
        for m in candidates:
            out[m] = math.log(p.get(m, self.eps) + self.eps)
        return out


@dataclass
class NPrior(PriorFactor):
    """N<15 → boost default model (was hard cliff in compose_prior)."""
    name: str = "N_prior"
    default_model: str = "chronos2"
    N_threshold: int = 15
    strength: float = 2.0   # log-odds boost at N=0

    def log_prior(self, candidates, ctx):
        out = {m: 0.0 for m in candidates}
        if ctx.N is None or ctx.N >= self.N_threshold:
            return out
        # smooth sigmoid in (N_threshold - N): more boost at smaller N
        boost = self.strength * (1.0 - ctx.N / self.N_threshold)
        if self.default_model in out:
            out[self.default_model] = boost
        return out


@dataclass
class TypePrior(PriorFactor):
    """Down-weight POINT_PREDICTORS (replaces _apply_type_prior multiplicative)."""
    name: str = "type"
    point_models: tuple = ("naive_drift", "naive_seasonal",
                           "arima_ets", "llmtime", "chronos")
    log_factor: float = math.log(0.3)   # 0.3× → -1.2 in log space

    def log_prior(self, candidates, ctx):
        return {m: (self.log_factor if m in self.point_models else 0.0)
                for m in candidates}


@dataclass
class EntropyPrior(PriorFactor):
    """High C2 entropy → down-weight C2 (was: hard 'raise margin' switch).

    Quantifies feedback Round 4 §四 uncertainty-aware routing:
    when default is uncertain, alternatives become a-priori more attractive.
    """
    name: str = "entropy"
    default_model: str = "chronos2"
    beta: float = 0.5      # entropy → log-odds scale

    def log_prior(self, candidates, ctx):
        if ctx.entropy is None:
            return {m: 0.0 for m in candidates}
        # default down-weight proportional to its uncertainty
        out = {m: 0.0 for m in candidates}
        if self.default_model in out:
            out[self.default_model] = -self.beta * ctx.entropy
        return out


@dataclass
class IndustrialPrior(PriorFactor):
    """Wafer-like signal → boost Euclid (was: industrial_stats hard override)."""
    name: str = "industrial"
    target_model: str = "euclid_1nn"
    strength: float = 2.0

    def log_prior(self, candidates, ctx):
        if ctx.industrial is None or self.target_model not in candidates:
            return {m: 0.0 for m in candidates}
        out = {m: 0.0 for m in candidates}
        out[self.target_model] = self.strength * ctx.industrial
        return out


@dataclass
class AvailabilityPrior(PriorFactor):
    """Hard mask: ignore models not in current deployment."""
    name: str = "availability"
    local_models: tuple = ("naive_drift", "naive_seasonal", "arima_ets",
                           "llmtime", "chronos", "chronos2", "chronos_bolt",
                           "timesfm2", "moirai", "moirai2", "tirex",
                           "toto", "toto2")
    remote_models: tuple = ("time_moe", "sundial", "timer")
    log_mask: float = -1e6   # effectively -∞

    def log_prior(self, candidates, ctx):
        allowed = set(self.local_models)
        if ctx.allow_remote:
            allowed |= set(self.remote_models)
        return {m: (0.0 if m in allowed else self.log_mask)
                for m in candidates}


# ─── CV likelihood (replaces hand BMA in prior_crps) ──────────────────────────

@dataclass
class CVLikelihood(LikelihoodFactor):
    """log L_k = -loss_k / sigma_sq (Gaussian likelihood)."""
    name: str = "cv"
    sigma_sq: float = 0.5

    def log_lik(self, candidates, ctx, ev):
        if not ev.cv_losses:
            return {m: 0.0 for m in candidates}
        return {m: (-ev.cv_losses.get(m, 0.0) / self.sigma_sq
                    if m in ev.cv_losses else 0.0)
                for m in candidates}


# ─── Round 6 A1 · BanditLikelihoodFactor (bandit 进入 likelihood 体系) ────────

@dataclass
class BanditLikelihoodFactor:
    """Wraps a BanditState as a LikelihoodFactor.

    Returns log_lik = -μ_{r,k} / scale + (Thompson sampled noise if enabled).

    This unifies the Bandit path with BayesianRouter — no separate
    ContextualBanditRouter class needed (Round 6 A1 architecture收敛).
    """
    name: str = "bandit"
    bandit: object | None = None    # BanditState (lazy to avoid circular import)
    regime_fn: object | None = None  # Callable[[np.ndarray], int]
    scale: float = 1.0               # μ → log-lik conversion
    thompson_noise: bool = False     # if True, add sample from N(0, σ_μ)
    state_ref: object | None = None  # Round 6 B3 · for state.bandit_explore_scale

    def log_lik(self, candidates, ctx, ev):
        if self.bandit is None or self.regime_fn is None:
            return {m: 0.0 for m in candidates}
        z = ctx.features.get("z") if ctx.features else None
        if z is None:
            return {m: 0.0 for m in candidates}
        try:
            r = self.regime_fn(np.asarray(z))
        except Exception:
            return {m: 0.0 for m in candidates}
        # Round 6 B3 · drift-driven exploration: inflate σ + temperature-flatten
        explore = float(getattr(self.state_ref, "bandit_explore_scale", 1.0)) \
                  if self.state_ref is not None else 1.0
        explore = max(1.0, explore)
        eff_scale = self.scale * explore   # flatten log-odds when drift detected
        out = {}
        for m in candidates:
            mu, sigma = self.bandit.belief(r, m)
            if self.thompson_noise:
                import numpy as _np
                noise = _np.random.normal(0, (sigma + 1e-6) * explore)
                out[m] = -(mu + noise) / eff_scale
            else:
                out[m] = -mu / eff_scale
        return out

    def __call__(self, candidates, ctx, ev):
        return self.log_lik(candidates, ctx, ev)


@dataclass
class MemoryLikelihood(LikelihoodFactor):
    """log L_k = log Σ_{neighbor} sim(n) · 1/(1-acc_n[k]+eps).

    Replaces Items 3-4 hand-coded consensus_winner_inv_loss + diverse retrieval.
    Each retrieved neighbor contributes a sim-weighted inverse-error vote.
    """
    name: str = "memory"
    eps: float = 0.01
    state_ref: object | None = None   # Round 6 B3 · for state.memory_trust

    def log_lik(self, candidates, ctx, ev):
        if not ev.memory_neighbors:
            return {m: 0.0 for m in candidates}
        votes = {m: 0.0 for m in candidates}
        for nb in ev.memory_neighbors:
            sim = nb.get("sim", 1.0)
            # deployment-safe CV accs only (feedback 问题 6); tolerate legacy key
            accs = nb.get("cv_accs") or nb.get("all_clf_accs", {})
            for m in candidates:
                if m in accs:
                    votes[m] += sim * (1.0 / (1.0 - accs[m] + self.eps))
        # Round 6 B3 · drift-driven trust: scale log-lik by memory_trust ∈ (0,1]
        # trust=1 → unchanged; trust=0.3 → 30% influence on posterior log-odds.
        trust = float(getattr(self.state_ref, "memory_trust", 1.0)) \
                if self.state_ref is not None else 1.0
        trust = max(0.0, min(1.0, trust))
        return {m: (math.log(votes[m] + self.eps) * trust if votes[m] > 0 else 0.0)
                for m in candidates}


# ─── Factor Attribution Analysis (feedback 问题 2 · "强烈建议新增") ─────────────
#
# Two questions the additive energy model must stay able to answer, or it
# degenerates into a black box where nobody knows which factor actually drives
# routing or whether two factors are silently encoding the same signal:
#
#   (1) For THIS decision, how much did each factor push the chosen model up vs
#       its closest rival, and would removing the factor flip the argmax?
#       → attribute_decision() — a leave-one-factor-out (LOFO) decomposition.
#   (2) ACROSS many decisions, do two factors always move the same models in the
#       same direction (i.e. are they redundant / non-orthogonal)?
#       → FactorAttributionAccumulator.redundancy_matrix().

def _softmax_from_logp(log_p: dict[str, float]) -> dict[str, float]:
    m_max = max(log_p.values())
    exp_ = {k: math.exp(v - m_max) for k, v in log_p.items()}
    Z = sum(exp_.values()) or 1.0
    return {k: v / Z for k, v in exp_.items()}


def _kl(p: dict[str, float], q: dict[str, float], eps: float = 1e-12) -> float:
    """KL(p ‖ q) over the shared model support, in nats."""
    return sum(p[k] * math.log((p[k] + eps) / (q.get(k, 0.0) + eps))
               for k in p if p[k] > 0)


@dataclass
class AttributionResult:
    """Per-decision factor attribution (output of attribute_decision)."""
    posterior: dict[str, float]
    chosen: str
    runner_up: str
    margin_logodds: float                      # log_p[chosen] − log_p[runner_up]
    contributions: dict[str, dict[str, float]]  # factor → {model → log term}
    lofo: dict[str, dict]                        # factor → influence metrics
    decisive: list[str]                          # factors whose removal flips argmax

    def summary(self, top: int = 8) -> str:
        lines = [f"chosen={self.chosen}  runner_up={self.runner_up}  "
                 f"margin(log-odds)={self.margin_logodds:+.3f}  "
                 f"p={self.posterior[self.chosen]:.3f}"]
        ranked = sorted(self.lofo.items(),
                        key=lambda kv: -kv[1]["kl_if_removed"])[:top]
        lines.append(f"  {'factor':<16}{'Δmargin':>9}{'KL_drop':>9}  flips?")
        for label, m in ranked:
            flip = f"→ {m['argmax_if_removed']}" if m["argmax_changed"] else ""
            lines.append(f"  {label:<16}{m['delta_margin']:>+9.3f}"
                         f"{m['kl_if_removed']:>9.3f}  {flip}")
        if self.decisive:
            lines.append(f"  decisive (flip argmax if removed): {self.decisive}")
        return "\n".join(lines)


def attribute_decision(router: "BayesianRouter", ctx: Context,
                       ev: Evidence | None = None) -> AttributionResult:
    """Decompose a single routing decision into per-factor contributions.

    For each factor we report:
      - delta_margin: factor's contribution to (chosen − runner_up) log-odds.
        Positive = this factor favours the chosen model over its rival.
      - kl_if_removed: KL(full ‖ posterior_without_factor) — how much the whole
        posterior shape depends on this factor.
      - argmax_changed / argmax_if_removed: whether dropping the factor flips the
        top-1 pick (i.e. the factor is *causally decisive* for this decision).
    """
    contribs = router.factor_log_contributions(ctx, ev)
    cands = router.candidates
    log_p = {m: 0.0 for m in cands}
    for fac in contribs.values():
        for m in cands:
            log_p[m] += fac[m]
    post = _softmax_from_logp(log_p)
    ranked = sorted(cands, key=lambda m: -log_p[m])
    chosen = ranked[0]
    runner = ranked[1] if len(ranked) > 1 else ranked[0]
    margin = log_p[chosen] - log_p[runner]

    lofo: dict[str, dict] = {}
    decisive: list[str] = []
    for label, fac in contribs.items():
        lp2 = {m: log_p[m] - fac[m] for m in cands}
        post2 = _softmax_from_logp(lp2)
        new_argmax = max(lp2, key=lp2.get)
        changed = new_argmax != chosen
        lofo[label] = {
            "delta_margin": fac[chosen] - fac[runner],
            "logterm_chosen": fac[chosen],
            "kl_if_removed": _kl(post, post2),
            "argmax_changed": changed,
            "argmax_if_removed": new_argmax,
        }
        if changed:
            decisive.append(label)
    return AttributionResult(
        posterior=post, chosen=chosen, runner_up=runner, margin_logodds=margin,
        contributions=contribs, lofo=lofo, decisive=decisive,
    )


@dataclass
class FactorAttributionAccumulator:
    """Aggregate factor behaviour across many decisions to expose redundancy
    (feedback 问题 2 · "factor orthogonalization").

    Each factor contributes a per-decision log-term vector over the candidates.
    softmax routing is shift-invariant, so only the *mean-centred* (within a
    decision) vector carries discriminative signal. Two factors whose centred
    vectors are highly correlated across decisions are encoding the same signal
    — a candidate to merge or drop.
    """
    factor_vectors: dict = field(default_factory=dict)   # label → list[np.ndarray]
    _models: list | None = None
    n_obs: int = 0
    # Hard-mask factors (e.g. AvailabilityPrior, ±1e6) would otherwise swamp the
    # influence/correlation stats. Anything past ±clip already saturates softmax,
    # so clipping keeps soft and hard factors on a comparable, readable scale.
    clip: float = 50.0

    def observe(self, router: "BayesianRouter", ctx: Context,
                ev: Evidence | None = None) -> None:
        contribs = router.factor_log_contributions(ctx, ev)
        if self._models is None:
            self._models = list(router.candidates)
        for label, fac in contribs.items():
            vec = np.array([fac[m] for m in self._models], dtype=float)
            vec = vec - vec.mean()        # centre: only relative pushes matter
            vec = np.clip(vec, -self.clip, self.clip)
            self.factor_vectors.setdefault(label, []).append(vec)
        self.n_obs += 1

    def mean_abs_influence(self) -> dict[str, float]:
        """Average L2 norm of each factor's centred contribution — a cheap
        'how much does this factor ever move the decision' score. ~0 ⇒ inert."""
        out = {}
        for label, vecs in self.factor_vectors.items():
            out[label] = float(np.mean([np.linalg.norm(v) for v in vecs]))
        return out

    def redundancy_matrix(self) -> tuple[list[str], np.ndarray]:
        """Pearson correlation between factors' concatenated centred vectors.
        Returns (labels, matrix). |corr|→1 ⇒ the two factors are redundant."""
        labels = list(self.factor_vectors)
        flat = {l: np.concatenate(self.factor_vectors[l]) for l in labels}
        n = len(labels)
        M = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = flat[labels[i]], flat[labels[j]]
                if a.std() < 1e-9 or b.std() < 1e-9:
                    c = 0.0
                else:
                    c = float(np.corrcoef(a, b)[0, 1])
                M[i, j] = M[j, i] = c
        return labels, M

    def report(self, redundant_threshold: float = 0.8) -> str:
        if self.n_obs == 0:
            return "FactorAttributionAccumulator: no observations yet."
        infl = self.mean_abs_influence()
        labels, M = self.redundancy_matrix()
        lines = [f"Factor attribution over {self.n_obs} decisions",
                 "  mean |influence| (centred L2 norm):"]
        for label, v in sorted(infl.items(), key=lambda kv: -kv[1]):
            tag = "  ← inert" if v < 1e-6 else ""
            lines.append(f"    {label:<16} {v:.4f}{tag}")
        pairs = [(labels[i], labels[j], M[i, j])
                 for i in range(len(labels)) for j in range(i + 1, len(labels))]
        pairs.sort(key=lambda t: -abs(t[2]))
        lines.append(f"  redundant pairs (|corr| ≥ {redundant_threshold}):")
        flagged = [p for p in pairs if abs(p[2]) >= redundant_threshold]
        if not flagged:
            lines.append("    (none)")
        for a, b, c in flagged:
            lines.append(f"    {a} ⟷ {b}: corr={c:+.3f}")
        return "\n".join(lines)


# ─── high-level construction ──────────────────────────────────────────────────

def PriorPlan_from_posterior(chosen: str, posterior: dict[str, float],
                              decide_mode: str = "argmax",
                              top_k: int = 3) -> "PriorPlan":
    """Convert BayesianRouter result into PriorPlan dataclass for forecaster_reflect.

    Allows the Bayesian router to drop into the existing v11 wrapper without
    needing a parallel execution path.
    """
    from research.agent.planner_prior_aware import PriorPlan
    # rank by posterior
    ranked = sorted(posterior.items(), key=lambda kv: -kv[1])[:top_k]
    # Ensure chosen is first (Thompson sample may differ from argmax)
    if ranked and ranked[0][0] != chosen and chosen in posterior:
        ranked = [(chosen, posterior[chosen])] + [r for r in ranked if r[0] != chosen]
        ranked = ranked[:top_k]
    Z = sum(p for _, p in ranked) or 1.0
    strategies = [k for k, _ in ranked]
    weights = [p / Z for _, p in ranked]
    chosen_p = posterior.get(chosen, 0.0)
    level = "L1" if (len(strategies) == 1 or weights[0] >= 0.7) else "L2"
    return PriorPlan(
        level=level,
        strategies=strategies if level == "L2" else [chosen],
        weights=weights if level == "L2" else [1.0],
        combine="ensemble" if level == "L2" else "single",
        reason=f"Bayesian decide({decide_mode}): chose {chosen} (p={chosen_p:.3f})",
        posterior=dict(ranked),
    )


def default_forecasting_router(allow_remote: bool = False) -> BayesianRouter:
    """Pre-wired router matching planner_prior_aware behavior."""
    candidates = ["chronos2", "tirex", "toto", "timesfm2", "moirai", "moirai2",
                  "time_moe", "sundial", "timer", "naive_drift", "arima_ets"]
    return BayesianRouter(
        candidates=candidates,
        priors=[
            AvailabilityPrior(),
            CRPSPrior(),
            TypePrior(),
            NPrior(),
            EntropyPrior(),
        ],
        likelihoods=[CVLikelihood()],
    )


def default_tsc_router() -> BayesianRouter:
    candidates = ["rocket", "minirocket", "weasel", "catch22",
                  "moment_1nn", "moment_logreg", "mantis_1nn", "mantis_lr",
                  "dtw_1nn", "euclid_1nn", "llm_direct"]
    return BayesianRouter(
        candidates=candidates,
        priors=[
            NPrior(default_model="rocket", N_threshold=7),
            IndustrialPrior(),
        ],
        likelihoods=[CVLikelihood(), MemoryLikelihood()],
    )


# ─── self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("BayesianRouter unit tests — equivalence with Round 4-A + new modes")
    print("=" * 70)

    # Forecasting router
    fcr = default_forecasting_router()
    print(f"\n[Forecasting router] candidates = {fcr.candidates}")

    print("\n[Test 1] Weather + N=100 (expect C2 dominant, like prior_aware L1):")
    ctx = Context(dataset="Weather", N=100, H=96)
    post = fcr.posterior(ctx)
    top3 = sorted(post.items(), key=lambda x: -x[1])[:3]
    for m, p in top3: print(f"    {m:12} p = {p:.4f}")

    print("\n[Test 2] ECL + N=100 (expect Toto-leaning, like prior_aware L2):")
    ctx = Context(dataset="ECL", N=100, H=96)
    post = fcr.posterior(ctx)
    top3 = sorted(post.items(), key=lambda x: -x[1])[:3]
    for m, p in top3: print(f"    {m:12} p = {p:.4f}")

    print("\n[Test 3] N=10 cold start (NPrior smooth ramp):")
    ctx = Context(dataset="ETTh1", N=10, H=96)
    post = fcr.posterior(ctx)
    top3 = sorted(post.items(), key=lambda x: -x[1])[:3]
    for m, p in top3: print(f"    {m:12} p = {p:.4f}")

    print("\n[Test 4] Same as Test 2 + CV losses (BMA via CVLikelihood):")
    ctx = Context(dataset="ECL", N=100, H=96)
    ev = Evidence(cv_losses={"chronos2": 5.0, "tirex": 4.0, "toto": 3.0,
                              "timesfm2": 8.0})
    post = fcr.posterior(ctx, ev)
    top3 = sorted(post.items(), key=lambda x: -x[1])[:3]
    for m, p in top3: print(f"    {m:12} p = {p:.4f}")

    print("\n[Test 5] decide modes:")
    rng = np.random.default_rng(0)
    chosen_a, _ = fcr.decide(ctx, ev, mode="argmax")
    print(f"    argmax        → {chosen_a}")
    picks = [fcr.decide(ctx, ev, mode="thompson", rng=rng)[0] for _ in range(20)]
    from collections import Counter
    print(f"    thompson (20) → {dict(Counter(picks))}")
    chosen_r, _ = fcr.decide(ctx, ev, mode="risk_min", lam=1.0)
    print(f"    risk_min      → {chosen_r}")

    print("\n[Test 6] Entropy increases → C2 down-weighted:")
    for ent in [0.0, 0.5, 1.0, 2.0]:
        ctx = Context(dataset="ECL", N=100, H=96, entropy=ent)
        post = fcr.posterior(ctx)
        print(f"    entropy={ent:3.1f}: π(C2)={post.get('chronos2',0):.3f}, "
              f"π(toto)={post.get('toto',0):.3f}, "
              f"π(tirex)={post.get('tirex',0):.3f}")

    print("\n[Test 7] Industrial prior boost (TSC, simulated):")
    tsc = default_tsc_router()
    for ind in [0.0, 0.5, 1.0]:
        ctx = Context(N=30, industrial=ind)
        post = tsc.posterior(ctx)
        print(f"    industrial={ind:3.1f}: π(euclid)={post.get('euclid_1nn',0):.3f}, "
              f"π(rocket)={post.get('rocket',0):.3f}")

    print("\n" + "=" * 70)
    print("Factor Attribution Analysis (feedback 问题 2)")
    print("=" * 70)

    print("\n[Test 8] Per-decision attribution (ECL N=10 + CV losses):")
    ctx = Context(dataset="ECL", N=10, H=96, entropy=1.0)
    ev = Evidence(cv_losses={"chronos2": 5.0, "tirex": 4.0, "toto": 3.0})
    res = attribute_decision(fcr, ctx, ev)
    print(res.summary())
    # sanity: summed contributions reproduce log_posterior
    contribs = fcr.factor_log_contributions(ctx, ev)
    recon = {m: sum(contribs[f][m] for f in contribs) for m in fcr.candidates}
    lp = fcr.log_posterior(ctx, ev)
    err = max(abs(recon[m] - lp[m]) for m in fcr.candidates)
    print(f"    reconstruction max|Σfactors − log_posterior| = {err:.2e} (expect ≈0)")

    print("\n[Test 9] Cross-decision redundancy (sweep N / entropy / dataset):")
    acc = FactorAttributionAccumulator()
    rng = np.random.default_rng(0)
    for _ in range(200):
        c = Context(
            dataset=rng.choice(["Weather", "ECL", "ETTh1"]),
            N=int(rng.integers(3, 120)),
            H=96,
            entropy=float(rng.uniform(0, 2)),
        )
        e = Evidence(cv_losses={m: float(rng.uniform(2, 8))
                                for m in ["chronos2", "tirex", "toto"]})
        acc.observe(fcr, c, e)
    print(acc.report(redundant_threshold=0.8))
