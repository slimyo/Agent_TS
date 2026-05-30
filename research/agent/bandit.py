"""Phase 2 · Contextual Bandit / Thompson Routing (feedback Round 4 §二).

Maintains per-(regime, model) belief over per-cell loss:
    p(ℓ_k | regime r) = N(μ_r,k, σ_r,k²)

Bayesian update after each observation (Gaussian-Gaussian conjugate, unknown mean
+ unknown variance via Normal-Inverse-Gamma; we use the empirical mean+std with
shrinkage prior, a practical NIG approximation):

    n_post   = n_prior + n_obs
    μ_post   = (n_prior · μ_prior + Σ ℓ_obs) / n_post
    σ²_post  = (n_prior · σ²_prior + Σ (ℓ_obs - μ_post)² + κ) / n_post

Thompson sample per candidate, choose argmin (loss minimization):

    r̃_k ~ N(μ_r,k, σ_r,k²)
    chosen = argmin_k r̃_k

Compared to Round 5 `decide(mode="thompson")`:
  - Round 5 thompson = sample from CURRENT posterior (over all models) - no
    history accumulation, no per-regime context.
  - Phase 2 bandit = per-(regime, model) belief accumulates over time, sample
    each model's r̃ independently → true contextual bandit.

Convergence (theoretical): under stationary regime, μ_r,k → true regime-k mean
loss at rate 1/√n; σ_r,k shrinks → Thompson collapse to argmin (exploitation).
Non-stationary regimes handled via decay (`decay` arg < 1).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional
import math

import numpy as np


@dataclass
class BanditState:
    """Per-(regime, model) running Gaussian belief over per-cell loss.

    Args:
        prior_mu:    prior mean loss for all (regime, model) pairs
        prior_var:   prior variance (uncertainty)
        prior_n:     effective sample count of prior (higher = stronger)
        decay:       multiplicative decay applied before each observe (1.0 = no decay).
                     Use 0.99 for non-stationary regimes.
    """
    prior_mu: float = 1.0
    prior_var: float = 1.0
    prior_n: float = 2.0
    decay: float = 1.0              # default / fallback decay
    # state: (regime, model) → (n_eff, sum_loss, sum_sq)
    _state: dict[tuple, tuple[float, float, float]] = field(default_factory=dict)
    # Round 8 M4 · per-regime decay override (empty = use scalar `decay`)
    regime_decay: dict = field(default_factory=dict)   # {regime → decay}

    def _effective_decay(self, regime: int) -> float:
        """Round 8 M4 · per-regime decay rate (falls back to scalar)."""
        return float(self.regime_decay.get(regime, self.decay))

    def set_regime_decay(self, regime: int, decay: float) -> None:
        """Round 8 M4 · update one regime's decay (clipped to (0, 1])."""
        self.regime_decay[int(regime)] = float(max(1e-6, min(1.0, decay)))

    def _get(self, key: tuple) -> tuple[float, float, float]:
        return self._state.get(key,
            (self.prior_n,
             self.prior_n * self.prior_mu,
             self.prior_n * (self.prior_mu ** 2 + self.prior_var)))

    def observe(self, regime: int, model: str, loss: float) -> None:
        """Bayesian update on a single observation.

        Round 8 M4: decay rate is per-regime when `regime_decay[r]` is set;
        else falls back to scalar `self.decay`.
        """
        key = (regime, model)
        n, s, sq = self._get(key)
        d = self._effective_decay(regime)
        if d < 1.0:
            n *= d; s *= d; sq *= d
        n_new = n + 1
        s_new = s + loss
        sq_new = sq + loss * loss
        self._state[key] = (n_new, s_new, sq_new)

    def belief(self, regime: int, model: str) -> tuple[float, float]:
        """Returns (μ, σ) posterior belief about per-cell loss for (regime, model)."""
        key = (regime, model)
        n, s, sq = self._get(key)
        mu = s / max(n, 1e-9)
        # variance of the posterior PREDICTIVE distribution
        var = max(1e-9, sq / max(n, 1e-9) - mu * mu)
        # epistemic uncertainty (uncertainty about μ): var / n
        sigma_mu = math.sqrt(var / max(n, 1e-9))
        return mu, sigma_mu

    def thompson_sample(self, regime: int, candidates: list[str],
                         rng: Optional[np.random.Generator] = None
                         ) -> dict[str, float]:
        """For each candidate, sample r̃_k ~ N(μ_r,k, σ_r,k²). Returns {model: r̃}.

        Caller does argmin(r̃) for loss minimization or argmax for reward.
        """
        if rng is None: rng = np.random.default_rng()
        out = {}
        for m in candidates:
            mu, sigma = self.belief(regime, m)
            out[m] = float(rng.normal(mu, sigma + 1e-6))
        return out

    def best_arm(self, regime: int, candidates: list[str]) -> str:
        """Exploitation choice: argmin posterior mean."""
        best = None; best_mu = float("inf")
        for m in candidates:
            mu, _ = self.belief(regime, m)
            if mu < best_mu: best, best_mu = m, mu
        return best  # type: ignore

    # ─── persistence ──────────────────────────────────────────────────────
    def save(self, path) -> None:
        """Dump state to jsonl. Each line: {regime, model, n, sum_loss, sum_sq}."""
        import json
        from pathlib import Path
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        meta = {"_meta": True, "prior_mu": self.prior_mu,
                "prior_var": self.prior_var, "prior_n": self.prior_n,
                "decay": self.decay,
                "regime_decay": {str(k): float(v)
                                  for k, v in self.regime_decay.items()}}
        with p.open("w") as fh:
            fh.write(json.dumps(meta) + "\n")
            for (r, m), (n, s, sq) in self._state.items():
                fh.write(json.dumps({"regime": int(r), "model": m,
                                     "n": n, "sum_loss": s, "sum_sq": sq}) + "\n")

    @classmethod
    def load(cls, path) -> "BanditState":
        """Restore from jsonl. Returns fresh state if file missing."""
        import json
        from pathlib import Path
        p = Path(path)
        if not p.exists():
            return cls()
        lines = p.read_text().splitlines()
        meta = json.loads(lines[0]) if lines and lines[0].startswith('{"_meta"') else {}
        state = cls(prior_mu=meta.get("prior_mu", 1.0),
                    prior_var=meta.get("prior_var", 1.0),
                    prior_n=meta.get("prior_n", 2.0),
                    decay=meta.get("decay", 1.0))
        # Round 8 M4 · per-regime decay
        for k, v in (meta.get("regime_decay") or {}).items():
            try: state.regime_decay[int(k)] = float(v)
            except Exception: pass
        for line in lines[1:]:
            try:
                r = json.loads(line)
                state._state[(r["regime"], r["model"])] = \
                    (r["n"], r["sum_loss"], r["sum_sq"])
            except Exception: pass
        return state

    def regret_so_far(self, history: list[dict]) -> dict[str, float]:
        """Summary stats from a list of {'regime', 'chosen', 'loss', 'oracle_loss'}."""
        if not history: return {}
        cum_loss = sum(h["loss"] for h in history)
        cum_oracle = sum(h["oracle_loss"] for h in history)
        cum_regret = cum_loss - cum_oracle
        return {"n": len(history), "cum_loss": cum_loss,
                "cum_oracle": cum_oracle, "cum_regret": cum_regret,
                "mean_regret": cum_regret / len(history)}


# ─── ContextualBanditRouter: wraps BayesianRouter + observe loop ──────────────

@dataclass
class ContextualBanditRouter:
    """Bandit routing on top of a regime-aware embedding.

    Pipeline:
        1. embed(series) → z
        2. regime label  = regime_fn(z)
        3. Thompson sample per (regime, candidate) from BanditState
        4. choose argmin r̃ (decide)
        5. after observing actual loss, call observe(z, chosen, loss)
    """
    candidates: list[str]
    bandit: BanditState
    regime_fn: Callable[[np.ndarray], int]   # z → regime id

    def decide(self, z: np.ndarray, mode: str = "thompson",
               rng: Optional[np.random.Generator] = None
               ) -> tuple[str, dict[str, float]]:
        """Pick a model for this series (by its embedding z)."""
        regime = self.regime_fn(z)
        if mode == "thompson":
            samples = self.bandit.thompson_sample(regime, self.candidates, rng)
            chosen = min(samples, key=samples.get)
            return chosen, samples
        elif mode == "greedy":
            chosen = self.bandit.best_arm(regime, self.candidates)
            beliefs = {m: self.bandit.belief(regime, m)[0] for m in self.candidates}
            return chosen, beliefs
        elif mode == "ucb":
            # UCB1-style: argmin(μ - β·σ) for loss minimization
            beta = 2.0
            scores = {}
            for m in self.candidates:
                mu, sigma = self.bandit.belief(regime, m)
                scores[m] = mu - beta * sigma
            chosen = min(scores, key=scores.get)
            return chosen, scores
        else:
            raise ValueError(f"unknown mode {mode}")

    def observe(self, z: np.ndarray, chosen: str, loss: float) -> None:
        """Update belief after observing actual loss for chosen model."""
        regime = self.regime_fn(z)
        self.bandit.observe(regime, chosen, loss)


# ─── Module-level singleton for forecaster_reflect integration ────────────────

_GLOBAL_ROUTER: Optional["ContextualBanditRouter"] = None
_GLOBAL_EMBEDDING = None
_GLOBAL_REGIME = None


def get_router(state_path: str = "research/results/bandit_state.jsonl",
               candidates: Optional[list[str]] = None,
               embedding_name: str = "hand25",
               K: int = 6,
               prior_mu: float = 1.0,
               prior_var: float = 1.0,
               prior_n: float = 2.0,
               decay: float = 1.0,
               ) -> tuple["ContextualBanditRouter", "object"]:
    """Lazy singleton: build (embedding, regime assigner, bandit) once per process.

    Returns (router, embedding). Use `embedding.embed(series)` to get z that
    matches router.regime_fn.
    """
    global _GLOBAL_ROUTER, _GLOBAL_EMBEDDING, _GLOBAL_REGIME
    if _GLOBAL_ROUTER is not None:
        return _GLOBAL_ROUTER, _GLOBAL_EMBEDDING
    from research.agent.representation import build_regime_pipeline
    emb, assigner = build_regime_pipeline(embedding_name, K=K)
    state = BanditState.load(state_path)
    if state.prior_mu == 1.0 and not state._state:    # fresh
        state = BanditState(prior_mu=prior_mu, prior_var=prior_var,
                            prior_n=prior_n, decay=decay)
    if candidates is None:
        candidates = ["chronos2", "tirex", "toto", "time_moe", "sundial",
                      "timesfm2", "moirai", "moirai2"]
    def regime_fn(z):
        if assigner._centroids is None: return 0
        return int(assigner.predict_label(z[None, :])[0])
    _GLOBAL_ROUTER = ContextualBanditRouter(
        candidates=candidates, bandit=state, regime_fn=regime_fn)
    _GLOBAL_EMBEDDING = emb
    _GLOBAL_REGIME = assigner
    return _GLOBAL_ROUTER, _GLOBAL_EMBEDDING


def reset_router():
    """For tests: clear singleton."""
    global _GLOBAL_ROUTER, _GLOBAL_EMBEDDING, _GLOBAL_REGIME
    _GLOBAL_ROUTER = None
    _GLOBAL_EMBEDDING = None
    _GLOBAL_REGIME = None


def persist_router(state_path: str = "research/results/bandit_state.jsonl"):
    if _GLOBAL_ROUTER is not None:
        _GLOBAL_ROUTER.bandit.save(state_path)


# ─── Self-test: synthetic 3-regime, 4-arm bandit ──────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Phase 2 · Contextual Bandit / Thompson Routing")
    print("=" * 70)

    # ─── Test 1: pure Gaussian update on single (regime, arm) ────────────────
    print("\n[Test 1] Single-arm posterior tightens with observations")
    state = BanditState(prior_mu=1.0, prior_var=1.0, prior_n=2.0)
    rng = np.random.default_rng(0)
    true_mu = 0.3
    for i in [0, 1, 5, 20, 100]:
        # observe i samples from N(0.3, 0.1²)
        for _ in range(i - len(state._state.get((0, "tirex"), (0,0,0))) and 0):
            pass
    state = BanditState(prior_mu=1.0, prior_var=1.0, prior_n=2.0)
    for n in [0, 1, 5, 20, 100]:
        # observe (n - so_far) more
        cur = state._state.get((0, "tirex"), (state.prior_n, 0, 0))[0] - state.prior_n
        more = int(n - cur)
        for _ in range(max(0, more)):
            obs = rng.normal(true_mu, 0.1)
            state.observe(0, "tirex", obs)
        mu, sigma = state.belief(0, "tirex")
        print(f"  n={n:>3}  μ={mu:.3f} (true 0.3)  σ_μ={sigma:.3f}")

    # ─── Test 2: full bandit simulation ──────────────────────────────────────
    print("\n[Test 2] 3-regime × 4-arm bandit, 200 episodes")
    print("        Synthetic true losses per regime (lower = better):")
    # truth: regime 0 favors arm A; regime 1 favors B; regime 2 favors C
    true_means = {
        0: {"A": 0.2, "B": 0.5, "C": 0.5, "D": 0.5},  # → A best
        1: {"A": 0.6, "B": 0.2, "C": 0.5, "D": 0.5},  # → B best
        2: {"A": 0.5, "B": 0.5, "C": 0.2, "D": 0.5},  # → C best
    }
    for r, ms in true_means.items():
        print(f"    regime {r}: " + " ".join(f"{m}={v:.1f}" for m, v in ms.items())
              + f"   (oracle = {min(ms.values()):.1f})")

    def run_policy(policy_fn, n_ep=200, seed=0):
        state = BanditState(prior_mu=0.5, prior_var=0.2, prior_n=2.0)
        rng = np.random.default_rng(seed)
        cands = list(true_means[0].keys())
        history = []
        for t in range(n_ep):
            regime = int(rng.integers(0, 3))
            chosen = policy_fn(state, regime, cands, rng)
            loss = rng.normal(true_means[regime][chosen], 0.1)
            oracle = min(true_means[regime].values())
            state.observe(regime, chosen, loss)
            history.append({"regime": regime, "chosen": chosen,
                            "loss": loss, "oracle_loss": oracle})
        return state.regret_so_far(history)

    policies = {
        "random":   lambda s, r, c, rng: c[int(rng.integers(0, len(c)))],
        "greedy":   lambda s, r, c, rng: s.best_arm(r, c),
        "thompson": lambda s, r, c, rng: min(s.thompson_sample(r, c, rng).items(),
                                             key=lambda kv: kv[1])[0],
        "ucb":      lambda s, r, c, rng: min(
                        {m: s.belief(r, m)[0] - 2.0 * s.belief(r, m)[1] for m in c}.items(),
                        key=lambda kv: kv[1])[0],
    }

    print(f"\n    {'policy':<10} {'cum_loss':>10} {'cum_oracle':>11} {'cum_regret':>11} {'mean_reg':>10}")
    for name, fn in policies.items():
        # average over 3 seeds
        regrets = [run_policy(fn, seed=s) for s in [0, 1, 2]]
        avg = {k: float(np.mean([r[k] for r in regrets])) for k in regrets[0]}
        print(f"    {name:<10} {avg['cum_loss']:>10.2f} {avg['cum_oracle']:>11.2f} "
              f"{avg['cum_regret']:>11.2f} {avg['mean_regret']:>10.4f}")

    print("\n    expected: thompson ≈ ucb < greedy ≪ random  (regret order)")

    # ─── Test 3: ContextualBanditRouter wired to RegimeAssigner ──────────────
    print("\n[Test 3] ContextualBanditRouter end-to-end (cached forecasting data)")
    try:
        from research.agent.representation import build_regime_pipeline
        emb, assigner = build_regime_pipeline("hand25", K=6)
        if assigner._centroids is None:
            print("  no cached cells — skip")
        else:
            def regime_fn(z):
                return int(assigner.predict_label(z[None, :])[0])
            bandit = BanditState(prior_mu=1.0, prior_var=1.0, prior_n=2.0)
            router = ContextualBanditRouter(
                candidates=["chronos2", "tirex", "toto", "time_moe", "sundial"],
                bandit=bandit, regime_fn=regime_fn,
            )
            # warm-start from cached cells
            import json
            from pathlib import Path
            cells_path = Path("research/results/gated_residual_cells.jsonl")
            base = [json.loads(l) for l in cells_path.read_text().splitlines()]
            # Build cell index → series + per-model loss
            loss_jsonls = ["tirex_vs_c2.jsonl", "toto_vs_c2.jsonl",
                           "time_moe_vs_c2.jsonl", "sundial_vs_c2.jsonl"]
            losses_per_key = {}
            for c in base:
                key = (c["dataset"], c["N"], c["seed"])
                y_true = np.array(c["y_true"]); c2 = np.array(c["c2_pred"])
                losses_per_key[key] = {"chronos2": float(np.mean(np.abs(y_true - c2)))}
            for fname in loss_jsonls:
                p = Path(f"research/results/{fname}")
                if not p.exists(): continue
                model = fname.replace("_vs_c2.jsonl", "")
                for line in p.read_text().splitlines():
                    r = json.loads(line)
                    key = (r["dataset"], r["N"], r["seed"])
                    if key in losses_per_key:
                        losses_per_key[key][model] = r[f"mae_{model}"]

            # warm-start: feed each cell to bandit
            for c in base:
                key = (c["dataset"], c["N"], c["seed"])
                z = emb.embed(np.array(c["history"], dtype=np.float32))
                if key not in losses_per_key: continue
                for m, l in losses_per_key[key].items():
                    bandit.observe(regime_fn(z), m, l)

            # now query a fresh cell
            test_c = base[0]
            test_z = emb.embed(np.array(test_c["history"], dtype=np.float32))
            chosen, samples = router.decide(test_z, mode="thompson")
            chosen_g, beliefs = router.decide(test_z, mode="greedy")
            print(f"  cell {test_c['dataset']}/N={test_c['N']}/seed={test_c['seed']}:")
            print(f"    regime = {regime_fn(test_z)}")
            print(f"    thompson chose: {chosen}  (samples: " +
                  " ".join(f"{m}={v:.2f}" for m, v in samples.items()) + ")")
            print(f"    greedy   chose: {chosen_g}  (beliefs: " +
                  " ".join(f"{m}={v:.2f}" for m, v in beliefs.items()) + ")")
    except Exception as e:
        import traceback
        print(f"  FAIL: {type(e).__name__}: {e}")
        traceback.print_exc()
