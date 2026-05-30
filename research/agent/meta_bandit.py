"""Round 8 M1 · Meta-bandit on decide_mode (feedback 前§2.C).

Treats `decide_mode ∈ {argmax, thompson, risk_min}` as 3 arms of a *meta-level*
bandit. The system tracks per-mode reward (= −outcome, lower-is-better loss)
and autonomously switches to the best-performing mode, removing the last
hand-tuned constant in `RouterConfig`.

Update:
    observe(mode, outcome):
        (n, s, sq) ← decay·(n, s, sq) + (1, outcome, outcome²)
    belief(mode) = (μ, σ) from Gaussian conjugate

Selection:
    cold start (n_total < cold_start_K × n_modes):
        round-robin
    otherwise:
        Thompson sample r_k ~ N(μ_k, σ_k); pick arg min r_k
        (or argmin μ_k when `mode='greedy'`)

Plays the same role as Round 5 `BanditState` but for *meta* arms.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math

import numpy as np


META_MODES = ("argmax", "thompson", "risk_min")


# ─── State ───────────────────────────────────────────────────────────────────


@dataclass
class MetaBanditState:
    """Per-mode (n, sum_loss, sum_sq) with optional exponential decay."""
    decay:       float = 0.995
    prior_mu:    float = 1.0
    prior_var:   float = 1.0
    prior_n:     float = 1.0
    counts: dict = field(default_factory=dict)   # mode → (n, s, sq)

    def _get(self, mode: str) -> tuple[float, float, float]:
        if mode not in self.counts:
            n = self.prior_n
            s = n * self.prior_mu
            sq = n * (self.prior_mu ** 2 + self.prior_var)
            self.counts[mode] = (n, s, sq)
        return self.counts[mode]

    def observe(self, mode: str, outcome: float) -> None:
        n, s, sq = self._get(mode)
        if self.decay < 1.0:
            n *= self.decay; s *= self.decay; sq *= self.decay
        self.counts[mode] = (n + 1.0, s + outcome, sq + outcome * outcome)

    def belief(self, mode: str) -> tuple[float, float]:
        n, s, sq = self._get(mode)
        mu = s / max(n, 1e-9)
        var = max(sq / max(n, 1e-9) - mu * mu, 1e-9) / max(n, 1e-9)
        return float(mu), float(math.sqrt(var))

    def n_per_mode(self) -> dict[str, float]:
        return {m: self.counts.get(m, (0, 0, 0))[0] for m in META_MODES}


# ─── Config + Selector ───────────────────────────────────────────────────────


@dataclass
class MetaBanditConfig:
    modes: tuple = META_MODES
    cold_start_K: int = 10        # min obs per mode before exploiting
    selection: str = "thompson"   # "thompson" | "greedy"


def select_mode(state: MetaBanditState,
                config: Optional[MetaBanditConfig] = None,
                rng: Optional[np.random.Generator] = None) -> tuple[str, dict]:
    """Pick a decide_mode for the upcoming decision.

    Returns (mode, info) where info has:
        per_mode_belief: {mode → (μ, σ)}
        per_mode_n:      {mode → n}
        selection:       "cold_start" | "thompson" | "greedy"
    """
    if config is None: config = MetaBanditConfig()
    if rng is None:    rng = np.random.default_rng()

    n_each = state.n_per_mode()
    cold_K = config.cold_start_K
    # Cold start: round-robin to fill min observations per mode
    under = [m for m in config.modes if n_each.get(m, 0) < cold_K]
    info = {
        "per_mode_belief": {m: state.belief(m) for m in config.modes},
        "per_mode_n":      n_each,
    }
    if under:
        # Pick the one with the lowest count (tie: first in list)
        chosen = min(under, key=lambda m: n_each.get(m, 0))
        info["selection"] = "cold_start"
        return chosen, info

    if config.selection == "thompson":
        samples = {}
        for m in config.modes:
            mu, sigma = state.belief(m)
            samples[m] = float(rng.normal(mu, max(sigma, 1e-6)))
        chosen = min(samples, key=samples.get)   # arg min loss sample
        info["selection"] = "thompson"
        info["samples"] = samples
    else:   # greedy
        beliefs = {m: state.belief(m)[0] for m in config.modes}
        chosen = min(beliefs, key=beliefs.get)
        info["selection"] = "greedy"
    return chosen, info


# ─── persistence helpers (state lives on RouterState.meta_bandit) ────────────


def to_dict(s: MetaBanditState) -> dict:
    return {
        "decay": s.decay, "prior_mu": s.prior_mu,
        "prior_var": s.prior_var, "prior_n": s.prior_n,
        "counts": {m: list(v) for m, v in s.counts.items()},
    }


def from_dict(d: dict) -> MetaBanditState:
    s = MetaBanditState(
        decay=d.get("decay", 0.995), prior_mu=d.get("prior_mu", 1.0),
        prior_var=d.get("prior_var", 1.0), prior_n=d.get("prior_n", 1.0),
    )
    for m, v in d.get("counts", {}).items():
        s.counts[m] = tuple(v)
    return s


# ─── smoke ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Round 8 M1 · Meta-bandit smoke")
    print("=" * 60)
    rng = np.random.default_rng(0)

    state = MetaBanditState(decay=1.0)
    cfg = MetaBanditConfig(cold_start_K=5)

    # Simulate a ground truth: thompson is best (μ=0.4), argmax middle (μ=0.6),
    # risk_min worst (μ=0.8). Run 200 decisions, watch convergence.
    truth = {"argmax": 0.6, "thompson": 0.4, "risk_min": 0.8}
    history = {m: 0 for m in META_MODES}

    for step in range(200):
        mode, info = select_mode(state, cfg, rng)
        # Ground-truth reward: N(truth[mode], 0.1)
        outcome = float(rng.normal(truth[mode], 0.1))
        state.observe(mode, outcome)
        history[mode] += 1
        if step in (5, 15, 50, 100, 199):
            n_each = state.n_per_mode()
            beliefs = {m: state.belief(m) for m in META_MODES}
            print(f"\nstep={step:>3} sel={info['selection']:>10} chose={mode}")
            for m in META_MODES:
                mu, sig = beliefs[m]
                print(f"  {m:<10} n={n_each[m]:>5.1f}  μ={mu:.3f}±{sig:.3f}  "
                      f"(truth={truth[m]:.2f})")

    print(f"\nfinal mode usage: {history}")
    print(f"  ↳ thompson (best) usage = {history['thompson']/200:.1%}")
    print(f"  ↳ risk_min  (worst) usage = {history['risk_min']/200:.1%}")

    # Persistence round-trip
    d = to_dict(state)
    s2 = from_dict(d)
    assert s2.n_per_mode() == state.n_per_mode()
    print("\npersistence round-trip OK")
