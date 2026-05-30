"""Round 6 D1 · Router Telemetry + Health Report (feedback 后§7).

Aggregates RouterState.telemetry into a structured health report:

    {
      "n_decisions": ...,
      "choice_distribution": {model: count},
      "regime_distribution": {regime: count},
      "mean_routing_entropy": ...,
      "mean_posterior_gap":   ...,
      "outcome_mean":         ...,
      "outcome_std":          ...,
      "calibration": {raw → calibrated curve},
      "drift_signals": {...},
      "factor_influence":  {factor: mean |log contrib|},
      "escalation_rate":   ...,
      "failure_rate":      ...,
    }

The report exposes: "为什么系统在退化" — feedback §7 explicit goal.

Use:
    from research.agent.telemetry import generate_report
    report = generate_report(state)
    print(report.to_markdown())

Saved as `research/results/router_health_<ts>.md` for periodic review.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path
import json
import math
import time

import numpy as np


def _entropy(probs: dict[str, float]) -> float:
    """Shannon entropy of a posterior distribution."""
    return -sum(p * math.log(p + 1e-12) for p in probs.values() if p > 0)


@dataclass
class HealthReport:
    timestamp: float
    n_decisions: int
    n_observations: int
    window_size: int                       # how many recent records analyzed
    choice_dist: dict
    regime_dist: dict
    mean_entropy: float
    mean_posterior_gap: float
    outcome_mean: Optional[float]
    outcome_std: Optional[float]
    factor_influence: dict
    escalation_rate: float
    drift_signals: dict
    notes: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        L = []
        L.append(f"# Router Health Report\n")
        L.append(f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))}_\n")
        L.append(f"## Overview")
        L.append(f"- Total decisions: {self.n_decisions}")
        L.append(f"- Total observations: {self.n_observations}")
        L.append(f"- Window analyzed: last {self.window_size} records")
        L.append(f"- Mean routing entropy: **{self.mean_entropy:.3f}**")
        L.append(f"- Mean posterior gap: **{self.mean_posterior_gap:.3f}**")
        if self.outcome_mean is not None:
            L.append(f"- Outcome mean: **{self.outcome_mean:.4f}**  ±  {self.outcome_std:.4f}")
        L.append(f"- Escalation rate: {self.escalation_rate:.1%}\n")

        L.append(f"## Choice distribution")
        for m, n in sorted(self.choice_dist.items(), key=lambda kv: -kv[1]):
            pct = 100 * n / max(1, sum(self.choice_dist.values()))
            L.append(f"- {m}: {n} ({pct:.1f}%)")

        L.append(f"\n## Regime distribution")
        for r, n in sorted(self.regime_dist.items(), key=lambda kv: -kv[1]):
            pct = 100 * n / max(1, sum(self.regime_dist.values()))
            L.append(f"- regime {r}: {n} ({pct:.1f}%)")

        L.append(f"\n## Prior/Likelihood Factor Influence (mean |log contrib|)")
        for f, v in sorted(self.factor_influence.items(), key=lambda kv: -kv[1]):
            L.append(f"- {f}: {v:.4f}")

        if self.drift_signals:
            L.append(f"\n## Drift Signals")
            for k, v in self.drift_signals.items():
                L.append(f"- {k}: {v}")

        if self.notes:
            L.append(f"\n## Notes")
            for n in self.notes: L.append(f"- {n}")

        return "\n".join(L)


def generate_report(state, window: int = 200) -> HealthReport:
    """Build HealthReport from state.telemetry."""
    from collections import Counter

    recent = list(state.telemetry)[-window:] if state.telemetry else []
    n_rec = len(recent)

    choice_dist = dict(Counter(r.chosen for r in recent))
    regime_dist = dict(Counter(str(r.ctx_summary.get("regime"))
                                 for r in recent if r.ctx_summary.get("regime") is not None))

    # Entropy + posterior gap
    entropies, gaps = [], []
    for r in recent:
        if r.posterior:
            entropies.append(_entropy(r.posterior))
            ps = sorted(r.posterior.values(), reverse=True)
            gaps.append(ps[0] - (ps[1] if len(ps) > 1 else 0.0))

    # Outcomes
    obs = [r.outcome for r in recent if r.outcome is not None]
    out_mean = float(np.mean(obs)) if obs else None
    out_std = float(np.std(obs)) if obs else None

    # Factor influence: mean |log contribution| across recent records
    factor_inf: dict[str, list[float]] = {}
    for r in recent:
        for f_name, contribs in {**r.prior_contribs, **r.lik_contribs}.items():
            for m, val in contribs.items():
                factor_inf.setdefault(f_name, []).append(abs(float(val)))
    factor_influence = {f: float(np.mean(v)) if v else 0.0
                         for f, v in factor_inf.items()}

    # Escalation rate placeholder (B1 reflective loop tags would feed in)
    escalation_rate = 0.0

    # Drift signals — Round 6 B3: full 4-signal Drift Engine
    drift = {}
    try:
        from research.agent.drift_engine import compute_drift, DriftConfig
        sig = compute_drift(state, DriftConfig())
        drift = {
            "feature_kl":        sig.feature_kl,
            "residual_ks":       sig.residual_ks,
            "routing_kl":        sig.routing_kl,
            "memory_mismatch":   sig.memory_mismatch,
            "detected":          sig.detected,
            "any_detected":      sig.any_detected(),
            "window_recent":     sig.window_recent_used,
            "window_history":    sig.window_history_used,
        }
    except Exception as e:
        # Fallback: legacy regime-KL signal so report never breaks
        if n_rec >= 20:
            half = n_rec // 2
            from collections import Counter
            first_dist = Counter(r.ctx_summary.get("regime") for r in recent[:half]
                                  if r.ctx_summary.get("regime") is not None)
            second_dist = Counter(r.ctx_summary.get("regime") for r in recent[half:]
                                  if r.ctx_summary.get("regime") is not None)
            from research.agent.memory_decay import regime_drift_kl
            kl = regime_drift_kl(dict(second_dist), dict(first_dist))
            drift["regime_kl_first_vs_second_half"] = round(kl, 4)
            drift["drift_detected"] = bool(kl > 0.5)
            drift["fallback_reason"] = str(e)

    notes = []
    if out_mean is not None and out_std is not None and out_std > out_mean * 2:
        notes.append("High outcome variance — investigate input scale or unstable models")
    if entropies and float(np.mean(entropies)) > 1.5:
        notes.append("High mean routing entropy — posterior is diffuse, consider stronger priors")

    return HealthReport(
        timestamp=time.time(),
        n_decisions=state.n_decisions,
        n_observations=state.n_observations,
        window_size=n_rec,
        choice_dist=choice_dist,
        regime_dist=regime_dist,
        mean_entropy=float(np.mean(entropies)) if entropies else 0.0,
        mean_posterior_gap=float(np.mean(gaps)) if gaps else 0.0,
        outcome_mean=out_mean, outcome_std=out_std,
        factor_influence=factor_influence,
        escalation_rate=escalation_rate,
        drift_signals=drift,
        notes=notes,
    )


def save_report(report: HealthReport,
                path: Optional[str] = None) -> str:
    if path is None:
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(report.timestamp))
        path = f"research/results/router_health_{ts}.md"
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(report.to_markdown())
    return str(p)


if __name__ == "__main__":
    print("=" * 60)
    print("Round 6 D1 · Telemetry + Health Report smoke")
    print("=" * 60)
    from research.agent.router_state import RouterState, TelemetryRecord
    import random

    # Build mock state with 50 decisions
    state = RouterState()
    random.seed(0)
    models = ["chronos2", "tirex", "toto", "naive_drift"]
    for i in range(50):
        regime = i % 3
        chosen = random.choice(models)
        post = {m: random.random() for m in models}
        total = sum(post.values())
        post = {m: v/total for m, v in post.items()}
        rec = TelemetryRecord(
            t=time.time() - (50 - i) * 60,
            ctx_summary={"dataset": "ETTh1", "N": 50, "H": 24, "regime": regime},
            chosen=chosen, posterior=post,
            prior_contribs={"NPrior": {m: random.random() - 0.5 for m in models},
                            "CRPSPrior": {m: -random.random() for m in models}},
            lik_contribs={"CV": {m: -random.random() for m in models}},
            decide_mode="argmax",
            outcome=random.uniform(0.2, 0.8) if random.random() > 0.1 else None,
        )
        state.telemetry.append(rec)
    state.n_decisions = 50
    state.n_observations = sum(1 for r in state.telemetry if r.outcome is not None)

    report = generate_report(state, window=50)
    print("\n" + report.to_markdown())

    out_path = save_report(report, "/tmp/test_health.md")
    print(f"\nsaved to {out_path}")
