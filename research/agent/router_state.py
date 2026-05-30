"""Round 6 A2 · Unified RouterState — single container for all persistent state.

Aggregates 4 sub-states that were previously scattered:
    1. BanditState (per-regime, per-model belief)
    2. ForecastMemory / ClfMemory (case retrieval)
    3. RegimeAssigner (k-means centroids + per-regime π)
    4. RouterTelemetry (decision history, prior contributions, drift signals)

Single .save() / .load() entry; future C1 Failure Memory + C2 Decay all
plug into this same container.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json
import time

import numpy as np

from research.agent.bandit import BanditState


@dataclass
class TelemetryRecord:
    """Per-decision audit log."""
    t: float                            # wall-clock timestamp
    ctx_summary: dict                   # {dataset, N, H, regime, ...}
    chosen: str
    posterior: dict                     # full posterior
    prior_contribs: dict                # {factor_name: dict[model, log_p]}
    lik_contribs: dict                  # {factor_name: dict[model, log_l]}
    decide_mode: str
    outcome: Optional[float] = None     # filled by observe() if available


@dataclass
class RouterState:
    """Single persistent container for all router state.

    Attributes:
        bandit: BanditState (per-regime, per-model belief)
        regime_centroids: ndarray [K, dim] — k-means centroids
        regime_priors: dict[int, dict[str, float]] — per-regime π_k
        memory_cases: list[dict] — historical (z, chosen, outcome, ...) cases
        telemetry: list[TelemetryRecord] — decision audit trail
        n_decisions: int — total decide() calls
        n_observations: int — total observe() calls
        last_save: float — last persist timestamp
    """
    bandit: BanditState = field(default_factory=BanditState)
    regime_centroids: Optional[np.ndarray] = None
    regime_priors: dict[int, dict[str, float]] = field(default_factory=dict)
    memory_cases: list[dict] = field(default_factory=list)
    telemetry: list[TelemetryRecord] = field(default_factory=list)
    drift_history: list[dict] = field(default_factory=list)   # Round 6 B3 · event log
    # Round 7 M2/M3 · self-evolving state
    culled: dict = field(default_factory=dict)                # regime → set[str]
    learned_prior_strengths: dict = field(default_factory=dict)  # factor_name → float
    # Round 8 M1 · meta-bandit on decide_mode (lazy import to avoid cycle)
    meta_bandit_dict: dict = field(default_factory=dict)         # serialized state
    n_decisions: int = 0
    n_observations: int = 0
    last_save: float = 0.0

    # ─── observe-side ─────────────────────────────────────────────────────
    def add_observation(self, ctx, z, chosen: str, outcome: float,
                        regime: Optional[int] = None) -> None:
        """Round 6 A2 entry point for `BayesianRouter.observe()`.

        Records a case under unified storage; the bandit factor still
        writes to .bandit directly (kept for backwards compat).
        """
        self.n_observations += 1
        case = {
            "t": time.time(),
            "chosen": chosen,
            "outcome": float(outcome),
            "regime": regime,
            "z": z.tolist() if z is not None and hasattr(z, "tolist") else None,
        }
        if ctx is not None:
            case["ctx"] = {
                "dataset": getattr(ctx, "dataset", None),
                "N": getattr(ctx, "N", None),
                "H": getattr(ctx, "H", None),
            }
        self.memory_cases.append(case)
        # tag last telemetry record (if any) with outcome
        if self.telemetry and self.telemetry[-1].outcome is None:
            self.telemetry[-1].outcome = float(outcome)

    def log_decision(self, ctx, chosen: str, posterior: dict,
                     prior_contribs: dict, lik_contribs: dict,
                     decide_mode: str) -> None:
        """Round 6 D1 hook (telemetry pre-built here)."""
        self.n_decisions += 1
        rec = TelemetryRecord(
            t=time.time(),
            ctx_summary={
                "dataset": getattr(ctx, "dataset", None),
                "N": getattr(ctx, "N", None),
                "H": getattr(ctx, "H", None),
                "regime": int(np.argmax([posterior.get(k, 0.0) for k in posterior]))
                if posterior else None,
            },
            chosen=chosen,
            posterior=dict(posterior),
            prior_contribs=prior_contribs,
            lik_contribs=lik_contribs,
            decide_mode=decide_mode,
        )
        self.telemetry.append(rec)

    # ─── persistence ──────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        """Dump entire state to jsonl. One section per key."""
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        # bandit goes to sidecar
        bandit_path = p.with_suffix("").as_posix() + "_bandit.jsonl"
        self.bandit.save(bandit_path)
        # main state file
        meta = {
            "_meta": True,
            "n_decisions": self.n_decisions,
            "n_observations": self.n_observations,
            "last_save": time.time(),
            "bandit_sidecar": Path(bandit_path).name,
            "n_telemetry": len(self.telemetry),
            "n_memory_cases": len(self.memory_cases),
            "n_drift_events": len(self.drift_history),
            "n_culled_regimes": len(self.culled),
            "n_learned_priors": len(self.learned_prior_strengths),
            "regime_K": int(self.regime_centroids.shape[0])
                       if self.regime_centroids is not None else 0,
        }
        with p.open("w") as fh:
            fh.write(json.dumps(meta) + "\n")
            if self.regime_centroids is not None:
                fh.write(json.dumps({"_section": "centroids",
                                     "data": self.regime_centroids.tolist()}) + "\n")
            for r, pi in self.regime_priors.items():
                fh.write(json.dumps({"_section": "regime_prior",
                                     "regime": int(r), "pi": pi}) + "\n")
            for case in self.memory_cases[-5000:]:  # cap memory persistence
                fh.write(json.dumps({"_section": "case", **case}) + "\n")
            for ev in self.drift_history[-500:]:   # cap drift event log
                fh.write(json.dumps({"_section": "drift_event", **ev}) + "\n")
            # Round 7 M2 · culled set per regime
            for r, models in self.culled.items():
                fh.write(json.dumps({"_section": "culled",
                                      "regime": int(r),
                                      "models": sorted(models)}) + "\n")
            # Round 7 M3 · learned prior strengths
            if self.learned_prior_strengths:
                fh.write(json.dumps({"_section": "learned_priors",
                                      "data": dict(self.learned_prior_strengths)}) + "\n")
            # Round 8 M1 · meta-bandit on decide_mode
            if self.meta_bandit_dict:
                fh.write(json.dumps({"_section": "meta_bandit",
                                      "data": self.meta_bandit_dict}) + "\n")
            for rec in self.telemetry[-2000:]:    # cap telemetry persistence
                fh.write(json.dumps({
                    "_section": "telemetry",
                    "t": rec.t, "ctx_summary": rec.ctx_summary,
                    "chosen": rec.chosen, "posterior": rec.posterior,
                    "prior_contribs": rec.prior_contribs,
                    "lik_contribs": rec.lik_contribs,
                    "decide_mode": rec.decide_mode,
                    "outcome": rec.outcome,
                }) + "\n")
        self.last_save = time.time()

    @classmethod
    def load(cls, path: str | Path) -> "RouterState":
        """Restore from jsonl. Returns fresh state if file missing."""
        p = Path(path)
        if not p.exists():
            return cls()
        state = cls()
        lines = p.read_text().splitlines()
        if not lines:
            return state
        meta = json.loads(lines[0])
        sidecar = meta.get("bandit_sidecar")
        if sidecar:
            bandit_path = p.parent / sidecar
            state.bandit = BanditState.load(bandit_path)
        state.n_decisions = meta.get("n_decisions", 0)
        state.n_observations = meta.get("n_observations", 0)
        state.last_save = meta.get("last_save", 0.0)
        for line in lines[1:]:
            try:
                r = json.loads(line)
                sec = r.get("_section")
                if sec == "centroids":
                    state.regime_centroids = np.array(r["data"], dtype=np.float64)
                elif sec == "regime_prior":
                    state.regime_priors[int(r["regime"])] = r["pi"]
                elif sec == "case":
                    state.memory_cases.append({k: v for k, v in r.items()
                                                if not k.startswith("_")})
                elif sec == "drift_event":
                    state.drift_history.append({k: v for k, v in r.items()
                                                 if not k.startswith("_")})
                elif sec == "culled":
                    state.culled[int(r["regime"])] = set(r.get("models", []))
                elif sec == "learned_priors":
                    state.learned_prior_strengths = dict(r.get("data", {}))
                elif sec == "meta_bandit":
                    state.meta_bandit_dict = dict(r.get("data", {}))
                elif sec == "telemetry":
                    state.telemetry.append(TelemetryRecord(
                        t=r["t"], ctx_summary=r["ctx_summary"], chosen=r["chosen"],
                        posterior=r["posterior"], prior_contribs=r["prior_contribs"],
                        lik_contribs=r["lik_contribs"], decide_mode=r["decide_mode"],
                        outcome=r.get("outcome"),
                    ))
            except Exception:
                pass
        return state

    # ─── summary / health ─────────────────────────────────────────────────
    def summary(self) -> dict:
        """One-shot snapshot for Routing Health Report (D1)."""
        from collections import Counter
        recent = self.telemetry[-200:] if self.telemetry else []
        choice_dist = Counter(r.chosen for r in recent)
        regime_dist = Counter(r.ctx_summary.get("regime") for r in recent)
        observed = [r.outcome for r in recent if r.outcome is not None]
        return {
            "n_decisions": self.n_decisions,
            "n_observations": self.n_observations,
            "n_memory_cases": len(self.memory_cases),
            "n_regimes": int(self.regime_centroids.shape[0])
                       if self.regime_centroids is not None else 0,
            "recent_choice_dist": dict(choice_dist),
            "recent_regime_dist": {str(k): v for k, v in regime_dist.items()},
            "recent_mean_outcome": float(np.mean(observed)) if observed else None,
            "recent_std_outcome":  float(np.std(observed))  if observed else None,
        }


# ─── Module-level singleton (replaces bandit.get_router singleton) ───────────

_GLOBAL_STATE: Optional[RouterState] = None
_GLOBAL_STATE_PATH: Optional[str] = None


def get_state(path: str = "research/results/router_state.jsonl") -> RouterState:
    """Lazy singleton load. Multiple modules share one state."""
    global _GLOBAL_STATE, _GLOBAL_STATE_PATH
    if _GLOBAL_STATE is None or _GLOBAL_STATE_PATH != path:
        _GLOBAL_STATE = RouterState.load(path)
        _GLOBAL_STATE_PATH = path
    return _GLOBAL_STATE


def persist_state(path: Optional[str] = None) -> None:
    if _GLOBAL_STATE is None: return
    _GLOBAL_STATE.save(path or _GLOBAL_STATE_PATH or
                       "research/results/router_state.jsonl")


def reset_state() -> None:
    global _GLOBAL_STATE, _GLOBAL_STATE_PATH
    _GLOBAL_STATE = None; _GLOBAL_STATE_PATH = None


if __name__ == "__main__":
    import tempfile
    print("=" * 60)
    print("Round 6 A2 · RouterState smoke")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "state.jsonl"
        state = RouterState()
        state.bandit.observe(0, "chronos2", 0.3)
        state.bandit.observe(0, "tirex", 1.2)
        state.regime_centroids = np.eye(3)
        state.regime_priors[0] = {"chronos2": 0.6, "tirex": 0.4}

        # log a fake decision + outcome
        class FakeCtx:
            dataset, N, H = "ETTh1", 50, 24
        ctx = FakeCtx()
        state.log_decision(ctx, "chronos2",
                            posterior={"chronos2": 0.6, "tirex": 0.4},
                            prior_contribs={"NPrior": {"chronos2": 0.5}},
                            lik_contribs={"bandit": {"chronos2": -0.3, "tirex": -1.2}},
                            decide_mode="argmax")
        state.add_observation(ctx, z=np.array([1.0, 0]), chosen="chronos2", outcome=0.25)

        state.save(path)
        print(f"saved to {path}")
        print(f"summary: {state.summary()}")

        state2 = RouterState.load(path)
        print(f"\nreloaded:")
        print(f"  bandit (0,chronos2) belief: {state2.bandit.belief(0,'chronos2')}")
        print(f"  centroids shape: {state2.regime_centroids.shape}")
        print(f"  regime_priors: {state2.regime_priors}")
        print(f"  telemetry: {len(state2.telemetry)} records")
        print(f"  memory_cases: {len(state2.memory_cases)}")
        print(f"  summary after reload: {state2.summary()}")
