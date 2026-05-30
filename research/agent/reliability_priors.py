"""Round 7 P0-1 + P0-3 · Production-grade reliability priors.

Two PriorFactor subclasses that protect against env-mismatch / load failures:

    CircuitBreakerPrior  (P0-1, feedback Round 5 §3.1):
        After N consecutive failures of model M in current env,
        hard-mask M for `cooldown_seconds` then auto-reset on next success.

    OperationalReliabilityPrior  (P0-3, F14 解药):
        Per-model success-rate based soft down-weight.
        log_prior += strength * log(success_rate + eps).

Both consume the same in-memory `ReliabilityTracker` to share signal.
State persists via RouterState.

Why P0-1 hard mask + P0-3 soft prior together:
    - Hard mask: respond fast to environmental failure (load_error / OOM)
    - Soft prior: continuous down-weight for partial failure (high_residual)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional
import json
import math
import time

import numpy as np


# ─── Tracker (singleton-style; reset_tracker() for tests) ─────────────────────

@dataclass
class ReliabilityRecord:
    n_total: int = 0
    n_success: int = 0
    n_fail_consecutive: int = 0
    last_fail_t: float = 0.0
    last_success_t: float = 0.0
    open_until: float = 0.0          # circuit breaker open period


@dataclass
class ReliabilityTracker:
    """Per-model running stats. Shared by CircuitBreaker + Operational priors.

    Stored under RouterState.reliability (added in load/save).
    """
    records: dict[str, ReliabilityRecord] = field(default_factory=dict)
    failure_threshold: int = 3        # N consecutive failures → open breaker
    cooldown_seconds: float = 300.0   # how long breaker stays open

    def _rec(self, model: str) -> ReliabilityRecord:
        if model not in self.records:
            self.records[model] = ReliabilityRecord()
        return self.records[model]

    def record_outcome(self, model: str, success: bool,
                       error_type: Optional[str] = None) -> None:
        """Update on every observe()."""
        rec = self._rec(model)
        rec.n_total += 1
        now = time.time()
        if success:
            rec.n_success += 1
            rec.n_fail_consecutive = 0
            rec.last_success_t = now
            # close breaker on success
            rec.open_until = 0.0
        else:
            rec.n_fail_consecutive += 1
            rec.last_fail_t = now
            # hard failures (env / load) trip breaker faster
            if error_type in ("load_error", "oom", "outlier_corruption"):
                threshold = max(1, self.failure_threshold // 2)
            else:
                threshold = self.failure_threshold
            if rec.n_fail_consecutive >= threshold:
                rec.open_until = now + self.cooldown_seconds

    def success_rate(self, model: str, prior_strength: float = 1.0) -> float:
        """Smoothed estimate: (n_success + prior) / (n_total + 2 * prior)."""
        rec = self._rec(model)
        if rec.n_total == 0:
            return 0.5
        return (rec.n_success + prior_strength) / (rec.n_total + 2 * prior_strength)

    def is_open(self, model: str) -> bool:
        rec = self._rec(model)
        return rec.open_until > time.time()

    def health(self, model: str) -> dict:
        rec = self._rec(model)
        return {
            "n_total": rec.n_total,
            "n_success": rec.n_success,
            "n_fail_consecutive": rec.n_fail_consecutive,
            "success_rate": self.success_rate(model),
            "is_open": self.is_open(model),
            "open_until": rec.open_until,
        }

    # ─── persistence (called by RouterState save/load) ─────────────────
    def to_jsonl(self) -> list[dict]:
        return [{"_section": "reliability", "model": m,
                 "n_total": r.n_total, "n_success": r.n_success,
                 "n_fail_consecutive": r.n_fail_consecutive,
                 "last_fail_t": r.last_fail_t, "last_success_t": r.last_success_t,
                 "open_until": r.open_until}
                for m, r in self.records.items()]

    def load_record(self, d: dict) -> None:
        self.records[d["model"]] = ReliabilityRecord(
            n_total=d["n_total"], n_success=d["n_success"],
            n_fail_consecutive=d["n_fail_consecutive"],
            last_fail_t=d.get("last_fail_t", 0),
            last_success_t=d.get("last_success_t", 0),
            open_until=d.get("open_until", 0),
        )


# ─── Module singleton (shared across priors) ──────────────────────────────────

_GLOBAL_TRACKER: Optional[ReliabilityTracker] = None


def get_tracker() -> ReliabilityTracker:
    global _GLOBAL_TRACKER
    if _GLOBAL_TRACKER is None:
        _GLOBAL_TRACKER = ReliabilityTracker()
    return _GLOBAL_TRACKER


def reset_tracker() -> None:
    global _GLOBAL_TRACKER
    _GLOBAL_TRACKER = None


# ─── CircuitBreakerPrior (P0-1) ───────────────────────────────────────────────

@dataclass
class CircuitBreakerPrior:
    """Hard-mask models with open circuit breaker.

    log_prior = -1e6 for open-breaker models, 0 otherwise.
    """
    name: str = "circuit_breaker"
    log_mask: float = -1e6

    def log_prior(self, candidates, ctx):
        tr = get_tracker()
        return {m: (self.log_mask if tr.is_open(m) else 0.0)
                for m in candidates}

    def __call__(self, candidates, ctx):
        return self.log_prior(candidates, ctx)


# ─── OperationalReliabilityPrior (P0-3, F14 解药) ─────────────────────────────

@dataclass
class OperationalReliabilityPrior:
    """Soft down-weight by smoothed success rate (F14 fix + F16 fix).

    F16: when n_total < min_obs_threshold, return 0 (no prior signal) to
    avoid cold-start bias toward default model. Only kicks in once we have
    reliable evidence.
    """
    name: str = "op_reliability"
    strength: float = 1.5
    eps: float = 0.05
    prior_strength: float = 2.0       # Bayes smoothing
    min_obs_threshold: int = 3        # F16: skip prior when n_total < this

    def log_prior(self, candidates, ctx):
        tr = get_tracker()
        out = {}
        for m in candidates:
            rec = tr._rec(m)
            if rec.n_total < self.min_obs_threshold:
                out[m] = 0.0   # F16: no signal yet, neutral
            else:
                out[m] = self.strength * math.log(
                    tr.success_rate(m, self.prior_strength) + self.eps)
        return out

    def __call__(self, candidates, ctx):
        return self.log_prior(candidates, ctx)


# ─── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Round 7 P0-1 + P0-3 · Reliability priors smoke")
    print("=" * 60)

    reset_tracker()
    tr = get_tracker()
    tr.failure_threshold = 3
    tr.cooldown_seconds = 60

    candidates = ["chronos2", "tirex", "moirai"]

    # ─── Test 1: 全成功 → 高 reliability prior
    print("\n[Test 1] all-success scenario")
    for _ in range(10):
        for m in candidates: tr.record_outcome(m, success=True)
    cb = CircuitBreakerPrior()
    op = OperationalReliabilityPrior(strength=1.0)
    ctx = type("Ctx", (), {})()
    print(f"  CB:  {cb(candidates, ctx)}")
    print(f"  OP:  {op(candidates, ctx)}")

    # ─── Test 2: moirai 连续失败 → 开闸 → CB hard-mask
    print("\n[Test 2] moirai 3 consecutive load_error fails")
    for _ in range(3):
        tr.record_outcome("moirai", success=False, error_type="load_error")
    print(f"  health(moirai): {tr.health('moirai')}")
    print(f"  CB:  {cb(candidates, ctx)}")
    print(f"  OP:  {op(candidates, ctx)}")

    # ─── Test 3: tirex 一次失败 → CB not open, OP slight down-weight
    print("\n[Test 3] tirex one fail (not enough to trip breaker)")
    tr.record_outcome("tirex", success=False, error_type="high_residual")
    print(f"  health(tirex): {tr.health('tirex')}")
    print(f"  CB:  {cb(candidates, ctx)}")
    print(f"  OP:  {op(candidates, ctx)}")

    # ─── Test 4: 持久化
    print("\n[Test 4] persistence round-trip")
    serialized = tr.to_jsonl()
    print(f"  serialized {len(serialized)} records")
    reset_tracker()
    tr2 = get_tracker()
    for d in serialized: tr2.load_record(d)
    print(f"  reloaded moirai health: {tr2.health('moirai')}")
    print(f"  CB still masks moirai: {cb(candidates, ctx)}")
