"""Round 6 C2 · Memory Decay (feedback 后§3.3 "旧 case 会害人").

Three decay schemes:

    1. **Exponential time decay** (most common):
         w_i(t) = exp(-Δt_i / τ)
       Older cases automatically lose influence.

    2. **Regime-drift decay**:
         If current series regime distribution shifts (KL divergence
         vs historical), down-weight ALL cases proportional to drift.

    3. **Seasonal forgetting** (utility / weather):
         Cases from > k seasons ago weighted by f(Δseason / season_period).
         (Not implemented here — for future quarterly data.)

API:
    weights = compute_decay_weights(cases, now, tau_days=30.0,
                                     regime_drift_score=0.0)
    # used by MemoryLikelihood / RepresentationLikelihood to re-weight neighbors

Time stored in seconds (Unix epoch); tau in days.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import math
import time

import numpy as np


# ─── core decay function ──────────────────────────────────────────────────────

def exp_time_decay(case_timestamps: np.ndarray,
                   now: Optional[float] = None,
                   tau_days: float = 30.0) -> np.ndarray:
    """Exponential time-decay weight ∈ (0, 1].

    Args:
        case_timestamps: [N] case timestamps (Unix seconds)
        now: reference time (default = current time)
        tau_days: half-life-like parameter; w drops to 1/e at this age
    """
    if now is None: now = time.time()
    age_seconds = now - np.asarray(case_timestamps)
    age_days = age_seconds / 86400.0
    tau_seconds = tau_days * 86400.0 / 86400.0  # same units
    return np.exp(-age_days / max(tau_days, 1e-6))


def regime_drift_kl(current_regime_dist: dict[int, int],
                    historical_regime_dist: dict[int, int],
                    smoothing: float = 0.5) -> float:
    """KL divergence between current vs historical regime distributions.

    Higher = more drift. Used to globally down-weight memory.
    """
    all_keys = set(current_regime_dist) | set(historical_regime_dist)
    if not all_keys: return 0.0
    def to_dist(d):
        total = sum(d.values()) + smoothing * len(all_keys)
        return {k: (d.get(k, 0) + smoothing) / total for k in all_keys}
    p = to_dist(current_regime_dist)
    q = to_dist(historical_regime_dist)
    return sum(p[k] * math.log(p[k] / q[k]) for k in all_keys)


def drift_decay_factor(drift_score: float, max_drift: float = 1.0) -> float:
    """Convert KL drift score → multiplicative decay factor ∈ (0, 1].

    Drift = 0  → factor = 1.0 (no down-weight)
    Drift = max_drift → factor = exp(-1) ≈ 0.37
    """
    return float(math.exp(-drift_score / max(max_drift, 1e-6)))


# ─── unified weight ───────────────────────────────────────────────────────────

@dataclass
class DecayConfig:
    tau_days: float = 30.0
    regime_drift_max: float = 1.0
    enable_time_decay: bool = True
    enable_drift_decay: bool = True


def compute_decay_weights(case_timestamps: np.ndarray,
                          now: Optional[float] = None,
                          regime_drift_score: float = 0.0,
                          config: Optional[DecayConfig] = None) -> np.ndarray:
    """Combine all decay sources into a single weight ∈ (0, 1] per case."""
    if config is None: config = DecayConfig()
    w = np.ones(len(case_timestamps), dtype=np.float64)
    if config.enable_time_decay:
        w *= exp_time_decay(case_timestamps, now, config.tau_days)
    if config.enable_drift_decay and regime_drift_score > 0:
        w *= drift_decay_factor(regime_drift_score, config.regime_drift_max)
    return w


# ─── drift detector hook (used by C2 and future B3 drift engine) ──────────────

def detect_regime_drift(recent_regimes: list[int],
                         historical_regimes: list[int],
                         smoothing: float = 0.5) -> dict:
    """Compute drift score + detection flag for caller."""
    from collections import Counter
    rec = dict(Counter(recent_regimes))
    hist = dict(Counter(historical_regimes))
    score = regime_drift_kl(rec, hist, smoothing)
    return {
        "kl_score": score,
        "detected": score > 0.5,
        "recent_dist": rec,
        "historical_dist": hist,
    }


if __name__ == "__main__":
    import numpy as np
    print("=" * 60)
    print("Round 6 C2 · Memory Decay smoke")
    print("=" * 60)

    # ─── 时间衰减 ──────────────────────────────────────────────
    now = time.time()
    case_times = np.array([
        now - 0 * 86400,    # 现在
        now - 7 * 86400,    # 1 周前
        now - 30 * 86400,   # 1 月前
        now - 90 * 86400,   # 3 月前
        now - 365 * 86400,  # 1 年前
    ])
    w = exp_time_decay(case_times, now, tau_days=30.0)
    print("\n[Time decay, τ=30 days]")
    for age, weight in zip([0, 7, 30, 90, 365], w):
        print(f"  age={age:>3} days  weight={weight:.4f}")

    # ─── regime drift ──────────────────────────────────────────
    print("\n[Regime drift KL]")
    drift_a = detect_regime_drift([0,0,0,1,1], [0,0,0,1,1])    # 完全一致
    drift_b = detect_regime_drift([2,2,2,3,3], [0,0,0,1,1])    # 完全不同
    drift_c = detect_regime_drift([0,0,2,1,2], [0,0,0,1,1])    # 部分偏移
    print(f"  identical regimes:  KL={drift_a['kl_score']:.4f}  detected={drift_a['detected']}")
    print(f"  total mismatch:     KL={drift_b['kl_score']:.4f}  detected={drift_b['detected']}")
    print(f"  partial drift:      KL={drift_c['kl_score']:.4f}  detected={drift_c['detected']}")

    # ─── combined ──────────────────────────────────────────────
    print("\n[Combined decay: time + drift]")
    cfg = DecayConfig(tau_days=30, regime_drift_max=1.0,
                       enable_time_decay=True, enable_drift_decay=True)
    w_no_drift = compute_decay_weights(case_times, now, 0.0, cfg)
    w_with_drift = compute_decay_weights(case_times, now, 0.8, cfg)
    print(f"  no drift:       {[f'{x:.3f}' for x in w_no_drift]}")
    print(f"  high drift 0.8: {[f'{x:.3f}' for x in w_with_drift]}")
