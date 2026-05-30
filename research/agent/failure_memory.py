"""Round 6 C1 · Failure Memory (feedback 后§3.2 "what predicted failure").

Stores not just "model X won" but "model X FAILED, under what conditions,
what signal predicted the failure". This lets routing evolve to avoid
failure modes rather than just chase winners.

Schema (FailureCase):
    timestamp:       float           # when observed
    model:           str
    regime:          int | None
    series_features: dict            # 25-d feature snapshot at decision time
    failure_type:    str             # "high_residual" | "OOM" | "load_error"
                                     # | "drift_mismatch" | "outlier_corruption"
    failure_signal:  dict[str, bool] # WHAT predicted failure (e.g.
                                     #  {high_freq_noise: True, low_N: True})
    actual_loss:     float
    expected_loss:   float           # from bandit belief / prior at time
    severity:        float           # (actual − expected) / expected
    notes:           str

Retrieval by feature similarity + failure_type filtering.

Future C4 Empirical Bayes: aggregate failure_signal frequencies per model
→ learn which features predict failure for which models.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
import json
import time

import numpy as np


# ─── failure type taxonomy ────────────────────────────────────────────────────

FAILURE_TYPES = (
    "high_residual",      # actual loss >> expected
    "load_error",         # model failed to load (env / weights)
    "oom",                # GPU memory error
    "drift_mismatch",     # prediction systematically off baseline
    "outlier_corruption", # output contains inf/nan or extreme magnitude
    "latency_violation",  # timeout
    "unknown",
)


def classify_failure(actual_loss: float, expected_loss: float,
                     error_str: Optional[str] = None,
                     series: Optional[np.ndarray] = None,
                     pred: Optional[np.ndarray] = None,
                     severity_threshold: float = 0.5,
                     ) -> tuple[str, float]:
    """Auto-classify failure type + severity.

    Returns (failure_type, severity_score ∈ [0, ∞)).
    """
    if error_str:
        es = error_str.lower()
        if "oom" in es or "out of memory" in es:
            return "oom", 1.0
        if "module" in es or "not found" in es or "noattribute" in es:
            return "load_error", 1.0
        if "timeout" in es:
            return "latency_violation", 1.0
        if "infinity" in es or "nan" in es:
            return "outlier_corruption", 1.0
    if pred is not None:
        if not np.all(np.isfinite(pred)):
            return "outlier_corruption", 1.0
    sev = (actual_loss - expected_loss) / max(abs(expected_loss), 1e-6)
    if sev > severity_threshold:
        return "high_residual", float(sev)
    return "unknown", float(sev)


# ─── failure-signal extraction ────────────────────────────────────────────────

def extract_failure_signal(features: dict, N: Optional[int] = None,
                           entropy: Optional[float] = None,
                           regime: Optional[int] = None,
                           ) -> dict[str, bool]:
    """Tag which factors likely contributed to the failure.

    Returns boolean dict so downstream Empirical Bayes can count co-occurrence.
    """
    sig = {}
    if features:
        # high-frequency / noise: spectral entropy > 0.8 or noise-fft high
        if features.get("spectral_entropy", 0) > 0.8:
            sig["high_freq_noise"] = True
        # low signal-to-noise: outlier ratio high
        if features.get("mad_outlier_frac", 0) > 0.15:
            sig["heavy_outliers"] = True
        # discontinuity / shift
        if features.get("step_count", 0) > 3:
            sig["level_shifts"] = True
        # near-constant: low std
        if features.get("std", 1) < 0.05:
            sig["near_constant"] = True
        # extreme magnitude
        if features.get("range", 0) > 1e4:
            sig["extreme_magnitude"] = True
    if N is not None and N <= 12:
        sig["low_N"] = True
    if entropy is not None and entropy > 1.5:
        sig["high_router_entropy"] = True
    return sig


# ─── FailureCase ──────────────────────────────────────────────────────────────

@dataclass
class FailureCase:
    timestamp: float
    model: str
    regime: Optional[int]
    series_features: dict
    failure_type: str
    failure_signal: dict
    actual_loss: float
    expected_loss: float
    severity: float
    notes: str = ""


# ─── FailureMemory store ──────────────────────────────────────────────────────

@dataclass
class FailureMemory:
    cases: list[FailureCase] = field(default_factory=list)

    def add(self, case: FailureCase) -> None:
        self.cases.append(case)

    def record(self, model: str, regime: Optional[int], features: dict,
               actual_loss: float, expected_loss: float,
               error_str: Optional[str] = None,
               pred: Optional[np.ndarray] = None,
               N: Optional[int] = None,
               entropy: Optional[float] = None,
               notes: str = "") -> Optional[FailureCase]:
        """One-shot record from a decide+observe outcome."""
        ftype, sev = classify_failure(actual_loss, expected_loss, error_str, None, pred)
        # If sev is small and ftype unknown, not a failure
        if ftype == "unknown" and sev <= 0.5:
            return None
        signal = extract_failure_signal(features, N=N, entropy=entropy,
                                         regime=regime)
        case = FailureCase(
            timestamp=time.time(), model=model, regime=regime,
            series_features=features, failure_type=ftype,
            failure_signal=signal,
            actual_loss=float(actual_loss),
            expected_loss=float(expected_loss),
            severity=float(sev), notes=notes,
        )
        self.cases.append(case)
        return case

    # ─── retrieval / aggregation ──────────────────────────────────────
    def by_model(self, model: str) -> list[FailureCase]:
        return [c for c in self.cases if c.model == model]

    def by_signal(self, signal_key: str) -> list[FailureCase]:
        return [c for c in self.cases if c.failure_signal.get(signal_key)]

    def signal_frequency(self, model: str) -> dict[str, int]:
        """For Empirical Bayes (C4): which signals co-occur with failures of model X?"""
        from collections import Counter
        cnt: Counter = Counter()
        for c in self.by_model(model):
            for k in c.failure_signal: cnt[k] += 1
        return dict(cnt)

    def failure_rate_per_model(self, n_total_per_model: dict[str, int]
                                ) -> dict[str, float]:
        """failure / total for each model."""
        from collections import Counter
        n_fail = Counter(c.model for c in self.cases)
        return {m: n_fail[m] / max(n_total_per_model.get(m, 1), 1)
                for m in n_total_per_model}

    def reliability_prior(self, model: str, n_total_per_model: dict[str, int]
                           ) -> float:
        """Round 6 → F14 解药 · 用于构造 OperationalReliabilityPrior。

        Returns success rate ∈ [0, 1] for model based on observed failures.
        """
        n_total = max(n_total_per_model.get(model, 1), 1)
        n_fail = sum(1 for c in self.cases if c.model == model)
        return 1.0 - (n_fail / n_total)

    # ─── persistence ──────────────────────────────────────────────────
    def save(self, path) -> None:
        from pathlib import Path
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as fh:
            for c in self.cases:
                fh.write(json.dumps(asdict(c)) + "\n")

    @classmethod
    def load(cls, path) -> "FailureMemory":
        from pathlib import Path
        p = Path(path)
        if not p.exists():
            return cls()
        mem = cls()
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
                mem.cases.append(FailureCase(**r))
            except Exception: pass
        return mem


if __name__ == "__main__":
    print("=" * 60)
    print("Round 6 C1 · Failure Memory smoke")
    print("=" * 60)
    mem = FailureMemory()

    # 模拟 5 个 failure 案例
    mem.record(model="tirex", regime=0,
                features={"spectral_entropy": 0.92, "mad_outlier_frac": 0.18,
                          "std": 0.5, "range": 12.0},
                actual_loss=3.5, expected_loss=1.0,
                error_str=None, N=10, entropy=1.8,
                notes="ETTh1 N=10 - high freq noise scenario")

    mem.record(model="tirex", regime=0,
                features={"spectral_entropy": 0.91, "mad_outlier_frac": 0.20,
                          "std": 0.6, "range": 14.0},
                actual_loss=4.2, expected_loss=1.1,
                error_str=None, N=10, entropy=1.7,
                notes="同模式重复")

    mem.record(model="toto", regime=2,
                features={"spectral_entropy": 0.4, "step_count": 5,
                          "std": 1.0, "range": 25.0},
                actual_loss=15.0, expected_loss=2.0,
                error_str=None, N=50, entropy=0.5,
                notes="ECL N=50 - drift")

    mem.record(model="time_moe", regime=1,
                features={"spectral_entropy": 0.3, "std": 0.1, "range": 2.0},
                actual_loss=None or 0,    # ignored when load error
                expected_loss=1.0,
                error_str="ModuleNotFoundError: time_moe",
                notes="env compat")

    mem.record(model="chronos2", regime=0,
                features={"std": 0.4, "range": 5.0, "spectral_entropy": 0.3},
                actual_loss=0.5, expected_loss=0.5,  # not a failure
                notes="正常情况")

    print(f"\nTotal failure cases recorded: {len(mem.cases)}")
    print(f"By failure_type:")
    from collections import Counter
    types = Counter(c.failure_type for c in mem.cases)
    for t, n in types.items(): print(f"  {t}: {n}")

    print(f"\nTirex signal frequency:")
    print(f"  {mem.signal_frequency('tirex')}")

    print(f"\nReliability prior (assume 10 attempts per model):")
    n_per = {"tirex": 10, "toto": 10, "time_moe": 5, "chronos2": 10}
    for m in ["tirex", "toto", "time_moe", "chronos2"]:
        print(f"  {m:10}: reliability = {mem.reliability_prior(m, n_per):.2f}")
