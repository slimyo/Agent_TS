"""Round 6 B2 · Confidence Calibration Layer (feedback 后§8).

Maps router's raw confidence (posterior_gap / posterior_max / agreement) to
calibrated P(correct | conf=c). Uses isotonic regression on historical
(confidence, success) pairs.

Inputs: state.telemetry records with .outcome filled in.
Output: a calibrator object with `.calibrate(conf) → P(correct)`.

Drives downstream behavior decisions:
    conf_calibrated >= 0.9 → fast single inference (L0 only)
    conf_calibrated ∈ [0.5, 0.9] → ensemble (L1)
    conf_calibrated ∈ [0.2, 0.5] → specialist escalation (L2)
    conf_calibrated < 0.2 → human-in-loop / abstain

Why this matters:
    - 工业系统真正核心 = "什么时候不能信自己"
    - Raw posterior_gap is uncalibrated heuristic; isotonic regression
      gives empirical probability of success at each confidence level
    - Future: per-regime / per-task calibration when enough data
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import json

import numpy as np


@dataclass
class CalibratorFit:
    """Result of fitting isotonic regression."""
    bin_edges: np.ndarray              # [B+1]
    bin_probs: np.ndarray              # [B] empirical P(correct) per bin
    n_per_bin: np.ndarray              # [B] samples per bin
    n_total: int
    mean_conf: float
    mean_correct: float


def _success_from_outcome(outcome: float, threshold: float) -> int:
    """Outcome (lower=better, e.g. MAE) → binary success.

    Threshold conventionally = median historical outcome, or per-cell baseline.
    """
    return 1 if outcome <= threshold else 0


class ConfidenceCalibrator:
    """Isotonic-regression calibrator over (confidence, success) pairs.

    Lightweight: no sklearn dependency required — implements simple
    monotone-binning with pooled adjacent violators (PAV) idea.
    """
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.fit_info: Optional[CalibratorFit] = None

    def fit(self, confidences: np.ndarray, successes: np.ndarray) -> CalibratorFit:
        """Fit calibration from raw (conf, success) pairs.

        Args:
            confidences: [N] floats in [0, 1]
            successes:   [N] 0/1
        """
        confidences = np.asarray(confidences, dtype=np.float64).clip(0, 1)
        successes = np.asarray(successes, dtype=np.float64)
        if len(confidences) < 5:
            # not enough data — return identity-like calibration
            self.fit_info = CalibratorFit(
                bin_edges=np.linspace(0, 1, self.n_bins + 1),
                bin_probs=np.linspace(0.5, 0.5, self.n_bins),
                n_per_bin=np.zeros(self.n_bins, dtype=int),
                n_total=len(confidences),
                mean_conf=float(confidences.mean()) if len(confidences) else 0.5,
                mean_correct=float(successes.mean()) if len(successes) else 0.5,
            )
            return self.fit_info

        edges = np.linspace(0, 1, self.n_bins + 1)
        bin_idx = np.clip(np.digitize(confidences, edges[1:-1]),
                          0, self.n_bins - 1)
        probs = np.zeros(self.n_bins, dtype=np.float64)
        counts = np.zeros(self.n_bins, dtype=int)
        for b in range(self.n_bins):
            mask = bin_idx == b
            counts[b] = mask.sum()
            if counts[b] > 0:
                probs[b] = float(successes[mask].mean())
            else:
                probs[b] = np.nan

        # Fill empty bins by interpolating from neighbors
        for b in range(self.n_bins):
            if np.isnan(probs[b]):
                left = next((probs[i] for i in range(b-1, -1, -1)
                              if not np.isnan(probs[i])), 0.5)
                right = next((probs[i] for i in range(b+1, self.n_bins)
                              if not np.isnan(probs[i])), 0.5)
                probs[b] = 0.5 * (left + right)

        # Enforce monotone non-decreasing via PAV
        for b in range(1, self.n_bins):
            if probs[b] < probs[b-1]:
                # pool with previous
                pool_n = counts[b] + counts[b-1] + 1e-9
                pooled = (probs[b] * (counts[b] + 1e-9) +
                          probs[b-1] * (counts[b-1] + 1e-9)) / pool_n
                probs[b] = probs[b-1] = pooled

        self.fit_info = CalibratorFit(
            bin_edges=edges, bin_probs=probs,
            n_per_bin=counts, n_total=len(confidences),
            mean_conf=float(confidences.mean()),
            mean_correct=float(successes.mean()),
        )
        return self.fit_info

    def calibrate(self, conf: float) -> float:
        """Map raw confidence → calibrated P(correct)."""
        if self.fit_info is None:
            return 0.5
        c = float(np.clip(conf, 0.0, 1.0))
        b = int(np.clip(np.digitize([c], self.fit_info.bin_edges[1:-1])[0],
                        0, self.n_bins - 1))
        return float(self.fit_info.bin_probs[b])

    # ─── tier 决策 (feedback 后§8 confidence table) ──────────────────────
    @staticmethod
    def behavior_tier(calibrated_conf: float) -> str:
        """Map calibrated conf → recommended downstream behavior."""
        if calibrated_conf >= 0.9: return "fast_single"
        if calibrated_conf >= 0.5: return "ensemble"
        if calibrated_conf >= 0.2: return "specialist_escalate"
        return "human_in_loop"

    # ─── persistence ────────────────────────────────────────────────────
    def save(self, path) -> None:
        from pathlib import Path
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        if self.fit_info is None:
            return
        with p.open("w") as fh:
            fh.write(json.dumps({
                "n_bins": self.n_bins,
                "bin_edges": self.fit_info.bin_edges.tolist(),
                "bin_probs": self.fit_info.bin_probs.tolist(),
                "n_per_bin": self.fit_info.n_per_bin.tolist(),
                "n_total": self.fit_info.n_total,
                "mean_conf": self.fit_info.mean_conf,
                "mean_correct": self.fit_info.mean_correct,
            }) + "\n")

    @classmethod
    def load(cls, path) -> "ConfidenceCalibrator":
        from pathlib import Path
        p = Path(path)
        if not p.exists():
            return cls()
        r = json.loads(p.read_text().splitlines()[0])
        c = cls(n_bins=r["n_bins"])
        c.fit_info = CalibratorFit(
            bin_edges=np.array(r["bin_edges"]),
            bin_probs=np.array(r["bin_probs"]),
            n_per_bin=np.array(r["n_per_bin"]),
            n_total=r["n_total"],
            mean_conf=r["mean_conf"], mean_correct=r["mean_correct"],
        )
        return c


# ─── fit from RouterState telemetry ───────────────────────────────────────────

def fit_from_state(state, threshold_quantile: float = 0.5,
                    metric: str = "posterior_max") -> ConfidenceCalibrator:
    """Fit calibrator from telemetry records.

    Args:
        state: RouterState with .telemetry filled in (each rec must have .outcome)
        threshold_quantile: outcomes ≤ this quantile → "success"
        metric: which confidence to calibrate. One of:
            "posterior_max" — max posterior across candidates
            "posterior_gap" — top1 - top2
    """
    confs, succ = [], []
    outcomes = [r.outcome for r in state.telemetry if r.outcome is not None]
    if len(outcomes) < 5:
        return ConfidenceCalibrator()
    threshold = float(np.quantile(outcomes, threshold_quantile))
    for r in state.telemetry:
        if r.outcome is None: continue
        post = r.posterior
        if not post: continue
        if metric == "posterior_max":
            c = max(post.values())
        elif metric == "posterior_gap":
            sorted_p = sorted(post.values(), reverse=True)
            c = sorted_p[0] - (sorted_p[1] if len(sorted_p) > 1 else 0.0)
        else:
            raise ValueError(metric)
        confs.append(c)
        succ.append(_success_from_outcome(r.outcome, threshold))
    cal = ConfidenceCalibrator()
    cal.fit(np.array(confs), np.array(succ))
    return cal


if __name__ == "__main__":
    print("=" * 60)
    print("Round 6 B2 · Calibration smoke")
    print("=" * 60)

    rng = np.random.default_rng(0)
    # Synthetic: high conf → likely success, low conf → low success
    N = 300
    confs = rng.uniform(0, 1, N)
    succ = (rng.uniform(0, 1, N) < confs).astype(int)   # ground-truth calibration

    cal = ConfidenceCalibrator(n_bins=10)
    info = cal.fit(confs, succ)
    print(f"\nFit on {N} synthetic samples (truth: P(succ|c)=c)")
    print(f"Mean conf: {info.mean_conf:.3f}, mean success: {info.mean_correct:.3f}")
    print(f"\nbin    edges         emp P(correct)   n")
    for b in range(cal.n_bins):
        print(f"  {b:2}  [{info.bin_edges[b]:.2f},{info.bin_edges[b+1]:.2f}]  "
              f"{info.bin_probs[b]:.3f}            {info.n_per_bin[b]}")

    print(f"\nbehavior tier examples:")
    for c in [0.05, 0.3, 0.6, 0.85, 0.95]:
        cc = cal.calibrate(c)
        print(f"  raw={c:.2f} → calibrated={cc:.3f} → behavior: "
              f"{ConfidenceCalibrator.behavior_tier(cc)}")
