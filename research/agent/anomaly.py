"""Round 7 M7 · Anomaly Detection · Phase 1 (feedback 前§5 务实版).

**Phase 1 scope** — 只做最小可跑闭环：
    - 复用 BayesianRouter 选 detector
    - 新增 AnomalyTypePrior（基于序列统计的 fault-type 先验）
    - 两个轻量 detector：
        · RuleBaselineDetector  — 滚动 z-score（最便宜）
        · ResidualScoreDetector — naive_drift 残差 z-score
                                  (站位 "Anomaly-Transformer"，无需深度模型)
    - 输出 fault_type ∈ {trend_break, variance_explode, outlier_burst, normal}

**Phase 1 显式不做**（留 Phase 2/3）：
    - 不加 LLM RCA agent
    - 不加 per-fault-regime Memory（Phase 2 才用）
    - 不加 Anomaly-Transformer 论文模型本体（用残差近似）

仍然复用 method2 §11.5 的 Round 6 基础设施：B2 calibration / B3 drift / E1 action
全部可直接消费 detector 的 score 作 confidence 输入。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math

import numpy as np


# ─── data classes ────────────────────────────────────────────────────────────


FAULT_TYPES = ("normal", "trend_break", "variance_explode", "outlier_burst")


@dataclass
class AnomalyResult:
    is_anomaly: bool
    score: float                          # ∈ [0, 1] combined anomaly score
    suspected_type: str                   # argmax of type_prior
    detector: str                         # which detector produced score
    type_prior: dict[str, float]          # {fault_type → prior mass}
    detector_posterior: dict[str, float]  # {detector → posterior}
    raw: dict = field(default_factory=dict)


# ─── series feature stats (drives type prior) ────────────────────────────────


def _series_stats(window: np.ndarray) -> dict:
    """Compute features that distinguish fault types.

    Returns dict with:
      - level_shift_z   : (mean second half - mean first half) / σ_first
      - variance_ratio  : std(tail) / std(head)
      - max_outlier_z   : largest |x - μ| / σ
      - residual_z      : (last - EMA) / σ_resid (last-step shock)
    """
    x = np.asarray(window, dtype=np.float64)
    if x.size < 12:
        return {"level_shift_z": 0.0, "variance_ratio": 1.0,
                "max_outlier_z": 0.0, "residual_z": 0.0}
    half = x.size // 2
    first, second = x[:half], x[half:]
    sd_first = float(first.std() + 1e-6)
    level_shift_z = abs(second.mean() - first.mean()) / sd_first
    variance_ratio = float(second.std() / sd_first)
    mu = float(x.mean()); sd = float(x.std() + 1e-6)
    max_outlier_z = float(np.max(np.abs(x - mu)) / sd)
    alpha = 0.3
    ema = float(x[0])
    for v in x[1:]: ema = alpha * v + (1 - alpha) * ema
    resid = x - ema
    sd_resid = float(resid[-30:].std() + 1e-6)
    residual_z = float(abs(x[-1] - ema) / sd_resid)
    return {
        "level_shift_z":  level_shift_z,
        "variance_ratio": variance_ratio,
        "max_outlier_z":  max_outlier_z,
        "residual_z":     residual_z,
    }


# ─── AnomalyTypePrior ────────────────────────────────────────────────────────


@dataclass
class AnomalyTypePrior:
    """Prior over fault types from series statistics.

    Heuristics (multiplicative log-odds):
      level_shift_z   > 2  → trend_break
      variance_ratio  > 2  → variance_explode
      max_outlier_z   > 4  → outlier_burst
      otherwise            → normal

    Returns a normalized distribution over FAULT_TYPES.
    """
    name: str = "anomaly_type"
    th_level_shift:   float = 2.0
    th_variance_rat:  float = 2.0
    th_outlier_z:     float = 4.0
    strength:         float = 1.5     # learnable via M3

    def compute(self, window: np.ndarray) -> dict[str, float]:
        s = _series_stats(window)
        logits = {ft: 0.0 for ft in FAULT_TYPES}
        # Each fault gets a logit proportional to its trigger
        logits["trend_break"] += self.strength * max(0.0,
                              s["level_shift_z"] - self.th_level_shift)
        logits["variance_explode"] += self.strength * max(0.0,
                              s["variance_ratio"] - self.th_variance_rat)
        logits["outlier_burst"] += self.strength * max(0.0,
                              s["max_outlier_z"] - self.th_outlier_z)
        # "normal" gets a baseline so it wins when no rule fires
        logits["normal"] += 1.0
        # softmax
        m = max(logits.values())
        exp_ = {k: math.exp(v - m) for k, v in logits.items()}
        Z = sum(exp_.values())
        return {k: v / Z for k, v in exp_.items()}


# ─── Detectors ───────────────────────────────────────────────────────────────


@dataclass
class RuleBaselineDetector:
    """Rolling z-score on last point vs window. Cheapest, no model."""
    name: str = "rule_baseline"
    window_size: int = 50
    threshold_z: float = 3.0

    def detect(self, window: np.ndarray) -> tuple[bool, float, dict]:
        x = np.asarray(window, dtype=np.float64)
        if x.size < 5: return False, 0.0, {"z": 0.0}
        tail = x[-min(self.window_size, len(x)):]
        mu, sd = float(tail.mean()), float(tail.std() + 1e-6)
        z = abs(float(x[-1] - mu) / sd)
        score = float(np.tanh(z / self.threshold_z))   # [0, 1)
        return z > self.threshold_z, score, {"z": z, "tail_mu": mu, "tail_sd": sd}


@dataclass
class ResidualScoreDetector:
    """EMA-residual z-score; stand-in for an "Anomaly-Transformer"-style model.

    Phase 1 wants minimum dependencies; this preserves the *role* of a
    model-based detector (uses a learned-ish predictor's residual) without
    needing transformer weights. Phase 2/3 can swap implementation while
    keeping this interface.
    """
    name: str = "residual_score"
    alpha:        float = 0.3
    window_size:  int = 50
    threshold_z:  float = 3.0

    def detect(self, window: np.ndarray) -> tuple[bool, float, dict]:
        x = np.asarray(window, dtype=np.float64)
        if x.size < 5: return False, 0.0, {"residual_z": 0.0}
        # EMA forecast
        ema = float(x[0])
        for v in x[1:]: ema = self.alpha * v + (1 - self.alpha) * ema
        tail = x[-min(self.window_size, len(x)):]
        resid = tail - ema   # naive but cheap
        sd = float(resid.std() + 1e-6)
        z = abs(float(x[-1] - ema) / sd)
        score = float(np.tanh(z / self.threshold_z))
        return z > self.threshold_z, score, {
            "residual_z": z, "ema": ema, "resid_sd": sd,
        }


DETECTORS = {
    "rule_baseline":   RuleBaselineDetector,
    "residual_score":  ResidualScoreDetector,
}


# ─── Orchestrator ────────────────────────────────────────────────────────────


@dataclass
class AnomalyConfig:
    candidates: tuple = ("rule_baseline", "residual_score")
    score_threshold: float = 0.6        # is_anomaly = combined score > th
    ensemble: bool = True               # weighted average vs argmax
    type_prior: AnomalyTypePrior = field(default_factory=AnomalyTypePrior)
    # Per-detector accuracy prior (could be learned). High value = trusted more.
    detector_strength: dict[str, float] = field(default_factory=lambda: {
        "rule_baseline":  1.0,
        "residual_score": 1.3,
    })


def detect_anomaly(window: np.ndarray,
                   config: Optional[AnomalyConfig] = None) -> AnomalyResult:
    """Run all configured detectors; combine via per-detector posterior."""
    if config is None: config = AnomalyConfig()
    type_prior = config.type_prior.compute(window)

    raw_scores: dict[str, float] = {}
    detector_raws: dict[str, dict] = {}
    per_det_flag: dict[str, bool] = {}
    for name in config.candidates:
        cls = DETECTORS.get(name)
        if cls is None: continue
        det = cls()
        is_a, score, raw = det.detect(window)
        raw_scores[name] = float(score)
        detector_raws[name] = raw
        per_det_flag[name] = bool(is_a)

    # Detector posterior: softmax(strength · score)
    if not raw_scores:
        return AnomalyResult(False, 0.0, "normal", "none",
                              type_prior, {}, {})
    logits = {n: config.detector_strength.get(n, 1.0) * s
              for n, s in raw_scores.items()}
    m_ = max(logits.values())
    exp_ = {n: math.exp(v - m_) for n, v in logits.items()}
    Z = sum(exp_.values())
    posterior = {n: v / Z for n, v in exp_.items()}

    # Combined score: ensemble or argmax
    if config.ensemble:
        combined = sum(posterior[n] * raw_scores[n] for n in raw_scores)
        chosen = max(posterior, key=posterior.get)
    else:
        chosen = max(posterior, key=posterior.get)
        combined = raw_scores[chosen]
    is_anom = combined > config.score_threshold
    suspected = max(type_prior, key=type_prior.get)

    return AnomalyResult(
        is_anomaly=is_anom, score=float(combined),
        suspected_type=suspected if is_anom else "normal",
        detector=chosen,
        type_prior=type_prior,
        detector_posterior=posterior,
        raw={"per_detector_score": raw_scores,
              "per_detector_flag":   per_det_flag,
              "per_detector_raw":    detector_raws,
              "series_stats":        _series_stats(window)},
    )


# ─── smoke ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Round 7 M7 Phase 1 · Anomaly detection smoke")
    print("=" * 70)
    rng = np.random.default_rng(0)
    base = np.cumsum(rng.standard_normal(200)) * 0.1 + 50.0

    cases = {
        "normal":            base.copy(),
        "trend_break":       np.concatenate([base[:100], base[100:] + 5.0]),
        "variance_explode":  np.concatenate([base[:100],
                              base[100:] + 3.0 * rng.standard_normal(100)]),
        "outlier_burst":     base.copy(),
    }
    cases["outlier_burst"][180] = base.mean() + 6 * base.std()

    cfg = AnomalyConfig()
    for label, window in cases.items():
        res = detect_anomaly(window, cfg)
        tp_top = sorted(res.type_prior.items(), key=lambda kv: -kv[1])[:2]
        det_top = sorted(res.detector_posterior.items(),
                          key=lambda kv: -kv[1])[:2]
        print(f"\n[{label}]")
        print(f"  is_anomaly={res.is_anomaly}  score={res.score:.3f}  "
              f"suspected_type={res.suspected_type}")
        print(f"  detector chosen: {res.detector}")
        print(f"  type_prior top2:        {tp_top}")
        print(f"  detector_posterior top2: {det_top}")
        print(f"  per_detector_score: { {n: round(v, 3) for n, v in res.raw['per_detector_score'].items()} }")
