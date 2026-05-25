"""A3 · 概率预测指标（CRPS / pinball / interval coverage）。

Chronos-2 输出 21 等距 quantile：levels = i/20, i=0..20。
对 point predictor（ARIMA/LLMTime/Naive），把 point forecast 当退化分布——
所有 quantile = point，pinball / coverage / CRPS 仍可算（CRPS 退化为 MAE）。

CRPS 公式（来自 sample-based 量级估计）：
  对 21 个等距 quantile q_i，CRPS(F, y) ≈ ∑_{i=1..n-1} (q_{i+1}-q_i) * |y - q_i| 的近似
  实际用更标准的：CRPS = ∫ (F(x)-1{x>=y})^2 dx
  对 sorted quantile representation, 可写为：
    CRPS = 2/n * ∑_i (y - q_i) * (1{y<q_i} - (i-0.5)/n)
  即 weighted pinball loss 求和后 / n 的 2 倍。
"""
from __future__ import annotations

import numpy as np


def pinball_loss(y_true: np.ndarray, q_pred: np.ndarray, alpha: float) -> float:
    """Quantile loss at level alpha. q_pred 形状 [H]，y_true 形状 [H]."""
    diff = y_true - q_pred
    loss = np.where(diff >= 0, alpha * diff, (alpha - 1) * diff)
    return float(np.mean(loss))


def crps_from_quantiles(y_true: np.ndarray, quantiles: np.ndarray,
                        levels: np.ndarray | None = None) -> float:
    """从 K 个 quantile 估 CRPS。quantiles 形状 [K, H]，levels 形状 [K]（默认等距）。

    用 Laio & Tamea 2007 / sample-based 公式：
      CRPS(F,y) ≈ 2/K * ∑_k pinball(y, q_k, level_k)
    这是 K quantile 表征下的 unbiased estimator。
    """
    K, H = quantiles.shape
    if levels is None:
        levels = (np.arange(K, dtype=np.float64) + 1) / (K + 1)  # midpoint convention
    total = 0.0
    for k in range(K):
        total += pinball_loss(y_true, quantiles[k], float(levels[k]))
    return 2.0 * total / K


def interval_coverage(y_true: np.ndarray, q_low: np.ndarray, q_high: np.ndarray) -> float:
    """覆盖率 = P(q_low <= y <= q_high)。"""
    inside = (y_true >= q_low) & (y_true <= q_high)
    return float(np.mean(inside))


def interval_width(q_low: np.ndarray, q_high: np.ndarray) -> float:
    return float(np.mean(q_high - q_low))


def point_as_degenerate_quantiles(y_pred: np.ndarray, n_q: int = 21) -> np.ndarray:
    """把 point forecast 当退化分布（所有 quantile = point）。"""
    return np.tile(y_pred[None, :], (n_q, 1))


def prob_metrics_from_quantiles(y_true: np.ndarray, quantiles: np.ndarray) -> dict[str, float]:
    """主入口：21 quantile -> {crps, pinball_q10, pinball_q50, pinball_q90,
                                coverage_80, width_80}。"""
    K = quantiles.shape[0]
    # 21 等距 quantile：index 2 ≈ q10, 10 ≈ q50, 18 ≈ q90
    idx_10 = int(0.1 * (K - 1))
    idx_50 = (K - 1) // 2
    idx_90 = int(0.9 * (K - 1))
    return {
        "crps":        crps_from_quantiles(y_true, quantiles),
        "pinball_q10": pinball_loss(y_true, quantiles[idx_10], 0.1),
        "pinball_q50": pinball_loss(y_true, quantiles[idx_50], 0.5),
        "pinball_q90": pinball_loss(y_true, quantiles[idx_90], 0.9),
        "coverage_80": interval_coverage(y_true, quantiles[idx_10], quantiles[idx_90]),
        "width_80":    interval_width(quantiles[idx_10], quantiles[idx_90]),
    }


if __name__ == "__main__":
    np.random.seed(0)
    y = np.random.randn(96)
    # 完美校准的预测：21 quantile of N(0,1)
    from scipy.stats import norm
    levels = (np.arange(21) + 1) / 22
    q = np.array([norm.ppf(l) for l in levels])
    q = np.tile(q[:, None], (1, 96))
    print(prob_metrics_from_quantiles(y, q))
