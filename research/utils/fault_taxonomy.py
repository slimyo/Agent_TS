"""P6.1 · 5-fault 时序失败根因检测器（plan §16.3.1 taxonomy）。

5 类 fault（按 plan §16.3.1）：
  F1 trend_break       — train 中段阶跃 (mean-shift > k*std)
  F2 seasonal_flip     — 季节性符号翻转 (ACF lag=m 符号变化)
  F3 variance_explode  — 后段方差爆炸 (late_std / early_std > k)
  F4 outlier_burst     — ≥1 个 |y - median| > 3σ 离群点
  F5 stationarity_flip — split-half ADF p 突变（一半 stationary、一半不）

每个序列可被打多个标签（top-1 + 第二位辅证）。
"""
from __future__ import annotations

import numpy as np

FAULT_NAMES = [
    "trend_break", "seasonal_flip", "variance_explode",
    "outlier_burst", "stationarity_flip",
]


def detect_faults(series: np.ndarray, season_m: int = 1,
                  trend_break_k: float = 2.0,
                  variance_ratio: float = 2.0,
                  outlier_z: float = 3.0) -> dict[str, float]:
    """返回每种 fault 的"强度分"（0~1），可用作 multi-label ground-truth。

    Returns:
        {"trend_break": 0.7, "outlier_burst": 0.3, ...}
        分数 0~1 越高越确信存在该 fault。
    """
    s = np.asarray(series, dtype=np.float64)
    n = len(s)
    if n < 8:
        return {k: 0.0 for k in FAULT_NAMES}

    scores = {}

    # F1 trend_break: max sliding-window mean shift vs global std
    half = n // 2
    early_mean = s[:half].mean()
    late_mean = s[half:].mean()
    global_std = s.std() + 1e-9
    shift = abs(late_mean - early_mean) / global_std
    scores["trend_break"] = float(np.clip(shift / trend_break_k, 0, 1))

    # F2 seasonal_flip: ACF at lag m 在前半 vs 后半符号翻转
    if season_m >= 2 and n >= 2 * season_m + 4:
        def acf_at_lag(x, lag):
            x = x - x.mean()
            denom = (x ** 2).sum() + 1e-9
            return float((x[:-lag] * x[lag:]).sum() / denom)
        a_early = acf_at_lag(s[:half], season_m)
        a_late = acf_at_lag(s[half:], season_m)
        # 符号翻转 + 强度差异
        if a_early * a_late < 0:
            scores["seasonal_flip"] = float(min(1.0, abs(a_early - a_late)))
        else:
            scores["seasonal_flip"] = 0.0
    else:
        scores["seasonal_flip"] = 0.0

    # F3 variance_explode: late_std / early_std
    early_std = s[:half].std() + 1e-9
    late_std = s[half:].std() + 1e-9
    ratio = late_std / early_std
    scores["variance_explode"] = float(np.clip((ratio - 1.0) / (variance_ratio - 1.0), 0, 1)) if ratio > 1 else 0.0

    # F4 outlier_burst: # points with |z| > outlier_z
    med = np.median(s)
    mad = np.median(np.abs(s - med)) + 1e-9
    z = np.abs(s - med) / (1.4826 * mad)
    n_outliers = int((z > outlier_z).sum())
    scores["outlier_burst"] = float(min(1.0, n_outliers / 3.0))  # 3+ outliers = 1.0

    # F5 stationarity_flip: 简化版（不调 ADF，太慢）—— 用 split-half mean+std 联合差异
    early_var = s[:half].var() + 1e-9
    late_var = s[half:].var() + 1e-9
    mean_diff = abs(s[:half].mean() - s[half:].mean()) / global_std
    var_diff = abs(np.log(late_var / early_var))
    scores["stationarity_flip"] = float(np.clip((mean_diff + var_diff) / 2.0, 0, 1))

    return scores


def top_faults(scores: dict[str, float], threshold: float = 0.5,
               max_k: int = 3) -> list[str]:
    """从 scores 取强度 >= threshold 的 top-K fault 名（按强度降序）。"""
    sorted_f = sorted(scores.items(), key=lambda kv: -kv[1])
    return [name for name, sc in sorted_f if sc >= threshold][:max_k]


def assign_ground_truth(train: np.ndarray, test: np.ndarray,
                        season_m: int = 1) -> dict:
    """对单个 (train, test) cell 同时检测 train + test 端的 fault。

    Returns:
        {
          "scores_train": {fault: score, ...},
          "scores_test": {...},
          "primary_fault": "trend_break",
          "secondary_faults": ["outlier_burst"],
          "all_above_thresh": [...]
        }
    """
    s_train = detect_faults(train, season_m)
    s_test = detect_faults(test, season_m)
    # 合并：取 max(train, test) per fault（哪一端有信号都算）
    s_max = {k: max(s_train[k], s_test[k]) for k in FAULT_NAMES}
    all_above = top_faults(s_max, threshold=0.5, max_k=5)
    primary = all_above[0] if all_above else "none"
    secondary = all_above[1:3]
    return {
        "scores_train": {k: round(v, 3) for k, v in s_train.items()},
        "scores_test": {k: round(v, 3) for k, v in s_test.items()},
        "scores_max": {k: round(v, 3) for k, v in s_max.items()},
        "primary_fault": primary,
        "secondary_faults": secondary,
        "all_above_thresh": all_above,
    }


if __name__ == "__main__":
    from research.utils.data_loader import load_series
    from research.utils.splitter import few_shot_split
    s, meta = load_series("ETTh1")
    sp = few_shot_split(s, N=50, H=96, seed=1)
    gt = assign_ground_truth(sp.train, sp.test, season_m=meta.season_m)
    print("ETTh1 N=50 seed=1 ground truth:")
    print(f"  primary: {gt['primary_fault']}")
    print(f"  secondary: {gt['secondary_faults']}")
    print(f"  scores_max: {gt['scores_max']}")
