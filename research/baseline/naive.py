"""B1 · Naive 基线（plan §三 B1）。

两种朴素法，按 val 集表现自动选优后用于 test：
  - mean   : 用训练集均值常数预测
  - drift  : 用最后两点斜率线性外推
  - seasonal: 用最后一个完整季节段循环平铺（season_m 由 runner 注入）

接口：predict(train, val, H, seed=42, season_m=1, **_) -> np.ndarray
"""
from __future__ import annotations

import numpy as np

from research.utils.metrics import mae


def _mean(train: np.ndarray, H: int) -> np.ndarray:
    return np.full(H, float(train.mean()))


def _drift(train: np.ndarray, H: int) -> np.ndarray:
    if len(train) < 2:
        return _mean(train, H)
    slope = (train[-1] - train[0]) / (len(train) - 1)
    return train[-1] + slope * np.arange(1, H + 1)


def _seasonal(train: np.ndarray, H: int, m: int) -> np.ndarray:
    if m <= 1 or len(train) < m:
        return _mean(train, H)
    last_season = train[-m:]
    reps = int(np.ceil(H / m))
    return np.tile(last_season, reps)[:H]


def predict(train: np.ndarray, val: np.ndarray, H: int,
            seed: int = 42, season_m: int = 1, **_) -> np.ndarray:
    """在 val 上挑出 mean/drift/seasonal 最优者，重新生成长度 H 的 test 预测。

    注意：候选预测在 train 末尾延伸 len(val)+H 步，前 len(val) 步用来比 val MAE，
    后 H 步当作 test 预测。这样确保所有候选用相同的"已知历史" = train。
    """
    candidates: dict[str, np.ndarray] = {
        "mean":     _mean(train, len(val) + H),
        "drift":    _drift(train, len(val) + H),
        "seasonal": _seasonal(train, len(val) + H, season_m),
    }
    # 用 val 段挑最优
    best_name, best_err = None, float("inf")
    for name, pred in candidates.items():
        err = mae(val, pred[: len(val)])
        if err < best_err:
            best_err, best_name = err, name
    return candidates[best_name][len(val):]
