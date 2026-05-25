"""AdaptTS-Agent v6 · 启用策略 promotion（plan §15.5 改进）。

与 Full (v5c) 唯一区别：enable_promotion=True。
反思过程中若某单策略在 val 上显著优于 best，触发更长 holdout 的二次 walk-forward CV；
若 promotion CV 也确认优势，则 promote 为单策略 best。

目标：解决 v5c case 2 揭示的"反思找到 winner 但用不上"问题。
"""
from __future__ import annotations
import numpy as np
from research.agent import adapt_ts


def predict(train: np.ndarray, val: np.ndarray, H: int,
            seed: int = 42, season_m: int = 1, **kwargs) -> np.ndarray:
    kwargs["enable_promotion"] = True
    y = adapt_ts.predict(train, val, H, seed=seed, season_m=season_m, **kwargs)
    predict.last_trace = adapt_ts.predict.last_trace   # type: ignore[attr-defined]
    return y
