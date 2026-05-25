"""A8 消融变体（plan §7.2 + §十五 N6）：w/o Model Cards。

与 Full（AdaptTS v5c）的唯一区别：反思 prompt 不嵌入 Model Cards 能力卡。
量化"模型先验知识"对反思 root_cause 文本质量的贡献。
MAE 预期接近 Full（best_plan 仍锁 walk-forward initial）。
"""
from __future__ import annotations
import numpy as np
from research.agent import adapt_ts


def predict(train: np.ndarray, val: np.ndarray, H: int,
            seed: int = 42, season_m: int = 1, **kwargs) -> np.ndarray:
    kwargs["use_model_cards"] = False
    y = adapt_ts.predict(train, val, H, seed=seed, season_m=season_m, **kwargs)
    predict.last_trace = adapt_ts.predict.last_trace   # type: ignore[attr-defined]
    return y
