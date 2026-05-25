"""A9 消融变体（plan §7.2 + §十五 N6）：w/o Diagnosis Revision。

与 Full 区别：保留 Model Cards，但反思 prompt 移除 diagnosis_revision 字段；
量化"诊断纠偏闭环"分支对 case study 价值的贡献（关闭后 diagnosis_revised 强制 None）。
"""
from __future__ import annotations
import numpy as np
from research.agent import adapt_ts


def predict(train: np.ndarray, val: np.ndarray, H: int,
            seed: int = 42, season_m: int = 1, **kwargs) -> np.ndarray:
    kwargs["allow_diagnosis_revision"] = False
    y = adapt_ts.predict(train, val, H, seed=seed, season_m=season_m, **kwargs)
    predict.last_trace = adapt_ts.predict.last_trace   # type: ignore[attr-defined]
    return y
