"""B7 · Chronos-Bolt（Amazon 2024-12，feedback 隐含的"升级版 Chronos-Small"）。

Chronos-Bolt 是 Chronos-Small 的直接升级：相同 T5 架构但 5-15x 推理加速。
patch tokenization；输出 9 quantile（取 median index 4）。
"""
from __future__ import annotations

import numpy as np
import torch

_PIPELINE = None


def _get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        from chronos import BaseChronosPipeline
        _PIPELINE = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-small",
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.float32,
        )
    return _PIPELINE


def predict(train: np.ndarray, val: np.ndarray, H: int,
            seed: int = 42, season_m: int = 1, **_) -> np.ndarray:
    pipe = _get_pipeline()
    torch.manual_seed(seed)
    ctx = torch.tensor(np.concatenate([train, val]), dtype=torch.float32)
    out = pipe.predict(ctx, prediction_length=H)   # [1, 9, H]
    median = out[0, out.shape[1] // 2, :].cpu().numpy()
    return median.astype(np.float64)
