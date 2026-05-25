"""B5 · Chronos-Small 零样本基线（plan §三 B5）。

使用 amazon/chronos-t5-small (60M)，将 train+val 拼接作为 context，
直接生成 H 步预测，取中位数作为点预测。

注意：模型权重首次会下载到 HF cache（约 250MB），可设置 HF_HOME 控制位置。
"""
from __future__ import annotations

import numpy as np
import torch

_PIPELINE = None


def _get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        from chronos import ChronosPipeline
        _PIPELINE = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-small",
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.float32,
        )
    return _PIPELINE


def predict(train: np.ndarray, val: np.ndarray, H: int,
            seed: int = 42, season_m: int = 1, **_) -> np.ndarray:
    pipe = _get_pipeline()
    torch.manual_seed(seed)
    context = torch.tensor(np.concatenate([train, val]), dtype=torch.float32)
    # forecast: shape [num_series=1, num_samples, H]
    forecast = pipe.predict(inputs=context, prediction_length=H, num_samples=20,
                            limit_prediction_length=False)
    median = np.median(forecast[0].cpu().numpy(), axis=0)
    return median.astype(np.float64)
