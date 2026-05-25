"""B6 · Chronos-2（Amazon 2025-10，feedback 增补）。

最新版 Chronos T5-based pretrained TSFM，支持 zero-shot 多变量 + 协变量预测。
本项目用单变量接口，取 quantile 中位（index 10/21）作为点预测。
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
            "amazon/chronos-2",
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.float32,
        )
    return _PIPELINE


def predict(train: np.ndarray, val: np.ndarray, H: int,
            seed: int = 42, season_m: int = 1, **_) -> np.ndarray:
    pipe = _get_pipeline()
    torch.manual_seed(seed)
    ctx = np.concatenate([train, val])
    x = torch.tensor(ctx, dtype=torch.float32).reshape(1, 1, -1)
    out = pipe.predict(x, prediction_length=H)
    quantiles = out[0]  # [1, 21, H]
    median = quantiles[0, quantiles.shape[1] // 2, :].cpu().numpy()
    # A1 副产品：把全部 21 quantile 暴露到模块级，给 entropy gating 用
    predict.last_quantiles = quantiles[0].cpu().numpy()   # type: ignore[attr-defined]
    return median.astype(np.float64)


def predict_with_uncertainty(train: np.ndarray, H: int,
                             seed: int = 42, season_m: int = 1) -> tuple[np.ndarray, float, np.ndarray]:
    """A1 (v12) · 同时返回点预测 + entropy proxy + 完整 21-quantile。
    entropy proxy = mean over H of (q90 - q10) / (|median| + train_std)，无量纲化处理。
    """
    pipe = _get_pipeline()
    torch.manual_seed(seed)
    x = torch.tensor(train, dtype=torch.float32).reshape(1, 1, -1)
    out = pipe.predict(x, prediction_length=H)
    q = out[0][0].cpu().numpy()  # [21, H]
    n_q = q.shape[0]
    median = q[n_q // 2, :]
    # 21 等距 quantile：index 2 ≈ q10, index 18 ≈ q90
    q10 = q[max(0, n_q // 10), :]
    q90 = q[min(n_q - 1, n_q - 1 - n_q // 10), :]
    scale = float(np.std(train)) + 1e-6
    entropy = float(np.mean((q90 - q10) / scale))
    return median.astype(np.float64), entropy, q
