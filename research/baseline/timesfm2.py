"""B8 · TimesFM 2.0（Google ICLR 2025，feedback 增补）。

Decoder-only TSFM 500M params，支持长上下文 + 多变量。
checkpoint: google/timesfm-2.0-500m-pytorch
"""
from __future__ import annotations

import numpy as np

_TFM = None


def _get_pipeline(horizon_len: int):
    global _TFM
    if _TFM is None:
        import timesfm
        _TFM = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="cpu", per_core_batch_size=1,
                horizon_len=horizon_len,
                context_len=512, num_layers=50,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
            ),
        )
    return _TFM


def predict(train: np.ndarray, val: np.ndarray, H: int,
            seed: int = 42, season_m: int = 1, **_) -> np.ndarray:
    pipe = _get_pipeline(horizon_len=H)
    ctx = np.concatenate([train, val]).astype(np.float32)
    fcst, _q = pipe.forecast(inputs=[ctx], freq=[0])
    # fcst: [batch=1, horizon_len]
    return np.asarray(fcst[0][:H], dtype=np.float64)
