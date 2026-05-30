"""task #69 · TiRex (NX-AI 2025) — xLSTM-based zero-shot forecasting TSFM.

API surface (matches research/baseline/chronos2.py for plug-in compatibility):
  - predict(train, val, H, seed, season_m) -> np.ndarray   (median forecast)
  - predict_with_uncertainty(train, H, seed, season_m) -> (median, entropy, quantiles)

Reference: https://huggingface.co/NX-AI/TiRex
pip package: tirex-ts (NOT `tirex` — name collision with SIREX library)
"""
from __future__ import annotations
import numpy as np
import torch

_MODEL = None
_QUANTILE_LEVELS = None


def _get_model():
    """Lazy load TiRex from HF."""
    global _MODEL, _QUANTILE_LEVELS
    if _MODEL is None:
        from tirex import load_model
        _MODEL = load_model("NX-AI/TiRex")
        # TiRex returns 9 quantiles by default
        _QUANTILE_LEVELS = np.linspace(0.1, 0.9, 9)
    return _MODEL


def predict(train: np.ndarray, val: np.ndarray, H: int,
            seed: int = 42, season_m: int = 1, **_) -> np.ndarray:
    """Median (q=0.5) point forecast.

    Compatible signature with chronos2.predict for drop-in router replacement.
    """
    model = _get_model()
    torch.manual_seed(seed)
    ctx = np.concatenate([train, val]).astype(np.float32)
    ctx_t = torch.tensor(ctx).reshape(1, -1)
    quantiles, median = model.forecast(context=ctx_t, prediction_length=H)
    # cache for entropy gating
    predict.last_quantiles = quantiles[0].detach().cpu().numpy()    # [H, 9]
    return median[0].detach().cpu().numpy().astype(np.float64)


def predict_with_uncertainty(train: np.ndarray, H: int,
                              seed: int = 42, season_m: int = 1
                              ) -> tuple[np.ndarray, float, np.ndarray]:
    """Median forecast + entropy proxy + full quantile matrix.

    entropy_proxy = mean over H of (q90 - q10) / (|median| + std(train))
    quantiles shape: [9, H] to match chronos2 convention (transposed from TiRex native).
    """
    model = _get_model()
    torch.manual_seed(seed)
    ctx_t = torch.tensor(train.astype(np.float32)).reshape(1, -1)
    quantiles, median = model.forecast(context=ctx_t, prediction_length=H)
    q = quantiles[0].detach().cpu().numpy()           # [H, 9]
    med = median[0].detach().cpu().numpy()             # [H]
    # Transpose to [9, H] for chronos2-style consumers
    q_T = q.T                                          # [9, H]
    q10 = q_T[0]                                       # 0.1 quantile
    q90 = q_T[-1]                                      # 0.9 quantile
    scale = float(np.std(train)) + 1e-6
    entropy = float(np.mean((q90 - q10) / scale))
    return med.astype(np.float64), entropy, q_T


if __name__ == "__main__":
    # Smoke test
    rng = np.random.default_rng(0)
    sig = np.sin(np.arange(200) * 0.1) + 0.1 * rng.standard_normal(200)
    train = sig[:170]; val = np.array([])
    pred = predict(train, val, H=24, seed=1)
    print(f"Median pred shape: {pred.shape}, range [{pred.min():.3f}, {pred.max():.3f}]")
    med, ent, q = predict_with_uncertainty(train, H=24)
    print(f"With uncertainty: med={med.shape}, entropy={ent:.3f}, quantiles={q.shape}")
