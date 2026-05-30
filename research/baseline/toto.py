"""task #69 · Toto (DataDog 2025/26) — observability-focused TSFM.

Open Base 1.0 (Datadog/Toto-Open-Base-1.0) is the only freely-loadable variant
that works in our env (torch 2.7 + Python 3.10). Toto 2.0 requires Python 3.12.

API (matches chronos2 for drop-in routing):
  - predict(train, val, H, seed, season_m) -> np.ndarray
  - predict_with_uncertainty(train, H, seed, season_m) -> (median, entropy, q)
"""
from __future__ import annotations
import numpy as np
import torch

_MODEL = None
_FORECASTER = None


def _get():
    """Lazy-load Toto Open Base 1.0."""
    global _MODEL, _FORECASTER
    if _FORECASTER is None:
        from toto.model.toto import Toto
        from toto.inference.forecaster import TotoForecaster
        _MODEL = Toto.from_pretrained("Datadog/Toto-Open-Base-1.0")
        _MODEL.eval()
        _FORECASTER = TotoForecaster(_MODEL.model)
    return _FORECASTER


def _make_ts(history: np.ndarray, time_interval_seconds: int = 3600):
    from toto.inference.forecaster import MaskedTimeseries
    L = len(history)
    return MaskedTimeseries(
        series=torch.tensor(history.astype(np.float32)).reshape(1, 1, -1),
        padding_mask=torch.ones(1, 1, L, dtype=torch.bool),
        id_mask=torch.zeros(1, 1, L, dtype=torch.int),
        timestamp_seconds=torch.zeros(1, 1, L, dtype=torch.int64),
        time_interval_seconds=torch.tensor([[time_interval_seconds]], dtype=torch.int64),
    )


def predict(train: np.ndarray, val: np.ndarray, H: int,
            seed: int = 42, season_m: int = 1, **_) -> np.ndarray:
    """Median point forecast (chronos2-compatible signature)."""
    fc = _get()
    torch.manual_seed(seed)
    ctx = np.concatenate([train, val]).astype(np.float32)
    ts = _make_ts(ctx)
    out = fc.forecast(ts, prediction_length=H, num_samples=20)
    # samples: [1, 1, H, 20] -> 21 quantiles for compatibility
    samples = out.samples[0, 0].detach().cpu().numpy()    # [H, 20]
    q = np.quantile(samples, np.linspace(0.05, 0.95, 21), axis=1).T  # [H, 21] → transpose
    median = out.median[0, 0].detach().cpu().numpy()      # [H]
    predict.last_quantiles = q.T                          # [21, H] to match chronos2
    return median.astype(np.float64)


def predict_with_uncertainty(train: np.ndarray, H: int,
                              seed: int = 42, season_m: int = 1
                              ) -> tuple[np.ndarray, float, np.ndarray]:
    """Median + entropy proxy + full quantile matrix (chronos2 shape: [21, H])."""
    fc = _get()
    torch.manual_seed(seed)
    ts = _make_ts(train.astype(np.float32))
    out = fc.forecast(ts, prediction_length=H, num_samples=30)
    samples = out.samples[0, 0].detach().cpu().numpy()                 # [H, 30]
    q = np.quantile(samples, np.linspace(0.05, 0.95, 21), axis=1)      # [21, H]
    median = out.median[0, 0].detach().cpu().numpy()
    q10 = q[2]; q90 = q[18]
    scale = float(np.std(train)) + 1e-6
    entropy = float(np.mean((q90 - q10) / scale))
    return median.astype(np.float64), entropy, q


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    sig = np.sin(np.arange(600) * 0.1) + 0.1 * rng.standard_normal(600)
    train = sig[:576]; val = np.array([])
    p = predict(train, val, H=24, seed=1)
    print(f"Toto median pred: shape={p.shape}, range[{p.min():.3f},{p.max():.3f}]")
    m, e, q = predict_with_uncertainty(train, H=24)
    print(f"Uncertainty: median={m.shape} entropy={e:.3f} quantiles={q.shape}")
