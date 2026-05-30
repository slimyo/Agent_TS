"""task #71 · Toto 2.0 4m (DataDog April 2026) — multivariate observability TSFM.

⚠ MUST be invoked from `tsci-py312` env (Python ≥ 3.12, transformers < 4.46).
Toto 2.0 strict dependency pins exclude main `tsci` env. Use either:

  (a) `conda activate tsci-py312 && python -m research.baseline.toto2 ...`
  (b) cross-env subprocess bridge (future work)

API (chronos2-compatible signature for eventual router integration):
  - predict(train, val, H, seed) -> np.ndarray (median forecast)
"""
from __future__ import annotations
import numpy as np
import torch

_MODEL = None
_GTS = None
_CTX_LEN = 512


def _get(prediction_length: int = 24, context_length: int = 512):
    """Lazy-load Toto 2.0 4m. (Heavier variants 313m/1B/2.5B also from_pretrained-able.)"""
    global _MODEL, _GTS, _CTX_LEN
    if _GTS is None or prediction_length != _GTS.config.prediction_length or context_length != _CTX_LEN:
        from toto2.model import Toto2Model, Toto2GluonTSModel
        from toto2.configuration import Toto2GluonTSModelConfig
        _MODEL = Toto2Model.from_pretrained("Datadog/Toto-2.0-4m")
        _MODEL.eval()
        cfg = Toto2GluonTSModelConfig(prediction_length=prediction_length,
                                       context_length=context_length, target_dim=1)
        _GTS = Toto2GluonTSModel(model=_MODEL, config=cfg)
        _GTS.eval()
        _CTX_LEN = context_length
    return _GTS


def predict(train: np.ndarray, val: np.ndarray, H: int,
            seed: int = 42, season_m: int = 1, **_) -> np.ndarray:
    """Median forecast via gluonts predictor (chronos2-compatible signature)."""
    import pandas as pd
    from gluonts.dataset.pandas import PandasDataset
    gts = _get(prediction_length=H, context_length=512)
    torch.manual_seed(seed)
    ctx = np.concatenate([train, val]).astype(np.float32)
    # Pad / truncate to 512
    if len(ctx) < 512:
        ctx = np.concatenate([np.zeros(512 - len(ctx), dtype=np.float32), ctx])
    else:
        ctx = ctx[-512:]
    df = pd.DataFrame({"target": ctx, "item_id": "s1"},
                       index=pd.date_range("2020-01-01", periods=512, freq="h"))
    predictor = gts.create_predictor(batch_size=1)
    ds = PandasDataset(df, target="target")
    fc = list(predictor.predict(ds))[0]
    return fc.median.astype(np.float64)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    sig = np.sin(np.arange(600) * 0.1) + 0.1 * rng.standard_normal(600)
    p = predict(sig[:512], np.array([]), H=24, seed=1)
    print(f"Toto 2.0 4m forecast: shape={p.shape}, sample={p[:5]}")
