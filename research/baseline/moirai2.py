"""Moirai 2.0 R-small (Salesforce Aug 2025) — universal masked transformer TSFM.

⚠ MUST run in `tsci-py312` env (Python 3.12 + uni2ts from-source).

Reference: https://github.com/SalesforceAIResearch/uni2ts ;
           https://huggingface.co/Salesforce/moirai-2.0-R-small
"""
from __future__ import annotations
import numpy as np
import torch

_MODEL = None
_PRED = None
_CTX = 512
_H = 24


def _get(prediction_length: int = 24, context_length: int = 512):
    global _MODEL, _PRED, _CTX, _H
    if _PRED is None or _H != prediction_length or _CTX != context_length:
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        module = Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small")
        model = Moirai2Forecast(
            module=module,
            prediction_length=prediction_length, context_length=context_length,
            target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
        )
        _MODEL, _PRED = model, model.create_predictor(batch_size=1)
        _CTX, _H = context_length, prediction_length
    return _PRED


def predict(train: np.ndarray, val: np.ndarray, H: int, seed: int = 42, **_) -> np.ndarray:
    """Median forecast via gluonts predictor."""
    import pandas as pd
    from gluonts.dataset.pandas import PandasDataset
    predictor = _get(prediction_length=H, context_length=512)
    torch.manual_seed(seed)
    ctx = np.concatenate([train, val]).astype(np.float32)
    if len(ctx) < 512:
        ctx = np.concatenate([np.zeros(512 - len(ctx), dtype=np.float32), ctx])
    else:
        ctx = ctx[-512:]
    df = pd.DataFrame({"target": ctx, "item_id": "s1"},
                       index=pd.date_range("2020-01-01", periods=512, freq="h"))
    ds = PandasDataset(df, target="target")
    fc = list(predictor.predict(ds))[0]
    return fc.median.astype(np.float64)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    sig = np.sin(np.arange(600) * 0.1) + 0.1 * rng.standard_normal(600)
    p = predict(sig[:512], np.array([]), H=24, seed=1)
    print(f"Moirai-2 R-small forecast: shape={p.shape}, sample={p[:5]}")
