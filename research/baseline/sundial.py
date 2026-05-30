"""Sundial (THUML ICML 2025 Oral) — generative TSFM with TimeFlow loss.

⚠ MUST run in `tsci-tx440` env (Python 3.10 + transformers 4.40.1).

Reference: https://github.com/thuml/Sundial, https://huggingface.co/thuml/sundial-base-128m

Quickstart requires lookback=1024 and 24+ horizon. Returns [B, num_samples, H].
"""
from __future__ import annotations
import numpy as np
import torch

_MODEL = None


def _get():
    global _MODEL
    if _MODEL is None:
        from transformers import AutoModelForCausalLM
        _MODEL = AutoModelForCausalLM.from_pretrained(
            "thuml/sundial-base-128m", trust_remote_code=True)
        _MODEL.eval()
    return _MODEL


def predict(train: np.ndarray, val: np.ndarray, H: int, seed: int = 42,
            num_samples: int = 20, **_) -> np.ndarray:
    """Median forecast across num_samples. chronos2-compatible signature."""
    model = _get()
    torch.manual_seed(seed)
    ctx = np.concatenate([train, val]).astype(np.float32)
    # Pad / truncate to 1024 (Sundial native)
    if len(ctx) < 1024:
        ctx = np.concatenate([np.zeros(1024 - len(ctx), dtype=np.float32), ctx])
    else:
        ctx = ctx[-1024:]
    ctx_t = torch.tensor(ctx).reshape(1, -1)
    # z-score normalize (model's internal RevIN insufficient for scale-extreme series)
    mean = ctx_t.mean(dim=-1, keepdim=True)
    std = ctx_t.std(dim=-1, keepdim=True).clamp(min=1e-6)
    ctx_norm = (ctx_t - mean) / std
    with torch.no_grad():
        out = model.generate(ctx_norm, max_new_tokens=H, num_samples=num_samples)
    # out shape: [B, num_samples, H] — denormalize
    samples = (out[0] * std + mean).detach().cpu().numpy()   # [num_samples, H]
    # quantiles
    q = np.quantile(samples, np.linspace(0.05, 0.95, 21), axis=0)   # [21, H]
    predict.last_quantiles = q
    return samples.mean(axis=0).astype(np.float64)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    sig = np.sin(np.arange(1100) * 0.1) + 0.1 * rng.standard_normal(1100)
    p = predict(sig[:1024], np.array([]), H=24, seed=1)
    print(f"Sundial forecast: shape={p.shape}, sample={p[:5]}")
