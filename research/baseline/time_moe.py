"""Time-MoE (ICLR 2025) — Mixture-of-Experts TSFM.

⚠ MUST run in `tsci-tx440` env (Python 3.10 + transformers 4.40.1).

Reference: https://huggingface.co/Maple728/TimeMoE-50M (also 200M variant available).
generate() returns [B, ctx+H] — last H values are the forecast.
"""
from __future__ import annotations
import numpy as np
import torch

_MODEL = None


def _get(repo: str = "Maple728/TimeMoE-50M"):
    global _MODEL
    if _MODEL is None:
        from transformers import AutoModelForCausalLM
        _MODEL = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)
        _MODEL.eval()
    return _MODEL


def predict(train: np.ndarray, val: np.ndarray, H: int, seed: int = 42, **_) -> np.ndarray:
    """Point forecast with manual z-score normalize (per Maple728 quickstart)."""
    model = _get()
    torch.manual_seed(seed)
    ctx = np.concatenate([train, val]).astype(np.float32)
    ctx_t = torch.tensor(ctx).reshape(1, -1)
    L = ctx.shape[0]
    # MANDATORY: z-score normalize (model trained on normalized inputs)
    mean = ctx_t.mean(dim=-1, keepdim=True)
    std = ctx_t.std(dim=-1, keepdim=True).clamp(min=1e-6)
    ctx_norm = (ctx_t - mean) / std
    with torch.no_grad():
        out_norm = model.generate(ctx_norm, max_new_tokens=H)
    out = out_norm[0, L:] * std.squeeze() + mean.squeeze()
    return out.detach().cpu().numpy().astype(np.float64)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    sig = np.sin(np.arange(200) * 0.1) + 0.1 * rng.standard_normal(200)
    p = predict(sig, np.array([]), H=24, seed=1)
    print(f"Time-MoE forecast: shape={p.shape}, last 5: {p[-5:]}")
