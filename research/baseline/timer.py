"""Timer-S1 (ByteDance, 2025) — 8.3B MoE TSFM, zero-shot quantile forecasting.

⚠ Large model (8.3B params, ~32 GB fp32 / 16 GB fp16). GPU needs >=40 GB VRAM;
on smaller GPUs / CPU we run with use_cache=False, batch_size=1, lookback<=2880.

Reference: https://huggingface.co/bytedance-research/Timer-S1
"""
from __future__ import annotations
import numpy as np
import torch

_MODEL = None
_DEVICE = None


def _get(repo: str = "bytedance-research/Timer-S1", prefer_cpu: bool = True):
    global _MODEL, _DEVICE
    if _MODEL is None:
        from transformers import AutoModelForCausalLM
        if prefer_cpu or not torch.cuda.is_available() or \
           torch.cuda.get_device_properties(0).total_memory < 40 * 1e9:
            _DEVICE = "cpu"
            _MODEL = AutoModelForCausalLM.from_pretrained(
                repo, trust_remote_code=True, torch_dtype=torch.float32
            )
        else:
            _DEVICE = "cuda"
            _MODEL = AutoModelForCausalLM.from_pretrained(
                repo, trust_remote_code=True, device_map="auto",
                torch_dtype=torch.float16,
            )
        _MODEL.config.use_cache = False
        _MODEL.eval()
    return _MODEL


# 9 quantile levels Timer-S1 outputs natively
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def predict(train: np.ndarray, val: np.ndarray, H: int, seed: int = 42,
            lookback_cap: int = 2880, **_) -> np.ndarray:
    """Point (median) forecast."""
    model = _get()
    torch.manual_seed(seed)
    ctx = np.concatenate([train, val]).astype(np.float32)
    if ctx.shape[0] > lookback_cap:
        ctx = ctx[-lookback_cap:]
    seqs = torch.tensor(ctx).reshape(1, -1).to(model.device)
    with torch.no_grad():
        out = model.generate(seqs, max_new_tokens=H, revin=True)
    # out shape: [B, 9 quantiles, H]
    return out[0, 4].detach().cpu().float().numpy().astype(np.float64)


def predict_with_uncertainty(train: np.ndarray, val: np.ndarray, H: int,
                              seed: int = 42, lookback_cap: int = 2880, **_):
    """Returns dict with median + 9 quantiles."""
    model = _get()
    torch.manual_seed(seed)
    ctx = np.concatenate([train, val]).astype(np.float32)
    if ctx.shape[0] > lookback_cap:
        ctx = ctx[-lookback_cap:]
    seqs = torch.tensor(ctx).reshape(1, -1).to(model.device)
    with torch.no_grad():
        out = model.generate(seqs, max_new_tokens=H, revin=True)
    q = out[0].detach().cpu().float().numpy().astype(np.float64)  # [9, H]
    return {"median": q[4], "quantiles": q, "quantile_levels": QUANTILES}


if __name__ == "__main__":
    import time
    rng = np.random.default_rng(0)
    sig = (np.sin(np.arange(400) * 0.1) + 0.1 * rng.standard_normal(400)).astype(np.float32)
    print("loading Timer-S1 (this may take a while on CPU)...")
    t0 = time.time()
    p = predict(sig, np.array([], dtype=np.float32), H=24, seed=1)
    print(f"forecast shape={p.shape}, last 5: {p[-5:]}, total {time.time()-t0:.1f}s")
