"""feedback Item 2 · CRPS 倒数静态性能先验 π_k + BMA 后验。

π_k = (1/loss_k) / Σ_j (1/loss_j)
  - loss_k = mean CRPS over validation cells (for probabilistic models)
  - loss_k = mean MAE over validation cells (for point predictors;
    equivalent to degenerate-quantile CRPS, matches feedback footnote)

BMA posterior (feedback §二.4):
  p(M_k|data) ∝ exp(-loss_k / σ²) · π_k

Aggregates losses from existing jsonl artifacts (no fresh sweep needed):
  - research/results/gated_residual_cells.jsonl  → c2_pred MAE on 34 cells
  - research/results/{tirex,toto,timer,time_moe,sundial}_vs_c2.jsonl
  - research/results/a3_prob_metrics*.jsonl       → real C2 CRPS

Output:
  - get_prior(): dict[str, π_k] (static, overall)
  - get_prior(dataset=...): conditional on dataset
  - bma_posterior(model_losses, sigma_sq): renormalize per-cell
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

RESULTS_DIR = Path("research/results")

# Maps loss-jsonl file → (model_key, loss_field)
LOSS_SOURCES = [
    ("tirex_vs_c2.jsonl",    "tirex",    "mae_tirex"),
    ("toto_vs_c2.jsonl",     "toto",     "mae_toto"),
    ("timer_vs_c2.jsonl",    "timer",    "mae_timer"),
    ("time_moe_vs_c2.jsonl", "time_moe", "mae_time_moe"),
    ("sundial_vs_c2.jsonl",  "sundial",  "mae_sundial"),
]


def _load_per_cell_losses() -> dict[str, dict[tuple, float]]:
    """Returns {model: {(dataset, N, seed): loss}}."""
    losses: dict[str, dict[tuple, float]] = defaultdict(dict)

    # C2 baseline from gated_residual_cells
    grc = RESULTS_DIR / "gated_residual_cells.jsonl"
    if grc.exists():
        for line in grc.read_text().splitlines():
            r = json.loads(line)
            key = (r["dataset"], r["N"], r["seed"])
            y_true = np.array(r["y_true"]); c2 = np.array(r["c2_pred"])
            losses["chronos2"][key] = float(np.mean(np.abs(y_true - c2)))

    # Each *_vs_c2 file
    for fname, key, field in LOSS_SOURCES:
        p = RESULTS_DIR / fname
        if not p.exists(): continue
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
                losses[key][(r["dataset"], r["N"], r["seed"])] = r[field]
                # also reinforce C2 from same cell (consistency check)
                losses["chronos2"][(r["dataset"], r["N"], r["seed"])] = r["mae_c2"]
            except Exception:
                pass

    return losses


def get_prior(dataset: str | None = None, eps: float = 1e-6) -> dict[str, float]:
    """π_k = (1/loss_k) / Σ_j (1/loss_j). Optionally conditioned on dataset.

    Returns {model_name: π_k}, summing to 1.
    If `dataset` set, restricts averaging to cells of that dataset.
    """
    losses = _load_per_cell_losses()
    mean_loss: dict[str, float] = {}
    for model, cell_losses in losses.items():
        vals = [v for (ds, _, _), v in cell_losses.items()
                if dataset is None or ds == dataset]
        if vals:
            mean_loss[model] = float(np.mean(vals))
    if not mean_loss: return {}
    inv = {m: 1.0 / (l + eps) for m, l in mean_loss.items()}
    Z = sum(inv.values())
    return {m: v / Z for m, v in inv.items()}


def bma_posterior(model_losses: dict[str, float],
                  sigma_sq: float = 1.0,
                  prior: dict[str, float] | None = None) -> dict[str, float]:
    """p(M_k|data) ∝ exp(-loss_k / σ²) · π_k.

    Args:
        model_losses: {model: per-cell loss (CRPS or MAE)}
        sigma_sq: temperature; smaller σ² → sharper posterior
        prior: optional π_k; defaults to uniform if None

    Returns: {model: posterior prob}, summing to 1.
    """
    if not model_losses: return {}
    keys = list(model_losses.keys())
    if prior is None:
        prior = {k: 1.0 / len(keys) for k in keys}
    log_lik = {k: -model_losses[k] / sigma_sq for k in keys}
    # add log prior; normalize via log-sum-exp for stability
    log_post = {k: log_lik[k] + np.log(prior.get(k, 1e-9) + 1e-12) for k in keys}
    m = max(log_post.values())
    exp_ = {k: np.exp(v - m) for k, v in log_post.items()}
    Z = sum(exp_.values())
    return {k: float(v / Z) for k, v in exp_.items()}


if __name__ == "__main__":
    print("=== Overall static prior π_k (1/MAE) ===")
    prior = get_prior()
    if not prior:
        print("  (no loss files found yet)")
    for m, p in sorted(prior.items(), key=lambda x: -x[1]):
        print(f"  {m:12} π = {p:.4f}")

    print("\n=== Per-dataset conditional priors ===")
    for ds in ["Exchange", "ECL", "ETTh1", "ETTh2", "Weather", "ILI"]:
        cond = get_prior(dataset=ds)
        if cond:
            print(f"  {ds:10}:", " ".join(f"{m}={p:.3f}" for m, p in cond.items()))

    print("\n=== BMA demo (sigma²=0.5) ===")
    # Fake per-cell losses to demonstrate
    demo_losses = {"chronos2": 1.2, "tirex": 0.8, "toto": 2.5}
    post = bma_posterior(demo_losses, sigma_sq=0.5, prior=prior or None)
    for m, p in sorted(post.items(), key=lambda x: -x[1]):
        print(f"  {m:12} posterior = {p:.4f}  (loss={demo_losses[m]:.2f})")
