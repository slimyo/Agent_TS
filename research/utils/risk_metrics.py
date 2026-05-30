"""Risk-sensitive evaluation (feedback Round 4 §四 risk_k = E[ℓ] + λ Var[ℓ]).

Augments per-model mean-loss with a risk term: a model with the same mean MAE
but higher loss-variance is **strictly worse** under risk-sensitive routing,
since the router cannot guarantee per-cell performance.

Two views:
  1. **Aggregate risk**: mean and std of loss over cells; risk = mean + λ·std
  2. **Per-dataset risk**: same per dataset, surfaces models with high
     domain-conditional variance (e.g. Sundial: high mean elsewhere, low on ETTh2)

Pareto front: (mean_loss, std_loss) plane; dominated models are NEVER chosen
under any λ ≥ 0. This complements 5-way oracle (which is over-optimistic by
assuming perfect routing knowledge).
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

RESULTS_DIR = Path("research/results")

LOSS_SOURCES = [
    ("tirex_vs_c2.jsonl",    "tirex",    "mae_tirex"),
    ("toto_vs_c2.jsonl",     "toto",     "mae_toto"),
    ("time_moe_vs_c2.jsonl", "time_moe", "mae_time_moe"),
    ("sundial_vs_c2.jsonl",  "sundial",  "mae_sundial"),
    ("timer_vs_c2.jsonl",    "timer",    "mae_timer"),
]


def load_per_cell_losses(include_c2: bool = True) -> dict[str, dict[tuple, float]]:
    """Returns {model: {(dataset, N, seed): mae}}."""
    out: dict[str, dict[tuple, float]] = defaultdict(dict)
    for fname, model, field in LOSS_SOURCES:
        p = RESULTS_DIR / fname
        if not p.exists(): continue
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
                key = (r["dataset"], r["N"], r["seed"])
                out[model][key] = r[field]
                if include_c2 and "mae_c2" in r:
                    out["chronos2"][key] = r["mae_c2"]
            except Exception: pass
    return out


def risk_score(losses: list[float], lam: float = 1.0) -> float:
    """risk = E[ℓ] + λ·std(ℓ)."""
    if not losses: return float("inf")
    arr = np.array(losses, dtype=np.float64)
    return float(arr.mean() + lam * arr.std())


def model_risk_table(lam: float = 1.0, dataset: str | None = None
                     ) -> dict[str, dict[str, float]]:
    """Returns {model: {n, mean, std, risk, min, max, median}} over (filtered) cells."""
    per_model = load_per_cell_losses()
    table = {}
    for model, cells in per_model.items():
        vals = [v for (ds, _, _), v in cells.items()
                if dataset is None or ds == dataset]
        if not vals: continue
        arr = np.array(vals, dtype=np.float64)
        table[model] = {
            "n": len(arr),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "risk": risk_score(vals, lam),
            "median": float(np.median(arr)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    return table


def pareto_front_mean_std(table: dict[str, dict[str, float]]
                           ) -> list[str]:
    """Returns model names on the (mean, std) Pareto front (both minimized)."""
    items = [(m, t["mean"], t["std"]) for m, t in table.items()]
    pareto = []
    for i, (m, mu, sig) in enumerate(items):
        dominated = False
        for j, (m2, mu2, sig2) in enumerate(items):
            if i == j: continue
            # m2 dominates m iff mu2 ≤ mu AND sig2 ≤ sig AND (strict in at least one)
            if mu2 <= mu and sig2 <= sig and (mu2 < mu or sig2 < sig):
                dominated = True; break
        if not dominated: pareto.append(m)
    return pareto


def lambda_sweep_winner(lambdas: list[float] | None = None,
                        dataset: str | None = None) -> dict[float, str]:
    """For each λ, return model with minimum risk."""
    if lambdas is None:
        lambdas = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0]
    out = {}
    for lam in lambdas:
        tbl = model_risk_table(lam=lam, dataset=dataset)
        if not tbl: out[lam] = None; continue
        out[lam] = min(tbl.items(), key=lambda kv: kv[1]["risk"])[0]
    return out


if __name__ == "__main__":
    print("=== Aggregate risk table (λ=1.0, all cells) ===")
    tbl = model_risk_table(lam=1.0)
    fmt = "{:10}  n={:>3}  mean={:>10.2f}  std={:>10.2f}  risk={:>10.2f}  median={:>10.2f}"
    for m in sorted(tbl, key=lambda x: tbl[x]["risk"]):
        t = tbl[m]
        print(fmt.format(m, t["n"], t["mean"], t["std"], t["risk"], t["median"]))

    print(f"\nPareto (mean, std) front: {pareto_front_mean_std(tbl)}")

    print("\n=== Risk-min model per λ ===")
    sweep = lambda_sweep_winner()
    for lam, m in sweep.items():
        print(f"  λ={lam:4.2f}  →  {m}")

    print("\n=== Per-dataset risk-min winner (λ=1.0) ===")
    for ds in ["Exchange", "ECL", "ETTh1", "ETTh2", "Weather", "ILI"]:
        tbl_ds = model_risk_table(lam=1.0, dataset=ds)
        if not tbl_ds: continue
        winner = min(tbl_ds, key=lambda m: tbl_ds[m]["risk"])
        t = tbl_ds[winner]
        print(f"  {ds:10}  →  {winner:10}  mean={t['mean']:>10.4f}  std={t['std']:>10.4f}")
