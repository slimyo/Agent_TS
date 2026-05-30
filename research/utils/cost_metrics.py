"""Cost-aware metric (feedback Round 4 §六 ℒ = ForecastError + λ·Cost).

Aggregates routing cost from 4 sources:
  1. **latency**:  observed median wall-time per cell (from sweep jsonl)
  2. **params**:   model parameter count (from MODEL_COSTS table)
  3. **env penalty**: cross-env / cross-host overhead in routing decision
  4. **VRAM**:     min GPU memory required (filters models from low-VRAM hosts)

Score:  cost = α·log(latency) + β·log(params) + γ·env_penalty + δ·log(VRAM)
        (log scale so 10× changes are 1 unit; coefficients tunable)

Pareto front: (mean_MAE, cost) plane. Dominated models are bad regardless of α.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

RESULTS_DIR = Path("research/results")

# Static cost table (params in M; vram in GB; env penalty 0=main / 1=alt-local / 2=remote)
MODEL_COSTS: dict[str, dict] = {
    # trivial / point
    "naive_drift":   {"params_m": 0,    "vram_gb": 0,   "env_penalty": 0, "ext_dep": False},
    "naive_seasonal":{"params_m": 0,    "vram_gb": 0,   "env_penalty": 0, "ext_dep": False},
    "arima_ets":     {"params_m": 0,    "vram_gb": 0,   "env_penalty": 0, "ext_dep": False},
    "llmtime":       {"params_m": 7000, "vram_gb": 0,   "env_penalty": 0, "ext_dep": True},  # external API
    # Chronos family
    "chronos":       {"params_m": 60,   "vram_gb": 1,   "env_penalty": 0, "ext_dep": False},
    "chronos_bolt":  {"params_m": 200,  "vram_gb": 2,   "env_penalty": 0, "ext_dep": False},
    "chronos2":      {"params_m": 120,  "vram_gb": 2,   "env_penalty": 0, "ext_dep": False},
    # local main env TSFMs
    "timesfm2":      {"params_m": 500,  "vram_gb": 4,   "env_penalty": 0, "ext_dep": False},
    "moirai":        {"params_m": 311,  "vram_gb": 3,   "env_penalty": 0, "ext_dep": False},
    "tirex":         {"params_m": 128,  "vram_gb": 2,   "env_penalty": 0, "ext_dep": False},
    "toto":          {"params_m": 151,  "vram_gb": 2,   "env_penalty": 0, "ext_dep": False},
    # alt-local env (tsci-py312)
    "moirai2":       {"params_m": 11,   "vram_gb": 1,   "env_penalty": 1, "ext_dep": False},
    "toto2":         {"params_m": 4,    "vram_gb": 1,   "env_penalty": 1, "ext_dep": False},
    # remote
    "time_moe":      {"params_m": 50,   "vram_gb": 2,   "env_penalty": 2, "ext_dep": True},
    "sundial":       {"params_m": 128,  "vram_gb": 2,   "env_penalty": 2, "ext_dep": True},
    "timer":         {"params_m": 8300, "vram_gb": 16,  "env_penalty": 2, "ext_dep": True},
}

LOSS_SOURCES = [
    ("tirex_vs_c2.jsonl",    "tirex"),
    ("toto_vs_c2.jsonl",     "toto"),
    ("time_moe_vs_c2.jsonl", "time_moe"),
    ("sundial_vs_c2.jsonl",  "sundial"),
    ("timer_vs_c2.jsonl",    "timer"),
]


def observed_latency(model: str) -> float | None:
    """Median wall-time per cell (seconds) from sweep jsonl. None if not measured."""
    for fname, m in LOSS_SOURCES:
        if m != model: continue
        p = RESULTS_DIR / fname
        if not p.exists(): return None
        ws = []
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
                if "wall" in r: ws.append(r["wall"])
            except Exception: pass
        return float(np.median(ws)) if ws else None
    return None


def cost_score(model: str,
               alpha: float = 1.0,    # latency weight
               beta:  float = 0.5,    # params weight
               gamma: float = 2.0,    # env penalty weight
               delta: float = 0.3,    # vram weight
               eps_log: float = 1e-3,
               default_latency: float = 0.1,
               ) -> float | None:
    """log-scale cost composite."""
    if model not in MODEL_COSTS: return None
    info = MODEL_COSTS[model]
    lat = observed_latency(model) or default_latency
    return (alpha * np.log10(lat + eps_log)
            + beta  * np.log10(info["params_m"] + 1)
            + gamma * info["env_penalty"]
            + delta * np.log10(info["vram_gb"] + 1))


def cost_mae_pareto(models: list[str] | None = None,
                    alpha: float = 1.0, beta: float = 0.5,
                    gamma: float = 2.0, delta: float = 0.3,
                    ) -> dict[str, dict[str, float]]:
    """Per-model (mean_mae, cost) for Pareto analysis."""
    if models is None:
        models = list(MODEL_COSTS.keys())
    out = {}
    # Load mae from sweep
    from research.utils.risk_metrics import load_per_cell_losses
    per_model = load_per_cell_losses()
    for m in models:
        c = cost_score(m, alpha, beta, gamma, delta)
        if c is None: continue
        mae = None
        if m in per_model and per_model[m]:
            mae = float(np.mean(list(per_model[m].values())))
        out[m] = {"mean_mae": mae, "cost": c,
                  "latency_s": observed_latency(m),
                  **MODEL_COSTS[m]}
    return out


def pareto_front_mae_cost(table: dict[str, dict]) -> list[str]:
    """Models on the (mean_mae, cost) Pareto front (both minimized)."""
    items = [(m, t["mean_mae"], t["cost"]) for m, t in table.items()
             if t["mean_mae"] is not None]
    pareto = []
    for i, (m, mae, c) in enumerate(items):
        dominated = False
        for j, (m2, mae2, c2) in enumerate(items):
            if i == j: continue
            if mae2 <= mae and c2 <= c and (mae2 < mae or c2 < c):
                dominated = True; break
        if not dominated: pareto.append(m)
    return pareto


if __name__ == "__main__":
    print("=== Cost-MAE table (5 measured + C2) ===")
    measured = ["chronos2", "tirex", "toto", "time_moe", "sundial"]
    tbl = cost_mae_pareto(measured)
    fmt = "{:10}  params={:>5}M  vram={:>2}GB  env={}  lat={:>6.2f}s  cost={:>5.2f}  mae={:>10.2f}"
    for m in sorted(tbl, key=lambda x: tbl[x]["cost"]):
        t = tbl[m]
        lat = t["latency_s"]
        lat_str = f"{lat:>6.2f}" if lat is not None else "  n/a "
        print(fmt.format(m, t["params_m"], t["vram_gb"], t["env_penalty"],
                         lat if lat else 0.0, t["cost"], t["mean_mae"]))

    print(f"\nPareto front (mean_mae, cost): {pareto_front_mae_cost(tbl)}")

    print("\n=== Cost-aware ranking under different α (latency weight) ===")
    print("  α=high → favor cheap models; α=low → favor accuracy")
    for alpha in [0.0, 0.5, 1.0, 2.0, 5.0]:
        # composite: total = mae_normalized + α · cost_normalized
        tbl_a = cost_mae_pareto(measured, alpha=alpha)
        mae_vals = np.array([t["mean_mae"] for t in tbl_a.values() if t["mean_mae"] is not None])
        cost_vals = np.array([t["cost"] for t in tbl_a.values()])
        # min-max normalize
        if mae_vals.max() > mae_vals.min():
            mae_n = {m: (t["mean_mae"] - mae_vals.min()) / (mae_vals.max() - mae_vals.min())
                     for m, t in tbl_a.items() if t["mean_mae"] is not None}
        else:
            mae_n = {m: 0.0 for m in tbl_a}
        if cost_vals.max() > cost_vals.min():
            cost_n = {m: (t["cost"] - cost_vals.min()) / (cost_vals.max() - cost_vals.min())
                      for m, t in tbl_a.items()}
        else:
            cost_n = {m: 0.0 for m in tbl_a}
        composite = {m: mae_n[m] + alpha * cost_n[m]
                     for m in mae_n if m in cost_n}
        winner = min(composite, key=composite.get)
        print(f"  α={alpha:4.1f}  winner={winner:10}  composite={composite[winner]:.3f}")

    print("\n=== Per-host availability (which models can a host run?) ===")
    for vram in [4, 6, 8, 16, 40]:
        ok = [m for m, c in MODEL_COSTS.items() if c["vram_gb"] <= vram]
        print(f"  VRAM={vram:>2}GB  → can run {len(ok)}/{len(MODEL_COSTS)}: {ok[-4:]} ...")
