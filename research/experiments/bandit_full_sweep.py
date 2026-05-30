"""Round 5 Phase 2 · Multi-dataset full sweep for ContextualBanditRouter.

Compares 3 planners on the same cell grid:
    - bandit (thompson)  — Round 5 Phase 2
    - bandit (greedy)    — exploitation only
    - prior_aware        — Round 4-A baseline (no online update)

Per cell: train+val split, run forecast, observe outcome, record MAE.
Saves to research/results/bandit_full_sweep.jsonl with one row per (planner, dataset, N, seed).

Each bandit policy maintains its OWN persistent state file so cross-policy
contamination is impossible.

Resume-safe: skips (planner, dataset, N, seed) already in jsonl.
"""
from __future__ import annotations
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np

DATASETS = ["ETTh1", "ETTh2", "ECL", "Exchange", "Weather", "ILI"]
NS = [10, 20, 50]
SEEDS = [1, 42, 123]
H = 24                     # short horizon for speed
OUT_PATH = Path("research/results/bandit_full_sweep.jsonl")
LOG_PATH = Path("research/results/bandit_full_sweep.log")

POLICIES = [
    ("bandit_thompson",  "bandit", "thompson", "research/results/bandit_state_thompson.jsonl"),
    ("bandit_greedy",    "bandit", "greedy",   "research/results/bandit_state_greedy.jsonl"),
    ("prior_aware",      "prior_aware", None,  None),
]

# Stable local candidates only — avoid TSFMs with env / cache compat issues
SAFE_CANDIDATES = ["chronos2", "chronos", "arima_ets", "naive_drift", "naive_seasonal"]


def run_one(policy_name: str, planner: str, decide: str | None,
            state_path: str | None,
            dataset: str, N: int, seed: int) -> dict:
    """Run forecast + observe; return row dict."""
    # set env per policy
    os.environ["ADAPTTS_PLANNER"] = planner
    if state_path: os.environ["ADAPTTS_BANDIT_PATH"] = state_path
    if decide:     os.environ["ADAPTTS_DECIDE"] = decide
    # IMPORTANT: reset bandit singleton between policies
    if planner == "bandit":
        from research.agent.bandit import get_router
        # Pre-build router with SAFE_CANDIDATES (singleton respects first build)
        get_router(state_path=state_path or "research/results/bandit_state.jsonl",
                   candidates=SAFE_CANDIDATES)

    from research.agent.forecaster_reflect import forecast_with_reflection, observe_outcome
    from research.agent.curator_uq import diagnose
    from research.utils.data_loader import load_series
    from research.utils.splitter import few_shot_split

    t0 = time.time()
    series, meta = load_series(dataset)
    sp = few_shot_split(series, N=N, H=H, seed=seed)
    diag = diagnose(sp.train, season_m=meta.season_m)
    try:
        pred, trace = forecast_with_reflection(
            train=sp.train, val=sp.val, H=H, diag=diag,
            season_m=meta.season_m, use_walk_forward=False, max_reflect=0,
            dataset=dataset,
        )
        mae = float(np.mean(np.abs(sp.test - pred)))
        if trace.bandit_handle is not None:
            observe_outcome(trace, y_true=sp.test, y_pred=pred, persist=True)
        chosen = trace.final_plan.strategies[0] if trace.final_plan.strategies else "?"
        regime = trace.bandit_handle["regime"] if trace.bandit_handle else None
        reason = trace.final_plan.reason[:120]
        err = None
    except Exception as e:
        mae = None; chosen = None; regime = None
        reason = ""; err = f"{type(e).__name__}: {e}"
    wall = time.time() - t0
    return {
        "policy": policy_name, "dataset": dataset, "N": N, "seed": seed,
        "H": H, "chosen": chosen, "regime": regime,
        "mae": mae, "wall": round(wall, 2),
        "reason": reason, "err": err,
    }


def already_done(out_path: Path) -> set:
    done = set()
    if not out_path.exists(): return done
    for line in out_path.read_text().splitlines():
        try:
            r = json.loads(line)
            done.add((r["policy"], r["dataset"], r["N"], r["seed"]))
        except Exception: pass
    return done


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    done = already_done(OUT_PATH)

    def logp(msg):
        with LOG_PATH.open("a") as fh: fh.write(msg + "\n")
        print(msg, flush=True)

    total = len(POLICIES) * len(DATASETS) * len(NS) * len(SEEDS)
    logp(f"=== bandit_full_sweep start === resuming, {len(done)} cells done / {total}")

    fh = OUT_PATH.open("a")
    # iterate policy-first so each bandit policy accumulates state coherently
    for pol_name, planner, decide, state_path in POLICIES:
        # clear singleton when switching policy
        from research.agent.bandit import reset_router
        reset_router()
        logp(f"\n--- policy={pol_name} (planner={planner} decide={decide}) ---")
        for ds in DATASETS:
            for n in NS:
                for s in SEEDS:
                    key = (pol_name, ds, n, s)
                    if key in done:
                        continue
                    row = run_one(pol_name, planner, decide, state_path, ds, n, s)
                    fh.write(json.dumps(row) + "\n"); fh.flush()
                    err_str = f" ERR={row['err']}" if row['err'] else ""
                    chose = str(row['chosen'] or '?')[:14]
                    mae_s = f"{row['mae']:.4f}" if row['mae'] is not None else "n/a"
                    logp(f"  {pol_name:18} {ds:9} N={n:<3} s={s:<3}  "
                         f"chose={chose:<14} mae={mae_s:>10}  "
                         f"({row['wall']:.1f}s){err_str[:60]}")
    fh.close()
    logp("=== bandit_full_sweep done ===")


if __name__ == "__main__":
    main()
