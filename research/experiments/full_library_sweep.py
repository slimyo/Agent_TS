"""Round 5 · Full library sweep over real TSFMs (Task 1 full coverage).

Extends bandit_full_sweep.py with broader candidate set; gracefully skips
individual model failures (DynamicCache / env compat / remote-only).

Candidates attempted (per-cell try/except, failure → fallback to chronos2):
    chronos2, tirex, toto, timesfm2, moirai, moirai2, naive_drift, arima_ets

Saves to research/results/full_library_sweep.jsonl.
"""
from __future__ import annotations
import json
import os
import time
from pathlib import Path

import numpy as np

DATASETS = ["ETTh1", "ETTh2", "ECL", "Exchange", "Weather", "ILI"]
NS = [10, 20, 50]
SEEDS = [1, 42, 123]
H = 24
OUT_PATH = Path("research/results/full_library_sweep.jsonl")
LOG_PATH = Path("research/results/full_library_sweep.log")

# 3 planners × wider candidates
POLICIES = [
    ("bandit_thompson",  "bandit", "thompson", "research/results/bandit_state_full_thompson.jsonl"),
    ("bandit_greedy",    "bandit", "greedy",   "research/results/bandit_state_full_greedy.jsonl"),
    ("prior_aware",      "prior_aware", None,  None),
]

# Full library — failures handled per-cell
FULL_CANDIDATES = ["chronos2", "tirex", "toto", "timesfm2", "moirai", "moirai2",
                   "naive_drift", "arima_ets"]
SAFE_FALLBACK = "chronos2"


def run_one(policy_name, planner, decide, state_path, dataset, N, seed) -> dict:
    os.environ["ADAPTTS_PLANNER"] = planner
    if state_path: os.environ["ADAPTTS_BANDIT_PATH"] = state_path
    if decide:     os.environ["ADAPTTS_DECIDE"] = decide
    if planner == "bandit":
        from research.agent.bandit import get_router
        get_router(state_path=state_path or "research/results/bandit_state.jsonl",
                   candidates=FULL_CANDIDATES)

    from research.agent.forecaster_reflect import forecast_with_reflection, observe_outcome
    from research.agent.curator_uq import diagnose
    from research.utils.data_loader import load_series
    from research.utils.splitter import few_shot_split

    t0 = time.time()
    series, meta = load_series(dataset)
    sp = few_shot_split(series, N=N, H=H, seed=seed)
    diag = diagnose(sp.train, season_m=meta.season_m)
    fallback_used = False
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
        # try fallback
        try:
            from research.baseline.chronos2 import predict
            pred = predict(train=sp.train, val=sp.val, H=H,
                           seed=seed, season_m=meta.season_m)
            mae = float(np.mean(np.abs(sp.test - pred)))
            chosen = SAFE_FALLBACK
            regime = None
            reason = f"FALLBACK after error: {type(e).__name__}"
            err = f"{type(e).__name__}: {str(e)[:120]}"
            fallback_used = True
        except Exception as e2:
            mae = None; chosen = None; regime = None
            reason = ""
            err = f"FALLBACK FAILED: {type(e2).__name__}: {str(e2)[:120]}"
    return {
        "policy": policy_name, "dataset": dataset, "N": N, "seed": seed,
        "H": H, "chosen": chosen, "regime": regime,
        "mae": mae, "wall": round(time.time() - t0, 2),
        "fallback": fallback_used, "reason": reason, "err": err,
    }


def already_done(path: Path) -> set:
    done = set()
    if not path.exists(): return done
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
            done.add((r["policy"], r["dataset"], r["N"], r["seed"]))
        except Exception: pass
    return done


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    done = already_done(OUT_PATH)
    def logp(msg):
        with LOG_PATH.open("a") as fh: fh.write(msg + "\n")
        print(msg, flush=True)
    total = len(POLICIES) * len(DATASETS) * len(NS) * len(SEEDS)
    logp(f"=== full_library_sweep start === resuming, {len(done)}/{total}")

    fh = OUT_PATH.open("a")
    for pol_name, planner, decide, state_path in POLICIES:
        from research.agent.bandit import reset_router
        reset_router()
        logp(f"\n--- policy={pol_name} (planner={planner} decide={decide}) ---")
        for ds in DATASETS:
            for n in NS:
                for s in SEEDS:
                    key = (pol_name, ds, n, s)
                    if key in done: continue
                    row = run_one(pol_name, planner, decide, state_path, ds, n, s)
                    fh.write(json.dumps(row) + "\n"); fh.flush()
                    err_str = f" ERR={row['err'][:50]}" if row['err'] else ""
                    fb = " (FB)" if row.get('fallback') else ""
                    chose = str(row['chosen'] or '?')[:14]
                    mae_s = f"{row['mae']:.4f}" if row['mae'] is not None else "n/a"
                    logp(f"  {pol_name:18} {ds:9} N={n:<3} s={s:<3}  "
                         f"chose={chose:<14} mae={mae_s:>10}{fb}  "
                         f"({row['wall']:.1f}s){err_str}")
    fh.close()
    logp("=== full_library_sweep done ===")


if __name__ == "__main__":
    main()
