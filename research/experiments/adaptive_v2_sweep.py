"""Round 7 验证 sweep · adaptive-v2 (with circuit breaker + safe predict)
vs Round 6 adaptive-v1 (no protection).

Same 54-cell grid. Compares:
    - adaptive_v2 (Round 7)
    - adaptive    (reuses Round 6 jsonl)
    - prior_aware (reuses Round 6 jsonl)
"""
from __future__ import annotations
import json, os, time
from pathlib import Path
import numpy as np

DATASETS = ["ETTh1", "ETTh2", "ECL", "Exchange", "Weather", "ILI"]
NS = [10, 20, 50]
SEEDS = [1, 42, 123]
H = 24
OUT_PATH = Path("research/results/adaptive_v2_sweep.jsonl")
LOG_PATH = Path("research/results/adaptive_v2_sweep.log")
STATE_PATH = "research/results/router_state_adaptive_v2.jsonl"


def run_one(dataset, N, seed) -> dict:
    os.environ["ADAPTTS_PLANNER"] = "adaptive"
    os.environ["ADAPTTS_STATE_PATH"] = STATE_PATH

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
        reason = trace.final_plan.reason[:140]
        layers = None
        if trace.bandit_handle and "ref_res" in trace.bandit_handle:
            rr = trace.bandit_handle["ref_res"]
            layers = rr.layers_used if rr else None
        err = None
    except Exception as e:
        mae = None; chosen = None; reason = ""; layers = None
        err = f"{type(e).__name__}: {str(e)[:120]}"

    return {
        "policy": "adaptive_v2", "dataset": dataset, "N": N, "seed": seed,
        "H": H, "chosen": chosen, "mae": mae, "wall": round(time.time() - t0, 2),
        "layers": layers, "reason": reason, "err": err,
    }


def already_done(path: Path) -> set:
    done = set()
    if not path.exists(): return done
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
            done.add((r["dataset"], r["N"], r["seed"]))
        except Exception: pass
    return done


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # clean Round 7 state for fair test
    for f in [STATE_PATH, STATE_PATH.replace(".jsonl", "_bandit.jsonl")]:
        if Path(f).exists(): Path(f).unlink()
    from research.agent.reliability_priors import reset_tracker
    reset_tracker()
    from research.agent.router_state import reset_state
    reset_state()

    done = already_done(OUT_PATH)
    total = len(DATASETS) * len(NS) * len(SEEDS)
    def logp(msg):
        with LOG_PATH.open("a") as fh: fh.write(msg + "\n")
        print(msg, flush=True)
    logp(f"=== adaptive_v2_sweep start === {len(done)}/{total}")

    fh = OUT_PATH.open("a")
    for ds in DATASETS:
        for n in NS:
            for s in SEEDS:
                if (ds, n, s) in done: continue
                row = run_one(ds, n, s)
                fh.write(json.dumps(row) + "\n"); fh.flush()
                chose = str(row['chosen'] or '?')[:14]
                mae_s = f"{row['mae']:.4f}" if row['mae'] is not None else "n/a"
                layers_s = f" L={row['layers']}" if row.get('layers') else ""
                err_s = f" ERR={row['err'][:50]}" if row['err'] else ""
                logp(f"  {ds:9} N={n:<3} s={s:<3}  chose={chose:<14} "
                     f"mae={mae_s:>10}{layers_s}  ({row['wall']:.1f}s){err_s}")
    fh.close()
    logp("=== adaptive_v2_sweep done ===")


if __name__ == "__main__":
    main()
