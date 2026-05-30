"""Round 6 Step 3 · adaptive vs bayesian vs prior_aware 162-cell sweep.

Same cell grid as bandit_full_sweep but 3 planners:
    - adaptive   (Round 6: reflective + bandit + telemetry)
    - bayesian   (Round 5: BayesianRouter argmax)
    - prior_aware (Round 4-A: hand prior stack)

Resume-safe. Each policy has its own state file (no contamination).
"""
from __future__ import annotations
import json, os, time
from pathlib import Path
import numpy as np

DATASETS = ["ETTh1", "ETTh2", "ECL", "Exchange", "Weather", "ILI"]
NS = [10, 20, 50]
SEEDS = [1, 42, 123]
H = 24
OUT_PATH = Path("research/results/adaptive_compare_sweep.jsonl")
LOG_PATH = Path("research/results/adaptive_compare_sweep.log")

POLICIES = [
    ("adaptive",   "adaptive",   "research/results/router_state_adaptive.jsonl"),
    ("bayesian",   "bayesian",   None),
    ("prior_aware","prior_aware",None),
]

SAFE_CANDIDATES = ["chronos2", "chronos", "arima_ets", "naive_drift", "naive_seasonal"]


def run_one(policy_name, planner, state_path, dataset, N, seed) -> dict:
    # set env
    os.environ["ADAPTTS_PLANNER"] = planner
    if state_path: os.environ["ADAPTTS_STATE_PATH"] = state_path

    # for adaptive: ensure state is loaded into singleton per policy
    if planner == "adaptive":
        from research.agent.router_state import reset_state, get_state
        reset_state()
        get_state(state_path)
        # ensure bandit is enabled (default)
        os.environ["ADAPTTS_BANDIT"] = "1"

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
        # observe outcome (closes the loop for adaptive/bandit)
        if trace.bandit_handle is not None:
            observe_outcome(trace, y_true=sp.test, y_pred=pred, persist=True)
        chosen = trace.final_plan.strategies[0] if trace.final_plan.strategies else "?"
        reason = trace.final_plan.reason[:140]
        err = None
        # adaptive layers info if present
        layers = None
        if trace.bandit_handle and "ref_res" in trace.bandit_handle:
            rr = trace.bandit_handle["ref_res"]
            layers = rr.layers_used if rr else None
    except Exception as e:
        mae = None; chosen = None; reason = ""; layers = None
        err = f"{type(e).__name__}: {str(e)[:120]}"
    return {
        "policy": policy_name, "dataset": dataset, "N": N, "seed": seed,
        "H": H, "chosen": chosen, "mae": mae, "wall": round(time.time() - t0, 2),
        "layers": layers, "reason": reason, "err": err,
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
    logp(f"=== adaptive_compare_sweep start === {len(done)}/{total}")

    fh = OUT_PATH.open("a")
    for pol_name, planner, state_path in POLICIES:
        logp(f"\n--- {pol_name} (planner={planner}) ---")
        for ds in DATASETS:
            for n in NS:
                for s in SEEDS:
                    key = (pol_name, ds, n, s)
                    if key in done: continue
                    row = run_one(pol_name, planner, state_path, ds, n, s)
                    fh.write(json.dumps(row) + "\n"); fh.flush()
                    chose = str(row['chosen'] or '?')[:14]
                    mae_s = f"{row['mae']:.4f}" if row['mae'] is not None else "n/a"
                    layers_s = f" L={row['layers']}" if row.get('layers') else ""
                    err_s = f" ERR={row['err'][:40]}" if row['err'] else ""
                    logp(f"  {pol_name:12} {ds:9} N={n:<3} s={s:<3}  "
                         f"chose={chose:<14} mae={mae_s:>10}{layers_s}  "
                         f"({row['wall']:.1f}s){err_s}")
    fh.close()
    logp("=== adaptive_compare_sweep done ===")


if __name__ == "__main__":
    main()
