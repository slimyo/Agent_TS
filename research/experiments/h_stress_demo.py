"""Round 6 R6-H · Multi-dataset stress test with fault injection.

Runs the R6-G end-to-end pipeline on 3 real datasets × 4 fault scenarios:

    Scenarios:
        clean              · canonical series, baseline behavior
        trend_break        · level shift at series mid-point (drift-like)
        variance_explode   · σ× 3.5 in second half (regime shift)
        outlier_burst      · 4 outliers at ±4σ (transient shocks)

Reports per-scenario:
    intervention counts (MONITOR/INSPECT/THROTTLE/SHUTDOWN/ESCALATE)
    mean h=1 MAE
    drift_history events fired
    safe_predict fallbacks triggered
    elapsed seconds

Sanity expectations:
    - trend_break / variance_explode → higher SHUTDOWN/INSPECT count vs clean
    - safe_predict fallbacks should be 0 on TSFM-stable datasets
    - drift_history may stay 0 if walk-forward is too short for window thresholds
      (B3 needs ≥120 obs by default; this test only runs ~8 obs per cell)

Run:
    mamba run -n tsci python -m research.experiments.h_stress_demo
"""
from __future__ import annotations
import argparse
import time
from collections import Counter
import numpy as np

from research.utils.data_loader import load_series
from research.utils.inject_fault import (
    inject_trend_break, inject_variance_explode, inject_outlier_burst,
)
from research.experiments.g_real_demo import run as run_pipeline
from research.agent.router_state import reset_state


SCENARIOS = {
    "clean":            lambda x, rng: x.copy(),
    "trend_break":      lambda x, rng: inject_trend_break(x, rng, shift_std=3.0),
    "variance_explode": lambda x, rng: inject_variance_explode(x, rng, ratio=3.5),
    "outlier_burst":    lambda x, rng: inject_outlier_burst(x, rng, n_outliers=6, z=4.0),
}


def stress_test(datasets: list[str], scenarios: list[str],
                n_steps: int = 6, warmup: int = 200, step: int = 100,
                H: int = 24, seed: int = 0,
                truncate: int = 1000) -> list[dict]:
    """Walk-forward stress test. `truncate` caps the series length so that
    fault injectors (which place faults at mid-point) land inside the
    walk-forward window [warmup, warmup + n_steps*step]."""
    rng = np.random.default_rng(seed)
    rows = []
    for ds in datasets:
        try:
            series, meta = load_series(ds)
        except Exception as e:
            print(f"[skip] {ds}: {e}")
            continue
        series = series.astype(np.float64)
        # Truncate so fault mid-point (~ len/2) overlaps walk window.
        # walk window covers [warmup, warmup + n_steps*step].
        target_len = min(truncate, len(series),
                          warmup + n_steps * step + H + 50)
        series = series[:target_len]
        # Compute clean threshold for this dataset once (use 95th percentile of
        # the warm-up window from the CANONICAL series for fairness across
        # scenarios).
        thresh = float(np.quantile(series[:warmup], 0.95))

        for sc in scenarios:
            if sc not in SCENARIOS:
                print(f"[skip] unknown scenario {sc}")
                continue
            injector = SCENARIOS[sc]
            faulted = injector(series, np.random.default_rng(seed))

            reset_state()
            state_path = f"research/results/router_state_h_{ds}_{sc}.jsonl"
            t0 = time.time()
            try:
                res = run_pipeline(
                    dataset=ds, H=H, n_steps=n_steps,
                    warmup=warmup, step=step,
                    upper_threshold=thresh,
                    state_path=state_path,
                    series_override=faulted,
                    label=sc,
                    verbose=False,
                )
            except Exception as e:
                rows.append({
                    "dataset": ds, "scenario": sc, "error": str(e)[:80],
                })
                continue
            elapsed = time.time() - t0

            tl = res["timeline"]
            state = res["state"]
            cnt = Counter(r["intervention"] for r in tl)
            n_fb = sum(1 for r in tl if r["fallback_used"])
            mae = float(np.mean([r["outcome_mae"] for r in tl])) if tl else float("nan")
            rows.append({
                "dataset": ds, "scenario": sc, "n_steps": len(tl),
                "MONITOR":  cnt.get("MONITOR", 0),
                "INSPECT":  cnt.get("INSPECT", 0),
                "THROTTLE": cnt.get("THROTTLE", 0),
                "SHUTDOWN": cnt.get("SHUTDOWN", 0),
                "ESCALATE": cnt.get("ESCALATE", 0),
                "mean_mae": round(mae, 3),
                "drift_events": len(state.drift_history),
                "fallbacks":    n_fb,
                "elapsed_s":    round(elapsed, 1),
            })
            reset_state()
    return rows


def _format(rows: list[dict]) -> str:
    if not rows:
        return "(empty)"
    cols = ["dataset", "scenario", "n_steps", "MONITOR", "INSPECT",
            "THROTTLE", "SHUTDOWN", "ESCALATE", "mean_mae",
            "drift_events", "fallbacks", "elapsed_s"]
    widths = {c: max(len(c), max((len(str(r.get(c, ""))) for r in rows), default=0))
              for c in cols}
    sep = "  "
    head = sep.join(c.rjust(widths[c]) for c in cols)
    body = []
    for r in rows:
        if "error" in r:
            body.append(f"{r['dataset']:>10} {r['scenario']:>20}   ERROR: {r['error']}")
            continue
        body.append(sep.join(str(r.get(c, "")).rjust(widths[c]) for c in cols))
    return head + "\n" + "\n".join(body)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["ETTh1", "ETTh2", "Exchange"])
    ap.add_argument("--scenarios", nargs="+",
                    default=["clean", "trend_break", "variance_explode",
                             "outlier_burst"])
    ap.add_argument("--n_steps", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--step", type=int, default=100)
    ap.add_argument("--H", type=int, default=24)
    args = ap.parse_args()

    print("=" * 100)
    print(f"R6-H stress · datasets={args.datasets}  scenarios={args.scenarios}  "
          f"n_steps={args.n_steps}")
    print("=" * 100)

    t0 = time.time()
    rows = stress_test(args.datasets, args.scenarios,
                        n_steps=args.n_steps, warmup=args.warmup,
                        step=args.step, H=args.H)
    print("\n" + _format(rows))

    print()
    # Per-scenario aggregates across datasets
    by_sc: dict[str, list[dict]] = {}
    for r in rows:
        if "error" in r: continue
        by_sc.setdefault(r["scenario"], []).append(r)
    print("─" * 100)
    print("Per-scenario aggregate (mean across datasets):")
    for sc, group in by_sc.items():
        n = len(group)
        if not n: continue
        avg_shutdown = np.mean([g["SHUTDOWN"] for g in group])
        avg_inspect  = np.mean([g["INSPECT"]  for g in group])
        avg_monitor  = np.mean([g["MONITOR"]  for g in group])
        avg_mae      = np.mean([g["mean_mae"] for g in group])
        avg_drift    = np.mean([g["drift_events"] for g in group])
        avg_fb       = np.mean([g["fallbacks"] for g in group])
        print(f"  {sc:>18}: MONITOR={avg_monitor:.1f}  INSPECT={avg_inspect:.1f}  "
              f"SHUTDOWN={avg_shutdown:.1f}  MAE={avg_mae:.2f}  "
              f"drift={avg_drift:.1f}  fbk={avg_fb:.1f}  (n={n})")
    print(f"\n(total elapsed: {time.time() - t0:.1f}s)")
