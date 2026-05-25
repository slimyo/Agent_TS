"""实验 runner 骨架。

用法：
  python -m research.experiments.runner \
      --dataset ETTh1 --N 20 --H 96 \
      --methods naive --seeds 1,42,123 \
      --out research/results/p0_naive.jsonl

输出每行一条 jsonl：{dataset, N, H, seed, method, mae, mse, mase, smape, wall_time, start_idx}
"""
from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

from research.utils.data_loader import load_series
from research.utils.splitter import few_shot_split
from research.utils.metrics import all_metrics


METHOD_REGISTRY = {
    "naive":     "research.baseline.naive",
    "arima_ets": "research.baseline.arima_ets",
    "chronos":   "research.baseline.chronos",
    "chronos2":  "research.baseline.chronos2",
    "chronos_bolt": "research.baseline.chronos_bolt",
    "timesfm2":  "research.baseline.timesfm2",
    "llmtime":   "research.baseline.llmtime",
    "tsci":      "research.baseline.tsci",
    "adapt_ts":  "research.agent.adapt_ts",
    "ablation_a8": "research.agent.ablation_a8",
    "ablation_a9": "research.agent.ablation_a9",
    "adapt_ts_v6": "research.agent.adapt_ts_v6",
}


def _parse_N(s: str) -> int | str:
    return s if s.lower() == "full" else int(s)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--N", type=_parse_N, default=20, help="train size, int or 'Full'")
    p.add_argument("--H", type=int, default=96)
    p.add_argument("--methods", default="naive", help="comma-separated method names")
    p.add_argument("--seeds", default="1,42,123", help="comma-separated ints")
    p.add_argument("--out", default=None, help="jsonl output path")
    return p.parse_args()


def main():
    args = parse_args()
    series, meta = load_series(args.dataset)
    methods = [m.strip() for m in args.methods.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    out_path = Path(args.out) if args.out else (
        Path(__file__).resolve().parents[1] / "results" / f"runner_{args.dataset}_N{args.N}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    with out_path.open("a") as fh:
        for seed in seeds:
            split = few_shot_split(series, N=args.N, H=args.H, seed=seed)
            for m in methods:
                mod = importlib.import_module(METHOD_REGISTRY[m])
                t0 = time.time()
                y_hat = mod.predict(
                    train=split.train, val=split.val, H=args.H,
                    seed=seed, season_m=meta.season_m,
                )
                wall = time.time() - t0
                metrics = all_metrics(split.test, y_hat, split.train, season_m=meta.season_m)
                # v11 memory 闭环：若 method 暴露 backfill_test_mae，则回填 test MAE
                if hasattr(mod, "backfill_test_mae"):
                    try:
                        mod.backfill_test_mae(metrics["mae"])
                    except Exception:
                        pass
                row = {
                    "dataset": args.dataset, "N": args.N, "H": args.H,
                    "seed": seed, "method": m,
                    "start_idx": split.start_idx, "wall_time": round(wall, 3),
                    **{k: round(v, 6) for k, v in metrics.items()},
                }
                fh.write(json.dumps(row) + "\n")
                n_rows += 1
                print(json.dumps(row))
    print(f"[runner] wrote {n_rows} rows -> {out_path}")


if __name__ == "__main__":
    main()
