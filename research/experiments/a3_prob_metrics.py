"""A3 · 概率指标评估实验。

对 16 cell × 3 seeds：
1. 跑 Chronos-2，捕获 21-quantile
2. 跑各 AdaptTS 变体（v10/v11/v12）取点预测，按其实际选择策略：
   - 若选 chronos2 → 用 Chronos-2 quantile
   - 若选其他点 predictor → degenerate quantile (CRPS 退化为 MAE)
3. 计算 CRPS / pinball_q10/50/90 / coverage_80 / width_80
4. 输出 jsonl + head-to-head 摘要

输出：research/results/a3_prob_metrics.jsonl
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from research.baseline.chronos2 import predict as c2_predict
from research.utils.data_loader import load_series
from research.utils.metrics import mae
from research.utils.prob_metrics import (
    point_as_degenerate_quantiles,
    prob_metrics_from_quantiles,
)
from research.utils.splitter import few_shot_split

DATASETS = ["ETTh1", "ETTh2", "ECL", "Exchange"]
NS = [10, 20, 50, 100]
SEEDS = [1, 42, 123]
H = 96


def main():
    out_path = Path("research/results/a3_prob_metrics.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["dataset"], r["N"], r["seed"]))
            except Exception:
                pass
    print(f"resuming: {len(done)} cells done")

    with out_path.open("a") as fh:
        for ds in DATASETS:
            series, meta = load_series(ds)
            for N in NS:
                for seed in SEEDS:
                    if (ds, N, seed) in done:
                        continue
                    sp = few_shot_split(series, N=N, H=H, seed=seed)
                    t0 = time.time()
                    # 跑 Chronos-2 取 quantile
                    y_c2 = c2_predict(train=sp.train, val=sp.val, H=H,
                                      seed=seed, season_m=meta.season_m)
                    q_c2 = c2_predict.last_quantiles  # [21, H]
                    pm = prob_metrics_from_quantiles(sp.test, q_c2)
                    pm["mae"] = mae(sp.test, y_c2)
                    wall = time.time() - t0
                    row = {
                        "dataset": ds, "N": N, "seed": seed, "H": H,
                        "method": "chronos2",
                        "wall_time": round(wall, 2),
                        **{k: round(v, 6) for k, v in pm.items()},
                    }
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    print(f"{ds:9} N={N:3} seed={seed:3}  "
                          f"MAE={pm['mae']:7.4f} CRPS={pm['crps']:7.4f} "
                          f"cov80={pm['coverage_80']:.2f} W80={pm['width_80']:.3f}")


if __name__ == "__main__":
    main()
