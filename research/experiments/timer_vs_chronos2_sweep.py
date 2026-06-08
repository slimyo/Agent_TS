"""Timer-S1 vs Chronos-2 head-to-head（远程 GPU）—— 决定是否更换预测 base 模型。

复现既有 72 forecasting cell（6 数据集 × N{10,20,50,100} × seed{1,42,123}, H=96），
用**完全相同的确定性 few_shot_split**（split 只依赖 dataset,N,H,seed），在远程同一 env 内
同管线跑 Timer-S1 与 Chronos-2，逐 cell 比 MAE。

决策口径：
  - per-cell rel = (mae_c2 - mae_timer)/mae_c2     （>0 表示 timer 更好）
  - win_rate = timer 严格优于 c2 的 cell 比例
  - 分数据集胜负 + 总体 mean rel-MAE
结论：仅当 timer 总体 rel-MAE 显著为正且跨数据集稳定时，才建议换 base。

设备：需 TIMER_FORCE_GPU=1（多卡 device_map=auto）。Chronos-2 自带 GPU 选择。
输出：research/results/timer_vs_chronos2.jsonl（首行 _summary）。
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SKIP_C2 = os.environ.get("SKIP_C2") == "1"   # 远程 env 无 chronos 包时，只跑 timer，本地再 merge

from research.utils.data_loader import load_series
from research.utils.splitter import few_shot_split

DATASETS = ["Weather", "ILI", "ETTh2", "Exchange", "ETTh1", "ECL"]
NS = [10, 20, 50, 100]
SEEDS = [1, 42, 123]
H = 96


def _mae(pred, test):
    pred = np.asarray(pred, dtype=np.float64).ravel()
    test = np.asarray(test, dtype=np.float64).ravel()
    L = min(len(pred), len(test))
    if L == 0:
        return float("nan")
    return float(np.mean(np.abs(pred[:L] - test[:L])))


def main():
    from research.baseline import timer as timer_mod
    c2_mod = None
    if not SKIP_C2:
        from research.baseline import chronos2 as c2_mod  # noqa: F811

    out = Path("research/results/timer_vs_chronos2.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    print("=== Timer-S1 vs Chronos-2 (72 cells, same split, remote GPU) ===", flush=True)

    # 预热各加载一次（打印设备）
    series_cache = {}
    for ds in DATASETS:
        series_cache[ds] = load_series(ds)

    for ds in DATASETS:
        series, meta = series_cache[ds]
        for N in NS:
            for seed in SEEDS:
                try:
                    sp = few_shot_split(series, N=N, H=H, seed=seed)
                except Exception as e:
                    print(f"  SKIP {ds} N={N} s={seed}: split {type(e).__name__}", flush=True)
                    continue
                try:
                    t0 = time.time()
                    p_timer = timer_mod.predict(sp.train, sp.val, H=H, seed=seed)
                    dt_timer = time.time() - t0
                except Exception as e:
                    print(f"  FAIL timer {ds} N={N} s={seed}: {type(e).__name__}: {e}", flush=True)
                    continue
                mae_t = _mae(p_timer, sp.test)
                mae_c = float("nan")
                dt_c2 = 0.0
                if c2_mod is not None:
                    try:
                        t0 = time.time()
                        p_c2 = c2_mod.predict(sp.train, sp.val, H=H, seed=seed,
                                              season_m=meta.season_m)
                        dt_c2 = time.time() - t0
                        mae_c = _mae(p_c2, sp.test)
                    except Exception as e:
                        print(f"  FAIL c2 {ds} N={N} s={seed}: {type(e).__name__}: {e}", flush=True)
                rel = (mae_c - mae_t) / mae_c if mae_c > 0 else float("nan")
                win = ("timer" if mae_t < mae_c else "c2") if mae_c == mae_c else None
                r = {"dataset": ds, "N": N, "seed": seed, "H": H,
                     "start_idx": int(sp.start_idx),
                     "mae_timer": round(mae_t, 6),
                     "mae_c2": (round(mae_c, 6) if mae_c == mae_c else None),
                     "rel_impr": (round(rel, 4) if rel == rel else None), "win": win,
                     "wall_timer": round(dt_timer, 2), "wall_c2": round(dt_c2, 2)}
                rows.append(r)
                print(f"  {ds:9} N={N:3} s={seed:3}: timer={mae_t:.4f} "
                      f"c2={mae_c if mae_c==mae_c else float('nan'):.4f} "
                      f"rel={rel if rel==rel else float('nan'):+.3f} -> {win}  "
                      f"({dt_timer:.1f}s/{dt_c2:.1f}s)", flush=True)

    # ---- 汇总（仅对有 chronos2 配对的 cell）----
    paired = [r for r in rows if r.get("mae_c2") is not None]
    summary = {"n_cells": len(rows), "n_paired": len(paired)}
    if paired:
        rows = paired
        rels = np.array([r["rel_impr"] for r in rows])
        summary["timer_win_rate"] = round(float(np.mean([r["win"] == "timer" for r in rows])), 3)
        summary["mean_rel_impr"] = round(float(rels.mean()), 4)
        summary["median_rel_impr"] = round(float(np.median(rels)), 4)
        summary["mean_mae_timer"] = round(float(np.mean([r["mae_timer"] for r in rows])), 4)
        summary["mean_mae_c2"] = round(float(np.mean([r["mae_c2"] for r in rows])), 4)
        per_ds = defaultdict(list)
        for r in rows:
            per_ds[r["dataset"]].append(r)
        summary["per_dataset"] = {
            ds: {"timer_win_rate": round(float(np.mean([x["win"] == "timer" for x in v])), 2),
                 "mean_rel_impr": round(float(np.mean([x["rel_impr"] for x in v])), 4),
                 "n": len(v)}
            for ds, v in sorted(per_ds.items())}
        # 决策建议
        switch = summary["mean_rel_impr"] > 0.03 and summary["timer_win_rate"] >= 0.55 and \
            sum(d["mean_rel_impr"] > 0 for d in summary["per_dataset"].values()) >= 4
        summary["recommend_switch_base"] = bool(switch)

    with out.open("w") as fh:
        fh.write(json.dumps({"_summary": summary}, ensure_ascii=False) + "\n")
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
