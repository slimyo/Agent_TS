"""全量预测 TSFM sweep —— 把整套预测库跑遍 forecasting cell，建全量 oracle 库。

复现确定性 few_shot_split（仅依赖 ds,N,H,seed），逐 cell 跑指定预测策略，算 MAE。
schema 与既有 p*/f4 一致：{dataset,N,seed,H,method,mae,start_idx} → research/results/taska_tsfm.jsonl

按 env 选模型（STRATEGY env / GROUP）：
  chronos/chronos2/chronos_bolt  → tsci-c2
  timer                          → tsci-remote   (TIMER_FORCE_GPU=1)
  sundial/time_moe               → tsci-remote-tx440
  timesfm2/tirex/toto            → tsci-tsfm      (HF_HUB_DISABLE_XET=1)
  toto2                          → tsci-toto2
  moirai2                        → 本地 tsci-moirai
  arima_ets/naive_*              → 任意
用 STRATEGY="timer,toto" 显式指定；OUT 覆盖输出文件名（避免多 env 并发写冲突）。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from research.utils.data_loader import load_series
from research.utils.splitter import few_shot_split

DATASETS = ["ETTh1", "ETTh2", "ECL", "Exchange", "Weather", "ILI", "ETTm1", "ETTm2"]
NS = [10, 20, 50, 100]
SEEDS = [1, 42, 123]
H = 96


def _mae(pred, test):
    pred = np.asarray(pred, float).ravel(); test = np.asarray(test, float).ravel()
    L = min(len(pred), len(test))
    return float(np.mean(np.abs(pred[:L] - test[:L]))) if L else float("nan")


def _direct_predict(name):
    """直接 import research.baseline.<name>.predict，绕开 forecaster_reflect 的重依赖链
    （llm/dotenv/openai/statsmodels）。用 inspect 过滤 predict 接受的 kwargs。"""
    import importlib
    import inspect
    mod = importlib.import_module(f"research.baseline.{name}")
    fn = mod.predict
    accept = set(inspect.signature(fn).parameters.keys())

    def call(train, val, H, season_m):
        kw = {}
        if "season_m" in accept:
            kw["season_m"] = season_m
        if "seed" in accept:
            kw["seed"] = 42
        return fn(train, val, H, **kw)
    return call


def main():
    strategies = [s for s in os.environ.get("STRATEGY", "").split(",") if s] or ["chronos2"]
    STRATEGY_FN = {}
    for s in strategies:
        try:
            STRATEGY_FN[s] = _direct_predict(s)
        except Exception as e:
            print(f"strategy {s} unavailable: {type(e).__name__}: {e}", flush=True)
    strategies = [s for s in strategies if s in STRATEGY_FN]
    out = Path(os.environ.get("OUT", "research/results/taska_tsfm.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            if l.strip():
                r = json.loads(l); done.add((r["dataset"], r["N"], r["seed"], r["method"]))
    print(f"strategies={strategies} | resuming {len(done)} rows", flush=True)
    series_cache = {}
    fh = out.open("a")
    for ds in DATASETS:
        if ds not in series_cache:
            try:
                series_cache[ds] = load_series(ds)
            except Exception as e:
                print(f"skip dataset {ds}: {e!r}", flush=True); series_cache[ds] = None
        if series_cache[ds] is None:
            continue
        series, meta = series_cache[ds]
        for N in NS:
            for seed in SEEDS:
                try:
                    sp = few_shot_split(series, N=N, H=H, seed=seed)
                except Exception as e:
                    print(f"  skip {ds} N={N} s={seed}: {e!r}", flush=True); continue
                for s in strategies:
                    if (ds, N, seed, s) in done:
                        continue
                    t0 = time.time()
                    try:
                        pred = STRATEGY_FN[s](sp.train, sp.val, H, meta.season_m)
                        mae = _mae(pred, sp.test)
                    except Exception as e:
                        print(f"  {ds} N={N} s={seed} {s}: FAIL {type(e).__name__}: {str(e)[:80]}", flush=True)
                        mae = float("nan")
                    fh.write(json.dumps({"dataset": ds, "N": N, "seed": seed, "H": H,
                                         "method": s, "start_idx": int(sp.start_idx),
                                         "mae": round(mae, 6) if mae == mae else None,
                                         "wall_time": round(time.time() - t0, 2)},
                                        ensure_ascii=False) + "\n")
                    fh.flush()
                print(f"  {ds:9} N={N:3} s={seed:3} done ({len(strategies)} strat)", flush=True)
    fh.close()
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
