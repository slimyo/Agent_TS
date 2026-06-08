"""独立 TSFM-嵌入分类 sweep —— 复刻"孤立加载成功"的最简模式，避开通用 sweep 的复杂性。

为什么单独写：通用 taskb_newlib_sweep 经 predict_with→_safe，模型加载失败被吞成 majority（污染）；
且并发/时序下 HF offline 解析偶发 LocalEntryNotFound。本脚本：
  1. 顶层**加载一次**模型（孤立 `python -c` 验证可行的同款路径），失败重试，仍失败则**退出**（绝不写 majority）。
  2. 逐 cell 直接调 tsfm_embed.<EMB>（不经 _safe），real acc 才写；异常则跳过（不写假值）。

用法：EMB=timesfm_emb OUT=research/results/taskb_newlib_timesfm.jsonl python -m research.experiments.emb_clf_sweep
（在对应远程 env + offline 下跑：timesfm_emb→tsci-tsfm / timer_emb→tsci-remote / chronos2_emb→tsci-c2）
"""
from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from research.baseline import tsfm_embed as TE
from research.experiments.taskb_newlib_sweep import UCR, UEA, N_PER_CLASS, SEEDS, MAX_TEST, _load

EMB = os.environ.get("EMB", "timesfm_emb")
METHOD = {"timesfm_emb": "E8_timesfm_emb", "timer_emb": "E9_timer_emb",
          "chronos2_emb": "E7_chronos2_emb"}[EMB]


def _warm():
    """顶层加载一次（带重试）。返回 True/False。"""
    fn = getattr(TE, EMB)
    w = np.asarray([np.sin(np.arange(128) * 0.1), np.cos(np.arange(128) * 0.1)], dtype=np.float32)
    for k in range(10):
        try:
            fn(w, np.array([0, 1]), w)
            print(f"WARM {EMB} OK (attempt {k+1})", flush=True)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"WARM {EMB} attempt {k+1} fail: {type(e).__name__}: {str(e)[:90]}", flush=True)
            time.sleep(6)
    return False


def main():
    out = Path(os.environ.get("OUT", f"research/results/taskb_newlib_{EMB}.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                if r.get("acc") is not None:
                    done.add((r["dataset"], r["N_per_class"], r["seed"]))
    print(f"{EMB} ({METHOD}) | resuming {len(done)} real rows", flush=True)
    if not _warm():
        print(f"[ABORT] {EMB} 加载失败 10 次，退出（不写任何 majority）", flush=True)
        return
    fn = getattr(TE, EMB)
    fh = out.open("a")
    n_new = 0
    for ds in UCR + UEA:
        for n in N_PER_CLASS:
            for seed in SEEDS:
                if (ds, n, seed) in done:
                    continue
                try:
                    X2tr, X3tr, ytr, X2te, X3te, yte = _load(ds, n, seed)
                except Exception as e:
                    print(f"  skip {ds} N={n} s={seed}: {e!r}", flush=True); continue
                t0 = time.time()
                try:
                    yp = fn(X2tr, ytr, X2te)
                    acc = float((np.asarray(yp) == yte).mean())
                except Exception as e:  # noqa: BLE001
                    print(f"  {ds} N={n} s={seed}: ERR {type(e).__name__}: {str(e)[:70]} (跳过)", flush=True)
                    continue
                fh.write(json.dumps({"dataset": ds, "N_per_class": n, "seed": seed, "method": METHOD,
                                     "n_test": int(len(yte)), "acc": round(acc, 4),
                                     "wall_time": round(time.time() - t0, 2)}, ensure_ascii=False) + "\n")
                fh.flush(); n_new += 1
        print(f"  {ds:24} done (new so far {n_new})", flush=True)
    fh.close()
    print(f"\nwrote {out} (+{n_new} rows)", flush=True)


if __name__ == "__main__":
    main()
