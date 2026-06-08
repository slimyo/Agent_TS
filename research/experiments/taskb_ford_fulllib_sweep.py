"""扩测试集（UCR）：把 FordA / FordB 两个此前未覆盖的 UCR 数据集，跑遍全量 10 分类器，
补进 oracle 库（与 taskb_libplus_ucr 同 schema）。UCR 22 → 24。

动机：用户要求"增加实验测试集 + 模型库全量"。Ford* 是长序列(L=500)二分类，
与现有 22 个不同分布，能给决策机制(双信号解耦)新的、独立的 LODO 域。

输出：research/results/taskb_ford_fulllib.jsonl（resume-safe）。
注：test 子采样 MAX_TEST=300（FordA test=1320 / FordB=810），保证 dtw O(L²) 可行，
与 UEA full sweep 的 MAX_TEST 口径一致；few-shot 训练集本就很小。
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from research.agent.clf_strategies import predict_with
from research.utils.ucr_loader import load_ucr_fewshot

DATASETS = ["FordA", "FordB"]
N_PER_CLASS = [3, 5, 10]
SEEDS = [1, 42]
MAX_TEST = 300
# Ford 序列长(L=500)，dtw O(L²) 单 cell ~600s 不可行 → 跳过（库剩 9 clf，仍 ≥8 过滤阈值、
# rocket base 在场）。与 UEA full sweep 的 length-gate 跳 dtw 口径一致。
SKIP_DTW = True
# 全量 10 分类器（method 名 → strategy 名），与 signal_router METHOD_TO_CLF 对齐
LIB = {
    "B1_dtw": "dtw_1nn", "B2_euclid": "euclid_1nn", "B3_rocket": "rocket",
    "B4a_moment_1nn": "moment_1nn", "B4b_moment_lr": "moment_logreg",
    "B7_catch22": "catch22", "B8_mantis_1nn": "mantis_1nn", "B9_mantis_lr": "mantis_lr",
    "B10_minirocket": "minirocket", "B11_weasel": "weasel",
}


def subsample(X, y, n_max, seed=0):
    if len(X) <= n_max:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=n_max, replace=False)
    return X[idx], y[idx]


def main():
    out = Path("research/results/taskb_ford_fulllib.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                done.add((r["dataset"], r["N_per_class"], r["seed"], r["method"]))
    print(f"resuming: {len(done)} rows done", flush=True)
    fh = out.open("a")
    for ds in DATASETS:
        for n in N_PER_CLASS:
            for seed in SEEDS:
                try:
                    Xtr, ytr, Xte, yte = load_ucr_fewshot(ds, n_per_class=n, seed=seed)
                except Exception as e:
                    print(f"skip {ds} N={n} s={seed}: {e!r}", flush=True)
                    continue
                Xte, yte = subsample(Xte, yte, MAX_TEST, seed)
                for method, clf in LIB.items():
                    if SKIP_DTW and method == "B1_dtw":
                        continue
                    if (ds, n, seed, method) in done:
                        continue
                    t0 = time.time()
                    try:
                        yp = predict_with(clf, Xtr, ytr, Xte, season_m=1)
                        acc = float((np.asarray(yp) == yte).mean())
                    except Exception as e:
                        print(f"  {ds} N={n} s={seed} {clf}: FAIL {e!r}", flush=True)
                        acc = float("nan")
                    fh.write(json.dumps({"dataset": ds, "N_per_class": n, "seed": seed,
                                         "method": method, "n_test": int(len(yte)),
                                         "acc": round(acc, 4),
                                         "wall_time": round(time.time() - t0, 2)},
                                        ensure_ascii=False) + "\n")
                    fh.flush()
                print(f"  {ds:10} N={n} s={seed} done", flush=True)
    fh.close()
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
