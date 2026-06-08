"""扩模型库（UEA 多变量）：UEA 此前只跑了 3 个多变量分类器(rocket/euclid/dtw)，
本 sweep 把其余 7 个单变量库分类器补上，使 UEA 也拥有全量 10-clf 库。

多变量→单变量适配（诚实声明）：predict_with 约定输入 [N,L]。对 UEA 的 [N,C,L]，
按 **channel-flatten**（沿通道拼接成长度 C×L 的单序列）降维后调用。这是标准、可复现的
baseline 降维口径；对个别基础模型可能次优，故每 cell 用 _safe 包裹，失败回退 majority（记 NaN）。
rocket/euclid/dtw 的多变量版已在 taskb_uea_full.jsonl，本文件只补 7 个。

输出：research/results/taskb_uea_libplus.jsonl（与 taskb_* 同 schema，resume-safe）。
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from research.agent.clf_strategies import predict_with
from research.utils.uea_loader import load_uea_fewshot

# 只取已在 taskb_uea_full 跑出 rocket 的 14 个数据集（保证 base 存在、库可合并）
DATASETS = ["BasicMotions", "ERing", "AtrialFibrillation", "Cricket", "Handwriting",
            "Libras", "UWaveGestureLibrary", "ArticularyWordRecognition", "Epilepsy",
            "NATOPS", "RacketSports", "HandMovementDirection", "FingerMovements", "Heartbeat"]
N_PER_CLASS = [3, 5, 10]
SEEDS = [1, 42]
MAX_TEST = 200  # 与 taskb_uea_full 口径一致
# 补的 7 个单变量库分类器
NEW = {"B4a_moment_1nn": "moment_1nn", "B4b_moment_lr": "moment_logreg",
       "B7_catch22": "catch22", "B8_mantis_1nn": "mantis_1nn", "B9_mantis_lr": "mantis_lr",
       "B10_minirocket": "minirocket", "B11_weasel": "weasel"}


def flatten_channels(X):
    """[N, C, L] -> [N, C*L]（channel-flatten 单变量化）。"""
    return X.reshape(X.shape[0], -1)


def subsample(X, y, n_max, seed=0):
    if len(X) <= n_max:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=n_max, replace=False)
    return X[idx], y[idx]


def main():
    out = Path("research/results/taskb_uea_libplus.jsonl")
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
                    Xtr, ytr, Xte, yte = load_uea_fewshot(ds, n_per_class=n, seed=seed)
                except Exception as e:
                    print(f"skip {ds} N={n} s={seed}: load {type(e).__name__}", flush=True)
                    continue
                Xte, yte = subsample(Xte, yte, MAX_TEST, seed)
                Xtr_f, Xte_f = flatten_channels(Xtr), flatten_channels(Xte)
                for method, clf in NEW.items():
                    if (ds, n, seed, method) in done:
                        continue
                    t0 = time.time()
                    try:
                        yp = predict_with(clf, Xtr_f, ytr, Xte_f, season_m=1)
                        acc = float((np.asarray(yp) == yte).mean())
                    except Exception as e:
                        print(f"  {ds} N={n} s={seed} {clf}: FAIL {type(e).__name__}", flush=True)
                        acc = float("nan")
                    fh.write(json.dumps({"dataset": ds, "N_per_class": n, "seed": seed,
                                         "method": method, "n_test": int(len(yte)),
                                         "acc": round(acc, 4),
                                         "n_channels": int(Xtr.shape[1]), "length": int(Xtr.shape[2]),
                                         "adapt": "channel_flatten",
                                         "wall_time": round(time.time() - t0, 2)},
                                        ensure_ascii=False) + "\n")
                    fh.flush()
                print(f"  {ds:28} N={n} s={seed} done", flush=True)
    fh.close()
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
