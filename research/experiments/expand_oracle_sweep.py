"""#89 · 扩元数据：在新 UCR 数据集上跑 5 分类器，扩充 oracle 标签库。

F-R9.6 钉死结论：离线 56-cell 下所有 router 范式都收敛 ≈Rocket，瓶颈是**元监督样本复杂度**。
本实验直接攻这个瓶颈：把 oracle 标签库从 10 数据集扩到 ~20+，再重跑 M10/M10c 看
learned router 是否随数据量上升而真正击败 Rocket（验证 F-R9.6 假说）。

输出：results/taskb_expand_ucr.jsonl（与 taskb_ucr / taskb_extended 同 schema：per (ds,N,seed,method) 一行 acc）
每个 cell 跑 {rocket, moment_1nn, moment_logreg, dtw_1nn, euclid_1nn}。
N_per_class ∈ {3,5,10}，seeds {1,42}。resume-safe（跳过已完成）。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from research.agent.clf_strategies import predict_with
from research.utils.ucr_loader import load_ucr_fewshot

# 新增数据集（本地未覆盖；少样本友好 + 多 domain，故意挑 expert-switching 候选）
NEW_DATASETS = [
    "ItalyPowerDemand",       # 2-class, L=24, power
    "SonyAIBORobotSurface1",  # 2-class, L=70, motion
    "SonyAIBORobotSurface2",  # 2-class, L=65, motion
    "MoteStrain",             # 2-class, L=84, sensor
    "ProximalPhalanxOutlineCorrect",  # 2-class, L=80, image
    "DistalPhalanxOutlineCorrect",    # 2-class, L=80, image
    "Plane",                  # 7-class, L=144, sensor
    "SyntheticControl",       # 6-class, L=60, simulated
    "FaceFour",               # 4-class, L=350, image
    "Trace",                  # 4-class, L=275, sensor
    "Lightning7",             # 7-class, L=319, EM
    "Chinatown",              # 2-class, L=24, traffic
]
CLFS = ["rocket", "moment_1nn", "moment_logreg", "dtw_1nn", "euclid_1nn"]
METHOD_NAME = {"rocket": "B3_rocket", "moment_1nn": "B4a_moment_1nn",
               "moment_logreg": "B4b_moment_lr", "dtw_1nn": "B1_dtw", "euclid_1nn": "B2_euclid"}
N_PER_CLASS = [3, 5, 10]
SEEDS = [1, 42]


def main():
    out = Path("research/results/taskb_expand_ucr.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                done.add((r["dataset"], r["N_per_class"], r["seed"], r["method"]))
    print(f"resuming: {len(done)} rows done", flush=True)

    fh = out.open("a")
    for ds in NEW_DATASETS:
        for n in N_PER_CLASS:
            for seed in SEEDS:
                try:
                    Xtr, ytr, Xte, yte = load_ucr_fewshot(ds, n_per_class=n, seed=seed)
                except Exception as e:
                    print(f"skip {ds} N={n} s={seed}: load fail {e!r}", flush=True)
                    continue
                for clf in CLFS:
                    method = METHOD_NAME[clf]
                    if (ds, n, seed, method) in done:
                        continue
                    t0 = time.time()
                    try:
                        yp = predict_with(clf, Xtr, ytr, Xte, season_m=1)
                        acc = float((np.asarray(yp) == yte).mean())
                    except Exception as e:
                        print(f"  {ds} N={n} s={seed} {clf}: FAIL {e!r}", flush=True)
                        acc = float("nan")
                    row = {"dataset": ds, "N_per_class": n, "seed": seed,
                           "method": method, "n_test": int(len(yte)),
                           "acc": round(acc, 4), "wall_time": round(time.time() - t0, 2)}
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
                print(f"  {ds:28} N={n} s={seed} done", flush=True)
    fh.close()
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
