"""扩候选模型库：把 5 个新分类器（catch22/mantis_1nn/mantis_lr/minirocket/weasel）
跑遍全部 22 数据集 × 3N × 2seed，扩充 oracle 标签库（从 5 clf → 10 clf）。

动机：E1/F-R9.7 显示 5-clf 库里 rocket 饱和、几乎无 expert-switching 头寸。扩到 10 clf
看是否出现新的"某分类器在某域占优" → 给决策机制(trust/相图)真正的偏离头寸。

输出追加到 results/taskb_libplus_ucr.jsonl（与 taskb_* 同 schema），resume-safe。
只跑新方法（已有 5 clf 在旧 jsonl 里，分析时合并）。
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

# 22 数据集（与既有 oracle 库一致）
DATASETS = ["BeetleFly", "BirdChicken", "Chinatown", "Coffee", "Crop",
            "DistalPhalanxOutlineCorrect", "ECG200", "ECG5000", "FaceFour", "GunPoint",
            "ItalyPowerDemand", "Lightning7", "MoteStrain", "Plane",
            "ProximalPhalanxOutlineCorrect", "SonyAIBORobotSurface1", "SonyAIBORobotSurface2",
            "Strawberry", "SyntheticControl", "Trace", "TwoLeadECG", "Wafer"]
N_PER_CLASS = [3, 5, 10]
SEEDS = [1, 42]
# 新增分类器（method 名 → strategy 名）；schema 用 method 名兼容 m10 METHOD_TO_CLF 扩展
NEW = {"B7_catch22": "catch22", "B8_mantis_1nn": "mantis_1nn", "B9_mantis_lr": "mantis_lr",
       "B10_minirocket": "minirocket", "B11_weasel": "weasel"}


def main():
    out = Path("research/results/taskb_libplus_ucr.jsonl")
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
                    print(f"skip {ds} N={n} s={seed}: {e!r}", flush=True); continue
                for method, clf in NEW.items():
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
                print(f"  {ds:28} N={n} s={seed} done", flush=True)
    fh.close()
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
