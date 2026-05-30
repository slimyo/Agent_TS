"""task #48 / P0-3 · UEA full sweep（扩 task #44 从 3 → 20 datasets）。

策略：
  - 20 datasets × 3 N-shot × 2 seeds × 3 methods = 360 cells
  - length > 1500 跳过 DTW (O(L²) 不可行)
  - 容错：dataset 下载失败/加载失败 skip
  - 增量 resume：已写入文件的不重跑
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from research.experiments.taskb_uea_sweep import (
    dtw_1nn_multivariate, euclid_1nn_multivariate, rocket_multivariate,
)
from research.utils.uea_loader import (
    UEA_DATASETS_RECOMMENDED, UEA_SKIP_DTW_IF_LENGTH, load_uea_fewshot,
)


DATASETS = list(UEA_DATASETS_RECOMMENDED.keys())  # 20 datasets
N_PER_CLASS = [3, 5, 10]
SEEDS = [1, 42]
MAX_TEST = 200  # 限制 test 子集 (LSST/large test 加速)


def subsample_test(X, y, n_max, seed=0):
    if len(X) <= n_max: return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=n_max, replace=False)
    return X[idx], y[idx]


def main():
    out = Path("research/results/taskb_uea_full.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            try:
                r = json.loads(l)
                done.add((r["dataset"], r["N_per_class"], r["seed"], r["method"]))
            except Exception: pass
    print(f"resuming: {len(done)} cells done")

    # 复用 task #44 已有数据
    if Path("research/results/taskb_uea.jsonl").exists():
        for l in open("research/results/taskb_uea.jsonl"):
            try:
                r = json.loads(l)
                done.add((r["dataset"], r["N_per_class"], r["seed"], r["method"]))
                # 也写入 full file 方便 aggregate
                row = {**r, "_from_task44": True}
                with out.open("a") as fh: fh.write(json.dumps(row) + "\n")
            except Exception: pass

    fh = out.open("a")
    for ds in DATASETS:
        meta = UEA_DATASETS_RECOMMENDED[ds]
        seq_len = meta["length"]
        skip_dtw = seq_len > UEA_SKIP_DTW_IF_LENGTH
        for n in N_PER_CLASS:
            for seed in SEEDS:
                try:
                    X_tr, y_tr, X_te, y_te = load_uea_fewshot(ds, n_per_class=n, seed=seed)
                except Exception as e:
                    print(f"  skip {ds} {n} {seed}: load failed {type(e).__name__}")
                    continue
                X_te_s, y_te_s = subsample_test(X_te, y_te, MAX_TEST, seed)

                methods = {"B3_rocket": rocket_multivariate,
                            "B2_euclid": euclid_1nn_multivariate}
                if not skip_dtw:
                    methods["B1_dtw"] = dtw_1nn_multivariate
                for name, fn in methods.items():
                    key = (ds, n, seed, name)
                    if key in done: continue
                    t0 = time.time()
                    try:
                        y_p = fn(X_tr, y_tr, X_te_s)
                        acc = float((y_p == y_te_s).mean())
                        from sklearn.metrics import f1_score
                        try: f1 = float(f1_score(y_te_s, y_p, average="macro"))
                        except Exception: f1 = 0.0
                        wall = time.time() - t0
                        row = {"dataset": ds, "N_per_class": n, "seed": seed,
                               "method": name, "n_test": len(y_te_s),
                               "acc": round(acc, 4), "macro_f1": round(f1, 4),
                               "wall_time": round(wall, 2),
                               "n_channels": int(X_tr.shape[1]),
                               "length": int(X_tr.shape[2])}
                        fh.write(json.dumps(row) + "\n"); fh.flush()
                        print(f"  {ds:30} n={n} seed={seed} {name:14}: acc={acc:.3f} f1={f1:.3f} ({wall:.1f}s)")
                    except Exception as e:
                        print(f"  FAIL {ds} {n} {seed} {name}: {type(e).__name__}")
    fh.close()


if __name__ == "__main__":
    main()
