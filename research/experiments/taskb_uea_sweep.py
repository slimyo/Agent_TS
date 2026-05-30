"""task #44 / P6.4e · UEA Multivariate sweep。

多变量 TSC 上测：
  - Rocket (sktime 原生支持多变量)
  - 1-NN Euclid (channel-flatten 后)
  - 1-NN DTW (channel-wise DTW 求和 — multivariate DTW 简化版)
  - MOMENT 不试（n_channels 切换复杂，留 future）
  - B7v3 router 用 rocket fallback (memory bank 仍是 UCR-5 build)

输出 research/results/taskb_uea.jsonl
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from research.utils.uea_loader import load_uea_fewshot

DATASETS = ["BasicMotions", "ERing", "AtrialFibrillation"]
N_PER_CLASS = [3, 5, 10]
SEEDS = [1, 42]


def rocket_multivariate(X_train, y_train, X_test, num_kernels=1000):
    """sktime RocketClassifier 原生支持 [N, C, L]."""
    from sktime.classification.kernel_based import RocketClassifier
    clf = RocketClassifier(num_kernels=num_kernels, random_state=0)
    clf.fit(X_train, y_train)
    return clf.predict(X_test)


def euclid_1nn_multivariate(X_train, y_train, X_test):
    """Flatten channels 后做 Euclidean 1-NN."""
    X_tr = X_train.reshape(X_train.shape[0], -1)
    X_te = X_test.reshape(X_test.shape[0], -1)
    diff = X_te[:, None, :] - X_tr[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=-1))
    return y_train[dists.argmin(axis=1)]


def dtw_1nn_multivariate(X_train, y_train, X_test):
    """Channel-wise DTW 求和 (简化多变量 DTW)."""
    from dtaidistance import dtw
    preds = []
    for x_te in X_test:
        dists = []
        for x_tr in X_train:
            d = sum(dtw.distance(x_te[c].astype(np.float64),
                                  x_tr[c].astype(np.float64))
                    for c in range(x_te.shape[0]))
            dists.append(d)
        preds.append(y_train[np.argmin(dists)])
    return np.array(preds)


def main():
    out = Path("research/results/taskb_uea.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            try:
                r = json.loads(l)
                done.add((r["dataset"], r["N_per_class"], r["seed"], r["method"]))
            except Exception:
                pass

    fh = out.open("a")
    for ds in DATASETS:
        for n in N_PER_CLASS:
            for seed in SEEDS:
                try:
                    X_tr, y_tr, X_te, y_te = load_uea_fewshot(ds, n_per_class=n, seed=seed)
                except Exception as e:
                    print(f"  skip {ds} {n} {seed}: load failed {e!r}")
                    continue
                print(f"\n{ds} n={n} seed={seed}: X_tr={X_tr.shape} X_te={X_te.shape}")

                methods = {
                    "B3_rocket":     rocket_multivariate,
                    "B2_euclid":     euclid_1nn_multivariate,
                    "B1_dtw":        dtw_1nn_multivariate,
                }
                for name, fn in methods.items():
                    key = (ds, n, seed, name)
                    if key in done: continue
                    t0 = time.time()
                    try:
                        y_p = fn(X_tr, y_tr, X_te)
                        acc = float((y_p == y_te).mean())
                        from sklearn.metrics import f1_score
                        try:
                            f1 = float(f1_score(y_te, y_p, average="macro"))
                        except Exception:
                            f1 = 0.0
                        wall = time.time() - t0
                        row = {"dataset": ds, "N_per_class": n, "seed": seed,
                               "method": name, "n_test": len(y_te),
                               "acc": round(acc, 4), "macro_f1": round(f1, 4),
                               "wall_time": round(wall, 2),
                               "n_channels": int(X_tr.shape[1]),
                               "length": int(X_tr.shape[2])}
                        fh.write(json.dumps(row) + "\n"); fh.flush()
                        print(f"  {name:14}: acc={acc:.3f} f1={f1:.3f} ({wall:.1f}s)")
                    except Exception as e:
                        print(f"  FAIL {name}: {e!r}")
    fh.close()


if __name__ == "__main__":
    main()
