"""检测任务（合成 4-fault）补全分类器库 → 全量 22 分类器。

复用 taskc_synth4class 的确定性数据生成（build_synthetic_dataset，同 seeds），
把检测库从 7 个补到全量 22（缺的 5 个 base + 新增 12 个），schema 与 taskc_synth4class 一致。

分组（env-gated，同 taskb_newlib）：
  LOCAL（numba/torch 本地 tsci 可跑）：B7-B11 base + E1-E6 深度/卷积 + E10-E12 多变量
  EMB（需 TSFM 权重 → 远程各 env）：E7 chronos2_emb / E8 timesfm_emb / E9 timer_emb
输出：research/results/taskc_newlib.jsonl
"""
from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from research.agent.clf_strategies import predict_with
from research.utils.data_loader import load_series
from research.utils.inject_fault import build_synthetic_dataset

DATASETS = ["ETTh1", "ECL"]
N_PER_CLASS_TRAIN = [3, 5, 10]
N_PER_CLASS_TEST = 20
SEEDS = [1, 42]
WINDOW_LEN = 96

BASE_MISSING = {"B7_catch22": "catch22", "B8_mantis_1nn": "mantis_1nn", "B9_mantis_lr": "mantis_lr",
                "B10_minirocket": "minirocket", "B11_weasel": "weasel"}
FLAT = {"E1_multirocket": "multirocket", "E2_arsenal": "arsenal", "E3_drcif": "drcif",
        "E4_fcn": "fcn", "E5_resnet": "resnet", "E6_inceptiontime": "inceptiontime"}
MV = {"E10_muse": "muse", "E11_rocket_mv": "rocket_mv", "E12_cif_mv": "cif_mv"}
EMB = {"E7_chronos2_emb": "chronos2_emb", "E8_timesfm_emb": "timesfm_emb", "E9_timer_emb": "timer_emb"}
ALL = {**BASE_MISSING, **FLAT, **MV, **EMB}


def _select():
    if os.environ.get("CLFS"):
        want = set(os.environ["CLFS"].split(","))
        return {m: c for m, c in ALL.items() if c in want or m in want}
    g = os.environ.get("GROUP", "local").lower()
    if g == "emb":
        return EMB
    if g == "all":
        return ALL
    return {**BASE_MISSING, **FLAT, **MV}   # local 默认（不含远程 EMB）


def main():
    sel = _select()
    deep_ep = int(os.environ.get("DEEP_EPOCHS", "100"))
    light = {"multirocket": {"num_kernels": 2000}, "arsenal": {"num_kernels": 1000, "n_estimators": 10},
             "drcif": {"n_estimators": 15}, "cif_mv": {"n_estimators": 15}, "rocket_mv": {"num_kernels": 4000}}
    out = Path(os.environ.get("OUT", "research/results/taskc_newlib.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            if l.strip():
                r = json.loads(l); done.add((r["dataset"], r["N_per_class"], r["seed"], r["method"]))
    print(f"detection sweep clfs={list(sel.values())} | resuming {len(done)}", flush=True)
    fh = out.open("a")
    for ds in DATASETS:
        series, meta = load_series(ds)
        for n in N_PER_CLASS_TRAIN:
            for seed in SEEDS:
                todo = {m: c for m, c in sel.items() if (ds, n, seed, m) not in done}
                if not todo:
                    continue
                Xtr, ytr = build_synthetic_dataset(series, window_len=WINDOW_LEN, n_per_class=n,
                                                   seed=seed * 1000, season_m=meta.season_m)
                Xte, yte = build_synthetic_dataset(series, window_len=WINDOW_LEN, n_per_class=N_PER_CLASS_TEST,
                                                   seed=seed * 1000 + 1, season_m=meta.season_m)
                X2tr = Xtr.reshape(Xtr.shape[0], -1); X2te = Xte.reshape(Xte.shape[0], -1)
                X3tr = X2tr[:, None, :]; X3te = X2te[:, None, :]
                for method, clf in todo.items():
                    native_mv = clf in MV.values()
                    Xtr_in, Xte_in = (X3tr, X3te) if native_mv else (X2tr, X2te)
                    kw = dict(light.get(clf, {}))
                    if clf in ("fcn", "resnet", "inceptiontime"):
                        kw["epochs"] = deep_ep
                    t0 = time.time()
                    try:
                        yp = predict_with(clf, Xtr_in, ytr, Xte_in, **kw)
                        acc = float((np.asarray(yp) == yte).mean())
                        from sklearn.metrics import f1_score
                        f1 = float(f1_score(yte, yp, average="macro"))
                    except Exception as e:
                        print(f"  {ds} N={n} s={seed} {clf}: FAIL {e!r}", flush=True); acc = float("nan"); f1 = 0.0
                    fh.write(json.dumps({"dataset": ds, "N_per_class": n, "seed": seed, "method": method,
                                         "n_test": int(len(yte)), "acc": round(acc, 4) if acc == acc else None,
                                         "macro_f1": round(f1, 4), "wall_time": round(time.time() - t0, 2)},
                                        ensure_ascii=False) + "\n")
                    fh.flush()
                print(f"  {ds} N={n} s={seed} done ({len(todo)} clf)", flush=True)
    fh.close()
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
