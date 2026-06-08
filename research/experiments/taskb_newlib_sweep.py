"""全量 sweep：把 baseline.md §1.1 新增的 12 个分类器跑遍 38 数据集（UCR24 + UEA14）
× N{3,5,10} × seed{1,42}，扩充 oracle 标签库，schema 与 taskb_* 一致，resume-safe。

分组（按输入表示 + 运行位置）：
  FLAT  (channel-flatten 2D 输入，与现有库口径一致)：
        类别一 multirocket/arsenal/drcif、类别二 fcn/resnet/inceptiontime  → 本地 tsci
  EMB   (FLAT 输入，但需 TSFM 权重 → 各自远程 env)：
        类别三 chronos2_emb/timesfm_emb/timer_emb
  MV    (原生多变量 3D 输入 [N,C,L]，UCR 退化 1 通道)：
        类别四 muse/rocket_mv/cif_mv  → 本地 tsci

用 env 选组/选分类器：
  CLFS="multirocket,drcif,..."   显式指定
  GROUP=flat|emb|mv|all          (default all 本地可跑 = flat+mv)
  DEEP_EPOCHS=100                深度网训练轮（sweep 提速）
输出：research/results/taskb_newlib.jsonl
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

UCR = ["BeetleFly", "BirdChicken", "Chinatown", "Coffee", "Crop",
       "DistalPhalanxOutlineCorrect", "ECG200", "ECG5000", "FaceFour", "FordA", "FordB",
       "GunPoint", "ItalyPowerDemand", "Lightning7", "MoteStrain", "Plane",
       "ProximalPhalanxOutlineCorrect", "SonyAIBORobotSurface1", "SonyAIBORobotSurface2",
       "Strawberry", "SyntheticControl", "Trace", "TwoLeadECG", "Wafer"]
UEA = ["BasicMotions", "ERing", "AtrialFibrillation", "Cricket", "Handwriting",
       "Libras", "UWaveGestureLibrary", "ArticularyWordRecognition", "Epilepsy",
       "NATOPS", "RacketSports", "HandMovementDirection", "FingerMovements", "Heartbeat"]
N_PER_CLASS = [3, 5, 10]
SEEDS = [1, 42]
MAX_TEST = 150  # 统一截断 test（UCR 个别 test 上万：Crop16800/Wafer6164/ECG5000 → 不截会拖死重模型）

# 轻量化参数（sweep 提速；保持模型有意义，少样本下足够）
LIGHT = {"multirocket": {"num_kernels": 2000}, "arsenal": {"num_kernels": 1000, "n_estimators": 10},
         "drcif": {"n_estimators": 15}, "cif_mv": {"n_estimators": 15},
         "rocket_mv": {"num_kernels": 4000}}

# method 名(写库) -> strategy 名(predict_with)
FLAT = {"E1_multirocket": "multirocket", "E2_arsenal": "arsenal", "E3_drcif": "drcif",
        "E4_fcn": "fcn", "E5_resnet": "resnet", "E6_inceptiontime": "inceptiontime"}
EMB = {"E7_chronos2_emb": "chronos2_emb", "E8_timesfm_emb": "timesfm_emb",
       "E9_timer_emb": "timer_emb"}
MV = {"E10_muse": "muse", "E11_rocket_mv": "rocket_mv", "E12_cif_mv": "cif_mv"}
ALL = {**FLAT, **EMB, **MV}


def _select():
    if os.environ.get("CLFS"):
        want = set(os.environ["CLFS"].split(","))
        return {m: c for m, c in ALL.items() if c in want or m in want}
    g = os.environ.get("GROUP", "local").lower()
    if g == "flat":
        return FLAT
    if g == "emb":
        return EMB
    if g == "mv":
        return MV
    if g == "all":
        return ALL
    return {**FLAT, **MV}   # local 默认（不含需远程权重的 emb）


def _load(ds, n, seed):
    """返回 (X2d_flat, X3d_native, y_train, X2d_te, X3d_te, y_test)。UCR: C=1。"""
    if ds in UEA:
        from research.utils.uea_loader import load_uea_fewshot
        Xtr, ytr, Xte, yte = load_uea_fewshot(ds, n_per_class=n, seed=seed)
        if len(Xte) > MAX_TEST:
            rng = np.random.default_rng(0); idx = rng.choice(len(Xte), MAX_TEST, replace=False)
            Xte, yte = Xte[idx], yte[idx]
        Xtr3d = Xtr if Xtr.ndim == 3 else Xtr[:, None, :]
        Xte3d = Xte if Xte.ndim == 3 else Xte[:, None, :]
    else:
        from research.utils.ucr_loader import load_ucr_fewshot
        Xtr, ytr, Xte, yte = load_ucr_fewshot(ds, n_per_class=n, seed=seed)
        if len(Xte) > MAX_TEST:
            rng = np.random.default_rng(0); idx = rng.choice(len(Xte), MAX_TEST, replace=False)
            Xte, yte = Xte[idx], yte[idx]
        Xtr3d = Xtr[:, None, :]; Xte3d = Xte[:, None, :]
    # 通道上限（控 UEA 高通道：Heartbeat 61ch / MotorImagery 等，区间森林按通道×区间爆炸；
    # 等距取 CCAP 个通道；对 channel-flatten/区间模型是必要的可行性折中）。
    ccap = int(os.environ.get("CCAP", "0"))
    if ccap and Xtr3d.shape[1] > ccap:
        idxC = np.linspace(0, Xtr3d.shape[1] - 1, ccap).astype(int)
        Xtr3d = Xtr3d[:, idxC, :]; Xte3d = Xte3d[:, idxC, :]
    # 时间维长度上限（控 UEA 超长序列：Cricket 1197/通道 等，沿时间等距下采样到 LCAP；
    # 与 TSFM 嵌入 _CAP=512 同口径，对这些模型几无精度损失但大幅提速）。
    lcap = int(os.environ.get("LCAP", "0"))
    if lcap and Xtr3d.shape[-1] > lcap:
        idxL = np.linspace(0, Xtr3d.shape[-1] - 1, lcap).astype(int)
        Xtr3d = Xtr3d[:, :, idxL]; Xte3d = Xte3d[:, :, idxL]
    Xtr2d = Xtr3d.reshape(Xtr3d.shape[0], -1); Xte2d = Xte3d.reshape(Xte3d.shape[0], -1)
    return Xtr2d, Xtr3d, ytr, Xte2d, Xte3d, yte


def main():
    sel = _select()
    deep_ep = int(os.environ.get("DEEP_EPOCHS", "100"))
    out = Path(os.environ.get("OUT", "research/results/taskb_newlib.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            if l.strip():
                r = json.loads(l); done.add((r["dataset"], r["N_per_class"], r["seed"], r["method"]))
    print(f"sweep clfs={list(sel.values())} | resuming {len(done)} rows", flush=True)

    # 预热 TSFM 嵌入模型（一次性加载，强重试）：避免 trust_remote_code 在 offline 下逐 cell
    # 重新解析 revision 抖动 → 单次抖动只影响预热、不污染整库（成功则全 cell 复用全局缓存）。
    emb_clfs = [c for c in sel.values() if c in ("chronos2_emb", "timesfm_emb", "timer_emb")]
    if emb_clfs:
        import numpy as _np
        import time as _t
        from research.baseline import tsfm_embed as _TE   # 直调（绕过 _safe，能感知失败）
        warm = _np.asarray([_np.sin(_np.arange(128) * 0.1), _np.cos(_np.arange(128) * 0.1)],
                           dtype=_np.float32)
        for c in emb_clfs:
            raw = getattr(_TE, c)   # tsfm_embed.timesfm_emb 等，失败会抛
            ok = False
            for attempt in range(8):
                try:
                    raw(warm, _np.array([0, 1]), warm)
                    ok = True
                    print(f"  prewarm {c} OK (attempt {attempt+1})", flush=True)
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"  prewarm {c} attempt {attempt+1} fail: {type(e).__name__}: {str(e)[:80]}", flush=True)
                    _t.sleep(5)
            if not ok:
                print(f"  [ABORT] {c} 预热失败 8 次 → 跳过该 emb 以免 majority 污染", flush=True)
                sel = {m: cc for m, cc in sel.items() if cc != c}

    fh = out.open("a")
    ds_list = UCR + UEA
    if os.environ.get("DATASETS"):                     # 并行分片：DATASETS=逗号分隔子集
        want = set(os.environ["DATASETS"].split(","))
        ds_list = [d for d in ds_list if d in want]
        print(f"  [shard] datasets={ds_list}", flush=True)
    for ds in ds_list:
        for n in N_PER_CLASS:
            for seed in SEEDS:
                todo = {m: c for m, c in sel.items() if (ds, n, seed, m) not in done}
                if not todo:
                    continue
                try:
                    X2tr, X3tr, ytr, X2te, X3te, yte = _load(ds, n, seed)
                except Exception as e:
                    print(f"skip {ds} N={n} s={seed}: {e!r}", flush=True); continue
                for method, clf in todo.items():
                    native_mv = clf in MV.values()
                    Xtr_in, Xte_in = (X3tr, X3te) if native_mv else (X2tr, X2te)
                    kw = dict(LIGHT.get(clf, {}))
                    if clf in ("fcn", "resnet", "inceptiontime"):
                        kw["epochs"] = deep_ep
                    t0 = time.time()
                    try:
                        yp = predict_with(clf, Xtr_in, ytr, Xte_in, **kw)
                        acc = float((np.asarray(yp) == yte).mean())
                    except Exception as e:
                        print(f"  {ds} N={n} s={seed} {clf}: FAIL {e!r}", flush=True)
                        acc = float("nan")
                    fh.write(json.dumps({"dataset": ds, "N_per_class": n, "seed": seed,
                                         "method": method, "n_test": int(len(yte)),
                                         "acc": round(acc, 4), "wall_time": round(time.time() - t0, 2)},
                                        ensure_ascii=False) + "\n")
                    fh.flush()
                print(f"  {ds:28} N={n} s={seed} done ({len(todo)} clf)", flush=True)
    fh.close()
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
