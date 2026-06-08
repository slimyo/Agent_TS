"""M11 · Learned belief routing on multivariate UEA（检验 saturation 是否 domain-invariant）。

F-R9.7：UCR 是 Rocket-饱和 benchmark，routing ceiling≈1.88pp，最优 router = 不 route。
reviewer/finish3 曾提"多变量 UEA DTW>Rocket"作为 expert-switching 候选。本实验在 **UEA-full
（14 数据集）** 上重做诚实 learned-belief LODO + 校准 gate，回答：

  saturation 是 UCR 特有，还是跨 univariate/multivariate domain-invariant？

候选 = {rocket, dtw, euclid}（UEA 现有 3 个多变量分类器；无 MOMENT 多变量）。
特征 z = 每通道 hand feature 跨通道 mean-pool（多变量适配 featurize_cell）。
诚实协议 = leave-one-dataset-out（同 M10）。

输出：results/m11_uea_belief.jsonl
"""
from __future__ import annotations

import json
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

from research.utils.uea_loader import load_uea_fewshot
from research.utils.series_features import extract_full_features, FEATURE_ORDER

METHOD_TO_CLF = {"B1_dtw": "dtw", "B2_euclid": "euclid", "B3_rocket": "rocket"}
CLF_ORDER = ["rocket", "dtw", "euclid"]
ROCKET_I = 0
SWEEP = "research/results/taskb_uea_full.jsonl"
NONMETA = [k for k in FEATURE_ORDER if not k.startswith("meta_")]


def _mv_featurize(X_cell, y):
    """多变量 cell [N,C,L] → 定长 z：每通道 hand feature 跨样本+通道 mean，加 meta。"""
    N, C, L = X_cell.shape
    feats = []
    for n in range(N):
        per_ch = []
        for c in range(C):
            f = extract_full_features(X_cell[n, c].astype(np.float64))
            per_ch.append([float(f.get(k, 0.0)) for k in NONMETA])
        feats.append(np.nanmean(np.array(per_ch), axis=0))
    avg = np.nanmean(np.array(feats), axis=0)
    nclass = len(set(y.tolist()))
    meta = [np.log1p(L), float(C), np.log1p(N / max(nclass, 1)),
            1.0 - max(Counter(y.tolist()).values()) / N, np.log1p(N)]
    z = np.concatenate([np.nan_to_num(avg), np.array(meta)])
    return z.astype(np.float32)


def build_dataset():
    by = defaultdict(dict)
    for l in open(SWEEP):
        r = json.loads(l)
        if r["method"] in METHOD_TO_CLF:
            by[(r["dataset"], r["N_per_class"], r["seed"])][METHOD_TO_CLF[r["method"]]] = r["acc"]
    X, A, info = [], [], []
    cache_path = Path("/tmp/m11_uea_z.npz")
    cache = {}
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        cache = {k: z[k] for k in z.files}
    new = dict(cache)
    for (ds, n, seed), accs in sorted(by.items()):
        if len(accs) < 2:
            continue
        key = f"{ds}_{n}_{seed}"
        if key in cache:
            z = cache[key]
        else:
            try:
                Xc, yc, _, _ = load_uea_fewshot(ds, n_per_class=n, seed=seed)
                z = _mv_featurize(Xc, yc)
                new[key] = z
                print(f"feat {key} (C={Xc.shape[1]},L={Xc.shape[2]})", flush=True)
            except Exception as e:
                print(f"skip {key}: {e!r}", flush=True)
                continue
        a = np.array([accs.get(c, np.nan) for c in CLF_ORDER], dtype=np.float64)
        X.append(z); A.append(a); info.append({"ds": ds, "N": n, "seed": seed, "accs": accs})
    if len(new) != len(cache):
        np.savez(cache_path, **new)
    return np.stack(X), np.stack(A), info


def lodo_belief(tau=0.0):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    X, A, info = build_dataset()
    win = np.array([int(np.where(~np.isnan(a))[0][np.argmax(a[~np.isnan(a)])]) for a in A])
    datasets = sorted(set(i["ds"] for i in info))
    recs = []
    for held in datasets:
        tr = [j for j, it in enumerate(info) if it["ds"] != held]
        te = [j for j, it in enumerate(info) if it["ds"] == held]
        if not te or not tr or len(set(win[tr].tolist())) < 2:
            for j in te:
                recs.append(_rec(info[j], "rocket", win[j]))
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(sc.transform(X[tr]), win[tr])
        proba = clf.predict_proba(sc.transform(X[te]))
        classes = list(clf.classes_)
        for row, j in enumerate(te):
            b = np.zeros(len(CLF_ORDER))
            for ci, c in enumerate(classes):
                b[c] = proba[row, ci]
            best = int(np.argmax(b))
            if tau > 0 and best != ROCKET_I and b[best] - b[ROCKET_I] <= tau:
                best = ROCKET_I
            recs.append(_rec(info[j], CLF_ORDER[best], win[j]))
    return recs


def _rec(it, chosen, oracle_idx):
    a = it["accs"]
    return {"ds": it["ds"], "N": it["N"], "seed": it["seed"], "chosen": chosen,
            "chosen_acc": float(a.get(chosen, 0.0)), "rocket_acc": float(a.get("rocket", 0.0)),
            "oracle_acc": float(max(a.values())), "oracle_clf": CLF_ORDER[oracle_idx],
            "deviated": chosen != "rocket",
            "correct_deviation": (chosen != "rocket") and a.get(chosen, 0.0) >= a.get("rocket", 0.0)}


def summ(recs, tau):
    n = len(recs)
    sel = sum(r["chosen_acc"] for r in recs) / n * 100
    roc = sum(r["rocket_acc"] for r in recs) / n * 100
    ora = sum(r["oracle_acc"] for r in recs) / n * 100
    dev = [r for r in recs if r["deviated"]]
    dok = [r for r in dev if r["correct_deviation"]]
    return {"tau": tau, "n": n, "belief_acc": round(sel, 2), "rocket_acc": round(roc, 2),
            "oracle_acc": round(ora, 2), "vs_rocket_pp": round(sel - roc, 2),
            "regret_pp": round(ora - sel, 2), "n_dev": len(dev),
            "dev_prec": round(len(dok) / len(dev), 3) if dev else float("nan")}


def main():
    out = Path("research/results/m11_uea_belief.jsonl")
    print("=== M11 UEA Learned Belief · LODO ===", flush=True)
    all_s, best = [], None
    for tau in [0.0, 0.1, 0.2, 0.3]:
        recs = lodo_belief(tau)
        s = summ(recs, tau); all_s.append(s)
        if tau == 0.0:
            best = recs
        print(f"tau={tau}: belief={s['belief_acc']}% vsRocket={s['vs_rocket_pp']:+}pp "
              f"oracle={s['oracle_acc']}% regret={s['regret_pp']}pp dev={s['n_dev']} prec={s['dev_prec']}",
              flush=True)
    # saturation descriptive
    _, A, info = build_dataset()
    win = Counter()
    ds_win = defaultdict(Counter)
    rk_is = 0
    for it in info:
        v = it["accs"]; b = max(v, key=v.get); win[b] += 1; ds_win[it["ds"]][b] += 1
        if abs(v.get("rocket", 0) - max(v.values())) < 1e-9:
            rk_is += 1
    nonrk = sorted(d for d, c in ds_win.items() if c.most_common(1)[0][0] != "rocket")
    print(f"\noracle dist: {dict(win)}", flush=True)
    print(f"rocket IS oracle: {rk_is}/{len(info)} = {rk_is/len(info)*100:.0f}%", flush=True)
    print(f"datasets rocket NOT top: {len(nonrk)}/{len(ds_win)} -> {nonrk}", flush=True)
    with out.open("w") as fh:
        for s in all_s:
            fh.write(json.dumps({"_summary": s}, ensure_ascii=False) + "\n")
        for r in best:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
