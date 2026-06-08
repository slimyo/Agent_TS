"""#86b · M10 with MOMENT embedding z（检验 F-R9.4 表示瓶颈假说）。

F-R9.4：M10 用 30-d 手特征 z，learned belief 输 Rocket（−0.8pp）。本实验把 z 换成
MOMENT-1-small frozen encoder 的 cell-level embedding（mean-pool over training samples, 512-d），
其余完全照 m10_learned_belief 的 LODO 协议，对比"表示升级"是否解锁正向 routing。

诚实性：leave-one-dataset-out 不变。MOMENT 是 frozen（不 fine-tune），cell z = mean_n embed(x_n)。
缓存：embedding 慢，cache 到 /tmp/m10b_moment_z.npz（key=ds_N_seed）。

输出：results/m10b_moment_belief.jsonl
对照基线：m10_learned_belief（hand-feature z）+ Rocket + Oracle。
"""
from __future__ import annotations

import json
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

from research.utils.ucr_loader import load_ucr_fewshot
from research.experiments.m10_learned_belief import (
    METHOD_TO_CLF, CLF_ORDER, ROCKET_I, SWEEPS, _shape_feats, _route_record, summarize,
)

CACHE = Path("/tmp/m10b_moment_z.npz")


def _cell_moment_z(X_train):
    """cell-level MOMENT z = mean-pool of per-sample 512-d embeddings。"""
    from research.baseline.moment_classifier import embed as _moment_embed
    emb = _moment_embed(X_train)                 # [N, 512]
    return emb.mean(axis=0).astype(np.float32)   # [512]


def build_dataset_moment():
    by_cell = defaultdict(dict)
    for fp in SWEEPS:
        if not Path(fp).exists():
            continue
        for line in open(fp):
            r = json.loads(line)
            if r["method"] not in METHOD_TO_CLF:
                continue
            by_cell[(r["dataset"], r["N_per_class"], r["seed"])][METHOD_TO_CLF[r["method"]]] = r["acc"]

    cache = {}
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        cache = {k: z[k] for k in z.files}

    X, A, info = [], [], []
    new_cache = dict(cache)
    for (ds, n, seed), accs in sorted(by_cell.items()):
        if len(accs) < 3:
            continue
        key = f"{ds}_{n}_{seed}"
        if key in cache:
            z = cache[key]
        else:
            try:
                Xtr, ytr, _, _ = load_ucr_fewshot(ds, n_per_class=n, seed=seed)
                z = _cell_moment_z(Xtr)
                new_cache[key] = z
                print(f"embedded {key} ({len(Xtr)} samples)", flush=True)
            except Exception as e:
                print(f"skip {key}: {e!r}", flush=True)
                continue
        a = np.array([accs.get(c, np.nan) for c in CLF_ORDER], dtype=np.float64)
        X.append(z); A.append(a); info.append({"ds": ds, "N": n, "seed": seed, "accs": accs})
    if len(new_cache) != len(cache):
        np.savez(CACHE, **new_cache)
        print(f"cached {len(new_cache)} embeddings → {CACHE}", flush=True)
    return np.stack(X), np.stack(A), info


def lodo_eval_moment(tau=0.0):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    X, A, info = build_dataset_moment()
    datasets = sorted(set(i["ds"] for i in info))
    win = np.full(len(info), -1, dtype=int)
    for i, a in enumerate(A):
        valid = np.where(~np.isnan(a))[0]
        win[i] = int(valid[np.argmax(a[valid])])

    records = []
    for held in datasets:
        tr = np.array([j for j, it in enumerate(info) if it["ds"] != held])
        te = np.array([j for j, it in enumerate(info) if it["ds"] == held])
        if len(te) == 0 or len(tr) == 0:
            continue
        scaler = StandardScaler().fit(X[tr])
        Xtr_z, Xte_z = scaler.transform(X[tr]), scaler.transform(X[te])
        ytr = win[tr]
        classes = sorted(set(ytr.tolist()))
        if len(classes) < 2:
            for j in te:
                records.append(_route_record(info[j], A[j], None, classes[0], tau, win[j]))
            continue
        clf = LogisticRegression(max_iter=3000, C=1.0)
        clf.fit(Xtr_z, ytr)
        proba = clf.predict_proba(Xte_z)
        for row, j in enumerate(te):
            b = np.zeros(len(CLF_ORDER))
            for ci, cls in enumerate(classes):
                b[cls] = proba[row, ci]
            records.append(_route_record(info[j], A[j], b, None, tau, win[j]))
    return records, summarize(records, tau)


def main():
    out = Path("research/results/m10b_moment_belief.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    print("=== M10b MOMENT-embedding Learned Belief · LODO UCR-10 ===", flush=True)
    all_summ, best = [], None
    for tau in [0.0, 0.1, 0.2, 0.3]:
        records, summ = lodo_eval_moment(tau=tau)
        all_summ.append(summ)
        if tau == 0.0:
            best = records
        print(f"tau={tau}: moment_belief={summ['learned_belief_acc']}% "
              f"(vs Rocket {summ['rocket_acc']}% = {summ['vs_rocket_pp']:+}pp) "
              f"oracle={summ['oracle_acc']}% regret={summ['regret_to_oracle_pp']}pp "
              f"| dev={summ['n_deviations']} prec={summ['deviation_precision']} "
              f"| belief(ok)={summ['belief_when_dev_correct']} bad={summ['belief_when_dev_wrong']}",
              flush=True)
    with out.open("w") as fh:
        for s in all_summ:
            fh.write(json.dumps({"_summary": s}, ensure_ascii=False) + "\n")
        for r in best:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    # per-dataset breakdown for tau=0
    byds = defaultdict(list)
    for r in best:
        byds[r["ds"]].append(r)
    print("\n=== per-dataset (tau=0) ===", flush=True)
    for ds in sorted(byds):
        rs = byds[ds]; nn = len(rs)
        roc = sum(x["rocket_acc"] for x in rs) / nn * 100
        lrn = sum(x["chosen_acc"] for x in rs) / nn * 100
        ora = sum(x["oracle_acc"] for x in rs) / nn * 100
        print(f"  {ds:12} n={nn} rocket={roc:.1f} moment_belief={lrn:.1f} "
              f"oracle={ora:.1f} ({lrn-roc:+.1f})", flush=True)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
