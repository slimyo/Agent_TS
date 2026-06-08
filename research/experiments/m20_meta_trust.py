"""E8 (method7 §5, #105) · Meta-trust + selective-action regret 曲线。

feedback_m6 方向3/5：当前 conformal 最好，但可探索 meta-trust = f(conformal, disagreement,
density, saturation) 是否更强；并画 selective-action 的 regret-vs-coverage 曲线（risk-coverage）。

LODO，10-clf 库。决策 cell = belief 想偏离。
  1. 四个 trust 源：conformal / ensemble-disagreement / feature-density(kNN) / saturation-gap
  2. meta-trust = logistic 融合（标签 = 偏离是否安全 a_star）
  3. 比各源 + meta 的 AUC（排"偏离会不会变差"）
  4. risk-coverage：按 trust 降序逐步放行偏离，画 cumulative safe-deviation-rate
输出：results/m20_meta_trust.jsonl
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("M10_EXPANDED", "1")
os.environ.setdefault("M10_LIBPLUS", "1")

from research.experiments.m10_learned_belief import build_dataset, CLF_ORDER, ROCKET_I
from research.experiments.m13_trust_vs_confidence import _fit_heads, _head_probs


def _auc(score, label):
    pos = [s for s, l in zip(score, label) if l]; neg = [s for s, l in zip(score, label) if not l]
    if not pos or not neg:
        return float("nan")
    w = sum(1 for p in pos for q in neg if p > q) + 0.5 * sum(1 for p in pos for q in neg if p == q)
    return w / (len(pos) * len(neg))


def run():
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LogisticRegression

    X, A, info = build_dataset()
    n = len(info)
    win = np.full(n, -1, dtype=int)
    gap = np.zeros(n)
    for i, a in enumerate(A):
        v = np.where(~np.isnan(a))[0]; win[i] = int(v[np.argmax(a[v])])
        base = a[ROCKET_I] if not np.isnan(a[ROCKET_I]) else 0.0
        gap[i] = a[v].max() - base
    datasets = sorted(set(it["ds"] for it in info))

    recs = []
    for held in datasets:
        tr = np.array([j for j, it in enumerate(info) if it["ds"] != held])
        te = np.array([j for j, it in enumerate(info) if it["ds"] == held])
        if len(tr) == 0 or len(te) == 0 or len(set(win[tr].tolist())) < 2:
            continue
        sc = StandardScaler().fit(X[tr]); Ztr, Zte = sc.transform(X[tr]), sc.transform(X[te])
        win_tr = win[tr]
        sat_reg = RandomForestRegressor(n_estimators=150, max_depth=4, random_state=0).fit(Ztr, gap[tr])
        m = len(tr); perm = np.random.default_rng(0).permutation(m)
        fit_i, cal_i = perm[: m // 2], perm[m // 2:]
        heads = _fit_heads(Ztr[fit_i], win_tr[fit_i]) or _fit_heads(Ztr, win_tr)
        cal_alpha = np.sort([1.0 - _head_probs(heads, Ztr[li]).mean(0).max() for li in cal_i]) \
            if len(cal_i) else np.array([1.0])

        def sources(z):
            P = _head_probs(heads, z); bbar = P.mean(0)
            conformal = float((cal_alpha >= (1.0 - bbar.max())).mean())
            disagree = -float(np.mean([np.std(P[:, k]) for k in range(P.shape[1])]))  # 越大越可信→取负std
            # density: 到训练点的平均 cos 相似（高=在已见区）
            d = -float(np.mean(np.linalg.norm(Ztr - z, axis=1)))
            satv = -float(sat_reg.predict(z.reshape(1, -1))[0])  # gap 小→饱和→该守base，作 trust 负向
            return conformal, disagree, d, satv, bbar

        # 训练 meta-trust：在 cal_i 上构 (4源, a_star)
        mx, my = [], []
        for li in cal_i:
            c, dis, d, sv, bb = sources(Ztr[li])
            bi = int(np.argmax(bb))
            if bi == ROCKET_I:
                continue
            base = A[tr[li]][ROCKET_I] if not np.isnan(A[tr[li]][ROCKET_I]) else 0.0
            dev = A[tr[li]][bi] if not np.isnan(A[tr[li]][bi]) else base
            mx.append([c, dis, d, sv]); my.append(1 if dev >= base else 0)
        meta = None
        if len(set(my)) == 2 and len(mx) >= 8:
            msc = StandardScaler().fit(mx)
            meta = (LogisticRegression(max_iter=1000).fit(msc.transform(mx), my), msc)

        for li, j in enumerate(te):
            c, dis, d, sv, bb = sources(Zte[li])
            bi = int(np.argmax(bb))
            if bi == ROCKET_I:
                continue
            base = A[j][ROCKET_I] if not np.isnan(A[j][ROCKET_I]) else 0.0
            dev = A[j][bi] if not np.isnan(A[j][bi]) else base
            mt = float(meta[0].predict_proba(meta[1].transform([[c, dis, d, sv]]))[0, 1]) if meta else 0.5
            recs.append({"ds": info[j]["ds"], "conformal": c, "disagree": dis, "density": d,
                         "sat": sv, "meta_trust": mt,
                         "safe": 1 if dev >= base else 0, "profit": 1 if dev > base else 0})
    return recs


def analyze(recs):
    out = {"n": len(recs)}
    safe = [r["safe"] for r in recs]; profit = [r["profit"] for r in recs]
    for src in ["conformal", "disagree", "density", "sat", "meta_trust"]:
        sc = [r[src] for r in recs]
        out[f"AUC_safe_{src}"] = round(_auc(sc, safe), 3)
        out[f"AUC_profit_{src}"] = round(_auc(sc, profit), 3)
    # risk-coverage：按 meta_trust 降序，cum safe rate at coverage 25/50/75%
    order = sorted(recs, key=lambda r: -r["meta_trust"])
    for cov in [0.25, 0.5, 0.75]:
        k = max(1, int(len(order) * cov))
        out[f"meta_safe@cov{int(cov*100)}"] = round(float(np.mean([r["safe"] for r in order[:k]])), 3)
    out["base_safe_rate"] = round(float(np.mean(safe)), 3)
    return out


def main():
    out = Path("research/results/m20_meta_trust.jsonl")
    print("=== E8 · Meta-trust + risk-coverage (LODO) ===", flush=True)
    recs = run()
    res = analyze(recs)
    for k in sorted(res):
        print(f"  {k:22}: {res[k]}", flush=True)
    with out.open("w") as fh:
        fh.write(json.dumps({"_summary": res}, ensure_ascii=False) + "\n")
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
