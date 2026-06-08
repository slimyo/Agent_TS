"""E11 (method7 §5, #105) · trust 跨域可迁移性 —— conformal trust 在域 A 标定后能否迁到域 B。

#105 未收口的一块：F-R12.5 已证 conformal 是最佳避险信号（同域 LODO AUC≈0.79）。本实验问**跨域**是否成立：
把 belief 头 + conformal 标定**只在源域**拟合，到**目标域**测"排序'偏离会不会变差'"的 Trust-AUC。
跨域 AUC 仍 >0.5 且接近同域 → trust 是**可迁移的避险信号**（method6 §2.5 理论界支撑）。

口径（signal_router 全量 22-clf 库 / 38 数据集；cell = belief 想偏离的决策 cell）：
  within_UCR / within_UEA : 同域留一数据集（基线，≈F-R12.5）
  cross_UCR_to_UEA / cross_UEA_to_UCR : 源域全部拟合 → 目标域全部测
  shuffle_ctrl : 目标标签打乱（AUC 应≈0.5）
输出：results/m23_trust_transfer.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test"))

import signal_router as S


def _auc(score, label):
    pos = [s for s, l in zip(score, label) if l]
    neg = [s for s, l in zip(score, label) if not l]
    if not pos or not neg:
        return float("nan")
    w = sum(1 for p in pos for q in neg if p > q) + 0.5 * sum(1 for p in pos for q in neg if p == q)
    return w / (len(pos) * len(neg))


def _prep():
    cells = S.load_oracle_library()
    keys = sorted(cells)
    CLF = S.CLF_ORDER; bi = S.ROCKET_I
    n = len(keys)
    acc = np.full((n, len(CLF)), np.nan)
    for i, k in enumerate(keys):
        for ci, c in enumerate(CLF):
            if c in cells[k]:
                acc[i, ci] = cells[k][c]
    win = np.array([int(np.nanargmax(acc[i])) for i in range(n)])
    X = np.array([[float(k[1]), float((~np.isnan(acc[i])).sum()), float(acc[i, bi]),
                   float(np.log1p(k[1]))] for i, k in enumerate(keys)])
    dom = np.array([S.domain_of(k[0]) for k in keys])
    ds = np.array([k[0] for k in keys])
    return X, acc, win, dom, ds, bi


def _fit_source(Xs, wins):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    sc = StandardScaler().fit(Xs)
    Zs = sc.transform(Xs)
    belief = LogisticRegression(max_iter=2000).fit(Zs, wins)
    bc = list(belief.classes_)
    # conformal 标定：源域一半算 nonconformity 分位
    m = len(Zs); perm = np.random.default_rng(0).permutation(m); cal = perm[m // 2:]
    P = belief.predict_proba(Zs[cal])
    alpha = sorted(1.0 - P[r].max() for r in range(len(cal))) or [1.0]
    return sc, belief, bc, np.array(alpha)


def _eval(sc, belief, bc, alpha, Xt, acc_t, bi):
    Zt = sc.transform(Xt); P = belief.predict_proba(Zt)
    trusts, labels = [], []
    for r in range(len(Zt)):
        p = P[r]; order = [bc[j] for j in np.argsort(-p)]
        prop = next((c for c in order if c != bi), bi)
        if prop == bi:
            continue
        trust = float((alpha >= (1.0 - float(p.max()))).mean())
        base = acc_t[r, bi] if not np.isnan(acc_t[r, bi]) else 0.0
        dev = acc_t[r, prop] if not np.isnan(acc_t[r, prop]) else base
        trusts.append(trust); labels.append(bool(dev >= base))
    return trusts, labels


def run():
    X, acc, win, dom, ds, bi = _prep()
    idx = {d: np.where(dom == d)[0] for d in ("UCR", "UEA")}
    print(f"cells: UCR={len(idx['UCR'])} UEA={len(idx['UEA'])}", flush=True)
    out = {}

    # within-domain LODO
    for d in ("UCR", "UEA"):
        sub = idx[d]; dss = sorted(set(ds[sub]))
        T, L = [], []
        for held in dss:
            tr = np.array([j for j in sub if ds[j] != held])
            te = np.array([j for j in sub if ds[j] == held])
            if len(tr) < 8 or len(te) == 0 or len(set(win[tr].tolist())) < 2:
                continue
            sc, bel, bc, al = _fit_source(X[tr], win[tr])
            t, l = _eval(sc, bel, bc, al, X[te], acc[te], bi)
            T += t; L += l
        out[f"within_{d}"] = {"auc": round(_auc(T, L), 3), "n": len(L),
                              "safe_rate": round(float(np.mean(L)), 3) if L else None}

    # cross-domain
    for src, tgt in [("UCR", "UEA"), ("UEA", "UCR")]:
        s, t = idx[src], idx[tgt]
        if len(set(win[s].tolist())) < 2:
            continue
        sc, bel, bc, al = _fit_source(X[s], win[s])
        T, L = _eval(sc, bel, bc, al, X[t], acc[t], bi)
        out[f"cross_{src}_to_{tgt}"] = {"auc": round(_auc(T, L), 3), "n": len(L),
                                        "safe_rate": round(float(np.mean(L)), 3) if L else None}

    # shuffle control (UCR->UEA labels shuffled)
    s, t = idx["UCR"], idx["UEA"]
    sc, bel, bc, al = _fit_source(X[s], win[s])
    T, L = _eval(sc, bel, bc, al, X[t], acc[t], bi)
    rng = np.random.default_rng(0); Ls = list(rng.permutation(np.array(L)))
    out["shuffle_ctrl_UCR_to_UEA"] = {"auc": round(_auc(T, Ls), 3), "n": len(Ls)}

    fp = ROOT / "research" / "results" / "m23_trust_transfer.jsonl"
    with fp.open("w") as fh:
        fh.write(json.dumps({"_summary": out}, ensure_ascii=False) + "\n")
    print("\n=== trust 跨域可迁移性 (Trust-AUC, avoid-harm) ===", flush=True)
    for k, v in out.items():
        print(f"  {k:26}: {v}", flush=True)
    print(f"\nwrote {fp}", flush=True)
    return out


if __name__ == "__main__":
    run()
