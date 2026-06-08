"""E1 (method6 §1A / §2.6.2) · Trust ≠ Confidence —— 攻击 F-R9.2 belief inversion。

核心问题：method4 实测 belief 强度是**反指标**（偏离错时 confidence 0.79 > 对时 0.51）。
本实验把"我觉得是X"(confidence)与"该不该信这次"(trust=1−epistemic)解耦，验证：

  按 trust 排序/门控后，"偏离对/错"的 confidence 能否从负相关**翻成正相关**？

三种 epistemic 估计（method6 §2.5.4，全 LODO，无 test 泄漏）：
  (a) MI      = H(b̄) − mean_j H(b^j)          （ensemble 互信息）
  (b) JS      = mean pairwise Jensen-Shannon   （ensemble 分歧）
  (c) conformal = 训练 split 上 winner 的 nonconformity 分位

主指标（method6 §2.6.5）：
  - Inversion-coef = corr(confidence, correctness)  over 偏离决策（目标：负→正）
  - Trust-AUC      = AUC(trust, correctness)         （trust 能否排序对错）
  - 经 trust 门控后偏离精度 / safe-deviation-rate 提升

复用 m10.build_dataset（z, per-clf acc A, info），UCR-22（M10_EXPANDED=1）默认。
输出：results/m13_trust_vs_confidence.jsonl
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("M10_EXPANDED", "1")   # 默认用 22 数据集（更多 cell 更稳）

from research.experiments.m10_learned_belief import build_dataset, CLF_ORDER, ROCKET_I

K_HEADS = 20
RNG0 = 0


# ───────────────────────── belief heads + epistemic ─────────────────────────

def _fit_heads(Z, win, K=K_HEADS, seed0=RNG0):
    from sklearn.linear_model import LogisticRegression
    heads = []
    rng = np.random.default_rng(seed0)
    n = len(Z)
    for _ in range(K):
        idx = rng.integers(0, n, n)
        if len(set(win[idx].tolist())) < 2:
            continue
        try:
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(Z[idx], win[idx])
            heads.append(clf)
        except Exception:
            pass
    return heads


def _head_probs(heads, z):
    """返回 [K, |CLF|] 每头在全候选上的 belief。"""
    P = np.zeros((len(heads), len(CLF_ORDER)))
    for j, h in enumerate(heads):
        p = h.predict_proba(z.reshape(1, -1))[0]
        for ci, c in enumerate(h.classes_):
            P[j, int(c)] = p[ci]
    return P


def _H(q):
    q = np.clip(q, 1e-12, 1.0)
    return float(-(q * np.log(q)).sum())


def _epistemic(P, kind):
    bbar = P.mean(0)
    if kind == "MI":
        return _H(bbar) - float(np.mean([_H(P[j]) for j in range(len(P))]))
    if kind == "JS":
        # mean pairwise JS divergence
        K = len(P)
        if K < 2:
            return 0.0
        tot, cnt = 0.0, 0
        for a in range(K):
            for b in range(a + 1, K):
                m = 0.5 * (P[a] + P[b])
                js = 0.5 * _kl(P[a], m) + 0.5 * _kl(P[b], m)
                tot += js; cnt += 1
        return tot / max(cnt, 1)
    raise ValueError(kind)


def _kl(p, q):
    p = np.clip(p, 1e-12, 1.0); q = np.clip(q, 1e-12, 1.0)
    return float((p * np.log(p / q)).sum())


# ───────────────────────── LODO 主流程 ─────────────────────────

def run(epi_kind="MI"):
    X, A, info = build_dataset()
    n = len(info)
    win = np.full(n, -1, dtype=int)
    for i, a in enumerate(A):
        v = np.where(~np.isnan(a))[0]
        win[i] = int(v[np.argmax(a[v])])
    datasets = sorted(set(it["ds"] for it in info))

    recs = []
    # conformal: 需要训练 split 上的 nonconformity 分布
    for held in datasets:
        tr = np.array([j for j, it in enumerate(info) if it["ds"] != held])
        te = np.array([j for j, it in enumerate(info) if it["ds"] == held])
        if len(tr) == 0 or len(te) == 0 or len(set(win[tr].tolist())) < 2:
            for j in te:
                recs.append(_rec(info[j], A[j], None, None, None))
            continue
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(X[tr])
        Ztr, Zte = sc.transform(X[tr]), sc.transform(X[te])   # 局部索引 0..len-1
        win_tr = win[tr]                                       # 与 Ztr 对齐
        # split tr（局部索引）→ fit / calib（conformal）
        m = len(tr)
        perm = np.random.default_rng(RNG0).permutation(m)
        fit_i, cal_i = perm[: m // 2], perm[m // 2:]
        heads = _fit_heads(Ztr[fit_i], win_tr[fit_i])
        if not heads:
            heads = _fit_heads(Ztr, win_tr)
        # 训练域 epistemic 分布（给 trust 归一）+ conformal nonconformity
        tr_epi = []
        for li in cal_i:
            P = _head_probs(heads, Ztr[li])
            tr_epi.append(_epistemic(P, epi_kind))
        tr_epi = np.array(tr_epi) if tr_epi else np.array([0.0])
        # conformal: α = 1 − b̄(true winner)
        cal_alpha = []
        for li in cal_i:
            P = _head_probs(heads, Ztr[li]); bbar = P.mean(0)
            cal_alpha.append(1.0 - bbar[win_tr[li]])
        cal_alpha = np.sort(np.array(cal_alpha)) if cal_alpha else np.array([1.0])

        def trust_from_epi(e):
            # rank-normalize against training epistemic → trust = 1 − rank
            r = float((tr_epi < e).mean())
            return 1.0 - r

        def trust_conformal(z):
            P = _head_probs(heads, z); bbar = P.mean(0)
            a_new = 1.0 - bbar.max()   # nonconformity of the *predicted* winner
            pval = float((cal_alpha >= a_new).mean())   # 高 pval = 更可能在已见分布内
            return pval

        for li, j in enumerate(te):            # li=局部行号, j=全局 info 下标
            P = _head_probs(heads, Zte[li])
            bbar = P.mean(0)
            e = _epistemic(P, epi_kind)
            trust = trust_from_epi(e)
            trust_c = trust_conformal(Zte[li])
            recs.append(_rec(info[j], A[j], bbar, trust, trust_c))
    return recs


def _rec(it, accs_row, bbar, trust, trust_c):
    accs = it["accs"]
    base = accs.get("rocket", 0.0)
    if bbar is None:
        chosen_i = ROCKET_I
        conf = 1.0
    else:
        chosen_i = int(np.argmax(bbar))
        conf = float(bbar.max())
    chosen = CLF_ORDER[chosen_i]
    deviated = chosen != "rocket"
    correct = bool(deviated and accs.get(chosen, 0.0) >= base)  # 偏离是否正确
    return {
        "ds": it["ds"], "N": it["N"], "seed": it["seed"],
        "chosen": chosen, "deviated": deviated,
        "confidence": round(conf, 4),
        "trust": round(trust, 4) if trust is not None else None,
        "trust_conformal": round(trust_c, 4) if trust_c is not None else None,
        "correct_deviation": correct,
        "chosen_acc": float(accs.get(chosen, 0.0)), "base_acc": float(base),
    }


# ───────────────────────── 指标 ─────────────────────────

def _corr(xs, ys):
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def _auc(scores, labels):
    # labels ∈ {0,1}; AUC = P(score_pos > score_neg)
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    wins = sum(1 for p in pos for q in neg if p > q) + 0.5 * sum(1 for p in pos for q in neg if p == q)
    return wins / (len(pos) * len(neg))


def analyze(recs, epi_kind):
    dev = [r for r in recs if r["deviated"]]
    out = {"epistemic": epi_kind, "n_cells": len(recs), "n_deviations": len(dev)}
    if dev:
        conf = [r["confidence"] for r in dev]
        corr = [1.0 if r["correct_deviation"] else 0.0 for r in dev]
        # F-R9.2 复现：raw confidence vs correctness（期望 ≤0，即 inversion）
        out["inversion_coef_raw"] = round(_corr(conf, corr), 3)
        # mean confidence when correct vs wrong（method4 报的 0.51 vs 0.79）
        cok = [c for c, k in zip(conf, corr) if k]
        cbad = [c for c, k in zip(conf, corr) if not k]
        out["conf_when_correct"] = round(float(np.mean(cok)), 3) if cok else None
        out["conf_when_wrong"] = round(float(np.mean(cbad)), 3) if cbad else None
        # trust 能否排序对错（核心：trust-AUC 应 >0.5）
        for tkey in ("trust", "trust_conformal"):
            tv = [r[tkey] for r in dev if r[tkey] is not None]
            tc = [1.0 if r["correct_deviation"] else 0.0 for r in dev if r[tkey] is not None]
            out[f"AUC_{tkey}"] = round(_auc(tv, tc), 3) if tv else None
        # trust-gated：只保留 trust 高的一半偏离，看偏离精度提升 + corrected inversion
        tv = np.array([r["trust"] for r in dev if r["trust"] is not None])
        if len(tv) >= 4:
            thr = float(np.median(tv))
            gated = [r for r in dev if r["trust"] is not None and r["trust"] >= thr]
            if gated:
                gc = [r["confidence"] for r in gated]
                gk = [1.0 if r["correct_deviation"] else 0.0 for r in gated]
                out["inversion_coef_trust_gated"] = round(_corr(gc, gk), 3)
                out["dev_precision_raw"] = round(np.mean(corr), 3)
                out["dev_precision_trust_gated"] = round(np.mean(gk), 3)
                out["n_dev_after_gate"] = len(gated)
    return out


def main():
    out = Path("research/results/m13_trust_vs_confidence.jsonl")
    print("=== E1 · Trust ≠ Confidence (LODO, UCR-22) ===", flush=True)
    summaries = []
    best_recs = None
    for epi in ["MI", "JS"]:   # conformal trust 在每条 rec 里单独带（trust_conformal）
        recs = run(epi_kind=epi)
        s = analyze(recs, epi)
        summaries.append(s)
        if epi == "MI":
            best_recs = recs
        print(f"\n[epistemic={epi}] dev={s['n_deviations']}/{s['n_cells']}", flush=True)
        print(f"  conf when correct={s.get('conf_when_correct')} wrong={s.get('conf_when_wrong')} "
              f"(F-R9.2: wrong>correct = inversion)", flush=True)
        print(f"  inversion_coef raw={s.get('inversion_coef_raw')} "
              f"→ trust_gated={s.get('inversion_coef_trust_gated')}", flush=True)
        print(f"  Trust-AUC(ensemble)={s.get('AUC_trust')} Trust-AUC(conformal)={s.get('AUC_trust_conformal')}", flush=True)
        print(f"  dev_precision raw={s.get('dev_precision_raw')} "
              f"→ trust_gated={s.get('dev_precision_trust_gated')} (n_after={s.get('n_dev_after_gate')})", flush=True)
    with out.open("w") as fh:
        for s in summaries:
            fh.write(json.dumps({"_summary": s}, ensure_ascii=False) + "\n")
        for r in (best_recs or []):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
