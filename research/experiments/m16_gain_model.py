"""E4 (method7 §1, #101) · Gain Model —— 补 F-R11.7 缺口（trust 只避险，gain 才获利）。

method6 证明 trust ≈ P(action safe) ≠ gain ≈ E[Δreward]。本实验直接建 gain model：
  gain(z, m) = acc(m) − acc(base)   （逐候选回归，LODO）
决策升级：deviate to m  iff  trust(m)≥τ_t  AND  pred_gain(z,m)≥τ_g。

对比四策略（10-clf un-saturated 库，LODO）：
  always-commit / trust-only / **trust+gain** / oracle-action
关键问句：trust+gain 能否把"获利"从 trust-only 的 ~0 提到可测正值？

复用 m10.build_dataset（z, per-clf acc）+ m13 的 heads/conformal-trust。
输出：results/m16_gain_model.jsonl
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


def run(tau_t=0.5, tau_g=0.0):
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor

    X, A, info = build_dataset()
    n = len(info)
    win = np.full(n, -1, dtype=int)
    for i, a in enumerate(A):
        v = np.where(~np.isnan(a))[0]
        win[i] = int(v[np.argmax(a[v])])
    datasets = sorted(set(it["ds"] for it in info))
    nC = len(CLF_ORDER)

    recs = []
    gain_pred_all, gain_true_all = [], []
    for held in datasets:
        tr = np.array([j for j, it in enumerate(info) if it["ds"] != held])
        te = np.array([j for j, it in enumerate(info) if it["ds"] == held])
        if len(tr) == 0 or len(te) == 0 or len(set(win[tr].tolist())) < 2:
            continue
        sc = StandardScaler().fit(X[tr]); Ztr, Zte = sc.transform(X[tr]), sc.transform(X[te])
        win_tr = win[tr]
        # per-候选 gain 回归：gain_m = acc_m − acc_base（只用训练 cell，标签来自历史 outcome）
        gain_reg = {}
        for ci, c in enumerate(CLF_ORDER):
            if ci == ROCKET_I:
                continue
            mask = ~np.isnan(A[tr][:, ci]) & ~np.isnan(A[tr][:, ROCKET_I])
            if mask.sum() < 8:
                continue
            g = A[tr][mask, ci] - A[tr][mask, ROCKET_I]
            gain_reg[c] = RandomForestRegressor(n_estimators=200, max_depth=4, random_state=0).fit(
                Ztr[mask], g)
        # belief heads + conformal trust
        m = len(tr); perm = np.random.default_rng(0).permutation(m)
        fit_i, cal_i = perm[: m // 2], perm[m // 2:]
        heads = _fit_heads(Ztr[fit_i], win_tr[fit_i]) or _fit_heads(Ztr, win_tr)
        cal_alpha = np.sort([1.0 - _head_probs(heads, Ztr[li]).mean(0).max() for li in cal_i]) \
            if len(cal_i) else np.array([1.0])

        for li, j in enumerate(te):
            P = _head_probs(heads, Zte[li]); bbar = P.mean(0)
            best_i = int(np.argmax(bbar))
            if best_i == ROCKET_I:
                continue   # belief 守 base，非决策 cell
            dev_m = CLF_ORDER[best_i]
            accs = A[j]; base = accs[ROCKET_I] if not np.isnan(accs[ROCKET_I]) else 0.0
            dev_acc = accs[best_i] if not np.isnan(accs[best_i]) else base
            true_gain = float(dev_acc - base)
            # trust
            trust = float((cal_alpha >= (1.0 - bbar.max())).mean())
            # pred gain for the belief-proposed model
            pg = float(gain_reg[dev_m].predict(Zte[li].reshape(1, -1))[0]) if dev_m in gain_reg else 0.0
            gain_pred_all.append(pg); gain_true_all.append(true_gain)
            recs.append({"ds": info[j]["ds"], "N": info[j]["N"], "seed": info[j]["seed"],
                         "dev_model": dev_m, "trust": round(trust, 4),
                         "pred_gain": round(pg, 4), "true_gain": round(true_gain, 4),
                         "base_acc": round(base, 4), "dev_acc": round(float(dev_acc), 4),
                         "a_star": "deviate" if true_gain > 0 else "commit-base"})
    return recs, np.array(gain_pred_all), np.array(gain_true_all)


def _realized(action, r):
    return r["dev_acc"] if action == "deviate" else r["base_acc"]


def _corr(x, y):
    if len(x) < 3 or len(set(x.tolist())) < 2 or len(set(y.tolist())) < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _auc(score, label):
    pos = [s for s, l in zip(score, label) if l]; neg = [s for s, l in zip(score, label) if not l]
    if not pos or not neg:
        return float("nan")
    w = sum(1 for p in pos for q in neg if p > q) + 0.5 * sum(1 for p in pos for q in neg if p == q)
    return w / (len(pos) * len(neg))


def analyze(recs, gp, gt):
    n = len(recs)
    out = {"n_decision_cells": n,
           "deviate_optimal_frac": round(float(np.mean([r["a_star"] == "deviate" for r in recs])), 3),
           "gain_pred_corr": round(_corr(gp, gt), 3),
           "gain_AUC": round(_auc(gp, [g > 0 for g in gt]), 3)}
    lab = [1 if r["a_star"] == "deviate" else 0 for r in recs]
    out["trust_AUC"] = round(_auc(np.array([r["trust"] for r in recs]), lab), 3)
    # 策略
    out["always-commit"] = round(float(np.mean([_realized("commit-base", r) for r in recs])), 4)
    out["always-deviate"] = round(float(np.mean([_realized("deviate", r) for r in recs])), 4)
    out["trust-only(.5)"] = round(float(np.mean([
        _realized("deviate" if r["trust"] >= 0.5 else "commit-base", r) for r in recs])), 4)
    # trust+gain 双门：扫 τ_g 选训练无关的固定阈值=0（pred_gain>0）
    for tg in [0.0, 0.01, 0.02]:
        acc = np.mean([_realized("deviate" if (r["trust"] >= 0.5 and r["pred_gain"] > tg) else "commit-base", r)
                       for r in recs])
        out[f"trust+gain(t.5,g{tg})"] = round(float(acc), 4)
    out["oracle-action"] = round(float(np.mean([_realized(r["a_star"], r) for r in recs])), 4)
    # 偏离精度
    def prec(pred_dev):
        yes = [r for r in recs if pred_dev(r)]
        return round(sum(r["a_star"] == "deviate" for r in yes) / len(yes), 3) if yes else float("nan"), len(yes)
    out["trust_only_prec"], out["trust_only_calls"] = prec(lambda r: r["trust"] >= 0.5)
    out["trust+gain_prec"], out["trust+gain_calls"] = prec(lambda r: r["trust"] >= 0.5 and r["pred_gain"] > 0)
    return out


def main():
    out = Path("research/results/m16_gain_model.jsonl")
    print("=== E4 · Gain Model (LODO, 10-clf lib) ===", flush=True)
    recs, gp, gt = run()
    res = analyze(recs, gp, gt)
    for k in ["n_decision_cells", "deviate_optimal_frac", "gain_pred_corr", "gain_AUC", "trust_AUC",
              "always-commit", "always-deviate", "trust-only(.5)",
              "trust+gain(t.5,g0.0)", "trust+gain(t.5,g0.01)", "trust+gain(t.5,g0.02)", "oracle-action",
              "trust_only_prec", "trust_only_calls", "trust+gain_prec", "trust+gain_calls"]:
        print(f"  {k:22}: {res.get(k)}", flush=True)
    with out.open("w") as fh:
        fh.write(json.dumps({"_summary": res}, ensure_ascii=False) + "\n")
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
