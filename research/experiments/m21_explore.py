"""E9 (method7 §6, #106) · 探索性偏离 —— "未知未知"区的信息增益。

feedback_m6 方向3（reviewer3）：当所有模型 trust 高但集体错（unknown-unknown），
任何"利用"型预测都无法获利；此时主动选低置信/最不同的候选去执行，价值在**信息增益**。

离线复盘（无在线反馈，纯诊断，10-clf 库）：
  1. 量化"未知未知"cell：belief 高置信(max≥0.5) 但 base 与 belief-top 都 ≠ oracle 的比例
  2. 反事实：这些 cell 上，"探索"(选 belief 最低的候选) vs commit-base，谁更接近 oracle？
  3. 信息价值代理：探索候选的 acc 是否覆盖了 base/belief-top 漏掉的 oracle？
输出：results/m21_explore.jsonl（诊断为主，判断"探索"是否值得作为独立动作）
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


def run():
    from sklearn.preprocessing import StandardScaler
    X, A, info = build_dataset()
    n = len(info)
    win = np.full(n, -1, dtype=int)
    for i, a in enumerate(A):
        v = np.where(~np.isnan(a))[0]; win[i] = int(v[np.argmax(a[v])])
    datasets = sorted(set(it["ds"] for it in info))

    recs = []
    for held in datasets:
        tr = np.array([j for j, it in enumerate(info) if it["ds"] != held])
        te = np.array([j for j, it in enumerate(info) if it["ds"] == held])
        if len(tr) == 0 or len(te) == 0 or len(set(win[tr].tolist())) < 2:
            continue
        sc = StandardScaler().fit(X[tr]); Ztr, Zte = sc.transform(X[tr]), sc.transform(X[te])
        heads = _fit_heads(Ztr, win[tr])
        if not heads:
            continue
        for li, j in enumerate(te):
            bb = _head_probs(heads, Zte[li]).mean(0)
            top_i = int(np.argmax(bb))
            low_i = int(np.argmin(bb))     # belief 最不看好的候选 = "探索"对象
            a = A[j]; valid = np.where(~np.isnan(a))[0]
            base = a[ROCKET_I] if not np.isnan(a[ROCKET_I]) else 0.0
            oracle = float(a[valid].max())
            top_acc = a[top_i] if not np.isnan(a[top_i]) else base
            low_acc = a[low_i] if not np.isnan(a[low_i]) else base
            # unknown-unknown：belief 高置信但 base 与 top 都不是 oracle
            uu = (bb.max() >= 0.5) and (abs(base - oracle) > 1e-9) and (abs(top_acc - oracle) > 1e-9)
            # 探索候选是否恰是 oracle（信息增益的最强情形）
            explore_is_oracle = abs(low_acc - oracle) < 1e-9
            recs.append({"ds": info[j]["ds"], "conf": round(float(bb.max()), 3),
                         "base": round(base, 4), "top": round(float(top_acc), 4),
                         "explore": round(float(low_acc), 4), "oracle": round(oracle, 4),
                         "unknown_unknown": bool(uu),
                         "explore_is_oracle": bool(explore_is_oracle)})
    return recs


def analyze(recs):
    n = len(recs)
    uu = [r for r in recs if r["unknown_unknown"]]
    out = {"n": n, "unknown_unknown_cells": len(uu),
           "uu_frac": round(len(uu) / n, 3) if n else 0.0}
    if uu:
        out["uu_base_acc"] = round(float(np.mean([r["base"] for r in uu])), 4)
        out["uu_oracle_acc"] = round(float(np.mean([r["oracle"] for r in uu])), 4)
        out["uu_explore_acc"] = round(float(np.mean([r["explore"] for r in uu])), 4)
        out["uu_top_acc"] = round(float(np.mean([r["top"] for r in uu])), 4)
        out["explore_is_oracle_frac"] = round(float(np.mean([r["explore_is_oracle"] for r in uu])), 3)
        # 探索是否优于 base（在 uu 区）
        out["explore_beats_base_frac"] = round(float(np.mean([r["explore"] > r["base"] for r in uu])), 3)
    return out


def main():
    out = Path("research/results/m21_explore.jsonl")
    print("=== E9 · Exploratory deviation (unknown-unknown, LODO) ===", flush=True)
    recs = run()
    res = analyze(recs)
    for k in sorted(res):
        print(f"  {k:24}: {res[k]}", flush=True)
    with out.open("w") as fh:
        fh.write(json.dumps({"_summary": res}, ensure_ascii=False) + "\n")
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
