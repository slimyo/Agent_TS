"""E6 (method7 §3, #103) · Portfolio —— "选一个专家" → "组合多个专家"。

feedback_m6 方向5（最有论文潜力）：per-cell 选 winner 不可学（E4/E5），换投资组合视角——
不选一个，而是给候选配权重。问句：分散化能否比"恒用 base"稳定更好？

实现（快速版，复用 m10 oracle 库，无逐 cell 重算）：用每 cell 已知的 per-classifier acc，
把"组合"近似为**加权 acc 期望**（凸组合上界——真实硬投票通常略低，但足以判断方向）：
  portfolio_acc(cell) = Σ_m w_m · acc_m(cell)
权重 w 由**训练域**均值 acc 决定（LODO，无 test）：
  - equal-topk：训练域 top-k 等权
  - gain-weighted：训练域均值 acc 的 softmax
对比 always-base / oracle-single。

注：加权 acc 是软组合的**线性近似/上界**；若它都赢不了 base，硬投票更赢不了 → 足以否证。
输出：results/m18_portfolio.jsonl
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("M10_EXPANDED", "1")
os.environ.setdefault("M10_LIBPLUS", "1")

from research.experiments.m10_learned_belief import build_dataset, CLF_ORDER, ROCKET_I


def run(topk=3):
    X, A, info = build_dataset()              # A: [n, |CLF|] per-cell acc（nan=缺）
    datasets = sorted(set(it["ds"] for it in info))
    rows = []
    for held in datasets:
        tr = [i for i, it in enumerate(info) if it["ds"] != held]
        te = [i for i, it in enumerate(info) if it["ds"] == held]
        if not tr or not te:
            continue
        mean_acc = {}
        for ci, c in enumerate(CLF_ORDER):
            vals = [A[i][ci] for i in tr if not np.isnan(A[i][ci])]
            if vals:
                mean_acc[c] = float(np.mean(vals))
        if "rocket" not in mean_acc:
            continue
        ks = list(mean_acc); arr = np.array([mean_acc[k] for k in ks])
        w = np.exp(15 * (arr - arr.max())); w = w / w.sum()
        gw = dict(zip(ks, w))                                # gain-weighted softmax
        topk_models = sorted(mean_acc, key=mean_acc.get, reverse=True)[:topk]
        eqw = {m: 1.0 / len(topk_models) for m in topk_models}   # equal top-k

        for i in te:
            a = A[i]
            accs = {c: a[ci] for ci, c in enumerate(CLF_ORDER) if not np.isnan(a[ci])}
            if "rocket" not in accs:
                continue
            base = accs["rocket"]
            oracle = max(accs.values())
            def port(weights):
                num = sum(weights[m] * accs[m] for m in weights if m in accs)
                den = sum(weights[m] for m in weights if m in accs)
                return num / den if den > 0 else base
            rows.append({"ds": info[i]["ds"], "N": info[i]["N"], "seed": info[i]["seed"],
                         "base": round(base, 4), "equal_topk": round(port(eqw), 4),
                         "gain_weighted": round(port(gw), 4), "oracle_single": round(oracle, 4)})
    return rows


def analyze(rows):
    n = len(rows)
    f = lambda k: round(float(np.mean([r[k] for r in rows])) * 100, 2)
    return {"n_cells": n, "base": f("base"), "equal_topk": f("equal_topk"),
            "gain_weighted": f("gain_weighted"), "oracle_single": f("oracle_single"),
            "equal_vs_base": round(f("equal_topk") - f("base"), 2),
            "gainw_vs_base": round(f("gain_weighted") - f("base"), 2),
            "note": "portfolio_acc = 加权 acc 期望（软组合线性上界）；< base 则硬投票更不可能赢"}


def main():
    out = Path("research/results/m18_portfolio.jsonl")
    print("=== E6 · Portfolio (库内加权-acc 近似, LODO, 10-clf) ===", flush=True)
    rows = run(topk=3)
    res = analyze(rows)
    for k in ["n_cells", "base", "equal_topk", "gain_weighted", "oracle_single",
              "equal_vs_base", "gainw_vs_base", "note"]:
        print(f"  {k:16}: {res.get(k)}", flush=True)
    with out.open("w") as fh:
        fh.write(json.dumps({"_summary": res}, ensure_ascii=False) + "\n")
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
