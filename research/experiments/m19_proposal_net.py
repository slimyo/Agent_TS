"""E7 (method7 §4, #104) · Proposal Network vs LLM defer。

feedback_m6 方向4：E3 证 LLM 价值在"提偏离候选"(proposal)而非推理。那么学一个轻量
proposal net（z → 该偏离到哪个候选）是否能匹配/超过 LLM？AlphaGo 式：proposal→trust verify。

LODO，10-clf 库。决策 cell = belief 想偏离。对比"谁提的候选更可能真获利"：
  - belief-argmax（method4 老法）
  - proposal-net：LogisticRegression(z) → 预测 oracle-winner（非 base），取其为候选
  - （LLM defer 的精度引用 E3/F-R11.8 = 0.267，作对照锚点）
指标：各 proposer 提出的候选"真是 oracle / 真 ≥ base"的精度；proposal-net 是否 ≥ LLM。
输出：results/m19_proposal_net.jsonl
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
    from sklearn.linear_model import LogisticRegression

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
        win_tr = win[tr]
        heads = _fit_heads(Ztr, win_tr)
        # proposal net: 学 oracle-winner（含 base），推理时取"最高概率的非 base"作 proposal
        pnet = LogisticRegression(max_iter=2000, C=1.0).fit(Ztr, win_tr)
        classes = list(pnet.classes_)
        for li, j in enumerate(te):
            bb = _head_probs(heads, Zte[li]).mean(0)
            belief_i = int(np.argmax(bb))
            if belief_i == ROCKET_I:
                continue   # 非决策 cell
            # proposal net 提议
            pp = pnet.predict_proba(Zte[li].reshape(1, -1))[0]
            order = [classes[k] for k in np.argsort(-pp)]
            pnet_prop = next((c for c in order if c != ROCKET_I), belief_i)
            a = A[j]; base = a[ROCKET_I] if not np.isnan(a[ROCKET_I]) else 0.0
            oracle_i = win[j]
            def quality(idx):
                acc = a[idx] if not np.isnan(a[idx]) else base
                return {"is_oracle": int(idx == oracle_i), "beats_base": int(acc >= base),
                        "profits": int(acc > base)}
            recs.append({"ds": info[j]["ds"],
                         "belief": quality(belief_i),
                         "pnet": quality(int(pnet_prop))})
    return recs


def analyze(recs):
    n = len(recs)
    out = {"n_decision_cells": n}
    for who in ["belief", "pnet"]:
        out[f"{who}_is_oracle"] = round(float(np.mean([r[who]["is_oracle"] for r in recs])), 3)
        out[f"{who}_beats_base"] = round(float(np.mean([r[who]["beats_base"] for r in recs])), 3)
        out[f"{who}_profits"] = round(float(np.mean([r[who]["profits"] for r in recs])), 3)
    out["LLM_ref_profits_prec(E3/F-R11.8)"] = 0.267
    return out


def main():
    out = Path("research/results/m19_proposal_net.jsonl")
    print("=== E7 · Proposal Network vs LLM (LODO, 10-clf lib) ===", flush=True)
    recs = run()
    res = analyze(recs)
    for k in sorted(res):
        print(f"  {k:34}: {res[k]}", flush=True)
    with out.open("w") as fh:
        fh.write(json.dumps({"_summary": res}, ensure_ascii=False) + "\n")
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
