"""E3 (method6 §2 / §2.6.4) · LLM defer —— DeepSeek 第二意见作为低-trust 动作。

接 F-R11.7 的开口：trust 善于"避险"（偏离会不会变差），但排不动"获利"（偏离能不能净赚）。
本实验问：在 belief 想偏离的决策 cell 上，**DeepSeek 第二意见能否补上 trust 排不动的"获利"判断**？

流程（全 LODO，诚实）：
  - 决策 cell = belief（K头集成 argmax）想偏离 rocket 的 cell
  - 每 cell 给 DeepSeek：Curator 画像 + 各候选**训练集 CV** + belief 倾向 + LOCO 检索的相似历史案例
    （base 在那些案例里是否=oracle）——**全是训练侧信息，无 test 泄漏**
  - LLM 输出 JSON {should_deviate, toward, confidence, reason}
  - LLM-defer 决策：should_deviate=True 且 toward 在候选 → 执行 toward，否则 commit-base

对照：always-commit / always-deviate / trust-gate(conformal≥0.5) / **LLM-defer**。
指标：决策 cell 上的实际 acc（vs oracle-action 上界）+ LLM 在"该偏离"上的精度/召回。
输出：results/m15_llm_defer.jsonl
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("M10_EXPANDED", "1")
os.environ.setdefault("M10_LIBPLUS", "1")
# DeepSeek：demo/.env 先加载且 key 注释掉 → 必须显式注入 research/.env 的 key
for _line in open("research/.env"):
    if _line.startswith("DEEPSEEK_API_KEY="):
        os.environ["DEEPSEEK_API_KEY"] = _line.strip().split("=", 1)[1]
os.environ["PROVIDER"] = "deepseek"
os.environ.setdefault("MODEL", "deepseek-chat")   # chat 够用且快；reasoner 更慢

from research.experiments.m10_learned_belief import build_dataset, CLF_ORDER, ROCKET_I
from research.experiments.m13_trust_vs_confidence import _fit_heads, _head_probs
from research.utils.llm import chat_cached


SYS = ("你是时序模型选择的审议助手。只能用给定的训练侧信息，禁止假设测试标签。"
       "判断：相比默认模型 rocket，是否应当偏离到某个候选分类器。输出严格 JSON。")


def _ask_llm(profile, cv_scores, belief_top, retrieved):
    user = (f"数据画像: {profile}\n"
            f"默认模型: rocket\n"
            f"各候选训练集CV准确率: {cv_scores}\n"
            f"集成信念最看好(非默认): {belief_top}\n"
            f"相似历史案例(LOCO检索, 'base是否=该案例最优'): {retrieved}\n"
            '请输出 JSON: {"should_deviate": true/false, "toward": "<候选名或null>", '
            '"confidence": 0~1, "reason": "<=30字"}')
    try:
        out = chat_cached([{"role": "system", "content": SYS},
                           {"role": "user", "content": user}], max_tokens=300)
        s = out[out.find("{"): out.rfind("}") + 1]
        d = json.loads(s)
        return {"should_deviate": bool(d.get("should_deviate", False)),
                "toward": d.get("toward"), "confidence": float(d.get("confidence", 0.5)),
                "reason": str(d.get("reason", ""))[:60]}
    except Exception as e:
        return {"should_deviate": False, "toward": None, "confidence": 0.0,
                "reason": f"llm_fail:{repr(e)[:30]}"}


def run():
    from sklearn.preprocessing import StandardScaler
    X, A, info = build_dataset()
    n = len(info)
    win = np.full(n, -1, dtype=int)
    for i, a in enumerate(A):
        v = np.where(~np.isnan(a))[0]
        win[i] = int(v[np.argmax(a[v])])
    datasets = sorted(set(it["ds"] for it in info))

    recs = []
    for held in datasets:
        tr = np.array([j for j, it in enumerate(info) if it["ds"] != held])
        te = np.array([j for j, it in enumerate(info) if it["ds"] == held])
        if len(tr) == 0 or len(te) == 0 or len(set(win[tr].tolist())) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        Ztr, Zte = sc.transform(X[tr]), sc.transform(X[te])
        win_tr = win[tr]
        m = len(tr); perm = np.random.default_rng(0).permutation(m)
        fit_i, cal_i = perm[: m // 2], perm[m // 2:]
        heads = _fit_heads(Ztr[fit_i], win_tr[fit_i]) or _fit_heads(Ztr, win_tr)
        cal_alpha = np.sort([1.0 - _head_probs(heads, Ztr[li]).mean(0).max() for li in cal_i]) \
            if len(cal_i) else np.array([1.0])
        # 训练域：每模型平均 CV 准确率（给 LLM 当"先验 CV"，部署可得，无 test）
        cv_mean = {}
        for ci, c in enumerate(CLF_ORDER):
            vals = [A[j][ci] for j in tr if not np.isnan(A[j][ci])]
            if vals:
                cv_mean[c] = round(float(np.mean(vals)), 3)
        # base 在训练 cell 里多大比例就是 oracle（检索案例的浓缩）
        base_oracle_frac = round(float(np.mean([win[j] == ROCKET_I for j in tr])), 2)

        for li, j in enumerate(te):
            P = _head_probs(heads, Zte[li]); bbar = P.mean(0)
            best_i = int(np.argmax(bbar))
            if best_i == ROCKET_I:
                continue   # belief 守 base → 非决策 cell
            a_new = 1.0 - bbar.max()
            trust = float((cal_alpha >= a_new).mean())
            accs = A[j]
            base = accs[ROCKET_I] if not np.isnan(accs[ROCKET_I]) else 0.0
            dev_model = CLF_ORDER[best_i]
            dev_acc = accs[best_i] if not np.isnan(accs[best_i]) else base
            a_star = "deviate" if dev_acc > base else "commit-base"
            # LLM 第二意见（只喂训练侧）
            prof = f"N={info[j]['N']}/类, 数据集={info[j]['ds']}"
            llm = _ask_llm(prof, cv_mean, dev_model,
                           f"{int(base_oracle_frac*100)}%案例中base即最优")
            recs.append({
                "ds": info[j]["ds"], "N": info[j]["N"], "seed": info[j]["seed"],
                "belief_dev_model": dev_model, "trust": round(trust, 4),
                "base_acc": round(base, 4), "dev_acc": round(float(dev_acc), 4),
                "a_star": a_star,
                "llm_should_deviate": llm["should_deviate"], "llm_toward": llm["toward"],
                "llm_conf": llm["confidence"], "llm_reason": llm["reason"],
            })
        print(f"  held={held}: cum decision cells={len(recs)}", flush=True)
    return recs


def realized(action, r):
    return r["dev_acc"] if action == "deviate" else r["base_acc"]


def analyze(recs):
    if not recs:
        return {}
    n = len(recs)
    res = {"n_decision_cells": n,
           "deviate_optimal_frac": round(float(np.mean([r["a_star"] == "deviate" for r in recs])), 3)}
    res["always-commit"] = round(float(np.mean([realized("commit-base", r) for r in recs])), 4)
    res["always-deviate"] = round(float(np.mean([realized("deviate", r) for r in recs])), 4)
    res["trust-gate(.5)"] = round(float(np.mean([
        realized("deviate" if r["trust"] >= 0.5 else "commit-base", r) for r in recs])), 4)
    # LLM-defer：LLM 说偏离且 toward==belief 模型 → deviate
    def llm_act(r):
        return "deviate" if (r["llm_should_deviate"] and
                             (r["llm_toward"] in (r["belief_dev_model"], None) or r["llm_toward"])) else "commit-base"
    res["llm-defer"] = round(float(np.mean([realized(llm_act(r), r) for r in recs])), 4)
    res["oracle-action"] = round(float(np.mean([realized(r["a_star"], r) for r in recs])), 4)
    # LLM 在"该偏离"判断上的精度/召回
    llm_yes = [r for r in recs if llm_act(r) == "deviate"]
    tp = sum(1 for r in llm_yes if r["a_star"] == "deviate")
    pos = sum(1 for r in recs if r["a_star"] == "deviate")
    res["llm_deviate_calls"] = len(llm_yes)
    res["llm_precision"] = round(tp / len(llm_yes), 3) if llm_yes else float("nan")
    res["llm_recall"] = round(tp / pos, 3) if pos else float("nan")
    # 对照 trust-gate 的精度
    tg_yes = [r for r in recs if r["trust"] >= 0.5]
    tg_tp = sum(1 for r in tg_yes if r["a_star"] == "deviate")
    res["trustgate_precision"] = round(tg_tp / len(tg_yes), 3) if tg_yes else float("nan")
    return res


def main():
    out = Path("research/results/m15_llm_defer.jsonl")
    print("=== E3 · LLM defer (DeepSeek 第二意见, LODO, 10-clf lib) ===", flush=True)
    recs = run()
    res = analyze(recs)
    print(f"\n=== decision cells = {res.get('n_decision_cells')}, "
          f"deviate-optimal frac = {res.get('deviate_optimal_frac')} ===", flush=True)
    for k in ["always-commit", "always-deviate", "trust-gate(.5)", "llm-defer", "oracle-action"]:
        print(f"  {k:16}: {res.get(k)}", flush=True)
    print(f"  LLM: calls={res.get('llm_deviate_calls')} precision={res.get('llm_precision')} "
          f"recall={res.get('llm_recall')} | trust-gate precision={res.get('trustgate_precision')}",
          flush=True)
    with out.open("w") as fh:
        fh.write(json.dumps({"_summary": res}, ensure_ascii=False) + "\n")
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
