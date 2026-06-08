"""M10 · Learned Belief Model（feedback_m3 一致 #1 缺陷：belief 不是学出来的）。

feedback_m3 三位 reviewer 一致结论：b(M)=softmax(−E) 的 E 全是人工 factor 之和，
是 "interpretable scoring function" 而非 "learned belief state"。M3 学 strength、
M8 学 attribution 都只在调 factor 权重，没改根本。

M10 直接学 belief：
    z = featurize_cell(X_train)            (30-dim 手特征，CPU，无 LLM/GPU)
    b(M|z) = softmax head over classifiers (logistic / gradient-boosting)
    训练目标 = cross-entropy 到 oracle-winner one-hot（reviewer 明确建议）
    路由 = argmax b(M|z)，或 confidence-gated（仅 belief margin>τ 才偏离 rocket）

诚实性：**leave-one-dataset-out (LODO)** —— held-out 数据集的 test acc 不进训练，
        无 per-cell 泄漏（对照 M9 的 leave-one-cell-out）。监督信号 = 历史 cell 的
        per-classifier test acc（部署时这是"已积累的离线经验库"，合法）。

同时输出 feedback_m3 缺陷 2/3 的分析料：
    - belief 形状特征：entropy / gini / top1-top2 gap（缺陷2 决策深化的输入）
    - belief miscalibration：偏离 rocket 且错时，belief 强度多高（缺陷3 "自信地犯错"）

对照：
    - Rocket-alone（B3）
    - 因子 router 诚实版（results/taskb_router_v3_honest_ucr.jsonl，仅 UCR-5 重叠 cell）
    - Oracle（per-cell 最优）

输出：results/m10_learned_belief.jsonl + 控制台 summary
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from research.utils.series_features import featurize_cell
from research.utils.ucr_loader import load_ucr_fewshot

import os as _os
METHOD_TO_CLF = {
    "B1_dtw": "dtw_1nn", "B2_euclid": "euclid_1nn", "B3_rocket": "rocket",
    "B4a_moment_1nn": "moment_1nn", "B4b_moment_lr": "moment_logreg",
}
CLF_ORDER = ["rocket", "moment_1nn", "moment_logreg", "dtw_1nn", "euclid_1nn"]
# M10_LIBPLUS=1：候选库扩到 10 个分类器（+catch22/mantis_1nn/mantis_lr/minirocket/weasel）
if _os.environ.get("M10_LIBPLUS", "") == "1":
    METHOD_TO_CLF.update({
        "B7_catch22": "catch22", "B8_mantis_1nn": "mantis_1nn", "B9_mantis_lr": "mantis_lr",
        "B10_minirocket": "minirocket", "B11_weasel": "weasel",
    })
    CLF_ORDER = CLF_ORDER + ["catch22", "mantis_1nn", "mantis_lr", "minirocket", "weasel"]
ROCKET_I = CLF_ORDER.index("rocket")
# #89: 设 M10_EXPANDED=1 把 oracle 库从 10 → 22 数据集（验证 F-R9.6 样本复杂度假说）
SWEEPS = ["research/results/taskb_ucr.jsonl",
          "research/results/taskb_extended_ucr.jsonl"]
if _os.environ.get("M10_EXPANDED", "") == "1":
    SWEEPS = SWEEPS + ["research/results/taskb_expand_ucr.jsonl"]
if _os.environ.get("M10_LIBPLUS", "") == "1":
    SWEEPS = SWEEPS + ["research/results/taskb_libplus_ucr.jsonl"]


def build_dataset():
    """返回 X(z), A(per-clf test acc, nan=missing), info[]。"""
    by_cell = defaultdict(dict)
    for fp in SWEEPS:
        if not Path(fp).exists():
            continue
        for line in open(fp):
            r = json.loads(line)
            if r["method"] not in METHOD_TO_CLF:
                continue
            by_cell[(r["dataset"], r["N_per_class"], r["seed"])][METHOD_TO_CLF[r["method"]]] = r["acc"]
    X, A, info = [], [], []
    for (ds, n, seed), accs in sorted(by_cell.items()):
        if len(accs) < 3:
            continue
        try:
            Xtr, ytr, _, _ = load_ucr_fewshot(ds, n_per_class=n, seed=seed)
        except Exception as e:
            print(f"skip {ds} {n} {seed}: {e!r}", flush=True)
            continue
        z = featurize_cell(Xtr, ytr)
        a = np.array([accs.get(c, np.nan) for c in CLF_ORDER], dtype=np.float64)
        X.append(z); A.append(a); info.append({"ds": ds, "N": n, "seed": seed, "accs": accs})
    return np.stack(X), np.stack(A), info


def _shape_feats(b: np.ndarray) -> dict:
    """belief 分布形状（缺陷2 决策深化的输入特征）。"""
    p = np.clip(b, 1e-12, 1.0); p = p / p.sum()
    ent = float(-(p * np.log(p)).sum())
    srt = np.sort(p)[::-1]
    gini = float(1.0 - (p ** 2).sum())  # 1 - Simpson; 高=分散
    top_gap = float(srt[0] - srt[1]) if len(srt) > 1 else float(srt[0])
    tail = float(srt[2:].sum()) if len(srt) > 2 else 0.0
    return {"entropy": ent, "gini": gini, "top1_top2_gap": top_gap, "tail_mass": tail}


def lodo_eval(tau: float = 0.0):
    """LODO：每次留一个数据集，用其余训练 belief head，在留出集上路由。
    tau>0 → confidence-gated（仅 belief(best)−belief(rocket)>τ 才偏离 rocket）。
    返回逐 cell 记录 + 汇总。"""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    X, A, info = build_dataset()
    datasets = sorted(set(i["ds"] for i in info))
    # oracle winner index per cell（只在已评测的 clf 里取 argmax）
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
            # degenerate: training winners all one class → belief = that class
            for j in te:
                forced = classes[0]
                records.append(_route_record(info[j], A[j], None, forced, tau, win[j]))
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0, multi_class="multinomial")
        clf.fit(Xtr_z, ytr)
        proba = clf.predict_proba(Xte_z)  # [te, len(classes)]
        # map to full 5-dim belief
        for row, j in enumerate(te):
            b = np.zeros(len(CLF_ORDER))
            for ci, cls in enumerate(classes):
                b[cls] = proba[row, ci]
            records.append(_route_record(info[j], A[j], b, None, tau, win[j]))
    return records, summarize(records, tau)


def _route_record(it, accs_row, belief, forced_idx, tau, oracle_idx):
    accs = it["accs"]
    if belief is None:
        chosen_i = forced_idx
        b = np.zeros(len(CLF_ORDER)); b[forced_idx] = 1.0
    else:
        b = belief
        # confidence-gated: deviate from rocket only if belief margin > tau
        best_i = int(np.argmax(b))
        if tau > 0 and best_i != ROCKET_I:
            if b[best_i] - b[ROCKET_I] <= tau:
                best_i = ROCKET_I
        chosen_i = best_i
    chosen = CLF_ORDER[chosen_i]
    shape = _shape_feats(b)
    return {
        "ds": it["ds"], "N": it["N"], "seed": it["seed"],
        "chosen": chosen,
        "chosen_acc": float(accs.get(chosen, 0.0)),
        "rocket_acc": float(accs.get("rocket", 0.0)),
        "oracle_acc": float(max(accs.values())),
        "oracle_clf": CLF_ORDER[oracle_idx] if oracle_idx >= 0 else None,
        "belief_chosen": float(b[chosen_i]),
        "belief_rocket": float(b[ROCKET_I]),
        "deviated": chosen != "rocket",
        "correct_deviation": (chosen != "rocket") and (accs.get(chosen, 0.0) >= accs.get("rocket", 0.0)),
        **{f"shape_{k}": v for k, v in shape.items()},
    }


def summarize(records, tau):
    n = len(records)
    sel = sum(r["chosen_acc"] for r in records) / n * 100
    roc = sum(r["rocket_acc"] for r in records) / n * 100
    ora = sum(r["oracle_acc"] for r in records) / n * 100
    dev = [r for r in records if r["deviated"]]
    dev_ok = [r for r in dev if r["correct_deviation"]]
    # belief miscalibration（缺陷3）：偏离且错 vs 偏离且对，平均 belief 强度
    dev_bad = [r for r in dev if not r["correct_deviation"]]
    bel_ok = (sum(r["belief_chosen"] for r in dev_ok) / len(dev_ok)) if dev_ok else float("nan")
    bel_bad = (sum(r["belief_chosen"] for r in dev_bad) / len(dev_bad)) if dev_bad else float("nan")
    return {
        "tau": tau, "n": n,
        "learned_belief_acc": round(sel, 2),
        "rocket_acc": round(roc, 2),
        "oracle_acc": round(ora, 2),
        "vs_rocket_pp": round(sel - roc, 2),
        "regret_to_oracle_pp": round(ora - sel, 2),
        "n_deviations": len(dev),
        "deviation_precision": round(len(dev_ok) / len(dev), 3) if dev else float("nan"),
        "belief_when_dev_correct": round(bel_ok, 3),
        "belief_when_dev_wrong": round(bel_bad, 3),
    }


def main():
    out = Path("research/results/m10_learned_belief.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    print("=== M10 Learned Belief · LODO across UCR-10 ===", flush=True)
    all_summ = []
    best_records = None
    for tau in [0.0, 0.1, 0.2, 0.3]:
        records, summ = lodo_eval(tau=tau)
        all_summ.append(summ)
        if tau == 0.0:
            best_records = records
        print(f"tau={tau}: learned_belief={summ['learned_belief_acc']}% "
              f"(vs Rocket {summ['rocket_acc']}% = {summ['vs_rocket_pp']:+}pp) "
              f"oracle={summ['oracle_acc']}% regret={summ['regret_to_oracle_pp']}pp "
              f"| dev={summ['n_deviations']} prec={summ['deviation_precision']} "
              f"| belief(dev_ok)={summ['belief_when_dev_correct']} "
              f"belief(dev_bad)={summ['belief_when_dev_wrong']}", flush=True)
    # persist tau=0 per-cell records + all summaries
    with out.open("w") as fh:
        for s in all_summ:
            fh.write(json.dumps({"_summary": s}, ensure_ascii=False) + "\n")
        for r in best_records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
