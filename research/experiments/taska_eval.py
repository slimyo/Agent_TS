"""P6.1 · TaskA RCA 评估指标 R1-R5（plan §16.3.1）。

R1 Top-1 accuracy:   pred.primary_fault == gt.primary_fault
R2 Top-3 inclusion:  gt.primary_fault in (pred.primary_fault + pred.secondary)
R3 LLM-as-Judge:     待人工或 GPT-4 评估（此脚本预留接口）
R4 关键词 F1:        在 pred.evidence 文本中提取 fault-related 关键词 vs gt
R5 人工 Cohen's κ:   多标注者一致性（不在此脚本计算）
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


# 关键词模板（每 fault 对应的诊断 vocabulary）
FAULT_KEYWORDS = {
    "trend_break":      ["阶跃", "断裂", "shift", "break", "突变", "mean shift", "趋势变化"],
    "seasonal_flip":    ["季节", "翻转", "周期", "seasonal", "ACF", "lag", "周期消失", "周期翻转"],
    "variance_explode": ["方差", "variance", "std", "波动", "explode", "ratio", "方差爆炸"],
    "outlier_burst":    ["离群", "outlier", "异常点", "z-score", "spike", "尖峰", "突发"],
    "stationarity_flip":["平稳", "stationary", "ADF", "非平稳", "split", "分布漂移", "distribution shift"],
}


def keyword_score(text: str, fault: str) -> float:
    """text 中 fault-关键词出现数 / 候选词总数。"""
    if not text:
        return 0.0
    kws = FAULT_KEYWORDS.get(fault, [])
    if not kws:
        return 0.0
    text_lower = text.lower()
    n_hit = sum(1 for k in kws if k.lower() in text_lower)
    return n_hit / len(kws)


def evaluate(predictions_path: str = "research/results/taska_rca_predictions.jsonl"):
    rows = [json.loads(l) for l in open(predictions_path)]
    if not rows:
        print("no predictions found")
        return

    # Aggregate metrics per method (agent / b1)
    methods = ["agent_pred", "b1_pred"]
    metrics = defaultdict(lambda: {"r1": 0, "r2": 0, "n": 0, "kw_total": 0, "kw_correct": 0})
    confusion = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for r in rows:
        gt_primary = r["ground_truth"]["primary_fault"]
        for m in methods:
            pred = r[m]
            p_primary = pred.get("primary_fault", "unknown")
            secondaries = pred.get("secondary_faults", [])
            evidence = pred.get("evidence", "")

            metrics[m]["n"] += 1
            if p_primary == gt_primary:
                metrics[m]["r1"] += 1
            top3 = [p_primary] + list(secondaries)
            if gt_primary in top3[:3]:
                metrics[m]["r2"] += 1
            # R4 keyword: 评估 evidence 是否引用正确 fault 的关键词
            metrics[m]["kw_total"] += 1
            metrics[m]["kw_correct"] += keyword_score(evidence, gt_primary)

            confusion[m][gt_primary][p_primary] += 1

    print(f"=== TaskA RCA — {len(rows)} cells × 2 methods ===\n")
    print(f"{'method':12} {'R1 Top-1':>10} {'R2 Top-3':>10} {'R4 Kw-F1':>10}")
    for m in methods:
        v = metrics[m]
        r1 = v["r1"] / v["n"] if v["n"] else 0
        r2 = v["r2"] / v["n"] if v["n"] else 0
        kw = v["kw_correct"] / v["kw_total"] if v["kw_total"] else 0
        print(f"{m:12} {r1:>10.3f} {r2:>10.3f} {kw:>10.3f}")

    # GT distribution
    print(f"\nGT primary fault distribution:")
    gt_dist = Counter(r["ground_truth"]["primary_fault"] for r in rows)
    for f, n in gt_dist.most_common():
        print(f"  {f}: {n}")

    # Confusion matrix per method
    for m in methods:
        print(f"\n=== {m} confusion (rows=GT, cols=pred) ===")
        all_faults = sorted(set(list(confusion[m].keys()) +
                                 [k for v in confusion[m].values() for k in v]))
        header = "GT/Pred"
        print(f"{header:<22}" + "".join(f"{f[:15]:>16}" for f in all_faults))
        for gt in all_faults:
            row = [f"{confusion[m][gt].get(p, 0):>16}" for p in all_faults]
            print(f"{gt:<22}" + "".join(row))


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "research/results/taska_rca_predictions.jsonl"
    evaluate(path)
