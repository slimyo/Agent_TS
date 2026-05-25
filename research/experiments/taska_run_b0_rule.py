"""task #30 / P6.4b-1 · B0-rule RCA baseline (feedback 建议补完整 ablation)。

设计：用 `fault_taxonomy.detect_faults` 仅在 train 端打分，取 top fault → primary_fault。
对照 Agent + B1（两者都有 train+test+prediction context）。

输出：research/results/taska_b0_rule_predictions.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

from research.utils.data_loader import load_series
from research.utils.fault_taxonomy import detect_faults, top_faults
from research.utils.splitter import few_shot_split


def b0_rule_rca(train, season_m: int = 1) -> dict:
    """B0-rule path: detect_faults on train only, return top fault."""
    scores = detect_faults(train, season_m=season_m)
    sorted_f = sorted(scores.items(), key=lambda kv: -kv[1])
    primary = sorted_f[0][0] if sorted_f[0][1] > 0 else "unknown"
    secondaries = [name for name, sc in sorted_f[1:4] if sc >= 0.3]
    return {
        "primary_fault": primary,
        "secondary_faults": secondaries,
        "supporting_evidence": f"rule-based on train; scores={scores}",
        "_path": "b0_rule",
    }


def main():
    failures = [json.loads(l) for l in open("research/results/taska_failures.jsonl")]
    print(f"loaded {len(failures)} failure cells\n")

    out = Path("research/results/taska_b0_rule_predictions.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists(): out.unlink()

    correct_r1 = 0
    correct_r2 = 0
    with out.open("a") as fh:
        for c in failures:
            series, meta = load_series(c["dataset"])
            sp = few_shot_split(series, N=c["N"], H=c["H"], seed=c["seed"])
            pred = b0_rule_rca(sp.train, season_m=meta.season_m)
            cell_id = f"{c['variant']}_{c['dataset']}_N{c['N']}_seed{c['seed']}"
            row = {
                "cell_id": cell_id,
                "ground_truth": c["ground_truth"],
                "b0_pred": pred,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            gt = c["ground_truth"]["primary_fault"]
            if pred["primary_fault"] == gt:
                correct_r1 += 1
            top3 = [pred["primary_fault"]] + pred["secondary_faults"]
            if gt in top3[:3]:
                correct_r2 += 1
            print(f"  {cell_id:35} GT={gt:18} B0={pred['primary_fault']:18} "
                  f"{'✓' if pred['primary_fault']==gt else '✗'}")

    n = len(failures)
    print(f"\n=== B0-rule (train-only fault scoring, no LLM) ===")
    print(f"  R1 Top-1 accuracy:  {correct_r1}/{n} = {correct_r1/n:.3f}")
    print(f"  R2 Top-3 inclusion: {correct_r2}/{n} = {correct_r2/n:.3f}")


if __name__ == "__main__":
    main()
