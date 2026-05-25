"""task #43 / P6.4d · Out-of-Taxonomy RCA — Agent niche positive。

设计：5 OOT faults × 5 cells × 2 datasets = 50 cells，GT = OOT fault label
（B0-rule 只能输出 5 in-taxonomy 类 → B0 R1 = 0% by construction）

评估：
  - **R1_OOT**: Agent 输出 'out_of_taxonomy' 的比例
  - **K-F1**: Agent 的 supporting_evidence 文本中包含该 OOT fault 关键词的比例
  - B0 keyword F1: 0% (只输出 class label)

输出 research/results/taska_oot_rca.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research.agent.rca import agent_rca, b1_direct_rca
from research.baseline.chronos2 import predict as c2_predict
from research.experiments.taska_run_b0_rule import b0_rule_rca
from research.utils.data_loader import load_series
from research.utils.inject_fault import (
    OOT_DESCRIPTIONS, OOT_INJECTORS, build_oot_rca_dataset,
)
from research.utils.metrics import mae


DATASETS = ["ETTh1", "ECL"]
N_PER_CLASS = 5  # 5 OOT × 5 × 2 = 50 cells
WINDOW_LEN = 96
H = 96


def keyword_match_score(text: str, keywords: list[str]) -> float:
    """文本中 fault 关键词命中数 / 关键词总数。"""
    if not text or not keywords:
        return 0.0
    text_lower = text.lower()
    n_hit = sum(1 for k in keywords if k.lower() in text_lower)
    return n_hit / len(keywords)


def main():
    out = Path("research/results/taska_oot_rca.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists(): out.unlink()

    all_results = []
    for ds in DATASETS:
        series, meta = load_series(ds)
        cells = build_oot_rca_dataset(
            series, window_len=WINDOW_LEN, n_per_class=N_PER_CLASS,
            seed=1, season_m=meta.season_m,
        )
        print(f"\n=== {ds}: built {len(cells)} OOT synthetic cells ===")

        for i, cell in enumerate(cells):
            cell_id = f"oot_{ds}_{cell['fault_label']}_k{cell['seed_idx']}"
            gt = cell["fault_label"]
            kws = cell["keywords"]
            train = cell["train"]
            test = cell["test"]

            try:
                y_hat = c2_predict(train=train, val=np.array([]), H=H,
                                   seed=1, season_m=meta.season_m)
                pred_mae = mae(test[:H], y_hat[:len(test)])
            except Exception as e:
                print(f"  skip {cell_id}: {e!r}")
                continue

            # B0 rule (forced to 5-class taxonomy)
            b0_pred = b0_rule_rca(train, season_m=meta.season_m)
            # B1 LLM-direct
            try:
                b1_pred = b1_direct_rca(
                    train=train, test=test, prediction=y_hat,
                    dataset=ds, N=len(train), seed=1, strategy="chronos2",
                    adapt_mae=pred_mae, c2_mae=pred_mae,
                )
            except Exception as e:
                b1_pred = {"primary_fault": "unknown", "secondary_faults": [], "supporting_evidence": ""}
            # Agent
            try:
                agent_pred = agent_rca(
                    train=train, val=np.array([]), test=test, prediction=y_hat,
                    dataset=ds, N=len(train), seed=1, H=H,
                    strategy="chronos2", adapt_mae=pred_mae, c2_mae=pred_mae,
                    season_m=meta.season_m,
                )
            except Exception as e:
                agent_pred = {"primary_fault": "unknown", "secondary_faults": [], "supporting_evidence": ""}

            # Score
            agent_evi = agent_pred.get("supporting_evidence", "") or ""
            b1_evi = b1_pred.get("supporting_evidence", "") or ""
            row = {
                "cell_id": cell_id,
                "ground_truth": {"primary_fault": gt, "keywords": kws,
                                  "description": cell["description"]},
                "b0_pred": {"primary_fault": b0_pred["primary_fault"],
                            "evidence": "(rule, no NL)",
                            "kw_match": 0.0,
                            "is_oot": False},
                "b1_pred": {"primary_fault": b1_pred.get("primary_fault", "unknown"),
                            "evidence": b1_evi[:200],
                            "kw_match": keyword_match_score(b1_evi, kws),
                            "is_oot": b1_pred.get("primary_fault") == "out_of_taxonomy"},
                "agent_pred": {"primary_fault": agent_pred.get("primary_fault", "unknown"),
                               "evidence": agent_evi[:300],
                               "kw_match": keyword_match_score(agent_evi, kws),
                               "is_oot": agent_pred.get("primary_fault") == "out_of_taxonomy"},
                "pred_mae": float(pred_mae),
            }
            all_results.append(row)
            with out.open("a") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  [{i+1:2}/{len(cells)}] {cell_id}: GT={gt:25}"
                  f"  Agent={agent_pred.get('primary_fault','?'):20} kw={row['agent_pred']['kw_match']:.2f}"
                  f"  B0={b0_pred['primary_fault']:18}"
                  f"  B1={b1_pred.get('primary_fault','?'):20}")

    # Aggregate
    n = len(all_results)
    print(f"\n=== OOT RCA Aggregate (n={n}) ===")
    agent_oot_recall = sum(1 for r in all_results if r["agent_pred"]["is_oot"]) / n
    b1_oot_recall = sum(1 for r in all_results if r["b1_pred"]["is_oot"]) / n
    b0_oot_recall = sum(1 for r in all_results if r["b0_pred"]["is_oot"]) / n  # =0 by construction

    agent_kw = sum(r["agent_pred"]["kw_match"] for r in all_results) / n
    b1_kw = sum(r["b1_pred"]["kw_match"] for r in all_results) / n
    b0_kw = sum(r["b0_pred"]["kw_match"] for r in all_results) / n   # =0

    print(f"  OOT-recall (predict 'out_of_taxonomy'):")
    print(f"    B0:    {b0_oot_recall:.3f} (impossible by construction)")
    print(f"    B1:    {b1_oot_recall:.3f}")
    print(f"    Agent: {agent_oot_recall:.3f}")
    print(f"  Keyword-F1 (evidence text matches GT keywords):")
    print(f"    B0:    {b0_kw:.3f} (no NL output)")
    print(f"    B1:    {b1_kw:.3f}")
    print(f"    Agent: {agent_kw:.3f}")

    # Per-fault breakdown
    from collections import Counter
    by_fault_n = Counter()
    by_fault_oot = Counter()
    by_fault_kw = Counter()
    for r in all_results:
        f = r["ground_truth"]["primary_fault"]
        by_fault_n[f] += 1
        if r["agent_pred"]["is_oot"]:
            by_fault_oot[f] += 1
        by_fault_kw[f] += r["agent_pred"]["kw_match"]
    print(f"\n  Per-OOT-fault Agent recall + keyword F1:")
    for f in OOT_DESCRIPTIONS:
        if by_fault_n[f]:
            print(f"    {f:28}: OOT-recall={by_fault_oot[f]/by_fault_n[f]:.2f}  "
                  f"kw-F1={by_fault_kw[f]/by_fault_n[f]:.3f}  n={by_fault_n[f]}")


if __name__ == "__main__":
    main()
