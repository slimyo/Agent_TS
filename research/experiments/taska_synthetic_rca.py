"""task #25 / P6.2 · RCA Synthetic 5-fault — fair Agent vs B0-rule 对比。

设计：用 inject_fault.py 5-injector 在 ETTh1+ECL 上构造 50 cells，clean GT = injected label
（不来自 rule detector），解决 task #30 B0 tautological 问题。

对照：B0-rule / B1 LLM-direct / B5 Agent (v2 Curator 12-dim)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research.agent.rca import agent_rca, b1_direct_rca
from research.baseline.chronos2 import predict as c2_predict
from research.experiments.taska_run_b0_rule import b0_rule_rca
from research.utils.data_loader import load_series
from research.utils.fault_taxonomy import detect_faults
from research.utils.inject_fault import RCA_FAULT_LABELS, build_rca_synthetic_dataset
from research.utils.metrics import mae


DATASETS = ["ETTh1", "ECL"]
N_PER_CLASS = 5  # 5 faults × 5 cells × 2 datasets = 50 cells
WINDOW_LEN = 96
H = 96


def main():
    out = Path("research/results/taska_synthetic_rca.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists(): out.unlink()

    all_results = []
    for ds in DATASETS:
        series, meta = load_series(ds)
        print(f"\n=== {ds} (season_m={meta.season_m}) ===")
        cells = build_rca_synthetic_dataset(
            series, window_len=WINDOW_LEN, n_per_class=N_PER_CLASS,
            seed=1, season_m=meta.season_m,
        )
        print(f"built {len(cells)} synthetic cells")

        for i, cell in enumerate(cells):
            cell_id = f"synth_{ds}_{cell['fault_label']}_k{cell['seed_idx']}"
            gt = cell["fault_label"]
            train = cell["train"]
            test = cell["test"]

            # 跑 Chronos-2 取一个 prediction（作为 Agent 的输入）
            try:
                y_hat = c2_predict(train=train, val=np.array([]), H=H,
                                   seed=1, season_m=meta.season_m)
                pred_mae = mae(test[:H], y_hat[:len(test)])
            except Exception as e:
                print(f"  skip {cell_id}: {e!r}")
                continue

            # B0 rule
            b0_pred = b0_rule_rca(train, season_m=meta.season_m)

            # B1 LLM-direct（用 baseline naive prediction 作为 input）
            try:
                b1_pred = b1_direct_rca(
                    train=train, test=test, prediction=y_hat,
                    dataset=ds, N=len(train), seed=1, strategy="chronos2",
                    adapt_mae=pred_mae, c2_mae=pred_mae,
                )
            except Exception as e:
                b1_pred = {"primary_fault": "unknown", "secondary_faults": []}
                print(f"  B1 fail: {e!r}")

            # Agent (B5 v2)
            try:
                agent_pred = agent_rca(
                    train=train, val=np.array([]), test=test, prediction=y_hat,
                    dataset=ds, N=len(train), seed=1, H=H,
                    strategy="chronos2", adapt_mae=pred_mae, c2_mae=pred_mae,
                    season_m=meta.season_m,
                )
            except Exception as e:
                agent_pred = {"primary_fault": "unknown", "secondary_faults": []}
                print(f"  Agent fail: {e!r}")

            row = {
                "cell_id": cell_id,
                "ground_truth": {"primary_fault": gt},
                "b0_pred": {"primary_fault": b0_pred["primary_fault"],
                            "secondary_faults": b0_pred["secondary_faults"]},
                "b1_pred": {"primary_fault": b1_pred.get("primary_fault", "unknown"),
                            "secondary_faults": b1_pred.get("secondary_faults", [])},
                "agent_pred": {"primary_fault": agent_pred.get("primary_fault", "unknown"),
                               "secondary_faults": agent_pred.get("secondary_faults", [])},
                "pred_mae": float(pred_mae),
            }
            all_results.append(row)
            with out.open("a") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  [{i+1}/{len(cells)}] {cell_id}: GT={gt:18}"
                  f"  B0={b0_pred['primary_fault']:18}"
                  f"  B1={b1_pred.get('primary_fault','?'):18}"
                  f"  Agent={agent_pred.get('primary_fault','?'):18}"
                  f"  c2_mae={pred_mae:.3f}")

    # Aggregate R1/R2 for each method
    print(f"\n=== Synthetic RCA Aggregate (n={len(all_results)}) ===")
    from collections import Counter
    for method_key in ["b0_pred", "b1_pred", "agent_pred"]:
        r1 = sum(1 for r in all_results
                 if r[method_key]["primary_fault"] == r["ground_truth"]["primary_fault"])
        r2 = sum(1 for r in all_results
                 if r["ground_truth"]["primary_fault"]
                 in [r[method_key]["primary_fault"]] + r[method_key].get("secondary_faults", []))
        print(f"  {method_key:12}: R1={r1/len(all_results):.3f}  R2={r2/len(all_results):.3f}")

    # Per-class breakdown
    print(f"\n=== Per-fault R1 (Agent) ===")
    by_fault = Counter()
    correct_by_fault = Counter()
    for r in all_results:
        gt = r["ground_truth"]["primary_fault"]
        by_fault[gt] += 1
        if r["agent_pred"]["primary_fault"] == gt: correct_by_fault[gt] += 1
    for f in RCA_FAULT_LABELS:
        if by_fault[f]:
            print(f"  {f:18}: {correct_by_fault[f]}/{by_fault[f]} = {correct_by_fault[f]/by_fault[f]:.2f}")


if __name__ == "__main__":
    main()
