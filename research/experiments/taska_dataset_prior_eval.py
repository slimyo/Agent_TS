"""task #17 / B5 · 测 dataset-name semantic prior 对 Agent 的影响。

复用 50 OOT cells (task #43)，对比：
  - Agent v3 (no prior): baseline
  - Agent + dataset_prior: 新增 semantic context

期望：semantic prior 能让 LLM 更好区分 "natural pattern" vs "anomalous injection"，
提升 OOT-recall。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research.agent.rca import agent_rca
from research.baseline.chronos2 import predict as c2_predict
from research.utils.data_loader import load_series
from research.utils.inject_fault import build_oot_rca_dataset
from research.utils.metrics import mae


def main():
    out = Path("research/results/taska_dataset_prior_eval.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists(): out.unlink()

    results = []
    for ds in ["ETTh1", "ECL"]:
        series, meta = load_series(ds)
        cells = build_oot_rca_dataset(series, window_len=96, n_per_class=5,
                                       seed=1, season_m=meta.season_m)
        for i, c in enumerate(cells):
            gt = c["fault_label"]
            train = c["train"]; test = c["test"]
            try:
                y_hat = c2_predict(train=train, val=np.array([]), H=96, seed=1, season_m=meta.season_m)
                pred_mae = mae(test[:96], y_hat[:len(test)])
            except Exception as e:
                continue
            # No prior (cached)
            no_prior = agent_rca(train=train, val=np.array([]), test=test, prediction=y_hat,
                                 dataset=ds, N=len(train), seed=1, H=96,
                                 strategy="chronos2", adapt_mae=pred_mae, c2_mae=pred_mae,
                                 season_m=meta.season_m, use_dataset_prior=False)
            # With prior (new prompt → new LLM call)
            with_prior = agent_rca(train=train, val=np.array([]), test=test, prediction=y_hat,
                                    dataset=ds, N=len(train), seed=1, H=96,
                                    strategy="chronos2", adapt_mae=pred_mae, c2_mae=pred_mae,
                                    season_m=meta.season_m, use_dataset_prior=True)
            row = {
                "cell_id": f"oot_{ds}_{c['fault_label']}_k{c['seed_idx']}",
                "ds": ds, "gt": gt,
                "no_prior_pred": no_prior.get("primary_fault"),
                "with_prior_pred": with_prior.get("primary_fault"),
                "no_prior_is_oot": no_prior.get("primary_fault") == "out_of_taxonomy",
                "with_prior_is_oot": with_prior.get("primary_fault") == "out_of_taxonomy",
                "no_prior_evidence": (no_prior.get("supporting_evidence") or "")[:150],
                "with_prior_evidence": (with_prior.get("supporting_evidence") or "")[:200],
            }
            results.append(row)
            with out.open("a") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i % 5 == 0:
                print(f"  [{ds}] {i}/{len(cells)}: GT={gt:25} no={row['no_prior_pred']:20} with={row['with_prior_pred']:20}")

    # Aggregate
    n = len(results)
    no_oot = sum(r["no_prior_is_oot"] for r in results)
    with_oot = sum(r["with_prior_is_oot"] for r in results)
    print(f"\n=== Aggregate (n={n}) ===")
    print(f"  Agent no prior:        OOT-recall = {no_oot}/{n} = {no_oot/n:.3f}")
    print(f"  Agent + dataset prior: OOT-recall = {with_oot}/{n} = {with_oot/n:.3f}")
    print(f"  Δ = {(with_oot - no_oot)/n*100:+.1f}pp")

    # Per-fault
    from collections import Counter
    fault_n = Counter()
    fault_no = Counter()
    fault_with = Counter()
    for r in results:
        f = r["gt"]; fault_n[f] += 1
        if r["no_prior_is_oot"]: fault_no[f] += 1
        if r["with_prior_is_oot"]: fault_with[f] += 1
    print("\nPer-OOT-fault breakdown:")
    for f in sorted(fault_n):
        print(f"  {f:30}: no_prior={fault_no[f]}/{fault_n[f]}  with_prior={fault_with[f]}/{fault_n[f]}")


if __name__ == "__main__":
    main()
