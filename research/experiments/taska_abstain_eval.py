"""task #46 · Abstain head on/off 对照评估。

复用 task #25 in-tax (50 cells) + task #43 OOT (50 cells) 共 100 cells。
对照：
  - Agent v3 (no abstain): baseline
  - Agent + abstain (override OOT)

输出：research/results/taska_abstain_eval.jsonl + 汇总表
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research.agent.rca import _apply_abstain_override, agent_rca
from research.baseline.chronos2 import predict as c2_predict
from research.utils.data_loader import load_series
from research.utils.inject_fault import (
    build_oot_rca_dataset, build_rca_synthetic_dataset,
)
from research.utils.metrics import mae
from research.utils.fault_taxonomy import FAULT_NAMES

H = 96
WINDOW_LEN = 96


def main():
    out = Path("research/results/taska_abstain_eval.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists(): out.unlink()

    all_cells = []  # list of (kind, gt, train, test, y_hat, mae)

    for ds in ["ETTh1", "ECL"]:
        series, meta = load_series(ds)
        # in-tax
        in_tax = build_rca_synthetic_dataset(
            series, window_len=WINDOW_LEN, n_per_class=5,
            seed=1, season_m=meta.season_m,
        )
        for c in in_tax:
            y_hat = c2_predict(train=c["train"], val=np.array([]), H=H, seed=1, season_m=meta.season_m)
            m = mae(c["test"][:H], y_hat[:len(c["test"])])
            all_cells.append({
                "ds": ds, "kind": "in_tax", "gt": c["fault_label"],
                "train": c["train"], "test": c["test"], "y_hat": y_hat, "mae": m,
                "season_m": meta.season_m,
                "cell_id": f"intax_{ds}_{c['fault_label']}_k{c['seed_idx']}",
            })
        # OOT
        oot = build_oot_rca_dataset(
            series, window_len=WINDOW_LEN, n_per_class=5,
            seed=1, season_m=meta.season_m,
        )
        for c in oot:
            y_hat = c2_predict(train=c["train"], val=np.array([]), H=H, seed=1, season_m=meta.season_m)
            m = mae(c["test"][:H], y_hat[:len(c["test"])])
            all_cells.append({
                "ds": ds, "kind": "oot", "gt": c["fault_label"],
                "train": c["train"], "test": c["test"], "y_hat": y_hat, "mae": m,
                "season_m": meta.season_m,
                "cell_id": f"oot_{ds}_{c['fault_label']}_k{c['seed_idx']}",
            })

    print(f"\n=== Run Agent w/ abstain on each cell (uses cached LLM responses) ===")
    print(f"Total cells: {len(all_cells)}")

    results = []
    for i, c in enumerate(all_cells):
        # Run agent — without abstain first (cached)
        agent_no_abstain = agent_rca(
            train=c["train"], val=np.array([]), test=c["test"], prediction=c["y_hat"],
            dataset=c["ds"], N=len(c["train"]), seed=1, H=H,
            strategy="chronos2", adapt_mae=c["mae"], c2_mae=c["mae"],
            season_m=c["season_m"], use_abstain=False,
        )
        # Apply abstain override to the same agent output
        agent_w_abstain = dict(agent_no_abstain)
        agent_w_abstain = _apply_abstain_override(c["train"], agent_w_abstain, threshold=0.5)

        row = {
            "cell_id": c["cell_id"], "kind": c["kind"], "gt": c["gt"],
            "agent_no_abstain": agent_no_abstain.get("primary_fault"),
            "agent_w_abstain": agent_w_abstain.get("primary_fault"),
            "abstain_proba": agent_w_abstain.get("_abstain_proba"),
            "abstain_override": agent_w_abstain.get("_abstain_override", False),
        }
        results.append(row)
        with out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        if i < 5 or i % 10 == 0:
            print(f"  [{i+1}/{len(all_cells)}] {c['cell_id']:42}: GT={c['gt']:25} "
                  f"no_abs={row['agent_no_abstain']:20} w_abs={row['agent_w_abstain']:20} "
                  f"p(OOT)={row['abstain_proba']:.2f}")

    # Aggregate
    print(f"\n=== Agent w/ vs w/o Abstain ===\n")

    # For OOT cells: R1 = how often we predict 'out_of_taxonomy'
    oot_rows = [r for r in results if r["kind"] == "oot"]
    in_rows = [r for r in results if r["kind"] == "in_tax"]

    oot_r1_no_abs = sum(1 for r in oot_rows if r["agent_no_abstain"] == "out_of_taxonomy") / len(oot_rows)
    oot_r1_w_abs = sum(1 for r in oot_rows if r["agent_w_abstain"] == "out_of_taxonomy") / len(oot_rows)
    # In-tax R1: did the LLM in-tax classification survive (no abstain override)?
    in_r1_no_abs = sum(1 for r in in_rows if r["agent_no_abstain"] == r["gt"]) / len(in_rows)
    in_r1_w_abs = sum(1 for r in in_rows if r["agent_w_abstain"] == r["gt"]) / len(in_rows)
    # In-tax over-abstain (wrongly fired OOT on in-tax)
    in_over_abstain = sum(1 for r in in_rows if r["agent_w_abstain"] == "out_of_taxonomy") / len(in_rows)

    print(f"OOT cells (n={len(oot_rows)}) — OOT-recall:")
    print(f"  Agent no abstain  : {oot_r1_no_abs:.3f}")
    print(f"  Agent + abstain   : {oot_r1_w_abs:.3f}  (Δ +{(oot_r1_w_abs - oot_r1_no_abs)*100:.1f}pp)")
    print(f"\nIn-tax cells (n={len(in_rows)}) — Top-1 accuracy (must NOT predict OOT):")
    print(f"  Agent no abstain  : {in_r1_no_abs:.3f}")
    print(f"  Agent + abstain   : {in_r1_w_abs:.3f}  (Δ {(in_r1_w_abs - in_r1_no_abs)*100:+.1f}pp)")
    print(f"  In-tax over-abstain (wrongly fired): {in_over_abstain:.3f}")

    # Per-OOT-class
    print(f"\nPer-OOT-class abstain recall:")
    from collections import Counter
    by_fault = Counter()
    correct_by_fault = Counter()
    for r in oot_rows:
        f = r["gt"]
        by_fault[f] += 1
        if r["agent_w_abstain"] == "out_of_taxonomy":
            correct_by_fault[f] += 1
    for f in sorted(by_fault):
        print(f"  {f:30}: {correct_by_fault[f]}/{by_fault[f]} = {correct_by_fault[f]/by_fault[f]:.2f}")


if __name__ == "__main__":
    main()
