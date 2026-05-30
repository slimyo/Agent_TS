"""task #47 / P1-6 · Cross-LLM RCA — 验证 specialist bias 是否 LLM-agnostic。

复用 task #43 OOT 50 cells，跑 3 个 LLM 模型 (glm-4-flash-250414 / glm-4-air / glm-4-plus)。
关键 question: 不同 LLM 是否都表现出 specialist bias？

输出：research/results/taska_cross_llm_rca.jsonl
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from research.agent.rca import agent_rca, b1_direct_rca
from research.baseline.chronos2 import predict as c2_predict
from research.utils.data_loader import load_series
from research.utils.inject_fault import build_oot_rca_dataset
from research.utils.metrics import mae

LLM_MODELS = ["glm-4-flash-250414", "glm-4-air", "glm-4-plus"]
DATASETS = ["ETTh1", "ECL"]
N_PER_CLASS = 5  # 5 OOT × 5 × 2 = 50 cells × 3 LLMs = 150 LLM runs
WINDOW_LEN = 96
H = 96


def main():
    out = Path("research/results/taska_cross_llm_rca.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["llm_model"], r["cell_id"]))
            except Exception: pass

    fh = out.open("a")
    for ds in DATASETS:
        series, meta = load_series(ds)
        cells = build_oot_rca_dataset(
            series, window_len=WINDOW_LEN, n_per_class=N_PER_CLASS,
            seed=1, season_m=meta.season_m,
        )
        # Pre-compute predictions (Chronos-2, only once)
        cell_preds = []
        for c in cells:
            try:
                y_hat = c2_predict(train=c["train"], val=np.array([]), H=H,
                                   seed=1, season_m=meta.season_m)
                m = mae(c["test"][:H], y_hat[:len(c["test"])])
            except Exception:
                continue
            cell_preds.append((c, y_hat, m))
        print(f"\n=== {ds}: {len(cell_preds)} cells ===")

        for model in LLM_MODELS:
            os.environ["MODEL"] = model
            print(f"  -- LLM = {model} --")
            for c, y_hat, pred_mae in cell_preds:
                cell_id = f"oot_{ds}_{c['fault_label']}_k{c['seed_idx']}"
                if (model, cell_id) in done: continue
                gt = c["fault_label"]
                try:
                    agent_pred = agent_rca(
                        train=c["train"], val=np.array([]), test=c["test"],
                        prediction=y_hat, dataset=ds, N=len(c["train"]), seed=1, H=H,
                        strategy="chronos2", adapt_mae=pred_mae, c2_mae=pred_mae,
                        season_m=meta.season_m, llm_model=model,
                    )
                except Exception as e:
                    print(f"    Agent FAIL {cell_id}: {e!r}")
                    continue
                try:
                    b1_pred = b1_direct_rca(
                        train=c["train"], test=c["test"], prediction=y_hat,
                        dataset=ds, N=len(c["train"]), seed=1, strategy="chronos2",
                        adapt_mae=pred_mae, c2_mae=pred_mae, llm_model=model,
                    )
                except Exception as e:
                    b1_pred = {"primary_fault": "unknown"}

                row = {
                    "llm_model": model,
                    "cell_id": cell_id,
                    "gt": gt,
                    "agent_primary": agent_pred.get("primary_fault", "unknown"),
                    "b1_primary": b1_pred.get("primary_fault", "unknown"),
                    "agent_is_oot": agent_pred.get("primary_fault") == "out_of_taxonomy",
                    "b1_is_oot": b1_pred.get("primary_fault") == "out_of_taxonomy",
                }
                fh.write(json.dumps(row) + "\n"); fh.flush()
    fh.close()

    # Aggregate
    print("\n=== Cross-LLM Aggregate ===")
    rows = [json.loads(l) for l in open("research/results/taska_cross_llm_rca.jsonl")]
    from collections import Counter, defaultdict
    by_model = defaultdict(list)
    for r in rows: by_model[r["llm_model"]].append(r)
    print(f'{"LLM":25}  {"Agent OOT-recall":>16}  {"B1 OOT-recall":>14}  {"n":>4}')
    for model in LLM_MODELS:
        rs = by_model.get(model, [])
        if not rs: continue
        a_oot = sum(r["agent_is_oot"] for r in rs) / len(rs)
        b_oot = sum(r["b1_is_oot"] for r in rs) / len(rs)
        print(f'{model:25}  {a_oot:>16.3f}  {b_oot:>14.3f}  {len(rs):>4}')

    print("\n=== Per-LLM Agent prediction distribution ===")
    for model in LLM_MODELS:
        rs = by_model.get(model, [])
        if not rs: continue
        dist = Counter(r["agent_primary"] for r in rs)
        print(f"  {model}: {dict(dist)}")


if __name__ == "__main__":
    main()
