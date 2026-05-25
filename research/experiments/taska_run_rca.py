"""P6.1 · 跑 RCA Agent (B5) vs LLM-direct (B1) 在 30 个 failure cells 上。

输出 research/results/taska_rca_predictions.jsonl
每行 = {cell_id, ground_truth, agent_pred, b1_pred}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research.agent.rca import agent_rca, b1_direct_rca
from research.utils.data_loader import load_series
from research.utils.splitter import few_shot_split


def reconstruct_prediction(dataset, N, H, seed, strategy_hint=None):
    """重跑 Chronos-2 baseline 拿预测（最简 baseline，作为 RCA 输入）。
    若有其他策略 jsonl，可读 winner 信息——这里偷懒：直接用 Chronos-2 重跑预测。
    """
    series, meta = load_series(dataset)
    sp = few_shot_split(series, N=N, H=H, seed=seed)
    from research.baseline.chronos2 import predict as c2_pred
    y_hat = c2_pred(train=sp.train, val=sp.val, H=H, seed=seed, season_m=meta.season_m)
    return sp, meta, y_hat


def main():
    failures = [json.loads(l) for l in open("research/results/taska_failures.jsonl")]
    print(f"loaded {len(failures)} failure cells")

    out_path = Path("research/results/taska_rca_predictions.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add(r["cell_id"])
            except Exception:
                pass
    print(f"resuming: {len(done)} cells done")

    with out_path.open("a") as fh:
        for i, c in enumerate(failures):
            cell_id = f"{c['variant']}_{c['dataset']}_N{c['N']}_seed{c['seed']}"
            if cell_id in done:
                continue
            print(f"\n[{i+1}/{len(failures)}] {cell_id} ratio={c['mae_ratio']:.2f}")
            try:
                sp, meta, y_hat = reconstruct_prediction(
                    c["dataset"], c["N"], c["H"], c["seed"]
                )
                agent_out = agent_rca(
                    train=sp.train, val=sp.val, test=sp.test, prediction=y_hat,
                    dataset=c["dataset"], N=c["N"], seed=c["seed"], H=c["H"],
                    strategy=c["variant"],
                    adapt_mae=c["adapt_mae"], c2_mae=c["chronos2_mae"],
                    season_m=meta.season_m,
                )
                b1_out = b1_direct_rca(
                    train=sp.train, test=sp.test, prediction=y_hat,
                    dataset=c["dataset"], N=c["N"], seed=c["seed"],
                    strategy=c["variant"],
                    adapt_mae=c["adapt_mae"], c2_mae=c["chronos2_mae"],
                )
                row = {
                    "cell_id": cell_id,
                    "ground_truth": c["ground_truth"],
                    "agent_pred": {
                        "primary_fault": agent_out["primary_fault"],
                        "secondary_faults": agent_out["secondary_faults"],
                        "evidence": agent_out.get("supporting_evidence", "")[:300],
                    },
                    "b1_pred": {
                        "primary_fault": b1_out["primary_fault"],
                        "secondary_faults": b1_out["secondary_faults"],
                        "evidence": b1_out.get("supporting_evidence", "")[:300],
                    },
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                print(f"  GT={c['ground_truth']['primary_fault']}  "
                      f"Agent={agent_out['primary_fault']}  "
                      f"B1={b1_out['primary_fault']}")
            except Exception as e:
                print(f"  FAIL: {e!r}")

    print("\nDone. Run taska_eval.py to compute R1-R5 metrics.")


if __name__ == "__main__":
    main()
