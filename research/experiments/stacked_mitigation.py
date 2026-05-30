"""P0-C · 三 mitigation paths 的 stacked 实验。

四种 config × 50 OOT cells × {default LLM, strong LLM}：
  baseline  : Agent default (no prior, no abstain)
  +prior    : Agent + dataset semantic prior
  +abstain  : Agent + abstain head
  +stack    : Agent + prior + abstain (all)

Strong LLM 用 glm-4-plus，结合 +prior / +abstain / +stack 看是否 super-additive。

复用 task #43/#46/#47 已 cached LLM 响应 + abstain head model。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research.agent.rca import agent_rca, _apply_abstain_override
from research.baseline.chronos2 import predict as c2_predict
from research.utils.data_loader import load_series
from research.utils.inject_fault import build_oot_rca_dataset
from research.utils.metrics import mae

LLM_MODELS = ["glm-4-flash-250414", "glm-4-plus"]
H = 96
WINDOW_LEN = 96


def main():
    out = Path("research/results/stacked_mitigation.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists(): out.unlink()

    # Build OOT cells (reuse task #43)
    cells = []
    for ds in ["ETTh1", "ECL"]:
        series, meta = load_series(ds)
        ds_cells = build_oot_rca_dataset(series, window_len=WINDOW_LEN,
                                          n_per_class=5, seed=1,
                                          season_m=meta.season_m)
        for c in ds_cells:
            try:
                y_hat = c2_predict(train=c["train"], val=np.array([]), H=H,
                                   seed=1, season_m=meta.season_m)
                pred_mae = mae(c["test"][:H], y_hat[:len(c["test"])])
            except Exception:
                continue
            cells.append((ds, c, y_hat, pred_mae, meta.season_m))

    print(f"Total OOT cells: {len(cells)}\n")

    results = []
    import os
    for model in LLM_MODELS:
        os.environ["MODEL"] = model
        print(f"\n=== LLM = {model} ===")
        for ds, c, y_hat, pmae, sm in cells:
            cell_id = f"oot_{ds}_{c['fault_label']}_k{c['seed_idx']}"
            gt = c["fault_label"]
            # 4 configs
            configs = {}
            for use_prior, use_abst in [(False,False), (True,False), (False,True), (True,True)]:
                try:
                    pred = agent_rca(
                        train=c["train"], val=np.array([]), test=c["test"], prediction=y_hat,
                        dataset=ds, N=len(c["train"]), seed=1, H=H,
                        strategy="chronos2", adapt_mae=pmae, c2_mae=pmae,
                        season_m=sm, llm_model=model,
                        use_dataset_prior=use_prior, use_abstain=use_abst,
                    )
                    name = ("+prior" if use_prior else "") + ("+abstain" if use_abst else "")
                    if not name: name = "baseline"
                    else: name = name.lstrip("+")
                    configs[name] = pred.get("primary_fault")
                except Exception as e:
                    print(f"  FAIL {cell_id} prior={use_prior} abst={use_abst}: {e!r}")
                    continue
            row = {
                "llm": model, "cell_id": cell_id, "gt": gt,
                "baseline":      configs.get("baseline"),
                "prior":         configs.get("prior"),
                "abstain":       configs.get("abstain"),
                "prior_abstain": configs.get("prior+abstain"),
            }
            results.append(row)
            with out.open("a") as fh:
                fh.write(json.dumps(row) + "\n")

    print(f"\n=== Aggregate (n={len(results)}) ===\n")
    # OOT-recall per config × LLM
    by_llm = {}
    for r in results:
        by_llm.setdefault(r["llm"], []).append(r)
    print(f'{"LLM":25}  {"baseline":>10}  {"+prior":>10}  {"+abstain":>10}  {"+stack":>10}')
    for llm, rs in by_llm.items():
        ootr = lambda key: sum(1 for r in rs if r.get(key) == "out_of_taxonomy") / len(rs)
        print(f'{llm:25}  {ootr("baseline"):>10.3f}  {ootr("prior"):>10.3f}  {ootr("abstain"):>10.3f}  {ootr("prior_abstain"):>10.3f}')


if __name__ == "__main__":
    main()
