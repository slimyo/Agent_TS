"""P6.4b-6 / task #35 · Agent-Router (B7) vs B1-B6 + Oracle 上界。

复用 task #26 的 UCR sweep 数据（已含 B1-B6），新跑 B7 Agent-Router。
也复用 task #27 合成 4-class 数据 → 跑 B7 看 routing 是否在 statistical-label
任务上比 Rocket alone 还要强。

实验设置（与 task #26 对齐）：
  5 UCR datasets × 3 N-shot × 2 seeds = 30 cells
  B7 Agent-Router 用 LOO CV + margin 0.10 + Rocket default + 不启用 memory
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from research.agent.clf_planner import b7_agent_router, classification_planner
from research.utils.ucr_loader import load_ucr_fewshot

UCR_DATASETS = ["Coffee", "ECG200", "TwoLeadECG", "BeetleFly", "BirdChicken"]
N_PER_CLASS = [3, 5, 10]
SEEDS = [1, 42]


def run_ucr():
    out = Path("research/results/taskb_router_ucr.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            try:
                r = json.loads(l)
                done.add((r["dataset"], r["N_per_class"], r["seed"]))
            except Exception:
                pass

    fh = out.open("a")
    for ds in UCR_DATASETS:
        for n in N_PER_CLASS:
            for seed in SEEDS:
                key = (ds, n, seed)
                if key in done: continue
                X_tr, y_tr, X_te, y_te = load_ucr_fewshot(ds, n_per_class=n, seed=seed)
                t0 = time.time()
                chosen, y_pred, trace = classification_planner(
                    X_tr, y_tr, X_te, season_m=1,
                    use_cv=True, cv_method="loo",
                    margin=0.10, default_classifier="rocket",
                )
                wall = time.time() - t0
                acc = float((y_pred == y_te).mean())
                from sklearn.metrics import f1_score
                try:
                    f1 = float(f1_score(y_te, y_pred, average="macro"))
                except Exception:
                    f1 = 0.0
                row = {
                    "dataset": ds, "N_per_class": n, "seed": seed,
                    "method": "B7_router",
                    "n_test": len(y_te),
                    "acc": round(acc, 4), "macro_f1": round(f1, 4),
                    "chosen_classifier": chosen,
                    "chosen_reason": trace.chosen_reason,
                    "cv_accs": trace.cv_accs,
                    "wall_time": round(wall, 2),
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                print(f"  {ds:13} n={n} seed={seed}: chose '{chosen}' "
                      f"→ acc={acc:.3f} f1={f1:.3f} ({wall:.1f}s)")
    fh.close()


if __name__ == "__main__":
    run_ucr()
