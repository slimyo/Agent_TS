"""M9 · 诚实版 B7v3 sweep（feedback 问题 6 数据泄漏修复后）。

与 taskb_router_v3_sweep.py 的唯一差异：
  1. memory bank = CV-based（best_classifier=CV-winner，投票用 cv_accs）
  2. 给 planner 传 dataset/seed → 触发 leave-one-cell-out（剔除查询 cell 自身）

输出与泄漏版分开：results/taskb_router_v3_honest_ucr.jsonl
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from research.agent.clf_planner import classification_planner
from research.utils.ucr_loader import load_ucr_fewshot

UCR_DATASETS = ["Coffee", "ECG200", "TwoLeadECG", "BeetleFly", "BirdChicken"]
N_PER_CLASS = [3, 5, 10]
SEEDS = [1, 42]
MEMORY_PATH = "/tmp/clf_memory_v2.jsonl"


def main():
    out = Path("research/results/taskb_router_v3_honest_ucr.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()   # always fresh (no skip-done)
    fh = out.open("a")
    accs = []
    for ds in UCR_DATASETS:
        for n in N_PER_CLASS:
            for seed in SEEDS:
                X_tr, y_tr, X_te, y_te = load_ucr_fewshot(ds, n_per_class=n, seed=seed)
                t0 = time.time()
                chosen, y_pred, trace = classification_planner(
                    X_tr, y_tr, X_te, season_m=1,
                    use_cv=True, cv_method="loo",
                    margin=0.10, default_classifier="rocket",
                    n_min_for_routing=7,
                    use_memory=True,
                    memory_path=MEMORY_PATH,
                    use_enhanced_features=True,
                    weighted_vote_min_ratio=0.55,
                    dataset=ds, seed=seed,        # ← enables leave-one-cell-out
                )
                wall = time.time() - t0
                acc = float((y_pred == y_te).mean())
                accs.append(acc)
                from sklearn.metrics import f1_score
                try:
                    f1 = float(f1_score(y_te, y_pred, average="macro"))
                except Exception:
                    f1 = 0.0
                row = {
                    "dataset": ds, "N_per_class": n, "seed": seed,
                    "method": "B7v3_honest_router",
                    "n_test": len(y_te),
                    "acc": round(acc, 4), "macro_f1": round(f1, 4),
                    "chosen_classifier": chosen,
                    "chosen_reason": trace.chosen_reason,
                    "mem_winner": trace.mem_winner,
                    "wall_time": round(wall, 2),
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                print(f"  {ds:13} n={n} seed={seed}: chose '{chosen}' "
                      f"→ acc={acc:.3f} ({wall:.1f}s)", flush=True)
    fh.close()
    print(f"\nHONEST B7v3 mean acc = {sum(accs)/len(accs)*100:.2f}%  ({len(accs)} cells)",
          flush=True)


if __name__ == "__main__":
    main()
