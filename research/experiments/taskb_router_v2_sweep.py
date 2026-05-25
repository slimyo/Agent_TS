"""task #37 · B7v2 Router (N<7 fallback) sweep。

复用 task #35 的设置，加 n_min_for_routing=7 参数。
输出独立文件 taskb_router_v2_ucr.jsonl 与 v1 对比。
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


def main():
    out = Path("research/results/taskb_router_v2_ucr.jsonl")
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
                    n_min_for_routing=7,   # v2: N<7 强制 fallback
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
                    "method": "B7v2_router",
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
    main()
