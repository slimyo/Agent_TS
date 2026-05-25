"""A8/A9 消融评估（plan §7.2 + §十五 N6 / §15.7 语义更新）。

对每个 (N, seed)，跑三个变体：
  - Full       AdaptTS v5c（含 Model Cards + Diagnosis Revision）
  - A8         w/o Model Cards
  - A9         w/o Diagnosis Revision

收集每个 reflect_step 的 root_cause，量化文本质量指标：
  - 长度（字符数）
  - 引用的 MAE 数字数（含小数点的数）
  - 引用的诊断词数（trend/season/stat/平稳/趋势/季节）
  - 引用的策略名数（chronos/arima/drift/seasonal/llmtime）
  - 引用的 Model Card 词数（assume/strength/weakness/typical_failure/假设/弱）

输出 JSON 到 results/a8_a9_trace.json + 简表打印。
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from importlib import import_module
from pathlib import Path

import numpy as np

from research.utils.data_loader import load_series
from research.utils.metrics import mae
from research.utils.splitter import few_shot_split


VARIANTS = {
    "Full": "research.agent.adapt_ts",
    "A8":   "research.agent.ablation_a8",
    "A9":   "research.agent.ablation_a9",
}


DIAG_WORDS = ["trend", "season", "stat", "stationar",
              "趋势", "季节", "周期", "平稳"]
STRAT_WORDS = ["chronos", "arima", "drift", "seasonal", "llmtime", "naive"]
CARD_WORDS = ["assume", "strength", "weakness", "typical_failure",
              "假设", "弱点", "失败"]


def quality_metrics(rc: str) -> dict:
    nums = re.findall(r"\d+\.\d+|\d+", rc)
    rc_l = rc.lower()
    return {
        "len_chars": len(rc),
        "n_numbers": len(nums),
        "n_diag_words": sum(rc_l.count(w) for w in DIAG_WORDS),
        "n_strategy_words": sum(rc_l.count(w) for w in STRAT_WORDS),
        "n_card_words": sum(rc_l.count(w) for w in CARD_WORDS),
    }


def run_variant(variant: str, N: int, seed: int, series, season_m: int) -> dict:
    mod = import_module(VARIANTS[variant])
    sp = few_shot_split(series, N=N, H=96, seed=seed)
    y = mod.predict(sp.train, sp.val, 96, seed=seed, season_m=season_m)
    t = mod.predict.last_trace
    reflects = []
    qm_sum = defaultdict(float)
    for step in t.reflect_steps:
        qm = quality_metrics(step.root_cause)
        reflects.append({
            "per_strat_mae": {k: round(v, 3) for k, v in step.per_strat_mae.items()},
            "root_cause": step.root_cause,
            "diagnosis_revision": step.diagnosis_revision,
            "plan_after": step.plan_after.strategies if step.plan_after else None,
            "quality": qm,
        })
        for k, v in qm.items():
            qm_sum[k] += v
    n_steps = max(1, len(reflects))
    return {
        "variant": variant,
        "N": N, "seed": seed, "start_idx": int(sp.start_idx),
        "test_mae": round(float(mae(sp.test, y)), 3),
        "final_plan": {"strategies": t.final_plan.strategies,
                       "weights": [round(w, 2) for w in t.final_plan.weights]},
        "diagnosis_revised": t.diagnosis_revised,
        "n_reflects": len(reflects),
        "reflects": reflects,
        "avg_quality": {k: round(v / n_steps, 2) for k, v in qm_sum.items()},
    }


def main():
    import time, os
    series, meta = load_series("ETTh1")
    # A8/A9 主要看 root_cause 文本质量（plan §15.7 语义更新）而非 MAE，
    # 单 seed 已足够验证（避免 chronos CPU 推理三倍开销）。需要 3 seeds 时设 A8A9_SEEDS=1,42,123
    seeds_env = os.environ.get("A8A9_SEEDS", "1")
    seeds = [int(s) for s in seeds_env.split(",")]
    N = 20      # plan §7.3 固定 N=20 做 ablation
    print(f"using seeds={seeds}", flush=True)

    # 增量 append jsonl（崩溃/卡死也保留已完成的 runs）
    fp_inc = Path("research/results/a8_a9_runs.jsonl")
    done = set()
    if fp_inc.exists():
        for line in fp_inc.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["variant"], r["seed"]))
            except Exception:
                pass
    print(f"resuming: already done {len(done)} runs", flush=True)

    out: list[dict] = []
    for variant in VARIANTS:
        for seed in seeds:
            if (variant, seed) in done:
                print(f"  skip {variant} seed={seed} (already in jsonl)", flush=True)
                continue
            print(f"running {variant} seed={seed} ...", flush=True)
            t0 = time.time()
            try:
                rec = run_variant(variant, N, seed, series, meta.season_m)
                out.append(rec)
                # 立即 append 落盘
                with fp_inc.open("a") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"  done in {time.time()-t0:.1f}s, test_mae={rec['test_mae']}", flush=True)
            except Exception as e:
                print(f"  FAIL: {e!r}", flush=True)

    # 最终再写一份完整 json（合并已 jsonl 中的）
    all_runs = []
    for line in fp_inc.read_text().splitlines():
        try:
            all_runs.append(json.loads(line))
        except Exception:
            pass
    fp = Path("research/results/a8_a9_trace.json")
    fp.write_text(json.dumps(all_runs, ensure_ascii=False, indent=2))
    print(f"\nfinal: {len(all_runs)} runs → {fp}", flush=True)
    out = all_runs

    # 汇总表
    print("\n=== A8/A9 消融汇总（ETTh1 N=20, 3 seeds） ===")
    agg = defaultdict(lambda: defaultdict(list))
    for r in out:
        v = r["variant"]
        agg[v]["test_mae"].append(r["test_mae"])
        agg[v]["n_reflects"].append(r["n_reflects"])
        for k, val in r["avg_quality"].items():
            agg[v]["avg_" + k].append(val)
        agg[v]["diag_revised_count"].append(1 if r["diagnosis_revised"] else 0)

    cols = ["test_mae", "n_reflects", "avg_len_chars", "avg_n_numbers",
            "avg_n_diag_words", "avg_n_strategy_words", "avg_n_card_words",
            "diag_revised_count"]
    print(f"{'variant':>8} " + " ".join(f"{c:>15}" for c in cols))
    for v in VARIANTS:
        row = [v]
        for c in cols:
            vals = agg[v][c]
            avg = sum(vals) / len(vals) if vals else 0
            row.append(f"{avg:>15.2f}")
        print(f"{row[0]:>8} " + " ".join(row[1:]))


if __name__ == "__main__":
    main()
