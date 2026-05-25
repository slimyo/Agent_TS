"""F2 · 跨 LLM 鲁棒性验证（plan §4.2 增补 + R6 部分回应）。

对比同账号 4 个 GLM 模型上跑 AdaptTS / TSci / LLMTime 的 MAE，
验证论文结论不绑定特定 LLM。

模型选择（覆盖差异化）：
  - glm-4-flash-250414: 默认非 reasoning（主实验用）
  - glm-4.7-flash:      reasoning model（content 走 reasoning_content）
  - glm-4-air:          非 reasoning 中端
  - glm-4-plus:         非 reasoning 旗舰

实验设置：
  - N=20（最具代表性的中等少样本）
  - 数据集 ETTh1 + ETTh2
  - 3 seeds

输出 results/f2_cross_llm.jsonl，每行 {model, method, dataset, N, H, seed, test_mae, wall_time}
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from research.utils.data_loader import load_series
from research.utils.metrics import all_metrics
from research.utils.splitter import few_shot_split

LLM_MODELS = [
    "glm-4-flash-250414",
    "glm-4.7-flash",
    "glm-4-air",
    "glm-4-plus",
]
METHODS = ["llmtime", "tsci", "adapt_ts"]
DATASETS = ["ETTh1", "ETTh2"]
N = 20
H = 96
SEEDS = [1, 42, 123]


def run_cell(model: str, method: str, dataset: str, seed: int) -> dict:
    # 关键：通过环境变量 override MODEL，让 utils/llm.py 拾取
    os.environ["MODEL"] = model
    # TSci 适配层在内部又设了 OPENAI_API_KEY/BASE_URL，无影响

    from importlib import import_module
    METHOD_MOD = {
        "llmtime":  "research.baseline.llmtime",
        "tsci":     "research.baseline.tsci",
        "adapt_ts": "research.agent.adapt_ts",
    }
    mod = import_module(METHOD_MOD[method])

    series, meta = load_series(dataset)
    sp = few_shot_split(series, N=N, H=H, seed=seed)
    t0 = time.time()
    y_hat = mod.predict(train=sp.train, val=sp.val, H=H,
                        seed=seed, season_m=meta.season_m)
    wall = time.time() - t0
    m = all_metrics(sp.test, y_hat, sp.train, season_m=meta.season_m)
    return {
        "model": model, "method": method, "dataset": dataset,
        "N": N, "H": H, "seed": seed, "wall_time": round(wall, 2),
        **{k: round(v, 6) for k, v in m.items()},
    }


def main():
    fp = Path("research/results/f2_cross_llm.jsonl")
    done = set()
    if fp.exists():
        for line in fp.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["model"], r["method"], r["dataset"], r["seed"]))
            except Exception: pass
    print(f"resuming: {len(done)} cells done", flush=True)

    total = len(LLM_MODELS) * len(METHODS) * len(DATASETS) * len(SEEDS)
    cnt = 0
    for model in LLM_MODELS:
        for method in METHODS:
            for ds in DATASETS:
                for seed in SEEDS:
                    cnt += 1
                    key = (model, method, ds, seed)
                    if key in done:
                        print(f"[{cnt}/{total}] skip {key}", flush=True)
                        continue
                    print(f"[{cnt}/{total}] {model:25} {method:10} {ds} seed={seed} ...", flush=True)
                    try:
                        r = run_cell(model, method, ds, seed)
                        with fp.open("a") as f:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")
                        print(f"  done in {r['wall_time']:.1f}s, MAE={r['mae']:.3f}", flush=True)
                    except Exception as e:
                        print(f"  FAIL: {e!r}", flush=True)
    print(f"\nall done. total rows in {fp}: {len(fp.read_text().splitlines())}")


if __name__ == "__main__":
    main()
