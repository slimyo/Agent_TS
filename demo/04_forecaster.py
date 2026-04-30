"""
Demo Step 4 — Forecaster Agent
对应 TSci 论文第 4 节：接收 Planner 选好的 top-k 调好参模型，由 LLM 决定
集成策略，输出最终预测并在 test 集上评估。

流程：
  load planner_state.json + 重建数据
  → 在 train+val 合并集上重训每个候选（让模型见到最近的数据）
  → 各模型 forecast H_TEST 步
  → LLM (glm-4.7-flash) 看 val MAPE 选集成策略
       single_best / performance_weighted / robust_aggregation
  → 应用集成（权重只能依赖 val，不能看 test）
  → 在 test 上算 MAPE，画图，写 forecaster_state.json

关键约束：集成权重在算 test 前必须定下来。LLM 只看 val MAPE 决策，
        否则就是数据泄漏。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openai import OpenAI


HERE = Path(__file__).parent


def _load_module(filename: str, alias: str):
    if alias in sys.modules:
        return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(alias, HERE / filename)
    m = importlib.util.module_from_spec(spec)
    sys.modules[alias] = m  # 必须先注册，dataclass/typing 会反查 sys.modules
    spec.loader.exec_module(m)
    return m


step1 = _load_module("01_curator_minimal.py", "step1")
step2 = _load_module("02_curator_visual.py", "step2")
step3 = _load_module("03_planner.py", "step3")


# ---------- 1. 数据 ----------

def prepare() -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    return step3.prepare_data()  # cleaned, train, val, test


# ---------- 2. 在 train+val 上重训每个候选并预测 H_TEST 步 ----------

def refit_and_forecast(
    candidates: list[dict], train: pd.Series, val: pd.Series, h: int
) -> dict[str, np.ndarray]:
    """返回 {model_name: forecast_array of shape (h,)}"""
    train_full = pd.concat([train, val])
    preds: dict[str, np.ndarray] = {}
    for c in candidates:
        name = c["model"]
        params = c["best_params"]
        # 注意：JSON 把 tuple 序列化成 list，要还原回来
        if "order" in params and isinstance(params["order"], list):
            params = {**params, "order": tuple(params["order"])}
        fit_fn = step3.MODEL_LIBRARY[name]["fit"]
        forecaster = fit_fn(train_full, params)
        preds[name] = np.asarray(forecaster(h))
        print(f"  [{name}] 已重训 + 预测 {h} 步")
    return preds


# ---------- 3. LLM 选集成策略 ----------

ENSEMBLE_SYSTEM = """你是时序集成预测专家。给定 top-k 候选模型和它们的 val MAPE，
从以下三种策略中选一种：

1. "single_best": 当某个模型在 val 上明显胜出（与第二名 MAPE 差距 > 30%）时直接用它。
2. "performance_weighted": 按 1/MAPE^beta 加权平均，再向均匀分布做 shrinkage（λ=0.1）。多数情况的稳妥选择。
3. "robust_aggregation": 模型很多 (k>=3) 且预测彼此分歧大时，用 median 抗极端预测。

仅输出严格 JSON：
{
  "strategy": "single_best" | "performance_weighted" | "robust_aggregation",
  "beta": <number, performance_weighted 时 1.0~3.0；其它策略填 0>,
  "reason": "<一句中文，必须引用具体 val_mape 数字作为依据>"
}
不要输出 JSON 之外的任何文字。"""


def llm_pick_ensemble(candidates: list[dict]) -> dict:
    brief = [
        {"model": c["model"], "val_mape": c["val_mape"], "best_params": c["best_params"]}
        for c in candidates
    ]
    user_msg = (
        f"k = {len(candidates)} 个候选模型及其 val MAPE：\n"
        f"{json.dumps(brief, ensure_ascii=False, indent=2)}"
    )

    client, model = step1.make_client()
    print(f"[集成决策模型: {model}]")

    messages = [
        {"role": "system", "content": ENSEMBLE_SYSTEM},
        {"role": "user", "content": user_msg},
    ]
    try:
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=0.2,
            response_format={"type": "json_object"},
        )
    except Exception:
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=0.2
        )
    raw = resp.choices[0].message.content or ""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        import re
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        return json.loads(m.group(0))


# ---------- 4. 三种集成实现 ----------

def ensemble_single_best(preds: dict, candidates: list[dict]) -> tuple[np.ndarray, dict]:
    best = min(candidates, key=lambda c: c["val_mape"])
    weights = {c["model"]: (1.0 if c["model"] == best["model"] else 0.0) for c in candidates}
    yhat = preds[best["model"]]
    return yhat, weights


def ensemble_perf_weighted(
    preds: dict, candidates: list[dict], beta: float, shrink: float = 0.1
) -> tuple[np.ndarray, dict]:
    eps = 1e-6
    raw = np.array([(c["val_mape"] + eps) ** (-beta) for c in candidates])
    w_perf = raw / raw.sum()
    k = len(candidates)
    w = (1 - shrink) * w_perf + shrink * (1.0 / k)
    w = w / w.sum()  # 数值再归一化
    weights = {c["model"]: float(w[i]) for i, c in enumerate(candidates)}

    stack = np.stack([preds[c["model"]] for c in candidates], axis=0)
    yhat = (stack * w[:, None]).sum(axis=0)
    return yhat, weights


def ensemble_robust(preds: dict, candidates: list[dict]) -> tuple[np.ndarray, dict]:
    stack = np.stack([preds[c["model"]] for c in candidates], axis=0)
    yhat = np.median(stack, axis=0)
    weights = {c["model"]: 1.0 / len(candidates) for c in candidates}  # 仅作记录
    return yhat, weights


def apply_ensemble(
    decision: dict, preds: dict, candidates: list[dict]
) -> tuple[np.ndarray, dict]:
    strategy = decision["strategy"]
    if strategy == "single_best":
        return ensemble_single_best(preds, candidates)
    if strategy == "performance_weighted":
        beta = float(decision.get("beta", 1.0))
        return ensemble_perf_weighted(preds, candidates, beta=beta)
    if strategy == "robust_aggregation":
        return ensemble_robust(preds, candidates)
    raise ValueError(f"未知集成策略: {strategy}")


# ---------- 5. 评估 + 画图 ----------

def plot_forecasts(
    train: pd.Series, val: pd.Series, test: pd.Series,
    preds: dict, ensemble: np.ndarray, out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    # 历史段（最近 60 天）
    hist = pd.concat([train, val]).iloc[-60:]
    ax.plot(hist.index, hist.values, color="#888", lw=1, label="history")
    # 真实 test
    ax.plot(test.index, test.values, color="black", lw=2, marker="o",
            markersize=4, label="test (truth)")
    # 各模型预测
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
    for (name, yhat), c in zip(preds.items(), colors):
        ax.plot(test.index, yhat, color=c, lw=1, linestyle="--",
                alpha=0.7, label=f"{name}")
    # 集成
    ax.plot(test.index, ensemble, color="red", lw=2.2,
            label="ensemble", marker="x", markersize=6)
    ax.axvline(test.index[0], color="gray", lw=0.5, linestyle=":")
    ax.set_title("Forecaster: per-model forecasts vs ensemble vs truth (test)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------- 6. 主流程 ----------

def main() -> None:
    print("=== Step 1: 加载 Planner 状态 ===")
    plan_path = HERE / "planner_state.json"
    if not plan_path.exists():
        raise FileNotFoundError("先跑 03_planner.py")
    plan = json.loads(plan_path.read_text())
    candidates = plan["candidates"]
    h = plan["task"]["horizon_test"]
    print(f"k = {len(candidates)} 候选, H_test = {h}")
    for c in candidates:
        print(f"  {c['model']:18s}  val_MAPE={c['val_mape']:.3f}  params={c['best_params']}")
    print()

    print("=== Step 2: 准备数据 ===")
    cleaned, train, val, test = prepare()
    print(f"train={len(train)}  val={len(val)}  test={len(test)}\n")

    print(f"=== Step 3: 在 train+val 上重训 + 预测 {h} 步 ===")
    preds = refit_and_forecast(candidates, train, val, h)
    print()

    print("=== Step 4: LLM 选集成策略 ===")
    decision = llm_pick_ensemble(candidates)
    print(json.dumps(decision, ensure_ascii=False, indent=2), "\n")

    print("=== Step 5: 应用集成 ===")
    ensemble, weights = apply_ensemble(decision, preds, candidates)
    print(f"集成权重: {json.dumps(weights, ensure_ascii=False)}\n")

    print("=== Step 6: 在 test 上评估 ===")
    test_mapes = {
        name: round(step3.mape(test.values, yhat), 4) for name, yhat in preds.items()
    }
    ens_mape = round(step3.mape(test.values, ensemble), 4)
    print("各模型 test MAPE:")
    for name, m in test_mapes.items():
        print(f"  {name:18s}  test_MAPE={m:.3f}")
    print(f"  {'ENSEMBLE':18s}  test_MAPE={ens_mape:.3f}")
    print()

    print("=== Step 7: 画图 + 写状态 ===")
    img_path = HERE / "forecaster_panel.png"
    plot_forecasts(train, val, test, preds, ensemble, img_path)
    print(f"图已保存 → {img_path}")

    state = {
        "decision": decision,
        "weights": weights,
        "test_mape_per_model": test_mapes,
        "ensemble_test_mape": ens_mape,
        "ensemble_forecast": ensemble.tolist(),
        "test_index": [str(t) for t in test.index],
        "test_truth": test.values.tolist(),
        "panel_path": str(img_path),
    }
    state_path = HERE / "forecaster_state.json"
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    print(f"状态已保存 → {state_path}")


if __name__ == "__main__":
    main()
