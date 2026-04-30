"""
Demo Step 6 — 用 LangGraph 把 4 个 Agent 串成有状态工作流
对应 TSci 论文的最终系统形态。

设计：
  - 一个共享 TypedDict 状态对象 PipelineState，从 START 一路流到 END
  - 4 个节点：curator / planner / forecaster / reporter
  - 每个节点是一个函数 (state) -> dict（dict 里的字段会合并进 state）
  - 节点内部完全复用 step 1-5 写好的工具，**不重复实现任何逻辑**

为什么这是真正的"多 Agent 系统"：
  - 状态显式可序列化（TypedDict）
  - 节点之间靠 schema 通信，不靠全局变量
  - LangGraph 自动管理执行顺序、错误恢复、可观测性
  - 想加循环 / 条件分支 / 反思机制只需改图结构，不动节点代码
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import TypedDict, Any

import numpy as np
import pandas as pd
from langgraph.graph import END, START, StateGraph


HERE = Path(__file__).parent


def _load_module(filename: str, alias: str):
    if alias in sys.modules:
        return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(alias, HERE / filename)
    m = importlib.util.module_from_spec(spec)
    sys.modules[alias] = m
    spec.loader.exec_module(m)
    return m


step1 = _load_module("01_curator_minimal.py", "step1")
step2 = _load_module("02_curator_visual.py", "step2")
step3 = _load_module("03_planner.py", "step3")
step4 = _load_module("04_forecaster.py", "step4")
step5 = _load_module("05_reporter.py", "step5")


# ---------- 1. 共享状态 schema ----------

class PipelineState(TypedDict, total=False):
    # —— Curator 写入 ——
    raw_series: pd.Series
    cleaned_series: pd.Series
    Q: dict
    strategy_pi: dict
    A: dict
    panel_path: str

    # —— Planner 写入 ——
    train: pd.Series
    val: pd.Series
    test: pd.Series
    candidates: list  # [{model, best_params, val_mape, reason, all_trials}]
    task_meta: dict

    # —— Forecaster 写入 ——
    preds_per_model: dict      # {name: np.ndarray}
    ensemble_decision: dict
    ensemble_weights: dict
    ensemble_forecast: list
    test_mape_per_model: dict
    ensemble_test_mape: float
    forecast_panel_path: str

    # —— Reporter 写入 ——
    prose: dict
    report_path: str


# ---------- 2. 四个节点 ----------

def node_curator(state: PipelineState) -> dict:
    print("\n▶ [node:curator] 数据诊断 + 清洗 + 可视化 + 视觉 LLM")
    raw = step2.make_synthetic_series()
    Q = step1.compute_quality_vector(raw)

    text_client, text_model = step1.make_client()
    pi = step1.ask_llm_for_strategy(text_client, text_model, Q)
    cleaned = step1.apply_strategy(raw, pi)

    png = step2.plot_curator_panel(cleaned, period=7)
    panel_path = HERE / "curator_panel.png"
    panel_path.write_bytes(png)

    vclient, vmodel = step2.make_vision_client()
    A = step2.ask_vlm_for_structure(vclient, vmodel, png)

    print(f"  Q.n={Q['n']}  A.trend={A.get('trend')}  A.period={A.get('seasonal_period')}")
    return {
        "raw_series": raw,
        "cleaned_series": cleaned,
        "Q": Q,
        "strategy_pi": pi,
        "A": A,
        "panel_path": str(panel_path),
    }


def node_planner(state: PipelineState) -> dict:
    print("\n▶ [node:planner] LLM 选候选 + 超参搜索")
    cleaned = state["cleaned_series"]
    test = cleaned.iloc[-step3.H_TEST:]
    val = cleaned.iloc[-(step3.H_TEST + step3.H_VAL):-step3.H_TEST]
    train = cleaned.iloc[:-(step3.H_TEST + step3.H_VAL)]

    pick = step3.llm_pick_candidates(state["Q"], state["A"])
    selected = [m for m in pick.get("selected", []) if m in step3.MODEL_LIBRARY] \
        or list(step3.MODEL_LIBRARY.keys())
    reasons = pick.get("reason_per_model", {})

    results = []
    for name in selected:
        r = step3.hp_search(name, train, val,
                             reason=reasons.get(name, "(no reason)"))
        if r is None:
            continue
        results.append(r)
    results.sort(key=lambda r: r.val_mape)

    candidates = [
        {
            "model": r.model,
            "best_params": r.best_params,
            "val_mape": r.val_mape,
            "reason": r.reason,
            "all_trials": [(p, s) for p, s in r.all_trials],
        }
        for r in results
    ]
    print("  候选: " + ", ".join(f"{c['model']}={c['val_mape']:.3f}" for c in candidates))

    return {
        "train": train,
        "val": val,
        "test": test,
        "candidates": candidates,
        "task_meta": {
            "horizon_test": step3.H_TEST,
            "horizon_val": step3.H_VAL,
            "period": step3.PERIOD,
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
        },
    }


def node_forecaster(state: PipelineState) -> dict:
    print("\n▶ [node:forecaster] 重训 + LLM 选集成 + test 评估")
    train, val, test = state["train"], state["val"], state["test"]
    candidates = state["candidates"]
    h = state["task_meta"]["horizon_test"]

    preds = step4.refit_and_forecast(candidates, train, val, h)
    decision = step4.llm_pick_ensemble(candidates)
    ensemble, weights = step4.apply_ensemble(decision, preds, candidates)

    test_mapes = {n: round(step3.mape(test.values, y), 4) for n, y in preds.items()}
    ens_mape = round(step3.mape(test.values, ensemble), 4)

    img_path = HERE / "forecaster_panel.png"
    step4.plot_forecasts(train, val, test, preds, ensemble, img_path)

    print(f"  策略={decision.get('strategy')}  ensemble_test_MAPE={ens_mape:.3f}")
    return {
        "preds_per_model": {k: v.tolist() for k, v in preds.items()},
        "ensemble_decision": decision,
        "ensemble_weights": weights,
        "ensemble_forecast": ensemble.tolist(),
        "test_mape_per_model": test_mapes,
        "ensemble_test_mape": ens_mape,
        "forecast_panel_path": str(img_path),
    }


def node_reporter(state: PipelineState) -> dict:
    print("\n▶ [node:reporter] LLM 写散文 + 代码组装 markdown")

    # 拼出 step5 期望的 states 形状
    states_for_step5 = {
        "curator": {
            "Q": state["Q"], "A": state["A"],
            "strategy_pi": state["strategy_pi"],
            "V_path": state["panel_path"],
        },
        "planner": {
            "task": state["task_meta"],
            "candidates": state["candidates"],
        },
        "forecaster": {
            "decision": state["ensemble_decision"],
            "weights": state["ensemble_weights"],
            "test_mape_per_model": state["test_mape_per_model"],
            "ensemble_test_mape": state["ensemble_test_mape"],
        },
    }
    facts = step5.build_facts_for_llm(states_for_step5)
    prose = step5.llm_write_prose(facts)
    md = step5.assemble_report(states_for_step5, prose)

    out = HERE / "final_report.md"
    out.write_text(md)
    print(f"  报告已写 → {out}")
    return {"prose": prose, "report_path": str(out)}


# ---------- 3. 构图 ----------

def build_graph():
    g = StateGraph(PipelineState)
    g.add_node("curator", node_curator)
    g.add_node("planner", node_planner)
    g.add_node("forecaster", node_forecaster)
    g.add_node("reporter", node_reporter)

    g.add_edge(START, "curator")
    g.add_edge("curator", "planner")
    g.add_edge("planner", "forecaster")
    g.add_edge("forecaster", "reporter")
    g.add_edge("reporter", END)

    return g.compile()


# ---------- 4. 主流程 ----------

def main() -> None:
    print("=== 构建 LangGraph 工作流 ===")
    app = build_graph()
    print("节点: curator → planner → forecaster → reporter")

    # 顺便把图结构画出来（仅在装了 graphviz 时成功）
    try:
        png = app.get_graph().draw_mermaid_png()
        (HERE / "pipeline_graph.png").write_bytes(png)
        print(f"图结构已画 → pipeline_graph.png")
    except Exception as e:
        print(f"(可选) 图可视化失败: {type(e).__name__}（不影响主流程）")

    print("\n=== 执行 ===")
    final_state: PipelineState = app.invoke({})

    print("\n=== 完成 ===")
    print(f"  Q: n={final_state['Q']['n']}, missing_ratio={final_state['Q']['missing_ratio']}")
    print(f"  A: {final_state['A']}")
    print(f"  candidates: {[c['model'] for c in final_state['candidates']]}")
    print(f"  ensemble strategy: {final_state['ensemble_decision'].get('strategy')}")
    print(f"  ensemble test MAPE: {final_state['ensemble_test_mape']}")
    print(f"  报告: {final_state['report_path']}")


if __name__ == "__main__":
    main()
