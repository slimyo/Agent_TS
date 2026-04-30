"""
Demo Step 5 — Reporter Agent
对应 TSci 论文第 5 节：把 Curator/Planner/Forecaster 的所有产物汇总，
让 LLM 只写散文段落，代码负责所有数字表格 + 图片插入。

设计原则：
  - 所有数字、表格、图片由代码生成（避免 LLM 编数字）
  - LLM 只写 5 段散文：执行摘要 / 数据洞察 / 选型解释 / 集成解释 / 局限性
  - 输出一份完整的 markdown 报告 final_report.md
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from openai import OpenAI


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


# ---------- 1. 加载所有状态 ----------

def load_all_states() -> dict:
    needed = ["curator_state.json", "planner_state.json", "forecaster_state.json"]
    out = {}
    for name in needed:
        p = HERE / name
        if not p.exists():
            raise FileNotFoundError(f"缺少 {name}，请按顺序跑完 step 1-4")
        out[name.replace("_state.json", "")] = json.loads(p.read_text())
    return out


# ---------- 2. 把所有事实摊平给 LLM 看（仅作输入，不让它编） ----------

def build_facts_for_llm(states: dict) -> dict:
    cur, plan, fc = states["curator"], states["planner"], states["forecaster"]
    return {
        "data_quality_Q": cur["Q"],
        "structure_profile_A": cur["A"],
        "preprocessing_strategy": cur["strategy_pi"],
        "task": plan["task"],
        "candidates": [
            {
                "model": c["model"],
                "best_params": c["best_params"],
                "val_mape": c["val_mape"],
                "all_trials": c["all_trials"],
                "selection_reason_from_planner": c["reason"],
            }
            for c in plan["candidates"]
        ],
        "ensemble_decision": fc["decision"],
        "ensemble_weights": fc["weights"],
        "test_mape_per_model": fc["test_mape_per_model"],
        "ensemble_test_mape": fc["ensemble_test_mape"],
    }


# ---------- 3. LLM 写散文（5 个固定字段） ----------

REPORTER_SYSTEM = """你是时序预测项目的高级数据科学家，正在给客户写一份**简洁严谨**的项目报告。
你**只能根据**给定的 facts JSON 写散文，**严禁编造任何数字**——所有数字必须直接复制 facts 中的值。

仅输出严格 JSON，含 5 个 markdown 散文字段（每段 80-180 字中文，不要 markdown 列表/表格/代码块）：

{
  "executive_summary": "<2-3 句话总览：什么数据、用了什么方法、最终精度如何。必须出现 ensemble_test_mape 数字。>",
  "data_insight": "<根据 Q 和 A 描述数据特征，必须引用 trend_slope、missing_ratio、A.seasonal_period 中至少 2 个数字。>",
  "model_selection_rationale": "<解释为何选这些候选模型，引用 candidates 中的 selection_reason_from_planner 和 val_mape 数字。>",
  "ensemble_decision_rationale": "<解释为何选这个集成策略，引用 ensemble_decision.reason 和具体权重。>",
  "limitations": "<诚实写出风险：训练数据是合成的、未做分布偏移检验、Curator 诊断没有置信度、模型库手工固定等。120-200 字。>"
}

不要输出 JSON 之外的任何文字。"""


def llm_write_prose(facts: dict) -> dict:
    client, model = step1.make_client()
    print(f"[报告生成模型: {model}]")

    messages = [
        {"role": "system", "content": REPORTER_SYSTEM},
        {"role": "user", "content": "facts:\n" + json.dumps(facts, ensure_ascii=False, indent=2)},
    ]
    try:
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=0.4,
            response_format={"type": "json_object"},
        )
    except Exception:
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=0.4
        )
    raw = resp.choices[0].message.content or ""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        import re
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        return json.loads(m.group(0))


# ---------- 4. 代码侧组装 markdown ----------

def fmt_q_table(q: dict) -> str:
    rows = [
        ("样本数 n", q["n"]),
        ("缺失数 / 比例", f"{q['missing_count']} ({q['missing_ratio']*100:.2f}%)"),
        ("均值 / 标准差", f"{q['mean']:.3f} / {q['std']:.3f}"),
        ("min / max", f"{q['min']:.3f} / {q['max']:.3f}"),
        ("线性趋势斜率", f"{q['trend_slope']:.6f}"),
        ("IQR 异常值数", q["outlier_count_iqr"]),
    ]
    out = ["| 指标 | 值 |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in rows]
    return "\n".join(out)


def fmt_a_table(a: dict) -> str:
    rows = [
        ("trend", a.get("trend")),
        ("seasonality", a.get("seasonality")),
        ("seasonal_period", a.get("seasonal_period")),
        ("stationarity", a.get("stationarity")),
    ]
    out = ["| 维度 | 判断 |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in rows]
    return "\n".join(out)


def fmt_candidates_table(candidates: list[dict]) -> str:
    out = ["| 模型 | 最优超参 | val_MAPE | Planner 选择理由 |",
           "|---|---|---|---|"]
    for c in candidates:
        out.append(
            f"| `{c['model']}` | `{c['best_params']}` | "
            f"{c['val_mape']:.3f} | {c['reason']} |"
        )
    return "\n".join(out)


def fmt_trials_table(candidates: list[dict]) -> str:
    out = ["| 模型 | 超参 | val_MAPE |", "|---|---|---|"]
    for c in candidates:
        for params, score in c["all_trials"]:
            out.append(f"| `{c['model']}` | `{params}` | {score:.3f} |")
    return "\n".join(out)


def fmt_test_results_table(per_model: dict, ens_mape: float, weights: dict) -> str:
    out = ["| 模型 | 集成权重 | test_MAPE |", "|---|---|---|"]
    for name, m in per_model.items():
        w = weights.get(name, 0.0)
        out.append(f"| `{name}` | {w:.2f} | {m:.3f} |")
    out.append(f"| **ensemble** | — | **{ens_mape:.3f}** |")
    return "\n".join(out)


def assemble_report(states: dict, prose: dict) -> str:
    cur, plan, fc = states["curator"], states["planner"], states["forecaster"]
    Q, A = cur["Q"], cur["A"]
    decision = fc["decision"]

    sections = [
        "# 时序预测项目报告（TSci Demo）",
        "",
        f"_自动生成于 Reporter Agent · 模型 = `glm-4.7-flash`_",
        "",
        "## 执行摘要",
        prose["executive_summary"],
        "",
        "---",
        "",
        "## 1. 数据画像",
        prose["data_insight"],
        "",
        "**质量向量 Q（来自 Curator §1）**：",
        "",
        fmt_q_table(Q),
        "",
        "**结构画像 A（多模态 LLM 看图判断）**：",
        "",
        fmt_a_table(A),
        "",
        f"_Curator 视觉诊断面板：_",
        "",
        "![curator panel](curator_panel.png)",
        "",
        f"**预处理策略 π**：缺失值 → `{cur['strategy_pi']['missing_strategy']}`；"
        f"异常值 → `{cur['strategy_pi']['outlier_strategy']}`",
        "",
        "---",
        "",
        "## 2. 模型选型",
        prose["model_selection_rationale"],
        "",
        "**候选模型（按 val_MAPE 排序）**：",
        "",
        fmt_candidates_table(plan["candidates"]),
        "",
        "---",
        "",
        "## 3. 集成决策",
        prose["ensemble_decision_rationale"],
        "",
        f"- **策略**: `{decision['strategy']}`"
        + (f"  (β = {decision.get('beta', 0)})"
           if decision["strategy"] == "performance_weighted" else ""),
        f"- **理由**: {decision.get('reason', '')}",
        f"- **权重**: " + ", ".join(f"`{k}`={v:.2f}" for k, v in fc["weights"].items()),
        "",
        "---",
        "",
        "## 4. 预测结果",
        "",
        fmt_test_results_table(fc["test_mape_per_model"],
                               fc["ensemble_test_mape"],
                               fc["weights"]),
        "",
        "_预测对比图（历史 + 各模型预测 + 集成 + 真值）：_",
        "",
        "![forecaster panel](forecaster_panel.png)",
        "",
        "---",
        "",
        "## 5. 假设与局限性",
        prose["limitations"],
        "",
        "---",
        "",
        "## 附录 A · 全部超参试验记录",
        "",
        fmt_trials_table(plan["candidates"]),
        "",
        "## 附录 B · 任务设置",
        "",
        f"- 训练 / 验证 / 测试 = "
        f"{plan['task']['n_train']} / {plan['task']['n_val']} / {plan['task']['n_test']}",
        f"- 季节周期 = {plan['task']['period']}",
        f"- 预测步长 (test horizon) = {plan['task']['horizon_test']}",
        "",
    ]
    return "\n".join(sections)


# ---------- 5. 主流程 ----------

def main() -> None:
    print("=== Step 1: 加载全部前置状态 ===")
    states = load_all_states()
    print("已加载: curator / planner / forecaster\n")

    print("=== Step 2: 摊平事实表给 LLM ===")
    facts = build_facts_for_llm(states)
    print(f"facts 字段: {list(facts.keys())}\n")

    print("=== Step 3: LLM 写 5 段散文 ===")
    prose = llm_write_prose(facts)
    for k, v in prose.items():
        print(f"\n[{k}]\n{v}")
    print()

    print("=== Step 4: 组装 markdown 报告 ===")
    md = assemble_report(states, prose)
    out = HERE / "final_report.md"
    out.write_text(md)
    print(f"已写入 → {out}（{len(md)} 字）")

    # 顺便保存 prose JSON 便于调试 / 复用
    (HERE / "reporter_prose.json").write_text(
        json.dumps(prose, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
