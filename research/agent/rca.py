"""P6.1 · Prediction Failure RCA Agent。

输入：(train, prediction, truth, optional diagnosis features)
输出：JSON {predicted_fault, supporting_evidence, hypothesis}

两条路径：
1. B5 Agent · diagnosis + Model Cards + structured prompt
2. B1 baseline · LLM 直接看数字 + 自由文本输出
"""
from __future__ import annotations

import json
import re

import numpy as np

from research.agent.curator_uq import diagnose
from research.agent.model_cards import MODEL_CARDS, render_cards_block
from research.utils.fault_taxonomy import FAULT_NAMES
from research.utils.llm import chat_cached

FAULT_TAXONOMY_DESCRIPTION = """\
5 类标准 fault（多选可，但 primary 单选）：

- **trend_break**: train/test 中段出现阶跃式 mean shift（≥2σ），导致后续模型无法捕捉非线性 trend
- **seasonal_flip**: 季节性符号翻转（ACF lag=m 前后段反号），模型套用旧周期假设错误
- **variance_explode**: 后段 std/前段 std > 2，方差非平稳，点预测会被新方差水平拉飞
- **outlier_burst**: ≥1 个 |z-score| > 3 离群点，把短验证集/测试段 MAE 拉爆
- **stationarity_flip**: train/test mean+var 明显不一致 (split-half) → 非平稳，ARIMA 假设违反，Chronos 训练分布外推

**特殊类 out_of_taxonomy**：如果你认为以上 5 类都不准确描述此 cell 的故障——例如缺失数据 (gap/flat) / 重噪声 (signal-to-noise drop) / 信号塌缩 (mode collapse to near-constant) / 频率变化 (chirp/frequency modulation) / 量化损失 (discrete levels / staircase) 等——请输出 `primary_fault: "out_of_taxonomy"` 并在 supporting_evidence 详细描述你看到的实际异常类型。
"""

AGENT_RCA_PROMPT = """\
你是时序预测失败根因分析专家。根据下面信息，判断这次预测失败属于哪类根因。

{dataset_prior}
【序列诊断】（来自 Curator 三路置信度）
{diag_text}

【模型 Model Cards（候选策略的能力描述）】
{cards}

【失败 cell 信息】
- 数据集: {dataset} N={N} seed={seed} H={H}
- 使用策略: {strategy}
- 测试 MAE: {adapt_mae:.4f}   Chronos-2 baseline MAE: {c2_mae:.4f}   恶化倍数: {ratio:.2f}×

【训练序列描述（first/last 8 点 + 统计）】
- first 8: {train_head}
- last 8: {train_tail}
- mean={train_mean:.3f} std={train_std:.3f}

【测试序列（真实值，first 8 + last 8）】
- first 8: {test_head}
- last 8: {test_tail}

【预测序列（first 8 + last 8）】
- first 8: {pred_head}
- last 8: {pred_tail}

任务：判断这次预测失败的主要根因。

**重要 — 决策规则（task #45 v4 fix bias）**：
1. 先看 5 类标准 fault（见 taxonomy）。**每一类的诊断阈值是硬约束**：variance_explode 需 variance_ratio ≥ 2；outlier_burst 需 outlier_count_z3 ≥ 1；trend_break 需 split-half mean shift ≥ 2σ。
2. **若你引用的诊断数字与某一类的硬约束矛盾**（如 variance_ratio=0.7 却要分类 variance_explode），**禁止该分类**——输出 `out_of_taxonomy` 并描述实际异常。
3. 若 ≥3 类同时强匹配 → 优先选 evidence 最强的那个。
4. 若没有任何一类的硬约束被满足 → 必须输出 `out_of_taxonomy`。

{taxonomy}

输出 JSON（仅 JSON，无其他文字）：
{{
  "primary_fault": "<one of: trend_break, seasonal_flip, variance_explode, outlier_burst, stationarity_flip, out_of_taxonomy>",
  "secondary_faults": ["...", "..."],
  "evidence_consistency_check": "<我引用的诊断数字: ...; 该数字是否满足上述 primary_fault 的硬约束? yes/no>",
  "supporting_evidence": "1-2 句解释（引用具体数字 / 诊断置信度 / Model Card 假设）；若为 out_of_taxonomy，描述实际异常模式",
  "hypothesized_repair": "如果你能选另一策略，应该选什么？为什么？"
}}
"""


B1_DIRECT_PROMPT = """\
你是时序预测失败根因分析专家。判断这次预测失败的根本原因。

【失败 cell】
- 数据集: {dataset} N={N} seed={seed}
- 使用策略: {strategy}
- 测试 MAE: {adapt_mae:.4f}   Chronos-2 baseline MAE: {c2_mae:.4f}

【训练序列】（first 8 + last 8）
- first: {train_head}
- last: {train_tail}

【测试真实值】
- first: {test_head}
- last: {test_tail}

【预测值】
- first: {pred_head}
- last: {pred_tail}

从以下 5 类中**严格选一**作为 primary_fault：

{taxonomy}

输出 JSON（仅 JSON）：
{{
  "primary_fault": "<one>",
  "secondary_faults": ["..."],
  "supporting_evidence": "..."
}}
"""


def _to_str_list(arr, n=8):
    a = np.asarray(arr).flatten()
    if len(a) <= 2 * n:
        return [f"{x:.3f}" for x in a]
    return [f"{x:.3f}" for x in a]


def _format_diag(d) -> str:
    """把 Diagnosis 转 readable text。v2 加 outlier_count + variance_ratio 显式展示。"""
    return (
        f"  n={d.n} mean={d.mean:.3f} std={d.std:.3f}\n"
        f"  trend_slope={d.trend_slope:.4g} trend_tstat={d.trend_tstat:.2f}\n"
        f"  ADF p-value={d.adf_pvalue:.3f}, ACF peak={d.acf_peak_value:.3f}@lag={d.acf_peak_lag}\n"
        f"  outlier_count_z3={getattr(d, 'outlier_count_z3', 0)}  ← MAD-based z>3 离群点数\n"
        f"  variance_ratio={getattr(d, 'variance_ratio', 1.0):.2f}  ← late_std/early_std (>2 = variance explode)\n"
        f"  trend_conf: stat={d.trend_conf_stat} llm={d.trend_conf_llm} xc={d.trend_conf_xc}\n"
        f"  season_conf: stat={d.season_conf_stat} llm={d.season_conf_llm} xc={d.season_conf_xc}\n"
        f"  stat_conf: stat={d.stat_conf_stat} llm={d.stat_conf_llm} xc={d.stat_conf_xc}"
    )


def _parse_rca_json(text: str) -> dict:
    """提取 JSON 块。若失败回退到正则提取关键字段。"""
    text = text.strip()
    # 去 markdown 围栏
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        pass
    # 正则兜底：找 primary_fault
    m = re.search(r'"primary_fault"\s*:\s*"([^"]+)"', text)
    primary = m.group(1) if m else "unknown"
    return {"primary_fault": primary, "secondary_faults": [],
            "supporting_evidence": text[:500]}


def _validate_fault(name: str) -> str:
    """规范化 fault 名（防 LLM 加引号 / 错拼）。允许 out_of_taxonomy。"""
    name = name.strip().lower().replace("-", "_").replace(" ", "_")
    allowed = list(FAULT_NAMES) + ["out_of_taxonomy", "oot"]
    if name in allowed:
        return "out_of_taxonomy" if name == "oot" else name
    # fuzzy match in-taxonomy
    for f in FAULT_NAMES:
        if f.replace("_", "") in name.replace("_", ""):
            return f
    # fuzzy match OOT
    for k in ["out_of", "oot", "unknown", "other", "neither"]:
        if k in name:
            return "out_of_taxonomy"
    return "unknown"


_ABSTAIN_CACHE = None


def _load_abstain_head():
    """task #46 · 加载 trained abstain head（in-tax / OOT 二分类）。"""
    global _ABSTAIN_CACHE
    if _ABSTAIN_CACHE is not None:
        return _ABSTAIN_CACHE
    import os, pickle
    p = "research/results/abstain_head.pkl"
    if not os.path.exists(p):
        _ABSTAIN_CACHE = None
        return None
    with open(p, "rb") as f:
        _ABSTAIN_CACHE = pickle.load(f)
    return _ABSTAIN_CACHE


def _apply_abstain_override(train_series, parsed: dict, threshold: float = 0.5) -> dict:
    """task #46 · 若 abstain head predict OOT (prob > threshold) → 覆盖 LLM 输出。"""
    head = _load_abstain_head()
    if head is None:
        return parsed
    try:
        from research.agent.abstain_head import extract_abstain_features
        feat = extract_abstain_features(train_series).reshape(1, -1)
        feat_z = head["scaler"].transform(feat)
        proba = head["clf"].predict_proba(feat_z)[0, 1]  # P(OOT)
        parsed["_abstain_proba"] = float(proba)
        if proba > threshold:
            parsed["_abstain_override"] = True
            parsed["_original_primary"] = parsed.get("primary_fault")
            parsed["primary_fault"] = "out_of_taxonomy"
            parsed["supporting_evidence"] = (
                f"[Abstain head override, p(OOT)={proba:.2f}] "
                + (parsed.get("supporting_evidence", "") or "")
            )[:400]
        else:
            parsed["_abstain_override"] = False
    except Exception:
        pass
    return parsed


def agent_rca(train: np.ndarray, val: np.ndarray, test: np.ndarray,
              prediction: np.ndarray, dataset: str, N: int, seed: int, H: int,
              strategy: str, adapt_mae: float, c2_mae: float,
              season_m: int = 1, llm_model: str | None = None,
              use_abstain: bool = False,
              use_dataset_prior: bool = False) -> dict:
    """B5 Agent path: diagnosis + Model Cards + structured prompt."""
    d = diagnose(train, season_m=season_m)
    cards = render_cards_block(["chronos2", "chronos_bolt", "llmtime",
                                 "arima_ets", "naive_drift"])
    # task #17 · dataset semantic prior
    if use_dataset_prior:
        from research.agent.dataset_priors import render_prior_block
        dataset_prior = render_prior_block(dataset)
    else:
        dataset_prior = ""
    prompt = AGENT_RCA_PROMPT.format(
        dataset_prior=dataset_prior,
        diag_text=_format_diag(d),
        cards=cards,
        dataset=dataset, N=N, seed=seed, H=H,
        strategy=strategy, adapt_mae=adapt_mae, c2_mae=c2_mae,
        ratio=adapt_mae / max(c2_mae, 1e-9),
        train_head=_to_str_list(train[:8]),
        train_tail=_to_str_list(train[-8:]),
        train_mean=float(train.mean()), train_std=float(train.std()),
        test_head=_to_str_list(test[:8]),
        test_tail=_to_str_list(test[-8:]),
        pred_head=_to_str_list(prediction[:8]),
        pred_tail=_to_str_list(prediction[-8:]),
        taxonomy=FAULT_TAXONOMY_DESCRIPTION,
    )
    messages = [{"role": "user", "content": prompt}]
    if llm_model:
        response = chat_cached(messages, model=llm_model)
    else:
        response = chat_cached(messages)
    parsed = _parse_rca_json(response)
    parsed["primary_fault"] = _validate_fault(parsed.get("primary_fault", "unknown"))
    parsed["secondary_faults"] = [_validate_fault(s)
                                  for s in parsed.get("secondary_faults", [])
                                  if _validate_fault(s) != "unknown"]
    parsed["_raw"] = response[:1000]
    parsed["_path"] = "agent"
    if use_abstain:
        parsed = _apply_abstain_override(train, parsed)
    return parsed


def b1_direct_rca(train: np.ndarray, test: np.ndarray, prediction: np.ndarray,
                  dataset: str, N: int, seed: int, strategy: str,
                  adapt_mae: float, c2_mae: float,
                  llm_model: str | None = None) -> dict:
    """B1 baseline: LLM 直接看数字，无诊断 / 无 Model Cards。"""
    prompt = B1_DIRECT_PROMPT.format(
        dataset=dataset, N=N, seed=seed, strategy=strategy,
        adapt_mae=adapt_mae, c2_mae=c2_mae,
        train_head=_to_str_list(train[:8]),
        train_tail=_to_str_list(train[-8:]),
        test_head=_to_str_list(test[:8]),
        test_tail=_to_str_list(test[-8:]),
        pred_head=_to_str_list(prediction[:8]),
        pred_tail=_to_str_list(prediction[-8:]),
        taxonomy=FAULT_TAXONOMY_DESCRIPTION,
    )
    messages = [{"role": "user", "content": prompt}]
    if llm_model:
        response = chat_cached(messages, model=llm_model)
    else:
        response = chat_cached(messages)
    parsed = _parse_rca_json(response)
    parsed["primary_fault"] = _validate_fault(parsed.get("primary_fault", "unknown"))
    parsed["secondary_faults"] = [_validate_fault(s)
                                  for s in parsed.get("secondary_faults", [])
                                  if _validate_fault(s) != "unknown"]
    parsed["_raw"] = response[:1000]
    parsed["_path"] = "b1_direct"
    return parsed
