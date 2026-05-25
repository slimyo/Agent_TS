"""Phase 3.2 · Adaptive Planner（plan §四层二）。

输入：Curator 诊断（含置信度）+ (train, val, H)
输出：候选策略列表 + 策略组合方式（single / ensemble / safe-fallback）

设计要点（plan §五 + §七）：
  - 低置信度 → 集成多个互补策略（plan §五 步骤三）
  - 高置信度 → 单一精细策略
  - 中置信度 → 2~3 策略弱集成
  - 极端短序列（N < 15）→ 强制保守（chronos + naive_seasonal 集成）

候选库（小而精，避免训练数据不足时过拟合）：
  - naive_seasonal: 季节朴素法
  - naive_drift:    漂移法（用最后两点斜率）
  - arima_ets:      ARIMA + ETS 自动选择
  - chronos:        预训练基础模型（最稳的少样本选择）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from research.agent.curator_uq import Diagnosis

Combine = Literal["single", "ensemble", "safe"]


@dataclass
class Plan:
    strategies: list[str]
    combine: Combine
    reason: str
    # 各策略的初始权重（ensemble 用），单策略时全部 1.0 到首个
    weights: list[float] = field(default_factory=list)


def _conf_score(conf: str) -> int:
    return {"low": 0, "mid": 1, "high": 2}[conf]


def make_plan(diag: Diagnosis, N: int, H: int,
              conf_source: Literal["stat", "llm", "xc"] = "xc") -> Plan:
    """根据置信度自适应选择策略组合。

    Args:
        conf_source: 用哪一路置信度做决策（plan §5.2 三种来源）。
                     默认 xc（双路交叉），论文中也会对比 stat/llm 单路。
    """
    suffix = f"_conf_{conf_source}"
    trend = getattr(diag, "trend" + suffix)
    season = getattr(diag, "season" + suffix)
    stat = getattr(diag, "stat" + suffix)

    # 是否启用 LLMTime 候选：N≤30 时 LLMTime 实测最优（finish.md §3.1）
    use_llmtime = N <= 30

    # 极端冷启动：强制保守。N≤12 时 LLM ICL 比 chronos 更优
    if N <= 12:
        if use_llmtime:
            return Plan(
                strategies=["llmtime", "chronos", "naive_drift"],
                combine="safe",
                reason=f"N={N} 极端冷启动，LLM ICL + 基础模型 + 漂移兜底",
                weights=[0.5, 0.3, 0.2],
            )
        return Plan(
            strategies=["chronos", "naive_drift"],
            combine="safe",
            reason=f"N={N} 极端冷启动",
            weights=[0.8, 0.2],
        )

    # 综合置信度：取三维度最低值，越低越要保守
    min_conf = min(_conf_score(trend), _conf_score(season), _conf_score(stat))

    # 决策矩阵
    if min_conf == 0:  # 任一维度低置信
        if use_llmtime:
            return Plan(
                strategies=["llmtime", "chronos", "arima_ets"],
                combine="ensemble",
                reason=f"诊断低置信（trend={trend},season={season},stat={stat}），N≤30 用 LLM+基础+ARIMA 三路集成",
                weights=[0.5, 0.3, 0.2],
            )
        if season == "low" and trend == "high":
            return Plan(
                strategies=["arima_ets", "chronos", "naive_drift"],
                combine="ensemble",
                reason=f"诊断有低置信项（trend={trend},season={season},stat={stat}），三路集成",
                weights=[0.4, 0.4, 0.2],
            )
        return Plan(
            strategies=["chronos", "arima_ets", "naive_seasonal"],
            combine="ensemble",
            reason=f"诊断低置信（trend={trend},season={season},stat={stat}），三路集成",
            weights=[0.5, 0.3, 0.2],
        )

    if min_conf == 1:  # 全部 mid 或更高
        if use_llmtime:
            return Plan(
                strategies=["llmtime", "chronos"],
                combine="ensemble",
                reason=f"诊断中等置信（trend={trend},season={season},stat={stat}），N≤30 双路弱集成 LLM+chronos",
                weights=[0.5, 0.5],
            )
        return Plan(
            strategies=["chronos", "arima_ets"],
            combine="ensemble",
            reason=f"诊断中等置信（trend={trend},season={season},stat={stat}），双路弱集成",
            weights=[0.5, 0.5],
        )

    # min_conf == 2: 全部 high → 单一精细策略
    # 季节性强 → ARIMA 季节版能利用周期；否则 chronos 更稳
    if season == "high":
        return Plan(
            strategies=["arima_ets"],
            combine="single",
            reason="所有维度高置信，且季节性强，选 ARIMA+ETS 单策略",
            weights=[1.0],
        )
    return Plan(
        strategies=["chronos"],
        combine="single",
        reason="所有维度高置信，无强季节，选 Chronos 单策略",
        weights=[1.0],
    )


if __name__ == "__main__":
    from research.agent.curator_uq import diagnose
    from research.utils.data_loader import load_series
    from research.utils.splitter import few_shot_split
    s, meta = load_series("ETTh1")
    for N in [10, 20, 50, 100]:
        sp = few_shot_split(s, N=N, H=96, seed=1)
        d = diagnose(sp.train, season_m=meta.season_m)
        p = make_plan(d, N=N, H=96, conf_source="xc")
        print(f"N={N}: conf(xc)=({d.trend_conf_xc},{d.season_conf_xc},{d.stat_conf_xc}) "
              f"→ {p.combine} {p.strategies} | {p.reason}")
