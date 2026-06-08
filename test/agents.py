"""多-Agent 工业少样本多任务自适应规划系统 · 核心 agents。

四个 agent（蒸馏自 research/method1-5）：
  1. CuratorAgent      — 感知：诊断序列结构（趋势/季节/平稳/复杂度/信噪），输出可解释画像
  2. SaturationPlanner — 决策：先判"该不该路由"(method5/F-R10)，再判"信不信 CV"(method3 v10/B7v2)
  3. RouterAgent       — 执行：在候选模型库里按 CV + N-fallback + honest-abstain 选模型
  4. ReporterAgent     — 解释：把每步决策汇成自然语言 + 结构化 trace

核心设计原则（来自 finish1-5 的血泪教训）：
  - Saturation-aware：base model 已近 oracle 时**主动 abstain**，不强行路由（F-R9.7/10.2）
  - N-conditional fallback：极少样本下 CV 噪声大 → 退回 default（method3 v10 / B7v2，+0.87pp）
  - Honest：决策只用部署可得信息（CV / 训练统计），绝不读 test（method3 M9 去泄漏）
  - 永不显著伤害 base：路由只在有把握时偏离（F-R10.2 损害控制）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np


# ════════════════════════════════════════════════════════════════════════════
# 共享数据结构
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Profile:
    """CuratorAgent 输出的序列画像（可解释）。"""
    n_train: int
    n_per_class: Optional[int] = None
    n_classes: Optional[int] = None
    length: int = 0
    n_channels: int = 1
    trend_strength: float = 0.0
    seasonality: float = 0.0
    stationarity: float = 1.0       # 0=非平稳 1=平稳
    noise_level: float = 0.0
    complexity: float = 0.0
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"N={self.n_train}"]
        if self.n_classes:
            parts.append(f"{self.n_classes}类×{self.n_per_class}/类")
        parts.append(f"L={self.length}")
        if self.n_channels > 1:
            parts.append(f"C={self.n_channels}")
        parts.append(f"trend={self.trend_strength:.2f}")
        parts.append(f"season={self.seasonality:.2f}")
        parts.append(f"noise={self.noise_level:.2f}")
        return " ".join(parts)


@dataclass
class Plan:
    """SaturationPlanner 输出的决策计划（可解释）。"""
    mode: str                       # "abstain" | "route" | "fallback"
    default_model: str
    candidates: list[str]
    reason: str
    confidence: float = 0.5
    sat_score: float = 0.0          # 预测饱和度（高=越该 abstain）


@dataclass
class Decision:
    """RouterAgent 输出的最终决策（可解释）。"""
    chosen_model: str
    reason: str
    cv_scores: dict[str, float] = field(default_factory=dict)
    deviated: bool = False
    trace: list[str] = field(default_factory=list)


# ════════════════════════════════════════════════════════════════════════════
# 1. CuratorAgent — 感知
# ════════════════════════════════════════════════════════════════════════════

class CuratorAgent:
    """诊断序列结构。多变量取通道均值。纯训练侧统计，无 test 泄漏。"""

    def profile_series(self, x: np.ndarray, season_m: int = 1) -> Profile:
        """单条 1-D 序列画像（forecasting / detection 用）。"""
        x = np.asarray(x, dtype=np.float64).ravel()
        L = len(x)
        p = Profile(n_train=L, length=L)
        if L < 3:
            p.notes.append("series too short")
            return p
        t = np.arange(L)
        # 趋势：线性拟合斜率的标准化 R²
        slope, intercept = np.polyfit(t, x, 1)
        fit = slope * t + intercept
        ss_tot = np.sum((x - x.mean()) ** 2) + 1e-12
        p.trend_strength = float(max(0.0, 1.0 - np.sum((x - fit) ** 2) / ss_tot))
        # 季节性：ACF at season_m（若给）否则峰值
        p.seasonality = float(_acf_peak(x, season_m))
        # 平稳性：前后半段均值/方差差异（小=平稳）
        h = L // 2
        m1, m2 = x[:h].mean(), x[h:].mean()
        s1, s2 = x[:h].std() + 1e-9, x[h:].std() + 1e-9
        drift = abs(m1 - m2) / (x.std() + 1e-9)
        p.stationarity = float(max(0.0, 1.0 - min(1.0, drift)))
        # 噪声：一阶差分能量占比
        p.noise_level = float(np.std(np.diff(x)) / (np.std(x) + 1e-9))
        # 复杂度：归一化排列熵代理（差分符号变化率）
        p.complexity = float(np.mean(np.abs(np.diff(np.sign(np.diff(x)))) > 0)) if L > 2 else 0.0
        return p

    def profile_cell(self, X: np.ndarray, y: np.ndarray) -> Profile:
        """few-shot 分类 cell 画像。X: [N,L] 或 [N,C,L]。"""
        X = np.asarray(X)
        if X.ndim == 3:           # 多变量 [N,C,L]
            N, C, L = X.shape
            flat = X.reshape(N, C, L)
            rep = flat.mean(axis=1)   # 通道均值作代表
        else:
            N, L = X.shape; C = 1
            rep = X
        classes = np.unique(y)
        nclass = len(classes)
        sub = [self.profile_series(rep[i]) for i in range(min(N, 30))]
        p = Profile(
            n_train=N, n_classes=nclass, n_per_class=int(N / max(nclass, 1)),
            length=L, n_channels=C,
            trend_strength=float(np.mean([s.trend_strength for s in sub])),
            seasonality=float(np.mean([s.seasonality for s in sub])),
            stationarity=float(np.mean([s.stationarity for s in sub])),
            noise_level=float(np.mean([s.noise_level for s in sub])),
            complexity=float(np.mean([s.complexity for s in sub])),
        )
        if p.n_per_class is not None and p.n_per_class < 5:
            p.notes.append(f"extreme few-shot (N/class={p.n_per_class})")
        if C > 1:
            p.notes.append(f"multivariate ({C} channels)")
        return p


def _acf_peak(x: np.ndarray, season_m: int) -> float:
    x = x - x.mean()
    denom = np.sum(x * x) + 1e-12
    if season_m and 1 < season_m < len(x):
        return float(abs(np.sum(x[season_m:] * x[:-season_m]) / denom))
    best = 0.0
    for lag in range(2, min(len(x) // 2, 50)):
        v = abs(np.sum(x[lag:] * x[:-lag]) / denom)
        best = max(best, v)
    return best


# ════════════════════════════════════════════════════════════════════════════
# 2. SaturationPlanner — 决策（method5 核心）
# ════════════════════════════════════════════════════════════════════════════

class SaturationPlanner:
    """决定 abstain / route / fallback。

    经验先验（来自 finish1-5 的跨任务饱和度实测）：
      - classification/detection：base(Rocket) 在 71-75% cell 是 oracle → 默认偏 abstain
      - forecasting：base(Chronos-2) 仅 25% cell 是 oracle → 默认偏 route
    再叠 N-fallback：N/class < n_min 时 CV 不可信 → 强制 default（method3 B7v2）。
    """

    # 跨任务饱和先验（finish4 §8 / finish5 §1 实测）
    SAT_PRIOR = {"forecasting": 0.25, "classification": 0.71, "detection": 0.75}

    def __init__(self, task: str, default_model: str, candidates: list[str],
                 n_min_for_routing: int = 7, sat_predictor: Optional[Callable] = None):
        self.task = task
        self.default_model = default_model
        self.candidates = candidates
        self.n_min = n_min_for_routing
        self.sat_predictor = sat_predictor   # 可选 ĝ(profile)->predicted gap

    def plan(self, prof: Profile) -> Plan:
        sat_prior = self.SAT_PRIOR.get(self.task, 0.5)
        # N-conditional fallback（极少样本 CV 噪声大，method3 v10/B7v2 实测 +0.87pp）
        npc = prof.n_per_class if prof.n_per_class is not None else prof.n_train
        if npc < self.n_min and self.task != "forecasting":
            return Plan(mode="fallback", default_model=self.default_model,
                        candidates=[self.default_model],
                        reason=f"N/class={npc}<{self.n_min}: CV 不可信，强制 default '{self.default_model}'"
                               f"（method3 B7v2 N-fallback）",
                        confidence=0.8, sat_score=sat_prior)
        # 饱和决策：sat_predictor 优先，否则用任务先验
        sat = sat_prior
        if self.sat_predictor is not None:
            try:
                sat = float(self.sat_predictor(prof))
            except Exception:
                pass
        if sat >= 0.65:
            return Plan(mode="abstain", default_model=self.default_model,
                        candidates=[self.default_model] + self.candidates,
                        reason=f"saturation={sat:.2f}≥0.65: base 大概率已近 oracle，"
                               f"偏向 abstain，仅在 CV 强烈支持时才偏离（F-R9.7/10.2 损害控制）",
                        confidence=0.6, sat_score=sat)
        return Plan(mode="route", default_model=self.default_model,
                    candidates=[self.default_model] + self.candidates,
                    reason=f"saturation={sat:.2f}<0.65: 有 routing 头寸，启用 CV 路由"
                           f"（forecasting 未饱和，F-R10.1）",
                    confidence=0.6, sat_score=sat)


# ════════════════════════════════════════════════════════════════════════════
# 3. RouterAgent — 执行（CV + margin gate + honest abstain）
# ════════════════════════════════════════════════════════════════════════════

class RouterAgent:
    """按 Plan 在候选库里选模型。margin 随 saturation 自适应：越饱和，偏离门槛越高。"""

    def __init__(self, cv_fn: Callable, base_margin: float = 0.05,
                 trust_fn: Optional[Callable] = None, trust_min: float = 0.5):
        # cv_fn(model_name) -> float（越大越好，部署可得，无 test）
        self.cv_fn = cv_fn
        self.base_margin = base_margin
        # trust_fn(best_model, cv_dict) -> float ∈ [0,1]（这次偏离可不可信，部署可得，无 test）
        # method6 F-R11.5：偏离需同时满足 CV margin AND trust≥trust_min（避险门控）。None=关闭。
        self.trust_fn = trust_fn
        self.trust_min = trust_min

    def route(self, plan: Plan) -> Decision:
        trace = [f"planner.mode={plan.mode}", plan.reason]
        if plan.mode == "fallback":
            return Decision(chosen_model=plan.default_model,
                            reason=plan.reason, deviated=False, trace=trace)

        # 计算候选 CV
        cv: dict[str, float] = {}
        for m in plan.candidates:
            try:
                v = self.cv_fn(m)
                if v == v:   # not nan
                    cv[m] = float(v)
            except Exception:
                pass
        trace.append("cv=" + ", ".join(f"{k}:{v:.3f}" for k, v in sorted(cv.items(), key=lambda kv: -kv[1])))

        default = plan.default_model
        if default not in cv:
            return Decision(chosen_model=default,
                            reason=f"default '{default}' 无 CV，直接用 default",
                            cv_scores=cv, deviated=False, trace=trace)

        # saturation-自适应 margin：abstain 模式门槛更高（F-R10.2 损害控制）
        margin = self.base_margin + (0.10 if plan.mode == "abstain" else 0.0)
        d_cv = cv[default]
        others = [(m, v) for m, v in cv.items() if m != default]
        if not others:
            return Decision(chosen_model=default, reason="无其它候选",
                            cv_scores=cv, deviated=False, trace=trace)
        best_m, best_v = max(others, key=lambda kv: kv[1])
        if best_v >= d_cv + margin:
            # method6 F-R11.5：CV margin 通过后，再过 trust 门（避险，挡掉"会变差"的偏离）
            if self.trust_fn is not None:
                try:
                    tr = float(self.trust_fn(best_m, cv))
                except Exception:
                    tr = 1.0
                if tr < self.trust_min:
                    trace.append(f"CV winner '{best_m}' 过 margin 但 trust={tr:.2f}<{self.trust_min} → trust 门拦截，守 default")
                    return Decision(chosen_model=default,
                                    reason=f"'{best_m}' CV 超默认 {best_v-d_cv:+.3f}，但可信度 trust={tr:.2f}"
                                           f"低于 {self.trust_min}（F-R11.5 避险门），守 default '{default}'",
                                    cv_scores=cv, deviated=False, trace=trace)
                trace.append(f"trust={tr:.2f}≥{self.trust_min} 通过")
            trace.append(f"CV winner '{best_m}'({best_v:.3f}) ≥ default({d_cv:.3f})+margin({margin:.2f}) → 偏离")
            return Decision(chosen_model=best_m,
                            reason=f"CV-winner '{best_m}' 超 default {best_v-d_cv:+.3f}≥margin{margin:.2f}"
                                   + (f" 且 trust 通过" if self.trust_fn is not None else "")
                                   + f"，在 {plan.mode} 模式下仍值得偏离",
                            cv_scores=cv, deviated=True, trace=trace)
        trace.append(f"best other '{best_m}'({best_v:.3f}) 未超 margin → 守 default")
        return Decision(chosen_model=default,
                        reason=f"信任 default '{default}'({d_cv:.3f})；最强候选 '{best_m}' "
                               f"仅 {best_v-d_cv:+.3f}<margin{margin:.2f}（honest abstain）",
                        cv_scores=cv, deviated=False, trace=trace)


# ════════════════════════════════════════════════════════════════════════════
# 4. ReporterAgent — 解释
# ════════════════════════════════════════════════════════════════════════════

class ReporterAgent:
    def report(self, task: str, prof: Profile, plan: Plan, dec: Decision) -> dict:
        nl = (f"[{task}] 画像: {prof.summary()}"
              + (f" | {';'.join(prof.notes)}" if prof.notes else "")
              + f"\n  决策: {plan.mode} (sat={plan.sat_score:.2f}) — {plan.reason}"
              + f"\n  选模型: {dec.chosen_model} ({'偏离' if dec.deviated else '守 default'}) — {dec.reason}")
        return {
            "task": task,
            "profile": prof.summary(),
            "notes": prof.notes,
            "plan_mode": plan.mode,
            "sat_score": round(plan.sat_score, 3),
            "chosen_model": dec.chosen_model,
            "deviated": dec.deviated,
            "cv_scores": {k: round(v, 4) for k, v in dec.cv_scores.items()},
            "trace": dec.trace,
            "nl_explanation": nl,
        }
