"""Phase 3.3 · Forecaster + 反思循环（plan §四层三）。

流程：
  1) 按 Plan 跑所有候选策略，在 val 上算 MAE
  2) 按权重 ensemble（或 single 取首个）→ 给出 val MAE
  3) 若 val MAE > 阈值 → 反思：让 LLM 看诊断 + 各策略 val MAE → 重新选策略
  4) 最多 3 次反思（plan §12 R2 硬上限）
  5) 用最终策略组合预测 test (H 步)

策略名 → baseline 模块：
  naive_seasonal / naive_drift → research.baseline.naive
  arima_ets                    → research.baseline.arima_ets
  chronos                      → research.baseline.chronos
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from research.agent.curator_uq import Diagnosis
from research.agent.model_cards import render_cards_block
from research.agent.planner_adaptive import Plan, make_plan
from research.utils.llm import chat_cached
from research.utils.metrics import mae


# ---------- 策略实现（轻量包装，复用 baseline 模块） ---------- #

def _naive_drift(train, val, H, season_m):
    if len(train) < 2:
        return np.full(H, float(train.mean()))
    slope = (train[-1] - train[0]) / (len(train) - 1)
    return train[-1] + slope * np.arange(1, H + 1)


def _naive_seasonal(train, val, H, season_m):
    m = season_m if season_m > 1 and len(train) >= season_m else 1
    if m == 1:
        return np.full(H, float(train.mean()))
    last = train[-m:]
    reps = int(np.ceil(H / m))
    return np.tile(last, reps)[:H]


def _arima_ets(train, val, H, season_m):
    from research.baseline.arima_ets import predict
    return predict(train=train, val=val, H=H, season_m=season_m)


def _chronos(train, val, H, season_m):
    from research.baseline.chronos import predict
    return predict(train=train, val=val, H=H, season_m=season_m)


def _llmtime(train, val, H, season_m):
    from research.baseline.llmtime import predict
    return predict(train=train, val=val, H=H, season_m=season_m)


def _chronos2(train, val, H, season_m):
    from research.baseline.chronos2 import predict
    return predict(train=train, val=val, H=H, season_m=season_m)


def _chronos_bolt(train, val, H, season_m):
    from research.baseline.chronos_bolt import predict
    return predict(train=train, val=val, H=H, season_m=season_m)


STRATEGY_FN: dict[str, Callable] = {
    "naive_drift":    _naive_drift,
    "naive_seasonal": _naive_seasonal,
    "arima_ets":      _arima_ets,
    "chronos":        _chronos,
    "llmtime":        _llmtime,
    # feedback 2026-05 增补
    "chronos2":       _chronos2,
    "chronos_bolt":   _chronos_bolt,
}

# Round 3+ extended baselines wired lazily — see baseline/{tirex,toto,toto2,
# timesfm2,moirai,moirai2,time_moe,sundial,timer}.py
def _make_lazy(modname: str):
    def _fn(train, val, H, season_m):
        from importlib import import_module
        return import_module(f"research.baseline.{modname}").predict(
            train=train, val=val, H=H, season_m=season_m)
    return _fn

for _m in ["tirex", "toto", "toto2", "timesfm2", "moirai", "moirai2",
           "time_moe", "sundial", "timer"]:
    STRATEGY_FN.setdefault(_m, _make_lazy(_m))

# v7: 当 ADAPTTS_CHRONOS=bolt 时，"chronos" 名义保留（trace 可读性），但实际派发到
# Chronos-Bolt 实现（同质量 + 50× 速度，feedback F4 sanity check 实证）。
import os as _os
if _os.environ.get("ADAPTTS_CHRONOS") == "bolt":
    STRATEGY_FN["chronos"] = _chronos_bolt
elif _os.environ.get("ADAPTTS_CHRONOS") == "2":
    STRATEGY_FN["chronos"] = _chronos2


# ---------- 反思：结构化根因推理（plan §十五） ---------- #

@dataclass
class ReflectStep:
    """单次反思的完整 trace（plan §15.3）。"""
    plan_before: Plan
    per_strat_mae: dict[str, float]
    root_cause: str                       # LLM 根因分析文本
    diagnosis_revision: dict | None       # 可选：诊断纠偏 {"trend":"high→mid","reason":"..."}
    plan_after: Plan | None               # None 表示反思失败回退


def _validate_root_cause(rc: str) -> bool:
    """合理性校验（plan §15.2.③）：root_cause 必须含数字 + 诊断/策略词，
    否则视为 LLM 幻觉，触发 safe fallback。"""
    if not rc or len(rc.strip()) < 10:
        return False
    has_number = bool(re.search(r"\d+(\.\d+)?", rc))
    diag_keywords = ["trend", "season", "stat", "stationar",
                     "趋势", "季节", "周期", "平稳",
                     "drift", "arima", "chronos", "naive", "llm"]
    has_keyword = any(w in rc.lower() for w in diag_keywords)
    return has_number and has_keyword


def _reflect(
    diag: Diagnosis,
    plan: Plan,
    val_maes: dict[str, float],
    previous_plans: list[Plan],
    H: int,
    use_model_cards: bool = True,
    allow_diagnosis_revision: bool = True,
) -> tuple[Plan | None, ReflectStep]:
    """让 LLM 看本轮 val 结果 → 结构化输出 (root_cause / diagnosis_revision / new_plan)。

    返回 (new_plan_or_None, ReflectStep) — None 表示反思失败/无效，调用方应触发回退。
    use_model_cards / allow_diagnosis_revision 是消融开关（对应 A8 / A9）。
    """
    tried = [p.strategies for p in previous_plans]
    pool = list(STRATEGY_FN.keys())

    # Model Cards 注入（A8 关闭则跳过）
    cards_block = render_cards_block(pool) if use_model_cards else \
        f"(no model cards; pool = {pool})"

    # diagnosis_revision 字段说明（A9 关闭则去掉）
    diag_rev_block = (
        '  "diagnosis_revision": null OR {"trend":"high→mid","season":"...","stat":"...","reason":"..."},'
        if allow_diagnosis_revision else
        '  // (diagnosis_revision disabled in this ablation),'
    )

    system_msg = (
        "You are a time-series forecasting strategy expert. "
        "You will receive (1) diagnosis with three-way confidence, (2) the previous plan "
        "and per-strategy val MAEs, (3) structured model cards describing each strategy's "
        "assumptions, strengths, and typical failures. "
        "Your job: do root-cause analysis (cite specific MAEs AND diagnosis terms), "
        "optionally revise the diagnosis if evidence contradicts it, "
        "and propose a new plan. Output ONLY a valid JSON object."
    )

    user_msg = (
        f"## Diagnosis (三路置信度, xc=cross-validated lower-bound)\n"
        f"  trend: {diag.trend_conf_xc} (stat={diag.trend_conf_stat}, llm={diag.trend_conf_llm}, "
        f"t-stat={diag.trend_tstat:.2f})\n"
        f"  season: {diag.season_conf_xc} (ACF peak={diag.acf_peak_value:.3f} at lag {diag.acf_peak_lag})\n"
        f"  stat: {diag.stat_conf_xc} (ADF p={diag.adf_pvalue:.3f})\n"
        f"  llm_reason: {diag.llm_reason}\n\n"
        f"## Strategy Library (Model Cards)\n{cards_block}\n\n"
        f"## Previous Plan\n"
        f"  strategies: {plan.strategies} ({plan.combine}); weights={plan.weights}\n"
        f"  reason: {plan.reason}\n\n"
        f"## Per-Strategy Val MAE (this round)\n{json.dumps(val_maes, indent=2)}\n\n"
        f"## Already-Tried Plans (do not repeat)\n{tried}\n\n"
        f"## Output schema (strict JSON, no markdown)\n"
        f"{{\n"
        f'  "root_cause": "MUST cite specific MAE numbers AND diagnosis terms; '
        f'explain WHY the previous plan failed by referring to model assumptions.",\n'
        f'{diag_rev_block}\n'
        f'  "new_plan": {{\n'
        f'    "strategies": ["..."],   // pick from {pool}\n'
        f'    "combine": "single | ensemble | safe",\n'
        f'    "weights": [0.5, 0.3, 0.2]\n'
        f'  }}\n'
        f"}}"
    )

    raw = chat_cached(
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2, max_tokens=2048,
    )
    m = re.search(r"\{.*\}", raw, flags=re.S)
    parsed: dict = {}
    if m:
        try:
            parsed = json.loads(m.group(0))
        except Exception:
            parsed = {}

    root_cause = str(parsed.get("root_cause", "")).strip()
    diag_revision = parsed.get("diagnosis_revision") if allow_diagnosis_revision else None
    if isinstance(diag_revision, dict) and not diag_revision:
        diag_revision = None
    new_plan_d = parsed.get("new_plan") or {}

    step = ReflectStep(
        plan_before=plan, per_strat_mae=val_maes,
        root_cause=root_cause, diagnosis_revision=diag_revision,
        plan_after=None,
    )

    # 合理性校验：root_cause 必须有数字 + 诊断/策略词
    if not _validate_root_cause(root_cause):
        return None, step

    strats = [s for s in new_plan_d.get("strategies", []) if s in STRATEGY_FN]
    if not strats:
        return None, step
    combine = new_plan_d.get("combine", "ensemble")
    if combine not in ("single", "ensemble", "safe"):
        combine = "ensemble"
    weights = new_plan_d.get("weights") or [1.0 / len(strats)] * len(strats)
    if len(weights) != len(strats):
        weights = [1.0 / len(strats)] * len(strats)
    new_plan = Plan(
        strategies=strats, combine=combine,
        reason=root_cause[:300],   # 直接把根因写到 plan.reason
        weights=list(weights),
    )
    step.plan_after = new_plan
    return new_plan, step


# ---------- 主入口 ---------- #

@dataclass
class ForecastTrace:
    plan_history: list[Plan]
    val_mae_history: list[float]
    final_plan: Plan
    final_val_mae: float
    reflect_steps: list[ReflectStep] = field(default_factory=list)   # 新增
    diagnosis_revised: dict | None = None                            # 新增：最终累计的诊断修正
    promotion: dict | None = None                                    # v6 新增：策略 promotion 信息
    bandit_handle: dict | None = None                                # Round 5 Phase 2: closure handle for observe_outcome


def observe_outcome(trace: "ForecastTrace", y_true: np.ndarray,
                    y_pred: np.ndarray | None = None,
                    persist: bool = True) -> dict | None:
    """Close the contextual bandit loop after observing actual test outcome.

    Call this after `forecast_with_reflection(...)` once y_true is available:

        pred, trace = forecast_with_reflection(...)
        # ... actual y_true revealed ...
        observe_outcome(trace, y_true=actual, y_pred=pred)

    Updates BanditState's per-(regime, chosen) belief with the observed MAE.
    No-op if trace was not produced by ADAPTTS_PLANNER=bandit.

    Args:
        persist: if True, save bandit state to disk after observe (recommended).
    Returns:
        {"regime", "chosen", "observed_mae", "updated_belief": (μ, σ)} or None.
    """
    if trace.bandit_handle is None: return None
    h = trace.bandit_handle
    if y_pred is None: return None
    mae = float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))
    # Round 6 · adaptive handle (preferred path)
    if "plan" in h and "state" in h:
        from research.agent.adaptive_planner import adaptive_observe
        from research.agent.router_state import persist_state
        res = adaptive_observe(h["state"], h["plan"], outcome=mae)
        if persist: persist_state()
        return {"adaptive": True, **res, "observed_mae": mae}
    # legacy bandit handle
    h["router"].observe(h["z"], h["chosen"], mae)
    if persist:
        h["router"].bandit.save(h["state_path"])
    return {"regime": h["regime"], "chosen": h["chosen"],
            "observed_mae": mae,
            "updated_belief": h["router"].bandit.belief(h["regime"], h["chosen"])}


def _run_plan(plan: Plan, train, val, H, season_m) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """跑出 plan 对应的组合预测 + 各策略原始预测。"""
    preds = {s: STRATEGY_FN[s](train, val, H, season_m) for s in plan.strategies}
    if plan.combine == "single":
        y = preds[plan.strategies[0]]
    else:
        w = np.array(plan.weights, dtype=np.float64)
        w = w / (w.sum() + 1e-12)
        y = sum(w[i] * preds[s] for i, s in enumerate(plan.strategies))
    return y, preds


def walk_forward_eval(
    strategy_names: list[str],
    series: np.ndarray,
    H_v: int,
    season_m: int,
    n_folds: int | None = None,
) -> dict[str, float]:
    """train 末尾多窗口滚动评估各策略平均 MAE。

    fold k: 用 series[: -k*H_v] 拟合，预测 series[-k*H_v : -(k-1)*H_v]，k=1..n_folds
    返回 {strategy_name: mean_MAE}；某策略某 fold 失败则该 fold 跳过。
    """
    n = len(series)
    if H_v < 1:
        H_v = 1
    if n_folds is None:
        # 默认 fold 数：序列越长越多 fold，至少 1，最多 5
        n_folds = max(1, min(5, (n - H_v) // H_v))
    maes: dict[str, list[float]] = {s: [] for s in strategy_names}
    for k in range(1, n_folds + 1):
        cut = n - k * H_v
        if cut < max(4, H_v):  # 训练段太短直接跳过
            break
        train_k = series[:cut]
        val_k = series[cut: cut + H_v]
        for s in strategy_names:
            try:
                pred = STRATEGY_FN[s](train_k, np.array([]), H_v, season_m)
                maes[s].append(float(np.mean(np.abs(val_k - pred))))
            except Exception:
                pass
    return {s: (float(np.mean(v)) if v else float("inf")) for s, v in maes.items()}


def softmax_weights(mae_dict: dict[str, float], tau: float = 0.5) -> list[tuple[str, float]]:
    """按 -MAE 做 softmax 得权重；MAE=inf 自动归零。"""
    items = [(s, m) for s, m in mae_dict.items() if np.isfinite(m)]
    if not items:
        return []
    mins = min(m for _, m in items)
    weights = np.exp(-(np.array([m for _, m in items]) - mins) / max(tau, 1e-6))
    weights = weights / weights.sum()
    return list(zip([s for s, _ in items], weights.tolist()))


def forecast_with_reflection(
    train: np.ndarray, val: np.ndarray, H: int,
    diag: Diagnosis, season_m: int = 1,
    max_reflect: int = 3, val_mae_threshold: float | None = None,
    conf_source: str = "xc",
    use_walk_forward: bool = True,
    use_model_cards: bool = True,
    allow_diagnosis_revision: bool = True,
    reflect_can_replace_best: bool = False,
    enable_promotion: bool = False,
    promotion_improve_frac: float = 0.30,
    dataset: str | None = None,        # feedback Item 6: forward to prior-aware planner
) -> tuple[np.ndarray, ForecastTrace]:
    """完整流程：初始 plan → walk-forward 重加权 → 反思 → 用最优 plan 预测 test。

    use_walk_forward=True 时，会用 train 末尾的多窗口 CV 对 plan.strategies 重新加权：
      - 候选库取 planner 给的 strategies + 若空则全库
      - 按 softmax(-MAE/τ) 权重，τ 随 N 调整（N 越大越倾向 winner-take-all）
    """
    N = len(train)
    # 初始 plan
    _planner = os.environ.get("ADAPTTS_PLANNER", "").lower()
    if _planner == "adaptive":
        # Round 6 · Full reflective adaptive runtime
        # Round 7 · + CircuitBreakerPrior + OperationalReliabilityPrior + safe_predict
        from research.agent.adaptive_planner import (
            adaptive_decide, adaptive_observe)
        from research.agent.reflective_loop import reflective_predict
        from research.agent.router_state import get_state
        from research.agent.bayesian_router import (
            RouterConfig, AvailabilityPrior, NPrior, TypePrior, CRPSPrior)
        from research.agent.reliability_priors import (
            CircuitBreakerPrior, OperationalReliabilityPrior)
        from research.agent.safe_predict import safe_predict
        candidates_a = [s for s in [
            "chronos2", "chronos", "tirex", "toto", "timesfm2",
            "moirai", "moirai2", "naive_drift", "arima_ets"]
            if s in STRATEGY_FN]
        cfg_a = RouterConfig(
            priors=[
                AvailabilityPrior(local_models=tuple(candidates_a), remote_models=()),
                CircuitBreakerPrior(),                       # Round 7 P0-1
                OperationalReliabilityPrior(strength=1.5),   # Round 7 P0-3
                CRPSPrior(),
                NPrior(default_model="chronos2", N_threshold=15, strength=2.0),
                TypePrior(),
            ],
            decide_mode=os.environ.get("ADAPTTS_DECIDE", "argmax").lower(),
            risk_lambda=float(os.environ.get("ADAPTTS_RISK_LAM", "1.0")),
            embedding_name=os.environ.get("ADAPTTS_EMBED", "hand25"),
            enable_remote=os.environ.get("ADAPTTS_ALLOW_REMOTE", "0") == "1",
            enable_bandit=os.environ.get("ADAPTTS_BANDIT", "1") == "1",
        )
        state_a = get_state(os.environ.get("ADAPTTS_STATE_PATH",
                            "research/results/router_state.jsonl"))
        plan_a = adaptive_decide(
            "forecast", train.astype(np.float32),
            candidates=candidates_a, config=cfg_a, state=state_a,
            dataset=dataset, N=N, H=H,
        )
        # Round 7 · safe predict wrapper around STRATEGY_FN
        def _raw_predict(m):
            return STRATEGY_FN[m](train, val, H, season_m)
        _safe_meta = {}   # records actual_model used per request (for F16 routing fix)
        def _safe_predict_fn(m):
            res = safe_predict(
                model_name=m, predict_fn=_raw_predict,
                H=H, history=train,
                fallback_model="naive_drift",
                fallback_predict_fn=_raw_predict,
                register_outcome=True,
            )
            _safe_meta[m] = {
                "fallback_used": res.fallback_used,
                "actual_model": res.chosen_model,
                "failure_type": res.failure_type,
            }
            return res.pred
        try:
            ref_res = reflective_predict(
                plan_a, _safe_predict_fn, history=train,
                tau_gap=float(os.environ.get("ADAPTTS_GAP_TAU", "0.10")),
                enable_l2=False, enable_l3=True,
            )
        except Exception as _re:
            ref_res = None

        # ─── Round 7 fix · pick a guaranteed-safe model for downstream _run_plan ─
        # The adaptive top-1 might be a model that failed (moirai/etc). safe_predict
        # already fell back; we must reflect that in plan.strategies so the
        # subsequent _run_plan call uses the same safe model, not the failed one.
        safe_chosen = plan_a.chosen
        if plan_a.chosen in _safe_meta:
            meta = _safe_meta[plan_a.chosen]
            if meta["fallback_used"]:
                safe_chosen = meta["actual_model"]
        if safe_chosen not in STRATEGY_FN or safe_chosen == "<zero>":
            safe_chosen = "chronos2"
        # Save bandit + telemetry state right away (persist for next call)
        try:
            from research.agent.router_state import persist_state
            persist_state()
        except Exception: pass

        from research.agent.planner_prior_aware import PriorPlan
        # Round 7 fix: always use safe_chosen (guaranteed available) for downstream
        plan = PriorPlan(
            level="L1",
            strategies=[safe_chosen],
            weights=[1.0],
            combine="single",
            reason=f"Adaptive(decide={cfg_a.decide_mode}) regime={plan_a.regime}; "
                   f"orig={plan_a.chosen} safe={safe_chosen}; "
                   f"layers={ref_res.layers_used if ref_res else ['L0']}; "
                   f"conf={ref_res.confidence:.3f}" if ref_res else "L0",
            posterior=dict(plan_a.top_k),
        )
        # stash adaptive handle for observe_outcome
        _ADAPTIVE_HANDLE = {
            "plan": plan_a, "state": state_a, "ref_res": ref_res,
        }
    elif _planner == "bandit":
        # Round 5 Phase 2 · Contextual bandit / Thompson Routing
        from research.agent.bandit import get_router, persist_router
        from research.agent.planner_prior_aware import PriorPlan
        router, emb = get_router(
            state_path=os.environ.get("ADAPTTS_BANDIT_PATH",
                                       "research/results/bandit_state.jsonl"),
            decay=float(os.environ.get("ADAPTTS_BANDIT_DECAY", "1.0")),
        )
        decide_mode = os.environ.get("ADAPTTS_DECIDE", "thompson").lower()
        z = emb.embed(train.astype(np.float32))
        chosen, scores = router.decide(z, mode=decide_mode)
        regime = router.regime_fn(z)
        # Filter chosen to STRATEGY_FN-known models
        if chosen not in STRATEGY_FN:
            chosen = "chronos2"  # safe default
        plan = PriorPlan(
            level="L1",
            strategies=[chosen],
            weights=[1.0],
            combine="single",
            reason=f"Bandit decide({decide_mode}) on regime={regime}; chose {chosen} "
                   f"(score={scores.get(chosen, 0):.3f})",
            posterior=scores,
        )
        # stash for post-prediction observe loop (set by external caller via observe_outcome)
        _BANDIT_HANDLE = {"z": z, "chosen": chosen, "router": router,
                          "regime": regime, "state_path":
                          os.environ.get("ADAPTTS_BANDIT_PATH",
                                          "research/results/bandit_state.jsonl")}
    elif _planner == "bayesian":
        # Round 5 · Bayesian unification — all heuristics → prior/likelihood factors
        from research.agent.bayesian_router import (
            default_forecasting_router, Context, Evidence, PriorPlan_from_posterior)
        allow_remote = os.environ.get("ADAPTTS_ALLOW_REMOTE", "0") == "1"
        decide_mode = os.environ.get("ADAPTTS_DECIDE", "argmax").lower()
        lam = float(os.environ.get("ADAPTTS_RISK_LAM", "1.0"))
        router = default_forecasting_router(allow_remote=allow_remote)
        ctx = Context(dataset=dataset, N=N, H=H, allow_remote=allow_remote)
        chosen, post = router.decide(ctx, mode=decide_mode, lam=lam)
        plan = PriorPlan_from_posterior(chosen, post, decide_mode=decide_mode)
    elif _planner == "prior_aware":
        # Round 4 · prior_aware (heuristic stack baseline for ablation)
        from research.agent.planner_prior_aware import make_prior_plan
        allow_remote = os.environ.get("ADAPTTS_ALLOW_REMOTE", "0") == "1"
        plan = make_prior_plan(dataset=dataset, N=N, H=H,
                               allow_remote=allow_remote)  # type: ignore
    else:
        plan = make_plan(diag, N=N, H=H, conf_source=conf_source)  # type: ignore

    # v10: N<15 跳过 walk-forward 时，若 ADAPTTS_DEFAULT 设置，仍无条件 fallback 到 default
    # 解决 N=10 cells 走 prefix-rule 导致的 catastrophic +85%~+162% 退步 (§3.1.21)
    _default_v10 = os.environ.get("ADAPTTS_DEFAULT")
    if N < 15 and _default_v10 and _default_v10 in STRATEGY_FN:
        plan = Plan(
            strategies=[_default_v10], combine="single",
            reason=f"v10 N<15 fallback to default={_default_v10} (skip walk-forward)",
            weights=[1.0],
        )

    # A1 / v12: ADAPTTS_GATE=entropy → Chronos-2 自身 quantile spread 作门控信号
    # 实测 entropy 分布与 v10 win/loss 不简单线性相关（finish.md §3.1.24）：
    #   - Low entropy（C2 confident）→ 信任 C2，跳过 CV
    #   - High entropy（C2 uncertain）→ 提高 deviation margin（CV 单一信号本身可疑，需要更大证据）
    #   - Mid entropy → v10 默认 margin
    # 这是 v12 的核心 insight：不是简单二元 gate，而是用 entropy 调制 v10 margin
    _ent_v12 = None
    if (os.environ.get("ADAPTTS_GATE") == "entropy"
        and N >= 15
        and _default_v10 == "chronos2"):
        try:
            from research.baseline.chronos2 import predict_with_uncertainty
            _, _ent_v12, _ = predict_with_uncertainty(train, H, seed=42, season_m=season_m)
            _ent_low = float(os.environ.get("ADAPTTS_ENT_LOW", "1.7"))
            if _ent_v12 < _ent_low:
                plan = Plan(
                    strategies=["chronos2"], combine="single",
                    reason=f"v12 entropy gate: C2 entropy={_ent_v12:.3f}<{_ent_low:.2f} → trust C2 (skip CV)",
                    weights=[1.0],
                )
                use_walk_forward = False  # 跳过下面的 walk-forward CV
        except Exception:
            pass

    # walk-forward 重加权（替代 prefix 规则）
    if use_walk_forward and N >= 15:
        # H_v：留 5-10 步做 holdout；fold 数让 strategy 至少能见 1 次
        H_v = max(3, min(10, N // 5))
        # 候选池：planner 的 strategies；N≤30 加 llmtime 兜底
        pool = list(dict.fromkeys(plan.strategies))
        if N <= 30 and "llmtime" not in pool:
            pool.append("llmtime")
        # v8: ADAPTTS_TSFM_POOL=expand → 若 chronos 在池中，同时加入 chronos_bolt/chronos2
        # 让 walk-forward CV 在多个 SOTA TSFM 间择优（feedback F4 + §3.1.19b finding #4 落地）
        if os.environ.get("ADAPTTS_TSFM_POOL") == "expand" and "chronos" in pool:
            for extra in ("chronos_bolt", "chronos2"):
                if extra not in pool:
                    pool.append(extra)
        cv_maes = walk_forward_eval(pool, train, H_v=H_v, season_m=season_m,
                                    n_folds=None)
        # v9: ADAPTTS_DEFAULT=chronos2 → "trust Chronos-2 unless proven otherwise" gating
        # 解决 v8 短 val CV 选错导致 +45% MAE 退步无救援的失败模式 (§3.1.20)
        # 规则：默认 chronos2；只有 best_other 以 ≥MARGIN 击败 chronos2 才偏离
        default_strat = os.environ.get("ADAPTTS_DEFAULT")
        if default_strat and default_strat in STRATEGY_FN:
            if default_strat not in pool:
                pool.append(default_strat)
                cv_maes = walk_forward_eval(pool, train, H_v=H_v, season_m=season_m, n_folds=None)
            margin = float(os.environ.get("ADAPTTS_DEFAULT_MARGIN", "0.20"))
            # v12: 用 Chronos-2 entropy 调制 margin
            # 高 entropy（C2 自身不确定）→ 提高 margin（CV 信号也可疑，需更强证据偏离）
            if _ent_v12 is not None:
                _ent_high = float(os.environ.get("ADAPTTS_ENT_HIGH", "3.0"))
                if _ent_v12 > _ent_high:
                    margin = max(margin, float(os.environ.get("ADAPTTS_MARGIN_HIGH_ENT", "0.45")))
            default_mae = cv_maes.get(default_strat, float("inf"))
            others = [(s, m) for s, m in cv_maes.items()
                      if s != default_strat and np.isfinite(m)]
            if others:
                others.sort(key=lambda kv: kv[1])
                best_s, best_m = others[0]
                if np.isfinite(default_mae) and best_m < default_mae * (1.0 - margin):
                    chosen = best_s
                    reason_extra = f"best_other={best_s}({best_m:.3f}) beats default={default_strat}({default_mae:.3f}) by ≥{margin*100:.0f}%"
                else:
                    chosen = default_strat
                    reason_extra = f"trust default={default_strat}({default_mae:.3f}); best_other={best_s}({best_m:.3f}) margin not met"
            else:
                chosen = default_strat
                reason_extra = f"only default={default_strat} valid"
            # v13 memory safety-net (替代 v11 双向 override)：
            # 只允许"revert deviation → default"方向，防止 self-reinforcing trap 消灭 v12 win
            # ADAPTTS_MEMORY_SKIP_QUERY=1 → Phase A 写入模式，不查询不 override
            mem_path = os.environ.get("ADAPTTS_MEMORY_PATH")
            mem_skip_query = os.environ.get("ADAPTTS_MEMORY_SKIP_QUERY") == "1"
            mem_override_applied = False
            if mem_path and not mem_skip_query and chosen != default_strat:
                try:
                    from research.agent.memory import Memory, case_features as _cf
                    _mem = Memory(mem_path)
                    K_MIN = int(os.environ.get("ADAPTTS_MEMORY_K_MIN", "5"))
                    if len(_mem) >= K_MIN:
                        feat = _cf(diag)
                        K = int(os.environ.get("ADAPTTS_MEMORY_K", "5"))
                        neighbors = _mem.query(feat, k=K)
                        # 统计邻居中 default 真实胜出的数量（test_mae 排第一）
                        # 这里 "default 胜出" 定义为该 case 的 final winner = default_strat
                        # 且实测 test_mae 不是太离谱
                        # 相似度加权投票：sim 高的邻居权重大
                        score_default = sum(sim for sim, c in neighbors
                                            if c.test_mae is not None
                                            and c.final_plan.get("strategies", [None])[0] == default_strat)
                        score_dev = sum(sim for sim, c in neighbors
                                        if c.test_mae is not None
                                        and c.final_plan.get("strategies", [None])[0] == chosen)
                        # safety-net 触发：default 加权得票 > chosen 加权得票 × THRESH
                        # THRESH > 1 让"近邻同选 chosen"能保住 chosen (保 win)
                        # default 跨多 cell 累计加权多时才 revert（保 loss-修复）
                        revert_thresh = float(os.environ.get("ADAPTTS_MEMORY_REVERT_THRESH", "1.0"))
                        if score_default > 1e-9 and score_default > score_dev * revert_thresh:
                            reason_extra = (f"v13 memory safety-net: CV chose {chosen}, "
                                            f"weighted votes default={score_default:.2f} > chosen={score_dev:.2f}*{revert_thresh:.1f} → revert.")
                            chosen = default_strat
                            mem_override_applied = True
                except Exception:
                    pass

            plan = Plan(
                strategies=[chosen], combine="single",
                reason=f"v9 gating: {reason_extra} | MAEs={cv_maes}",
                weights=[1.0],
            )
        # v8: ADAPTTS_TOP1=1 → 直接选 CV MAE 最低的 single 策略（不 ensemble 平均）
        # 解决 v7 ensemble 拉低 finding (§3.1.19b finding #1)
        elif os.environ.get("ADAPTTS_TOP1") == "1":
            valid = [(s, m) for s, m in cv_maes.items() if np.isfinite(m)]
            if valid:
                valid.sort(key=lambda kv: kv[1])
                top_s, top_m = valid[0]
                plan = Plan(
                    strategies=[top_s], combine="single",
                    reason=f"walk-forward CV top-1 (H_v={H_v}, MAEs={cv_maes})",
                    weights=[1.0],
                )
        else:
            # softmax τ：N 大用小 τ（winner-take-all），N 小用大 τ（更分散）
            tau = 0.3 if N >= 50 else 0.6
            sw = softmax_weights(cv_maes, tau=tau)
            if sw:
                # 仅保留权重 > 5% 的策略，避免引入烂候选
                sw = [(s, w) for s, w in sw if w >= 0.05]
                tot = sum(w for _, w in sw)
                sw = [(s, w / tot) for s, w in sw]
                plan = Plan(
                    strategies=[s for s, _ in sw],
                    combine="ensemble" if len(sw) > 1 else "single",
                    reason=f"walk-forward CV reweight (H_v={H_v}, τ={tau}, MAEs={cv_maes})",
                    weights=[w for _, w in sw],
                )

    plan_hist: list[Plan] = []
    val_hist: list[float] = []
    reflect_log: list[ReflectStep] = []        # 新增
    last_diag_revision: dict | None = None     # 新增
    best_plan, best_val, best_pred = None, float("inf"), None

    # 阈值收紧：默认 val.std × 0.5（避免在 noise 上做反思）
    threshold = val_mae_threshold if val_mae_threshold is not None else (
        float(val.std()) * 0.5 if len(val) > 1 else float("inf")
    )
    # 短 val 的 std 噪声大，再加绝对下限：至少和 train 的均值水平相关
    if len(val) <= 10 and val.std() < train.std() * 0.3:
        threshold = max(threshold, float(train.std()) * 0.3)

    # plan 切换的"改善幅度"门槛：新 plan 的 val MAE 至少比当前 best 改善 N% 才采纳
    # v5 实测 0.20 在 Model Cards 启用后过松（短 val 噪声被反思放大），收紧到 0.40。
    SWITCH_IMPROVE_FRAC = 0.40
    # N 极小时禁用反思（val 噪声压倒任何信号）
    if len(train) <= 12:
        max_reflect = 0

    for _ in range(max_reflect + 1):
        y_val_combo, per_strat = _run_plan(
            plan, train, np.array([]), len(val), season_m
        )
        v_mae = mae(val, y_val_combo) if len(val) > 0 else 0.0
        plan_hist.append(plan)
        val_hist.append(v_mae)
        per_strat_mae = {s: mae(val, p) for s, p in per_strat.items()} if len(val) > 0 else {}

        # best 选取：第一个 plan 总是"基线 best"
        # v5c 默认：reflect_can_replace_best=False → walk-forward initial plan 锁定为 best
        #         反思仍跑（产出 root_cause 用于可解释性），但不替换 best_plan
        # v5b/v4：reflect_can_replace_best=True → 后续候选满足改善门槛才替换
        if best_plan is None:
            best_plan, best_val = plan, v_mae
        elif reflect_can_replace_best and v_mae < best_val * (1.0 - SWITCH_IMPROVE_FRAC):
            best_plan, best_val = plan, v_mae

        if v_mae <= threshold:
            break  # 已足够好

        new_plan, step = _reflect(
            diag, plan, per_strat_mae, plan_hist, H,
            use_model_cards=use_model_cards,
            allow_diagnosis_revision=allow_diagnosis_revision,
        )
        reflect_log.append(step)
        if step.diagnosis_revision:
            last_diag_revision = step.diagnosis_revision
        if new_plan is None or new_plan.strategies == plan.strategies:
            break
        plan = new_plan

    # ===== v6 · 策略 promotion 机制（plan §15.5 + case 2 改进） =====
    # 在所有 reflect 完成后，扫描历史中"单策略 (combine=single 或 strategies=[s])"
    # 且其 per_strat_mae 在 reflect_log 出现过显著优于当前 best 的情形 → 触发二次验证。
    # 二次验证用更稳定的"长 holdout 多窗口 walk-forward CV"，胜出才 promote。
    promotion_info: dict | None = None
    if enable_promotion and N >= 20:
        # 收集所有 reflect_log 中出现过的"在 val 上优于 best_val 的单策略"
        candidate_strats: dict[str, float] = {}
        for step in reflect_log:
            for s, m_val in step.per_strat_mae.items():
                if m_val < best_val * (1.0 - promotion_improve_frac):
                    if s not in candidate_strats or m_val < candidate_strats[s]:
                        candidate_strats[s] = m_val
        if candidate_strats:
            # 用更长的 holdout 但要适应 N：H_v ≤ N//3 保证至少 2 fold
            H_v_promo = max(5, min(N // 3, H // 4, 24))
            n_folds_promo = max(2, min(5, (N - H_v_promo) // H_v_promo))
            promo_pool = list(candidate_strats) + [s for s in best_plan.strategies if s not in candidate_strats]
            promo_maes = walk_forward_eval(promo_pool, train, H_v=H_v_promo,
                                           season_m=season_m, n_folds=n_folds_promo)
            # 计算 best_plan 在 promotion CV 上的"组合 MAE"作 baseline
            # 简化：取 best_plan strategies 的加权 promo MAE
            best_promo = sum(w * promo_maes.get(s, float("inf"))
                             for s, w in zip(best_plan.strategies, best_plan.weights))
            for s, val_m in candidate_strats.items():
                if (s in promo_maes and np.isfinite(promo_maes[s])
                        and promo_maes[s] < best_promo * (1.0 - promotion_improve_frac)):
                    # 通过二次验证 → promote
                    promotion_info = {
                        "promoted_strategy": s,
                        "val_mae": float(val_m),
                        "promo_cv_mae": float(promo_maes[s]),
                        "baseline_promo_cv_mae": float(best_promo),
                        "H_v_promo": H_v_promo, "n_folds_promo": n_folds_promo,
                    }
                    best_plan = Plan(
                        strategies=[s], combine="single",
                        reason=f"promoted from reflect: val={val_m:.3f}, promo_cv={promo_maes[s]:.3f}",
                        weights=[1.0],
                    )
                    best_val = float(val_m)
                    break  # 只 promote 一次（避免连环切换）

    # 用最优 plan 跑 test（train+val 一起作为已知历史）
    history = np.concatenate([train, val]) if len(val) > 0 else train
    y_test, _ = _run_plan(best_plan, history, np.array([]), H, season_m)
    _bh = locals().get("_BANDIT_HANDLE")
    _ah = locals().get("_ADAPTIVE_HANDLE")
    # Round 6: adaptive handle takes precedence; we stash it under bandit_handle slot
    handle = _ah if _ah is not None else _bh
    return y_test, ForecastTrace(
        plan_history=plan_hist, val_mae_history=val_hist,
        final_plan=best_plan, final_val_mae=best_val,
        reflect_steps=reflect_log,
        diagnosis_revised=last_diag_revision,
        promotion=promotion_info,
        bandit_handle=handle,
    )
