"""三大任务 pipeline：把 4 个 agent 串成端到端自适应规划。

每个 pipeline：Curator → SaturationPlanner → RouterAgent(CV) → 执行选中模型 → Reporter。
全部诚实：CV 只用训练集，最终预测才用 test 算指标（评估，不进决策）。

- Classification：候选 {rocket, dtw_1nn, euclid_1nn, moment_1nn, moment_logreg}，CV=训练集 LOO/kfold
- Detection：合成 4-class fault 分类（taskc 设置），同分类 router
- Forecasting：候选 {chronos2, chronos_bolt, arima_ets, naive_seasonal, naive_drift}，CV=walk-forward train-tail
"""
from __future__ import annotations

import numpy as np

from test.agents import (
    CuratorAgent, SaturationPlanner, RouterAgent, ReporterAgent,
)


# ─── Classification / Detection（共用，都是 few-shot 分类）──────────────────────

CLF_CANDIDATES = ["rocket", "moment_1nn", "moment_logreg", "dtw_1nn", "euclid_1nn"]


def run_classification_cell(X_tr, y_tr, X_te, y_te, task="classification",
                            season_m=1, n_min=7):
    """单 cell 端到端。返回 (acc, report)。X 可 [N,L] 或 [N,C,L]（多变量自动展平给分类器）。"""
    from research.agent.clf_strategies import predict_with
    from research.agent.clf_planner import loo_cv_acc, kfold_cv_acc

    curator = CuratorAgent()
    prof = curator.profile_cell(X_tr, y_tr)

    # 多变量 → 展平给现有单变量分类器（[N,C,L]→[N,C*L]）
    def _flat(X):
        X = np.asarray(X)
        return X.reshape(X.shape[0], -1) if X.ndim == 3 else X
    Xtr_f, Xte_f = _flat(X_tr), _flat(X_te)

    # CV 缓存（部署可得，无 test）：同时缓存 per-fold 正确性向量供 trust 估计
    _cv_cache: dict[str, float] = {}
    _fold_cache: dict[str, list] = {}
    def cv_fn(model):
        if model in _cv_cache:
            return _cv_cache[model]
        npc = prof.n_per_class or len(Xtr_f)
        try:
            if npc <= 6:
                a, folds = loo_cv_acc(Xtr_f, y_tr, model)
            else:
                a, folds = kfold_cv_acc(Xtr_f, y_tr, model, k=3, seed=0)
        except Exception:
            a, folds = float("nan"), []
        _cv_cache[model] = a
        _fold_cache[model] = list(folds)
        return a

    def trust_fn(best_m, cv):
        """trust（部署可得，无 test）= 两道惩罚的乘积，直击 Round 9 的 CV↔test 背离：
        (1) **CV 饱和惩罚**（F-R8.8/F-R9.2 核心）：few-shot 下 best 的 CV 接近 1.0 时，
            CV 已失去判别力（饱和到天花板），此时高 CV 反而**不可信** → trust↓。
        (2) **fold 稳定性**：bootstrap 下 best 相对 default 仍胜的比例（避免单 fold 蒙）。
        二者都高才 trust 高。"""
        fb = _fold_cache.get(best_m); fd = _fold_cache.get("rocket")
        if not fb or not fd or len(fb) != len(fd) or len(fb) < 3:
            return 0.5
        cb = float(np.mean(fb))                       # best 的 CV
        # (1) 饱和惩罚：CV≥0.95 视为饱和，trust 线性压到 0；CV≤0.8 不罚
        sat_pen = float(np.clip((0.95 - cb) / 0.15, 0.0, 1.0))
        # (2) fold 稳定性
        diff = np.array(fb, dtype=float) - np.array(fd, dtype=float)
        rng = np.random.default_rng(0); n = len(diff)
        stab = np.mean([1.0 if diff[rng.integers(0, n, n)].mean() > 0 else 0.0 for _ in range(200)])
        return float(sat_pen * stab)

    planner = SaturationPlanner(task, "rocket", [c for c in CLF_CANDIDATES if c != "rocket"],
                                n_min_for_routing=n_min)
    plan = planner.plan(prof)
    router = RouterAgent(cv_fn, base_margin=0.05, trust_fn=trust_fn, trust_min=0.6)
    dec = router.route(plan)

    # 执行选中模型
    try:
        y_pred = predict_with(dec.chosen_model, Xtr_f, y_tr, Xte_f, season_m=season_m)
        acc = float((np.asarray(y_pred) == np.asarray(y_te)).mean())
    except Exception as e:
        # 兜底：base
        y_pred = predict_with("rocket", Xtr_f, y_tr, Xte_f, season_m=season_m)
        acc = float((np.asarray(y_pred) == np.asarray(y_te)).mean())
        dec.trace.append(f"executor fallback to rocket: {e!r}")

    rep = ReporterAgent().report(task, prof, plan, dec)
    rep["acc"] = round(acc, 4)
    return acc, rep


# ─── Forecasting ──────────────────────────────────────────────────────────────

# 偏离候选只保留强 TSFM（chronos_bolt）：finish.md §4 实证 naive_*/arima 在小 walk-forward
# 验证窗上"假赢"是 v8 catastrophic 的根因（short-val overfitting）。trivial 预测器只作 base 安全网，
# 不进偏离候选 → 体现"wrapper around strong TSFM, defer to base"教训（v10/v11）。
FC_CANDIDATES = ["chronos2", "chronos_bolt"]


def _walk_forward_mae(model, train, H, season_m, n_folds=2):
    """train-tail walk-forward CV（部署可得，无 test）。返回 -mean_MAE（越大越好）。

    few-shot 友好：验证 horizon 自适应为 min(H, L//3)，保证 N 小也能切出 fold。
    """
    from research.agent.forecaster_reflect import STRATEGY_FN
    fn = STRATEGY_FN.get(model)
    if fn is None:
        return float("nan")
    L = len(train)
    H_val = max(1, min(H, L // 3))           # 自适应验证窗口
    maes = []
    for k in range(n_folds, 0, -1):
        cut = L - k * H_val
        if cut < max(6, season_m):
            continue
        tr_k = train[:cut]
        val_k = train[cut:cut + H_val]
        if len(val_k) < 1:
            continue
        try:
            pred = fn(tr_k, np.array([]), len(val_k), season_m)
            maes.append(float(np.mean(np.abs(np.asarray(pred)[:len(val_k)] - val_k))))
        except Exception:
            pass
    if not maes:
        return float("nan")
    return -float(np.mean(maes))


def run_forecasting_cell(train, val, test, H, season_m, task="forecasting"):
    """单 cell 端到端预测。返回 (mae, report)。"""
    from research.agent.forecaster_reflect import STRATEGY_FN

    curator = CuratorAgent()
    prof = curator.profile_series(train, season_m=season_m)

    _cv_cache: dict[str, float] = {}
    def cv_fn(model):
        if model not in _cv_cache:
            _cv_cache[model] = _walk_forward_mae(model, train, H, season_m)
        return _cv_cache[model]

    planner = SaturationPlanner(task, "chronos2",
                                [c for c in FC_CANDIDATES if c != "chronos2"],
                                n_min_for_routing=0)   # 预测不做 N-fallback（未饱和）
    plan = planner.plan(prof)
    # margin 用相对值（WF-CV 是 -MAE，量纲跨数据集差异大）：要求候选 WF-MAE 比 base 好 ≥8%
    # 才偏离，挡掉 walk-forward 小验证窗的噪声"假赢"（finish.md §4 short-val overfitting 教训）。
    base_p = cv_fn("chronos2")
    rel_margin = abs(base_p) * 0.08 if base_p == base_p else 0.02
    router = RouterAgent(cv_fn, base_margin=rel_margin)
    dec = router.route(plan)

    fn = STRATEGY_FN.get(dec.chosen_model) or STRATEGY_FN["chronos2"]
    try:
        pred = np.asarray(fn(train, val, H, season_m))[:H]
        mae = float(np.mean(np.abs(pred - np.asarray(test)[:H])))
    except Exception as e:
        pred = np.asarray(STRATEGY_FN["chronos2"](train, val, H, season_m))[:H]
        mae = float(np.mean(np.abs(pred - np.asarray(test)[:H])))
        dec.trace.append(f"executor fallback to chronos2: {e!r}")

    rep = ReporterAgent().report(task, prof, plan, dec)
    rep["mae"] = round(mae, 4)
    return mae, rep


# ========================================================================== #
# method8 端到端：异构专家池 + 在线反馈路由（4-agent 闭环，真实执行模型）
# ========================================================================== #

def run_hetero_online_stream(stream, H=12, season_m=12, discount=0.9, seed=0,
                             use_feedback=True, use_affinity=True, prior_table=None):
    """端到端工业 agent：一条部署流（每元素=一条真实序列 (train,test)）顺序到达。

    每步：CuratorAgent 画像→task signature→regime tag → 在线 bandit(capability 先验暖启动)
          选专家 → **真实执行该专家**得预测→MAE → reward=(base_MAE−MAE)/base_MAE → 更新后验。
    后验按**检测到的 regime tag**分组(episode/状态)，折扣 discount 抗漂移。L1：base 锚点兜底。
    返回逐步记录（含 chosen / mae / base_mae / reward / regime）。
    """
    import numpy as _np
    from test.agents import CuratorAgent, ReporterAgent
    from test.experts import EXPERTS, BASE as EBASE, regime_tag, affinity_prior, task_signature
    from test.online_router import TrustAwareOnlineRouter

    curator = CuratorAgent()
    models = list(EXPERTS.keys())
    router = TrustAwareOnlineRouter(models, EBASE, cap_prior={m: 0.0 for m in models},
                                    discount=discount, seed=seed)
    recs = []
    for (train, test) in stream:
        train = _np.asarray(train, float); test = _np.asarray(test, float)
        ctx = train
        tag = regime_tag(ctx)                              # 部署可得的 regime 分组键
        if prior_table is not None:                       # 学出的 capability profile（按 regime 查表）
            router.cap = {m: prior_table.get(tag, {}).get(m, 0.0) for m in models}
        elif use_affinity:                                # 手工 capability-affinity 暖启动
            router.cap = affinity_prior(ctx)
        cands = models
        chosen = router.act(tag, cands, topk=5)
        # 真实执行专家
        def _mae(name):
            try:
                p = _np.asarray(EXPERTS[name](train, _np.array([]), H, season_m), float)[:H]
                return float(_np.mean(_np.abs(p - test[:H])))
            except Exception:
                return float(_np.mean(_np.abs(test[:H])))
        base_mae = _mae(EBASE); ch_mae = _mae(chosen)
        reward = (base_mae - ch_mae) / (base_mae + 1e-9)
        if use_feedback:
            router.update(tag, chosen, reward)
        recs.append({"regime": tag, "chosen": chosen, "mae": round(ch_mae, 4),
                     "base_mae": round(base_mae, 4), "reward_relMAE": round(reward, 4),
                     "deviated": chosen != EBASE})
    return recs
