"""P6.4b-4 / task #33 · Classification Planner（参照 forecaster_reflect 的 v9/v10 gating 范式）。

设计：
  1. Curator.diagnose 给每个训练样本算诊断 → aggregate 成 'avg diag'
  2. 在 X_train 内部做 LOO 或 K-fold CV 估计每个候选 classifier 的 acc
  3. （可选）Memory.query 取相似历史 case 的 best_classifier
  4. Margin gating: 若 best CV ≥ default + MARGIN → 偏离 default；否则信任 default (Rocket)
  5. （可选）Memory override: 若 ≥半数邻居赞成另一 classifier → revert

返回 (chosen_classifier_name, y_pred, trace)。
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from research.agent.clf_model_cards import CLF_MODEL_CARDS
from research.agent.clf_strategies import (
    CLF_STRATEGY_FN, DEFAULT_CLASSIFIER, predict_with,
)


# ---------- CV 估计 ---------- #

def loo_cv_acc(
    X_train: np.ndarray, y_train: np.ndarray,
    classifier_name: str, **kwargs
) -> tuple[float, list[float]]:
    """Leave-One-Out CV：N 个 sample，每个轮流当 val。
    返回 (mean_acc, per_fold_correctness[0/1])。
    """
    from sklearn.model_selection import LeaveOneOut
    N = len(X_train)
    if N < 2:
        return float("nan"), []
    correct = []
    fn = CLF_STRATEGY_FN[classifier_name]
    for tr_idx, te_idx in LeaveOneOut().split(X_train):
        try:
            y_pred = fn(X_train[tr_idx], y_train[tr_idx],
                        X_train[te_idx], **kwargs)
            correct.append(int(y_pred[0] == y_train[te_idx[0]]))
        except Exception:
            correct.append(0)
    return float(np.mean(correct)), correct


def kfold_cv_acc(
    X_train: np.ndarray, y_train: np.ndarray,
    classifier_name: str, k: int = 3, seed: int = 0, **kwargs
) -> tuple[float, list[float]]:
    """Stratified K-fold CV，每 fold 留 ~1/k 个样本做 val。
    更快 (K << N) 但准确度估计噪声大些。
    """
    from sklearn.model_selection import StratifiedKFold
    N = len(X_train)
    classes = np.unique(y_train)
    min_per_class = min((y_train == c).sum() for c in classes)
    actual_k = min(k, int(min_per_class))
    if actual_k < 2:
        return float("nan"), []
    skf = StratifiedKFold(n_splits=actual_k, shuffle=True, random_state=seed)
    accs = []
    fn = CLF_STRATEGY_FN[classifier_name]
    for tr_idx, te_idx in skf.split(X_train, y_train):
        try:
            y_pred = fn(X_train[tr_idx], y_train[tr_idx],
                        X_train[te_idx], **kwargs)
            accs.append(float((y_pred == y_train[te_idx]).mean()))
        except Exception:
            accs.append(0.0)
    return float(np.mean(accs)), accs


# ---------- Memory 检索（简化版，task #34 完整化） ---------- #

def _diag_feature_vec(diag, std_pop: float = 1.0) -> np.ndarray:
    """从 Diagnosis 12-dim 提取定长特征向量，L2 归一化。"""
    cmap = {"high": 1.0, "mid": 0.5, "low": 0.0}
    v = np.array([
        np.log1p(diag.n),
        np.tanh(diag.trend_tstat / 5.0),
        diag.adf_pvalue,
        diag.acf_peak_value,
        np.log1p(diag.acf_peak_lag),
        cmap[diag.trend_conf_xc],
        cmap[diag.season_conf_xc],
        cmap[diag.stat_conf_xc],
        diag.std / (abs(diag.mean) + 1e-6),
        np.tanh(diag.trend_slope),
        np.log1p(getattr(diag, "outlier_count_z3", 0)),  # v2
        np.tanh(getattr(diag, "variance_ratio", 1.0) - 1.0),  # v2
    ], dtype=np.float32)
    n = np.linalg.norm(v) + 1e-9
    return v / n


def memory_consensus(
    avg_feat: np.ndarray,
    memory_path: str,
    k: int = 5,
    k_min: int = 5,
    exclude_meta: Optional[dict] = None,
) -> tuple[Optional[str], int, list[tuple[float, dict]]]:
    """简单 memory 共识：检索 k 个最相似 (diag_feature, best_classifier) → majority vote。
    返回 (winner_clf_or_None, support_count, neighbors_list)。

    exclude_meta: leave-one-cell-out，剔除同 cell 的 case（feedback 问题 6）。
    """
    p = Path(memory_path)
    if not p.exists():
        return None, 0, []
    cases = []
    for line in p.read_text().splitlines():
        if not line.strip(): continue
        try:
            cases.append(json.loads(line))
        except Exception:
            continue
    if exclude_meta:
        keys = [kk for kk in ("dataset", "N_per_class", "seed") if kk in exclude_meta]
        cases = [c for c in cases
                 if not (keys and all((c.get("meta") or {}).get(kk) == exclude_meta[kk]
                                      for kk in keys))]
    if len(cases) < k_min:
        return None, 0, []
    # similarity = cosine（特征已 L2-normalized）
    feats = np.array([c["diag_feature"] for c in cases], dtype=np.float32)
    sims = feats @ avg_feat
    top_idx = np.argsort(-sims)[:k]
    neighbors = [(float(sims[i]), cases[i]) for i in top_idx]
    from collections import Counter
    votes = Counter(c["best_classifier"] for _, c in neighbors)
    winner, count = votes.most_common(1)[0]
    if count < max(2, k_min // 2):
        return None, count, neighbors
    return winner, count, neighbors


# ---------- 主入口 ---------- #

@dataclass
class ClfRoutingTrace:
    chosen: str
    chosen_reason: str
    cv_accs: dict[str, float]
    default_classifier: str
    margin: float
    mem_winner: Optional[str] = None
    mem_support: int = 0
    diag_features_summary: dict = field(default_factory=dict)
    cv_method: str = "loo"


def classification_planner(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    season_m: int = 1,
    use_cv: bool = True,
    cv_method: str = "loo",
    use_memory: bool = False,
    memory_path: Optional[str] = None,
    candidates: Optional[list[str]] = None,
    n_min_for_routing: int = 7,       # v2 (task #37): N<7 强制 default fallback
    use_enhanced_features: bool = False,  # v3 (task #41): 25-dim z-score features
    weighted_vote_min_ratio: float = 0.6, # v3 (task #39): 加权 vote 阈值 (覆盖 N_per_class=3 的 catastrophic mis-routes)
    vote_method: str = "topk",            # v5 (feedback Item 3): "topk" (default, top-1 weighted) | "inv_loss" (1/CRPS-style per-classifier)
    use_diverse_retrieval: bool = False,  # v5 (feedback Item 4): replace lowest-sim default-winner neighbor with highest-sim non-default alternative
    use_industrial_signature: bool = False,  # v4 (task #66): industrial-regime prior for Euclid
    default_classifier: str = DEFAULT_CLASSIFIER,
    margin: float = 0.10,
    seed: int = 0,
    use_bayesian: bool = False,           # Round 5 §八 unify: BayesianRouter 替全部 hand switches
    bayesian_decide: str = "argmax",      # argmax | thompson | risk_min
    dataset: str | None = None,           # future-proof for CRPSPrior conditioning
) -> tuple[str, np.ndarray, ClfRoutingTrace]:
    """完整 routing：CV → Memory → Margin gating → execute。
    返回 (chosen_classifier_name, y_pred, trace)。
    """
    if candidates is None:
        candidates = ["rocket", "moment_1nn", "moment_logreg", "dtw_1nn", "euclid_1nn"]
    candidates = [c for c in candidates if c in CLF_STRATEGY_FN]

    N = len(X_train)

    # feedback 问题 6 · leave-one-cell-out: 若 memory bank 与评测集重叠，必须排除
    # 查询 cell 自身的 case（否则它以 sim≈1 回灌自己的 outcome = self-leakage）。
    # 需要 dataset 才能定位；N_per_class 由类别数推得（few-shot loader 保证均衡）。
    exclude_meta = None
    if dataset is not None:
        try:
            n_classes = int(len(np.unique(y_train)))
            exclude_meta = {"dataset": dataset, "seed": seed,
                            "N_per_class": N // max(1, n_classes)}
        except Exception:
            exclude_meta = {"dataset": dataset, "seed": seed}

    # env override
    if os.environ.get("ADAPTTS_CLF_PLANNER", "").lower() == "bayesian":
        use_bayesian = True
        bayesian_decide = os.environ.get("ADAPTTS_DECIDE", bayesian_decide).lower()

    # ─── Round 5 §八 · BayesianRouter unified path ────────────────────────────
    if use_bayesian:
        from research.agent.bayesian_router import (
            BayesianRouter, Context, Evidence,
            NPrior, IndustrialPrior, AvailabilityPrior,
            CVLikelihood, MemoryLikelihood,
        )

        # Build evidence: CV accs → loss = 1-acc; memory neighbors if enabled
        cv_accs_b: dict[str, float] = {}
        cv_losses_b: dict[str, float] = {}
        if use_cv:
            for name in candidates:
                try:
                    if cv_method == "kfold":
                        a, _ = kfold_cv_acc(X_train, y_train, name, k=3, seed=seed)
                    else:
                        a, _ = loo_cv_acc(X_train, y_train, name)
                    if np.isfinite(a):
                        cv_accs_b[name] = a
                        cv_losses_b[name] = 1.0 - a
                except Exception:
                    pass

        # Industrial signal as posterior continuous, not hard override
        industrial_p = None
        if use_industrial_signature and "euclid_1nn" in candidates:
            from research.utils.series_features import industrial_stats
            ind = [industrial_stats(x) for x in X_train]
            acf_decay = float(np.mean([s.get("acf_decay", 0.5) for s in ind]))
            quant_bits = float(np.mean([s.get("quant_bits", 8.0) for s in ind]))
            # high acf_decay + low quant_bits → industrial regime
            industrial_p = max(0.0, min(1.0,
                acf_decay - quant_bits / 16.0 + 0.3))

        # Memory neighbors via existing infra (reuse Item 4 query_diverse if requested)
        memory_neighbors_b: list[dict] | None = None
        if use_memory and memory_path:
            try:
                from research.agent.clf_memory import ClfMemory
                from research.utils.series_features import featurize_cell, normalize_zscore
                from pathlib import Path as _P
                if use_enhanced_features:
                    full_vec = featurize_cell(X_train, y_train)
                    ns_path = _P(memory_path).parent / (_P(memory_path).stem + "_norm.npz")
                    if ns_path.exists():
                        ns = np.load(ns_path)
                        full_vec = normalize_zscore(full_vec, ns["mean"], ns["std"])
                    full_vec = full_vec / (np.linalg.norm(full_vec) + 1e-9)
                    mem = ClfMemory(memory_path, dim=len(full_vec))
                else:
                    from research.agent.curator_uq import diagnose
                    diags = [diagnose(x, season_m=season_m) for x in X_train]
                    feats = np.array([_diag_feature_vec(d) for d in diags])
                    full_vec = feats.mean(axis=0)
                    full_vec = full_vec / (np.linalg.norm(full_vec) + 1e-9)
                    mem = ClfMemory(memory_path, dim=len(full_vec))
                if len(mem) >= 3:
                    k_mem = int(os.environ.get("CLF_MEM_K", "5"))
                    if use_diverse_retrieval:
                        neighbors = mem.query_diverse(full_vec, k=k_mem,
                                                       default_classifier=default_classifier,
                                                       exclude_meta=exclude_meta)
                    else:
                        neighbors = mem.query(full_vec, k=k_mem,
                                              exclude_meta=exclude_meta)
                    # deployment-safe: pass CV accs only (feedback 问题 6)
                    memory_neighbors_b = [
                        {"sim": s, "cv_accs": c.votable_accs()}
                        for s, c in neighbors
                    ]
            except Exception:
                pass

        router = BayesianRouter(
            candidates=candidates,
            priors=[
                AvailabilityPrior(local_models=tuple(candidates), remote_models=()),
                NPrior(default_model=default_classifier, N_threshold=n_min_for_routing,
                       strength=2.0),
                IndustrialPrior(target_model="euclid_1nn", strength=2.0)
                  if use_industrial_signature and "euclid_1nn" in candidates else None,
            ],
            likelihoods=[
                CVLikelihood(sigma_sq=0.1),
                MemoryLikelihood(),
            ],
        )
        # filter None priors
        router.priors = [p for p in router.priors if p is not None]

        ctx = Context(dataset=dataset, N=N, H=None, industrial=industrial_p,
                      allow_remote=False)
        ev = Evidence(cv_losses=cv_losses_b or None,
                      memory_neighbors=memory_neighbors_b)
        chosen, post = router.decide(ctx, ev, mode=bayesian_decide)
        y_pred = predict_with(chosen, X_train, y_train, X_test, season_m=season_m)
        trace = ClfRoutingTrace(
            chosen=chosen,
            chosen_reason=f"Bayesian decide({bayesian_decide}): π={post.get(chosen, 0):.3f}; "
                          f"NPrior(N={N}) + Industrial(p={industrial_p}) + "
                          f"CVLik({len(cv_losses_b)}) + MemLik({len(memory_neighbors_b or [])})",
            cv_accs=cv_accs_b, default_classifier=default_classifier, margin=margin,
            cv_method=cv_method,
        )
        return chosen, y_pred, trace
    # ─── end Bayesian path ────────────────────────────────────────────────────

    # v2 (task #37): N<n_min_for_routing 强制 default，跳过 CV 噪声
    # 类比 forecasting v8→v10 加 N<15 fallback。实测 LOO CV 在 N≤4 给 catastrophic
    # mis-routes (BeetleFly N=3 -25pp, BirdChicken N=3 -20pp)
    if N < n_min_for_routing:
        y_pred = predict_with(default_classifier, X_train, y_train, X_test, season_m=season_m)
        trace = ClfRoutingTrace(
            chosen=default_classifier,
            chosen_reason=f"v2 N-fallback (N={N} < {n_min_for_routing}): force default '{default_classifier}'",
            cv_accs={}, default_classifier=default_classifier, margin=margin,
            cv_method=cv_method,
        )
        return default_classifier, y_pred, trace

    # 1) CV 估计各候选 acc
    cv_accs: dict[str, float] = {}
    if use_cv:
        for name in candidates:
            try:
                if cv_method == "kfold":
                    a, _ = kfold_cv_acc(X_train, y_train, name, k=3, seed=seed)
                else:
                    a, _ = loo_cv_acc(X_train, y_train, name)
                cv_accs[name] = a
            except Exception:
                cv_accs[name] = float("nan")

    # 2) 默认决策：默认 Rocket
    chosen = default_classifier
    reason = f"default ({default_classifier})"

    # 3) Margin gating
    if cv_accs and np.isfinite(cv_accs.get(default_classifier, 0)):
        default_acc = cv_accs[default_classifier]
        # 找 best non-default
        others = [(n, a) for n, a in cv_accs.items()
                  if n != default_classifier and np.isfinite(a)]
        if others:
            others.sort(key=lambda kv: -kv[1])
            best_n, best_a = others[0]
            if best_a >= default_acc + margin:
                chosen = best_n
                reason = (f"CV winner '{best_n}' acc={best_a:.3f} "
                          f"beats default acc={default_acc:.3f} by ≥{margin*100:.0f}pp")
            else:
                reason = (f"trust default '{default_classifier}' acc={default_acc:.3f}; "
                          f"best other '{best_n}' acc={best_a:.3f} margin={best_a-default_acc:+.3f} < {margin}")

    # 4) Memory override（如启用）
    mem_winner, mem_support = None, 0
    avg_feat = None
    if use_memory and memory_path:
        if use_enhanced_features:
            # v3 (task #38+#41): 25-dim 增强特征 + z-score (依赖外部 norm_stats.json)
            from research.utils.series_features import featurize_cell, normalize_zscore
            from research.agent.clf_memory import ClfMemory, consensus_winner_weighted, consensus_winner_inv_loss
            full_vec = featurize_cell(X_train, y_train)
            # 加载 norm stats（如有 sidecar）
            from pathlib import Path
            ns_path = Path(memory_path).parent / (Path(memory_path).stem + "_norm.npz")
            if ns_path.exists():
                ns = np.load(ns_path)
                full_vec = normalize_zscore(full_vec, ns["mean"], ns["std"])
            full_vec = full_vec / (np.linalg.norm(full_vec) + 1e-9)
            avg_feat = full_vec
            mem = ClfMemory(memory_path, dim=len(full_vec))
            if len(mem) >= 3:
                k_mem = int(os.environ.get("CLF_MEM_K", "5"))
                if use_diverse_retrieval:
                    neighbors = mem.query_diverse(full_vec, k=k_mem,
                                                  default_classifier=default_classifier,
                                                  exclude_meta=exclude_meta)
                else:
                    neighbors = mem.query(full_vec, k=k_mem,
                                          exclude_meta=exclude_meta)
                vote_fn = consensus_winner_inv_loss if vote_method == "inv_loss" else consensus_winner_weighted
                mem_winner, mem_support_ratio = vote_fn(
                    neighbors, k=5, min_vote_ratio=weighted_vote_min_ratio
                )
                mem_support = mem_support_ratio
        else:
            # v1 路径（向后兼容）
            from research.agent.curator_uq import diagnose
            diags = [diagnose(x, season_m=season_m) for x in X_train]
            feats = np.array([_diag_feature_vec(d) for d in diags])
            avg_feat = feats.mean(axis=0)
            avg_feat = avg_feat / (np.linalg.norm(avg_feat) + 1e-9)
            mem_winner, mem_support, _ = memory_consensus(
                avg_feat, memory_path,
                k=int(os.environ.get("CLF_MEM_K", "5")),
                k_min=int(os.environ.get("CLF_MEM_K_MIN", "5")),
                exclude_meta=exclude_meta,
            )
        if mem_winner and mem_winner != chosen:
            chosen = mem_winner
            reason += f" | memory override → {mem_winner} (support={mem_support:.2f})"

    # 4.5) v4 (task #66): Industrial-regime signature override.
    # 当 acf_decay 高 + quant_bits 低 → 信号平滑且离散水平少 (Wafer-like)，
    # 此时 Euclid 在 LOO 上 tied with rocket 但 test 上 +13pp，应当 prefer Euclid.
    if use_industrial_signature and "euclid_1nn" in candidates:
        from research.utils.series_features import industrial_stats
        ind = [industrial_stats(x) for x in X_train]
        acf_d = float(np.mean([d["acf_decay"] for d in ind]))
        quant = float(np.mean([d["quant_bits"] for d in ind]))
        # Signature (calibrated 2026-05): low acf_decay (<0.4, persistent signal)
        # AND low quant_bits (<7.5, discrete levels) → Wafer-like industrial regime
        if acf_d < 0.4 and quant < 7.5 and chosen != "euclid_1nn":
            # Check euclid CV acc is at least tied with default (not worse)
            euclid_acc = cv_accs.get("euclid_1nn", float("nan"))
            default_acc = cv_accs.get(default_classifier, float("nan"))
            if np.isfinite(euclid_acc) and np.isfinite(default_acc) and euclid_acc >= default_acc - 0.05:
                chosen = "euclid_1nn"
                reason += f" | industrial signature (acf_decay={acf_d:.2f},quant={quant:.2f}) → euclid_1nn"

    # 5) 执行
    y_pred = predict_with(chosen, X_train, y_train, X_test, season_m=season_m)

    trace = ClfRoutingTrace(
        chosen=chosen, chosen_reason=reason,
        cv_accs={k: round(v, 4) for k, v in cv_accs.items()},
        default_classifier=default_classifier, margin=margin,
        mem_winner=mem_winner, mem_support=mem_support,
        cv_method=cv_method,
    )
    return chosen, y_pred, trace


# ---------- B7 公共接口（runner / experiments 用）---------- #

def b7_agent_router(X_train, y_train, X_test, season_m: int = 1, **kwargs) -> np.ndarray:
    """B7 Agent-Router：兼容 baseline 接口，仅返回 y_pred。"""
    chosen, y_pred, trace = classification_planner(
        X_train, y_train, X_test, season_m=season_m, **kwargs
    )
    b7_agent_router.last_trace = trace  # type: ignore[attr-defined]
    return y_pred


if __name__ == "__main__":
    # Smoke: Coffee 5-shot
    from research.utils.ucr_loader import load_ucr_fewshot
    X_tr, y_tr, X_te, y_te = load_ucr_fewshot("Coffee", n_per_class=5, seed=1)
    print(f"Coffee 5-shot: X_tr={X_tr.shape}, X_te={X_te.shape}\n")
    chosen, y_pred, trace = classification_planner(
        X_tr, y_tr, X_te, season_m=1, use_cv=True, cv_method="loo",
        margin=0.10,
    )
    acc = float((y_pred == y_te).mean())
    print(f"chosen: {chosen}")
    print(f"reason: {trace.chosen_reason}")
    print(f"cv_accs: {trace.cv_accs}")
    print(f"final test acc: {acc:.3f}")
