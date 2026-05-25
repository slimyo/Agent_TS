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
) -> tuple[Optional[str], int, list[tuple[float, dict]]]:
    """简单 memory 共识：检索 k 个最相似 (diag_feature, best_classifier) → majority vote。
    返回 (winner_clf_or_None, support_count, neighbors_list)。
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
    default_classifier: str = DEFAULT_CLASSIFIER,
    margin: float = 0.10,
    seed: int = 0,
) -> tuple[str, np.ndarray, ClfRoutingTrace]:
    """完整 routing：CV → Memory → Margin gating → execute。
    返回 (chosen_classifier_name, y_pred, trace)。
    """
    if candidates is None:
        candidates = ["rocket", "moment_1nn", "moment_logreg", "dtw_1nn", "euclid_1nn"]
    candidates = [c for c in candidates if c in CLF_STRATEGY_FN]

    N = len(X_train)

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
            from research.agent.clf_memory import ClfMemory, consensus_winner_weighted
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
                neighbors = mem.query(full_vec, k=int(os.environ.get("CLF_MEM_K", "5")))
                mem_winner, mem_support_ratio = consensus_winner_weighted(
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
            )
        if mem_winner and mem_winner != chosen:
            chosen = mem_winner
            reason += f" | memory override → {mem_winner} (support={mem_support:.2f})"

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
