"""P6.4b-5 / task #34 · 分类 Memory layer。

参照 `agent/memory.py` (forecasting) 的设计，但 Case 字段重定义为分类语义：
  - feature: 12-dim 诊断特征（avg over training samples 的 diag）
  - meta: dataset / N_per_class / seed
  - best_classifier: 该 cell 上实际 test acc 最高的 classifier name
  - test_acc: best classifier 的 test acc
  - all_clf_accs: {clf_name: test_acc} 完整 dict
  - chosen_by_planner: planner 实际选了哪个（可能 ≠ best）

Memory consensus override 规则：
  - 检索 k 个最相似邻居
  - 若 ≥ K_MIN/2 个邻居 best_classifier 一致 → 用该 classifier
  - 与 planner CV 决策对照：
    - mem_winner == cv_winner → 共识增强
    - mem_winner ≠ cv_winner → 单向 revert (mem 投票更多则覆盖 CV)
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class ClfCase:
    # 注意：字段名 'diag_feature' 与 clf_planner.memory_consensus 期望一致
    diag_feature: list[float]     # 12-dim avg diag feature (L2-normalized)
    best_classifier: str          # post-hoc winner
    test_acc: float
    meta: dict = field(default_factory=dict)  # {dataset, N_per_class, seed}
    all_clf_accs: dict = field(default_factory=dict)
    chosen_by_planner: Optional[str] = None
    cv_accs: dict = field(default_factory=dict)

    # 向后兼容：允许 'feature' 别名读取
    @property
    def feature(self) -> list[float]:
        return self.diag_feature


class ClfMemory:
    """faiss 索引 + jsonl 持久化（每条 case 一行）。"""

    def __init__(self, path: str | Path, dim: int = 12, k_cap: Optional[int] = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self.k_cap = k_cap
        self._cases: list[ClfCase] = []
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                self._cases.append(ClfCase(**d))
            except Exception:
                continue

    def add(self, case: ClfCase):
        self._cases.append(case)
        if self.k_cap is not None and len(self._cases) > self.k_cap:
            drop = len(self._cases) - self.k_cap
            self._cases = self._cases[drop:]
            with self.path.open("w") as fh:
                for c in self._cases:
                    fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        else:
            with self.path.open("a") as fh:
                fh.write(json.dumps(asdict(case), ensure_ascii=False) + "\n")

    def query(self, feat: np.ndarray, k: int = 5) -> list[tuple[float, ClfCase]]:
        """numpy brute-force kNN（cosine 通过 IP，假设 feat 与 case.feature 都 L2-normalized）。"""
        if not self._cases:
            return []
        mat = np.array([c.diag_feature for c in self._cases], dtype=np.float32)
        q = feat.astype(np.float32).reshape(-1)
        sims = mat @ q
        top_idx = np.argsort(-sims)[:k]
        return [(float(sims[i]), self._cases[i]) for i in top_idx]

    def __len__(self) -> int:
        return len(self._cases)


def consensus_winner_weighted(
    neighbors: list[tuple[float, ClfCase]],
    k: int = 5,
    min_vote_ratio: float = 0.6,
) -> tuple[Optional[str], float]:
    """task #39 · 加权投票（feedback 推荐）：sim 作为权重，去掉硬阈值。
    返回 (winner_or_None, support_ratio)。
    """
    if not neighbors:
        return None, 0.0
    nbrs = neighbors[:k]
    total = sum(sim for sim, _ in nbrs)
    if total < 1e-9:
        return None, 0.0
    vote_w = {}
    for sim, c in nbrs:
        vote_w[c.best_classifier] = vote_w.get(c.best_classifier, 0.0) + sim
    winner = max(vote_w.items(), key=lambda kv: kv[1])
    support_ratio = winner[1] / total
    if support_ratio < min_vote_ratio:
        return None, support_ratio
    return winner[0], support_ratio


def consensus_winner(
    neighbors: list[tuple[float, ClfCase]],
    k_min: int = 3,
    similarity_threshold: float = 0.85,
) -> tuple[Optional[str], int]:
    """从邻居 list 提取 majority best_classifier。
    返回 (winner_name_or_None, support_count)。

    规则：
      - 只看 similarity >= threshold 的邻居（避免远邻居稀释）
      - majority vote on best_classifier
      - 若 winner 支持数 < k_min → 不输出
    """
    close = [(sim, c) for sim, c in neighbors if sim >= similarity_threshold]
    if not close:
        return None, 0
    votes = Counter(c.best_classifier for _, c in close)
    winner, count = votes.most_common(1)[0]
    if count < k_min:
        return None, count
    return winner, count


def write_case_from_sweep_row(memory_path: str, row: dict,
                              diag_feature: np.ndarray):
    """从一个 sweep row (含 method-per-row) 不够；
    实际写入需在 sweep aggregator 一次写一个 cell 的全部 classifier accs。
    """
    raise NotImplementedError("Use ClfMemory.add() with pre-aggregated ClfCase.")


# ---------- 工具：从已有 sweep results 构建 memory ---------- #

def build_memory_from_sweep(
    memory_path: str,
    sweep_jsonl: str,
    diag_feature_fn,            # callable: (dataset, N_per_class, seed) → feature
    overwrite: bool = True,
):
    """读 taskb_ucr.jsonl 风格的 sweep results（per-row method/acc），
    按 (dataset, N_per_class, seed) 聚合，每 cell 取 max-acc classifier 作 best，
    写入 ClfMemory。
    """
    from collections import defaultdict
    rows = [json.loads(l) for l in open(sweep_jsonl)]
    by_cell = defaultdict(dict)  # (ds, N, seed) → {method: acc}
    for r in rows:
        key = (r["dataset"], r["N_per_class"], r["seed"])
        by_cell[key][r["method"]] = r["acc"]

    p = Path(memory_path)
    if overwrite and p.exists():
        p.unlink()
    mem = ClfMemory(memory_path)

    # 把 method 名映射到 strategy name（taskb_run.py 使用的命名）
    METHOD_TO_CLF = {
        "B1_dtw": "dtw_1nn",
        "B2_euclid": "euclid_1nn",
        "B3_rocket": "rocket",
        "B4a_moment_1nn": "moment_1nn",
        "B4b_moment_lr": "moment_logreg",
    }

    for (ds, n, seed), accs in by_cell.items():
        clf_accs = {METHOD_TO_CLF.get(m, m): a
                    for m, a in accs.items()
                    if m in METHOD_TO_CLF}
        if not clf_accs:
            continue
        best = max(clf_accs.items(), key=lambda kv: kv[1])
        try:
            feat = diag_feature_fn(ds, n, seed)
        except Exception as e:
            print(f"skip {ds} {n} {seed}: feature fn failed: {e!r}")
            continue
        case = ClfCase(
            diag_feature=feat.tolist(),
            meta={"dataset": ds, "N_per_class": n, "seed": seed},
            best_classifier=best[0],
            test_acc=float(best[1]),
            all_clf_accs={k: float(v) for k, v in clf_accs.items()},
        )
        mem.add(case)

    return mem


if __name__ == "__main__":
    # Smoke: 从 taskb_ucr.jsonl 构建 memory
    from research.utils.ucr_loader import load_ucr_fewshot
    from research.agent.curator_uq import diagnose
    from research.agent.clf_planner import _diag_feature_vec

    def make_feature(ds, n, seed):
        X_tr, y_tr, _, _ = load_ucr_fewshot(ds, n_per_class=n, seed=seed)
        diags = [diagnose(x, season_m=1) for x in X_tr]
        feats = np.array([_diag_feature_vec(d) for d in diags])
        avg = feats.mean(axis=0)
        return avg / (np.linalg.norm(avg) + 1e-9)

    mem = build_memory_from_sweep(
        "/tmp/clf_memory_test.jsonl",
        "research/results/taskb_ucr.jsonl",
        make_feature,
        overwrite=True,
    )
    print(f"Built memory with {len(mem)} cases")
    # Sample query: BeetleFly N=3 seed=1 (v1 失败 cell)
    feat = make_feature("BeetleFly", 3, 1)
    neighbors = mem.query(feat, k=5)
    print(f"\nTop-5 neighbors for BeetleFly N=3 seed=1:")
    for sim, c in neighbors:
        print(f"  sim={sim:.3f} best={c.best_classifier:14} "
              f"acc={c.test_acc:.3f} meta={c.meta}")
    winner, support = consensus_winner(neighbors, k_min=3, similarity_threshold=0.85)
    print(f"\nConsensus winner: {winner} (support={support})")
