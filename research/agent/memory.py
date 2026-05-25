"""Phase 3.6 · 层四记忆库（plan §四层四 + §八）。

骨架版：faiss-cpu 内积索引 + jsonl 案例库。E5 实验时会用到。

案例 schema（Case）：
  - feature: 序列特征向量（用诊断量做指纹，无需额外 LLM 调用）
  - diag:    Diagnosis dataclass 序列化
  - final_plan: 最终采用的策略组合
  - test_mae:   对应 test MAE（事后回填）
  - meta:       {dataset, N, H, seed, start_idx}

主接口：
  Memory(path).add(case)             写一条
  Memory(path).query(feat, k)        返回 top-k 相似案例
  case_features(diag)                把 Diagnosis 转成定长向量

E5 实验需要的"分布突变检测" / "K 容量调参"留作 hook，先把读写跑通。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ---------- 特征向量：从诊断直接构造（不调 LLM） ---------- #

def case_features(diag) -> np.ndarray:
    """把 Diagnosis 映射成 10 维特征向量，置信度 high/mid/low → 1.0/0.5/0.0。"""
    cmap = {"high": 1.0, "mid": 0.5, "low": 0.0}
    v = np.array([
        np.log1p(diag.n),                     # 规模感
        np.tanh(diag.trend_tstat / 5.0),      # 趋势强度（截尾）
        diag.adf_pvalue,                       # 平稳性概率
        diag.acf_peak_value,                   # 季节强度
        np.log1p(diag.acf_peak_lag),          # 季节周期粗略
        cmap[diag.trend_conf_xc],
        cmap[diag.season_conf_xc],
        cmap[diag.stat_conf_xc],
        diag.std / (abs(diag.mean) + 1e-6),   # 变异系数
        np.tanh(diag.trend_slope),             # 趋势方向
    ], dtype=np.float32)
    # L2 归一化（faiss 内积 ≈ 余弦相似）
    n = np.linalg.norm(v) + 1e-9
    return v / n


@dataclass
class Case:
    feature: list[float]
    diag: dict
    final_plan: dict
    test_mae: float | None = None
    meta: dict = field(default_factory=dict)


class Memory:
    """faiss 索引 + jsonl 持久化（每条案例一行）。"""

    def __init__(self, path: str | Path, dim: int = 10, k_cap: int | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self.k_cap = k_cap          # 容量上限（None 表示无限）
        self._cases: list[Case] = []
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                self._cases.append(Case(**d))
            except Exception:
                continue

    def _build_index(self):
        import faiss
        idx = faiss.IndexFlatIP(self.dim)
        if self._cases:
            mat = np.array([c.feature for c in self._cases], dtype=np.float32)
            idx.add(mat)
        return idx

    def add(self, case: Case):
        self._cases.append(case)
        # 容量裁剪：先到先出（FIFO），更复杂的"价值评分淘汰"留给 E5
        if self.k_cap is not None and len(self._cases) > self.k_cap:
            drop = len(self._cases) - self.k_cap
            self._cases = self._cases[drop:]
            # 重写文件
            with self.path.open("w") as fh:
                for c in self._cases:
                    fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        else:
            with self.path.open("a") as fh:
                fh.write(json.dumps(asdict(case), ensure_ascii=False) + "\n")

    def query(self, feat: np.ndarray, k: int = 3) -> list[tuple[float, Case]]:
        if not self._cases:
            return []
        idx = self._build_index()
        D, I = idx.search(feat.reshape(1, -1).astype(np.float32), min(k, len(self._cases)))
        return [(float(D[0][j]), self._cases[I[0][j]]) for j in range(len(I[0])) if I[0][j] >= 0]

    def update_last_test_mae(self, test_mae: float) -> bool:
        """v11: 回填最近一次 add 的 case 的 test_mae，并重写 jsonl 持久化。"""
        if not self._cases:
            return False
        self._cases[-1].test_mae = float(test_mae)
        with self.path.open("w") as fh:
            for c in self._cases:
                fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        return True

    def __len__(self) -> int:
        return len(self._cases)


if __name__ == "__main__":
    # 烟测
    from research.agent.curator_uq import diagnose
    from research.utils.data_loader import load_series
    from research.utils.splitter import few_shot_split

    tmp = Path("/tmp/test_memory.jsonl")
    if tmp.exists():
        tmp.unlink()
    mem = Memory(tmp, k_cap=10)

    s, meta = load_series("ETTh1")
    for seed in [1, 42, 123]:
        sp = few_shot_split(s, N=20, H=96, seed=seed)
        d = diagnose(sp.train, season_m=meta.season_m)
        feat = case_features(d)
        mem.add(Case(
            feature=feat.tolist(),
            diag=asdict(d),
            final_plan={"strategies": ["arima_ets", "chronos"], "combine": "ensemble"},
            test_mae=2.0 + seed * 0.01,
            meta={"dataset": "ETTh1", "N": 20, "H": 96, "seed": seed, "start_idx": sp.start_idx},
        ))
    print("memory size:", len(mem))

    # 用 seed=1 的特征查最近邻
    sp = few_shot_split(s, N=20, H=96, seed=1)
    d = diagnose(sp.train, season_m=meta.season_m)
    feat = case_features(d)
    results = mem.query(feat, k=3)
    for sim, c in results:
        print(f"sim={sim:.4f} -> meta={c.meta} test_mae={c.test_mae}")
