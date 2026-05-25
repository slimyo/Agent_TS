"""少样本切割协议（对应 plan §2.2）。

切法（按时间顺序，无 shuffle）：
  raw -> 取一段长度为 (N + val_len + H) 的连续窗口
       -> train = 前 N 步
          val   = 紧接的 val_len 步
          test  = 紧接的 H 步

N=10 时 val_len 默认降到 3（plan §2.2 指定）。
seed 用于随机选窗口起点（同一 (dataset, N, seed) 始终给出相同窗口，方便复现）。
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Split:
    train: np.ndarray
    val:   np.ndarray
    test:  np.ndarray
    start_idx: int   # 训练集首个时间步在原序列的下标，便于排错


def default_val_len(N: int) -> int:
    if N <= 10:
        return 3
    return 10


def few_shot_split(
    series: np.ndarray,
    N: int | str,
    H: int = 96,
    val_len: int | None = None,
    seed: int = 42,
) -> Split:
    """从 series 里切一窗 (N, val_len, H)。

    Args:
        N: 整数训练长度；或字符串 "Full" 表示拿到 test 之前的全量历史作为 train。
        H: 测试步长。
        val_len: 验证长度，缺省按 default_val_len。
        seed: 控制窗口起点选取。
    """
    n = len(series)
    if isinstance(N, str) and N.lower() == "full":
        # 全量：留出最后 H 步当 test，再前面 val_len 步当 val，其余全是 train
        vl = default_val_len(10_000) if val_len is None else val_len
        if n < vl + H + 50:
            raise ValueError(f"series too short for Full split (len={n})")
        test = series[-H:]
        val = series[-(H + vl): -H]
        train = series[: -(H + vl)]
        return Split(train, val, test, start_idx=0)

    N = int(N)
    vl = default_val_len(N) if val_len is None else val_len
    win = N + vl + H
    if n < win:
        raise ValueError(f"series len {n} < required window {win} for N={N}, H={H}")

    rng = np.random.default_rng(seed)
    # 起点范围：[0, n - win]，端点闭区间
    start = int(rng.integers(0, n - win + 1))
    train = series[start: start + N]
    val   = series[start + N: start + N + vl]
    test  = series[start + N + vl: start + N + vl + H]
    return Split(train, val, test, start_idx=start)


if __name__ == "__main__":
    from .data_loader import load_series
    s, _ = load_series("ETTh1")
    for N in [10, 20, 50, 100, "Full"]:
        sp = few_shot_split(s, N=N, H=96, seed=1)
        print(f"N={N}: train={len(sp.train)}, val={len(sp.val)}, test={len(sp.test)}, start={sp.start_idx}")
