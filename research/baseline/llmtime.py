"""B3 · LLMTime（Gruver et al., NeurIPS 2023）。

核心思想：把数字序列编码成字符串，直接交给 LLM 续写，再解析回数字。
最简实现版（够做基线对比）：
  - 归一化到 [0, 99]（min-max scaling，保留 2 位整数）
  - 用空格分隔每个数字（让 tokenizer 不至于把多位数当成一个 token）
  - prompt：上下文 + "Continue the sequence with H values, separated by spaces:"
  - 解析失败 / 数量不对：用最后一个有效值填充
"""
from __future__ import annotations

import re

import numpy as np

from research.utils.llm import chat_cached


def _encode(x: np.ndarray) -> tuple[list[str], float, float]:
    """min-max 缩放到 [0, 99] 整数；返回 (tokens, lo, hi)。"""
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        scaled = np.zeros_like(x)
    else:
        scaled = (x - lo) / (hi - lo) * 99
    tokens = [f"{int(round(v))}" for v in scaled]
    return tokens, lo, hi


def _decode(tokens: list[float], lo: float, hi: float) -> np.ndarray:
    arr = np.asarray(tokens, dtype=np.float64)
    return arr / 99 * (hi - lo) + lo


def predict(train: np.ndarray, val: np.ndarray, H: int,
            seed: int = 42, season_m: int = 1, **_) -> np.ndarray:
    ctx = np.concatenate([train, val])
    tokens, lo, hi = _encode(ctx)
    ctx_str = " ".join(tokens)

    prompt = (
        "You are a time-series forecaster. Below is a univariate sequence of "
        f"{len(tokens)} integers in [0, 99]. Continue the sequence with exactly "
        f"{H} more integers, separated by single spaces, no other text.\n\n"
        f"Sequence: {ctx_str}\n\nContinuation:"
    )
    out = chat_cached(
        messages=[
            {"role": "system", "content": "Output only space-separated integers, nothing else."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        # reasoning model 会先输出思考过程，给足够的预算
        max_tokens=8 * H + 1024,
    )

    # 解析：reasoning model 输出中夹杂大量分析文本，
    # 策略 = 从字符串末尾出发，连续 token 中 numeric 占比 < 0.5 时停止
    toks = re.split(r"[\s,，、;；]+", out.strip())
    tail: list[int] = []
    bad_streak = 0
    for t in reversed(toks):
        if re.fullmatch(r"-?\d+", t):
            tail.append(int(t))
            bad_streak = 0
        else:
            bad_streak += 1
            if bad_streak >= 3 and len(tail) >= 5:
                break
    nums = list(reversed(tail))
    if len(nums) >= H:
        nums = nums[-H:]   # 取最后 H 个，避免吃到推理过程里的中间数字
    else:
        last = nums[-1] if nums else (int(tokens[-1]) if tokens else 50)
        nums = nums + [last] * (H - len(nums))
    # 钳制到 [0, 99]
    nums = [max(0, min(99, n)) for n in nums]
    return _decode(nums, lo, hi)
