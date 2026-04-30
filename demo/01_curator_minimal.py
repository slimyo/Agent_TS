"""
Demo Step 1 — 最小化 Curator Agent
对应论文 TSci 中 Curator 的"质量诊断 + 预处理"那一小步。

流程：读 CSV → 用 numpy/pandas 算质量向量 Q → 把 Q 喂给 LLM 拿策略 π
       → 用代码按策略清洗数据得到 D̃。

LLM 优先用免费/开源 OpenAI 兼容服务（默认 SiliconFlow）。换服务只需改
.env 里的 PROVIDER / *_API_KEY 两个值。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


# ---------- 0. LLM 客户端：从 .env 选一个 OpenAI 兼容 provider ----------

PROVIDERS = {
    # 国内免费/低价 OpenAI 兼容服务，按需切换
    "zhipu": {
        # 智谱 BigModel：glm-4.7-flash 免费文本，glm-4.6v-flash 免费多模态
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "env_key": "ZHIPU_API_KEY",
        "model": "glm-4.7-flash",
        "vision_model": "glm-4.6v-flash",
    },
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "env_key": "SILICONFLOW_API_KEY",
        "model": "Qwen/Qwen2.5-7B-Instruct",
    },
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "env_key": "DASHSCOPE_API_KEY",
        "model": "qwen-turbo",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "env_key": "DEEPSEEK_API_KEY",
        "model": "deepseek-chat",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "env_key": "OLLAMA_API_KEY",  # 任意非空字符串即可
        "model": "qwen2.5:7b",
    },
}


def make_client() -> tuple[OpenAI, str]:
    load_dotenv()
    provider = os.getenv("PROVIDER", "zhipu").lower()
    if provider not in PROVIDERS:
        raise ValueError(f"未知 provider={provider}，可选: {list(PROVIDERS)}")
    cfg = PROVIDERS[provider]
    api_key = os.getenv(cfg["env_key"])
    if not api_key:
        raise RuntimeError(
            f"请在 .env 里设置 {cfg['env_key']}（当前 PROVIDER={provider}）"
        )
    client = OpenAI(api_key=api_key, base_url=cfg["base_url"])
    model = os.getenv("MODEL", cfg["model"])
    return client, model


# ---------- 1. 工具函数：算质量向量 Q（论文公式 1） ----------

def compute_quality_vector(series: pd.Series) -> dict:
    """确定性统计：LLM 不擅长精算，这一步必须由代码完成。"""
    s = series.astype(float)
    n = int(len(s))
    missing = int(s.isna().sum())

    clean = s.dropna()
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = int(((clean < low) | (clean > high)).sum())

    # 线性趋势斜率（最小二乘）
    x = np.arange(len(clean))
    slope = float(np.polyfit(x, clean.values, 1)[0]) if len(clean) >= 2 else 0.0

    return {
        "n": n,
        "missing_count": missing,
        "missing_ratio": round(missing / n, 4) if n else 0.0,
        "mean": round(float(clean.mean()), 4),
        "std": round(float(clean.std()), 4),
        "min": round(float(clean.min()), 4),
        "max": round(float(clean.max()), 4),
        "trend_slope": round(slope, 6),
        "outlier_count_iqr": outliers,
    }


# ---------- 2. LLM 调用：拿预处理策略 π（论文公式 2） ----------

SYSTEM_PROMPT = """你是时序数据预处理专家。仅根据输入的统计向量 Q，输出一个严格的 JSON 对象，不要写任何额外文字。
JSON 字段固定为：
- missing_strategy: 取值 ∈ {"linear_interpolation", "ffill", "bfill", "mean", "drop"}
- outlier_strategy: 取值 ∈ {"clip_iqr", "drop", "keep"}
- reason: 一句话中文解释，说明为什么这样选。"""


def ask_llm_for_strategy(client: OpenAI, model: str, q: dict) -> dict:
    user_msg = f"质量向量 Q（JSON）：\n{json.dumps(q, ensure_ascii=False, indent=2)}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    kwargs = dict(model=model, messages=messages, temperature=0.2)

    # 优先用 JSON mode；不支持的服务退回普通模式后用正则抠 JSON
    try:
        resp = client.chat.completions.create(
            **kwargs, response_format={"type": "json_object"}
        )
    except Exception:
        resp = client.chat.completions.create(**kwargs)

    raw = resp.choices[0].message.content or ""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # 兜底：从文本里抠出第一个 {...} 块
        import re
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            raise ValueError(f"LLM 没返回 JSON：{raw!r}")
        return json.loads(m.group(0))


# ---------- 3. 工具函数：执行变换 D → D̃（论文公式 3） ----------

def apply_strategy(series: pd.Series, strategy: dict) -> pd.Series:
    s = series.astype(float).copy()

    ms = strategy.get("missing_strategy", "linear_interpolation")
    if ms == "linear_interpolation":
        s = s.interpolate(method="linear", limit_direction="both")
    elif ms == "ffill":
        s = s.ffill().bfill()
    elif ms == "bfill":
        s = s.bfill().ffill()
    elif ms == "mean":
        s = s.fillna(s.mean())
    elif ms == "drop":
        s = s.dropna()
    else:
        raise ValueError(f"未知 missing_strategy={ms}")

    os_ = strategy.get("outlier_strategy", "clip_iqr")
    if os_ == "clip_iqr":
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        s = s.clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)
    elif os_ == "drop":
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        s = s[(s >= q1 - 1.5 * iqr) & (s <= q3 + 1.5 * iqr)]
    elif os_ == "keep":
        pass
    else:
        raise ValueError(f"未知 outlier_strategy={os_}")

    return s


# ---------- 4. 主流程 ----------

def main() -> None:
    csv_path = Path(__file__).parent / "sample_data.csv"
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    series = df["value"]

    print("=== Step 1: 加载数据 ===")
    print(f"shape={series.shape}, 缺失值={int(series.isna().sum())}, 类型={series.dtype}\n")

    print("=== Step 2: 计算质量向量 Q ===")
    q = compute_quality_vector(series)
    print(json.dumps(q, ensure_ascii=False, indent=2), "\n")

    print("=== Step 3: 调 LLM 拿预处理策略 π ===")
    client, model = make_client()
    print(f"[使用模型: {model}]")
    strategy = ask_llm_for_strategy(client, model, q)
    print()

    print("=== Step 4: 解析 LLM 输出 ===")
    print(json.dumps(strategy, ensure_ascii=False, indent=2), "\n")

    print("=== Step 5: 执行变换得到 D̃ ===")
    cleaned = apply_strategy(series, strategy)
    print(f"干净数据 shape={cleaned.shape}, 缺失值={int(cleaned.isna().sum())}")
    print(f"min={cleaned.min():.3f}, max={cleaned.max():.3f}, mean={cleaned.mean():.3f}")


if __name__ == "__main__":
    main()
