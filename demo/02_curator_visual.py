"""
Demo Step 2 — Curator: 可视化 + 多模态 LLM 看图
对应 TSci 论文 Curator 的子步骤 2-3：生成可视化套件 V，让多模态 LLM
看图给出结构画像 A = {trend, seasonality, stationarity}。

流程：
  load → clean (复用 Step 1 工具) → 画 4 合 1 图（总览 + STL + ACF + PACF）
  → base64 编码 → 发给 glm-4.6v-flash → 解析 JSON → 打印 A

为什么生成合成数据：sample_data.csv 只有 41 个点，做不出像样的 STL/ACF。
本步骤换一份"含趋势 + 周季节性 + 噪声"的 365 天合成序列，让图有信号可讨论。
"""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无 GUI 环境也能画
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL


HERE = Path(__file__).parent

# ---------- 0. 复用 Step 1 的工具 ----------
# 文件名以数字开头，正常 import 不行，用 importlib 动态加载

def _load_step1():
    spec = importlib.util.spec_from_file_location(
        "step1", HERE / "01_curator_minimal.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["step1"] = mod  # dataclass / typing 等需要从 sys.modules 反查
    spec.loader.exec_module(mod)
    return mod


step1 = _load_step1()


# ---------- 1. 准备一段"有信号"的时序 ----------

def make_synthetic_series(n: int = 365, seed: int = 0) -> pd.Series:
    """趋势(线性) + 周季节性(7 天) + 高斯噪声 + 几个缺失/异常点。"""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend = 0.05 * t
    seasonal = 3.0 * np.sin(2 * np.pi * t / 7.0)
    noise = rng.normal(0, 0.6, n)
    y = 10.0 + trend + seasonal + noise

    # 注入几个缺失值和异常值，让 Curator 有事可做
    for i in [30, 87, 200, 240]:
        y[i] = np.nan
    y[120] = y[120] + 25  # 正异常
    y[300] = y[300] - 20  # 负异常

    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series(y, index=idx, name="value")


# ---------- 2. 画 4 合 1 可视化（论文里的 V） ----------

def plot_curator_panel(series: pd.Series, period: int = 7) -> bytes:
    """生成 4 个子图：总览 / STL 趋势+季节性+残差 / ACF / PACF。返回 PNG bytes。"""
    s = series.dropna()

    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(4, 2, hspace=0.6, wspace=0.25)

    # (A) 总览：原始 + 滚动均值 + 滚动标准差
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(s.index, s.values, lw=0.8, label="raw")
    ax0.plot(s.rolling(14).mean(), lw=1.5, label="rolling mean (14d)")
    ax0.plot(s.rolling(14).std(), lw=1.0, label="rolling std (14d)", linestyle="--")
    ax0.set_title("A. Overview: raw + rolling mean/std")
    ax0.legend(loc="upper left", fontsize=8)

    # (B/C/D) STL 分解
    stl = STL(s, period=period, robust=True).fit()
    ax1 = fig.add_subplot(gs[1, :])
    ax1.plot(stl.trend); ax1.set_title("B. STL — trend")
    ax2 = fig.add_subplot(gs[2, 0])
    ax2.plot(stl.seasonal); ax2.set_title(f"C. STL — seasonal (period={period})")
    ax3 = fig.add_subplot(gs[2, 1])
    ax3.plot(stl.resid); ax3.axhline(0, color="k", lw=0.5)
    ax3.set_title("D. STL — residual")

    # (E/F) ACF / PACF
    ax4 = fig.add_subplot(gs[3, 0])
    plot_acf(s.values, ax=ax4, lags=min(40, len(s) // 2 - 1))
    ax4.set_title("E. ACF")
    ax5 = fig.add_subplot(gs[3, 1])
    plot_pacf(s.values, ax=ax5, lags=min(40, len(s) // 2 - 1), method="ywm")
    ax5.set_title("F. PACF")

    fig.suptitle("Curator Visual Suite V", fontsize=12, y=0.995)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ---------- 3. 调用多模态 LLM 拿结构画像 A ----------

VISION_SYSTEM = """你是时序分析专家。我会给你一张包含 6 个子图的诊断面板：
  A) 原始序列 + 滚动均值/标准差
  B) STL 趋势分量
  C) STL 季节性分量
  D) STL 残差
  E) ACF
  F) PACF

请结合所有子图，输出严格 JSON：
{
  "trend": "increasing" | "decreasing" | "none",
  "seasonality": "yes" | "no",
  "seasonal_period": <整数, 没有就填 0>,
  "stationarity": "stationary" | "non_stationary",
  "reason": "<一句中文解释，引用具体子图作为证据，例如 '子图B显示...'>"
}
不要输出 JSON 之外的任何文字。"""


def make_vision_client() -> tuple[OpenAI, str]:
    """复用 .env 但用 vision_model。当前只支持 zhipu，其它 provider 后续再加。"""
    load_dotenv()
    provider = os.getenv("PROVIDER", "zhipu").lower()
    cfg = step1.PROVIDERS.get(provider, {})
    vision_model = cfg.get("vision_model")
    if not vision_model:
        raise RuntimeError(
            f"provider={provider} 在 PROVIDERS 里没配 vision_model。"
            f" 把 PROVIDER 改成支持视觉的服务（目前: zhipu）。"
        )
    api_key = os.getenv(cfg["env_key"])
    if not api_key:
        raise RuntimeError(f"请在 .env 里设置 {cfg['env_key']}")

    client = OpenAI(api_key=api_key, base_url=cfg["base_url"])
    model = os.getenv("VISION_MODEL", vision_model)
    return client, model


def ask_vlm_for_structure(client: OpenAI, model: str, png_bytes: bytes) -> dict:
    b64 = base64.b64encode(png_bytes).decode()
    data_url = f"data:image/png;base64,{b64}"

    messages = [
        {"role": "system", "content": VISION_SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": "请基于这张图给出 JSON 结构画像。"},
            ],
        },
    ]

    # 视觉模型对 response_format 支持参差，先试再退
    try:
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=0.2,
            response_format={"type": "json_object"},
        )
    except Exception:
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=0.2
        )

    raw = resp.choices[0].message.content or ""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        import re
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            raise ValueError(f"VLM 没返回 JSON：\n{raw}")
        return json.loads(m.group(0))


# ---------- 4. 主流程 ----------

def main() -> None:
    print("=== Step 1: 生成 / 加载时序 ===")
    series = make_synthetic_series()
    print(f"shape={series.shape}, 缺失={int(series.isna().sum())}, "
          f"min={np.nanmin(series):.2f}, max={np.nanmax(series):.2f}\n")

    print("=== Step 2: 复用 Curator Step 1 清洗 ===")
    q = step1.compute_quality_vector(series)
    print("Q:", json.dumps(q, ensure_ascii=False))
    text_client, text_model = step1.make_client()
    print(f"[文本模型: {text_model}]")
    strategy = step1.ask_llm_for_strategy(text_client, text_model, q)
    print("π:", json.dumps(strategy, ensure_ascii=False))
    cleaned = step1.apply_strategy(series, strategy)
    print(f"清洗后 shape={cleaned.shape}, 缺失={int(cleaned.isna().sum())}\n")

    print("=== Step 3: 生成可视化套件 V ===")
    png = plot_curator_panel(cleaned, period=7)
    out_path = HERE / "curator_panel.png"
    out_path.write_bytes(png)
    print(f"已保存 → {out_path}（{len(png)/1024:.1f} KB）\n")

    print("=== Step 4: 多模态 LLM 看图给结构画像 A ===")
    vclient, vmodel = make_vision_client()
    print(f"[视觉模型: {vmodel}]")
    structure = ask_vlm_for_structure(vclient, vmodel, png)
    print(json.dumps(structure, ensure_ascii=False, indent=2), "\n")

    print("=== Step 5: 汇总 Curator 状态 C = {Q, V, A} ===")
    state = {
        "Q": q,
        "V_path": str(out_path),
        "A": structure,
        "strategy_pi": strategy,
    }
    state_path = HERE / "curator_state.json"
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    print(f"完整状态已保存 → {state_path}")


if __name__ == "__main__":
    main()
