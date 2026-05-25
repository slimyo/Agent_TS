"""Phase 3.1 · Curator with Uncertainty Quantification（plan §五）。

输出三类置信度（高/中/低）× 三个维度（trend / season / stationarity），来源：
  - 方案 A · 统计检验：ADF p-value + ACF 峰值 + 线性回归 t-stat
  - 方案 B · LLM 主观：让 LLM 看统计量给出"语言置信度"
  - 方案 C · 双路交叉：A、B 都低才判低（最稳）

plan §5.2 明确要对比 A/B/C 三种来源的 CMR。这里同时输出三个版本。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Literal

import numpy as np
from scipy import signal, stats
from statsmodels.tsa.stattools import acf, adfuller

from research.utils.llm import chat_cached

Conf = Literal["high", "mid", "low"]


@dataclass
class Diagnosis:
    # 统计量
    n: int
    mean: float
    std: float
    trend_slope: float
    trend_tstat: float           # 线性回归 slope 的 t 统计量
    adf_pvalue: float            # 平稳性检验 p-value
    acf_peak_lag: int            # ACF 最大值（lag>1）对应的 lag
    acf_peak_value: float        # 上述峰值

    # 置信度 · 方案 A（统计）
    trend_conf_stat: Conf
    season_conf_stat: Conf
    stat_conf_stat: Conf

    # 置信度 · 方案 B（LLM 主观）
    trend_conf_llm: Conf
    season_conf_llm: Conf
    stat_conf_llm: Conf

    # 置信度 · 方案 C（双路交叉）
    trend_conf_xc: Conf
    season_conf_xc: Conf
    stat_conf_xc: Conf

    # LLM 文本理由（可解释性）
    llm_reason: str

    # v2 (Curator-12d, P6.1 task #24 弱点修复 — outlier 0/6 + variance 6/10 误判)
    outlier_count_z3: int = 0        # |z|>3 离群点数（MAD-based z-score）
    variance_ratio: float = 1.0      # late_std / early_std (split-half), >2 = variance explode


def _stat_confidence(x: np.ndarray, season_m: int = 24) -> dict:
    """方案 A：统计层面置信度。"""
    n = len(x)
    # 线性回归 → trend slope + t-stat
    if n >= 3:
        res = stats.linregress(np.arange(n), x)
        slope = float(res.slope)
        tstat = float(res.slope / (res.stderr + 1e-12))
    else:
        slope, tstat = 0.0, 0.0
    # ADF p-value
    try:
        adf_p = float(adfuller(x, autolag="AIC")[1]) if n >= 4 else 1.0
    except Exception:
        adf_p = 1.0
    # ACF peak（lag 2..min(season_m*2, n-2)）
    max_lag = min(max(season_m * 2, 4), n - 2)
    if max_lag >= 2:
        ac = acf(x, nlags=max_lag, fft=True)
        peak_lag = int(np.argmax(np.abs(ac[2:])) + 2)
        peak_val = float(ac[peak_lag])
    else:
        peak_lag, peak_val = 0, 0.0

    # 阈值映射 → high/mid/low
    def _trend_conf() -> Conf:
        a = abs(tstat)
        if a >= 3.0: return "high"
        if a >= 1.5: return "mid"
        return "low"

    def _season_conf() -> Conf:
        a = abs(peak_val)
        # 短序列 ACF 估计不稳，需要更高门槛
        if n < 30:
            if a >= 0.7: return "high"
            if a >= 0.4: return "mid"
            return "low"
        if a >= 0.5: return "high"
        if a >= 0.3: return "mid"
        return "low"

    def _stationarity_conf() -> Conf:
        # ADF：p 越小，越显著拒绝单位根→越平稳
        if adf_p <= 0.01: return "high"
        if adf_p <= 0.1:  return "mid"
        return "low"

    # v2: outlier count (MAD-based z>3) + split-half variance ratio
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) + 1e-9
    z = np.abs(x - med) / (1.4826 * mad)
    outlier_count_z3 = int((z > 3.0).sum())
    half = n // 2
    early_std = float(np.std(x[:half])) + 1e-9 if half >= 2 else 1e-9
    late_std = float(np.std(x[half:])) + 1e-9 if (n - half) >= 2 else 1e-9
    variance_ratio = late_std / early_std

    return {
        "trend_slope": slope, "trend_tstat": tstat,
        "adf_pvalue": adf_p,
        "acf_peak_lag": peak_lag, "acf_peak_value": peak_val,
        "trend_conf_stat": _trend_conf(),
        "season_conf_stat": _season_conf(),
        "stat_conf_stat": _stationarity_conf(),
        # v2
        "outlier_count_z3": outlier_count_z3,
        "variance_ratio": variance_ratio,
    }


def _llm_confidence(x: np.ndarray, stat: dict, season_m: int) -> dict:
    """方案 B：让 LLM 看统计量 + 序列摘要给主观置信度。"""
    n = len(x)
    # 给 LLM 看 head/tail + 统计量，避免长序列 token 浪费
    head = ", ".join(f"{v:.2f}" for v in x[: min(10, n)])
    tail = ", ".join(f"{v:.2f}" for v in x[-min(10, n):])
    prompt = (
        f"你是时间序列诊断专家。基于以下统计量评估三个维度的置信度（每个必须取 high/mid/low）：\n"
        f"  - trend: 趋势是否真实存在（不是随机游走）\n"
        f"  - season: 季节性是否真实存在（不是噪声幅值）\n"
        f"  - stationarity: 序列是否平稳\n\n"
        f"统计量：\n"
        f"  样本数 n={n}, 季节周期假设 m={season_m}\n"
        f"  趋势斜率={stat['trend_slope']:.4g}, t-stat={stat['trend_tstat']:.3f}\n"
        f"  ADF p-value={stat['adf_pvalue']:.4f}\n"
        f"  ACF 最大峰值={stat['acf_peak_value']:.3f} (lag={stat['acf_peak_lag']})\n"
        f"  序列前 10 点: [{head}]\n  序列后 10 点: [{tail}]\n\n"
        f"返回纯 JSON，仅包含 4 个字段：trend, season, stationarity, reason。\n"
        f"reason 用一两句中文解释。示例：\n"
        f'{{"trend":"mid","season":"low","stationarity":"high","reason":"..."}}'
    )
    raw = chat_cached(
        messages=[
            {"role": "system", "content": "Output only JSON, no markdown fences."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1, max_tokens=2048,
    )
    # 抽 JSON
    import re
    m = re.search(r"\{[^{}]*\}", raw, flags=re.S)
    parsed: dict = {}
    if m:
        try:
            parsed = json.loads(m.group(0))
        except Exception:
            parsed = {}

    def _norm(v) -> Conf:
        if isinstance(v, str) and v.lower() in ("high", "mid", "low"):
            return v.lower()  # type: ignore
        return "mid"

    return {
        "trend_conf_llm":  _norm(parsed.get("trend")),
        "season_conf_llm": _norm(parsed.get("season")),
        "stat_conf_llm":   _norm(parsed.get("stationarity")),
        "llm_reason":      str(parsed.get("reason", ""))[:300],
    }


def _cross(a: Conf, b: Conf) -> Conf:
    """方案 C：双路交叉。规则 = 两者取较低（保守）。"""
    order = {"low": 0, "mid": 1, "high": 2}
    return min([a, b], key=lambda c: order[c])


def diagnose(series: np.ndarray, season_m: int = 24) -> Diagnosis:
    x = np.asarray(series, dtype=np.float64)
    stat = _stat_confidence(x, season_m=season_m)
    llm = _llm_confidence(x, stat, season_m=season_m)

    return Diagnosis(
        n=len(x),
        mean=float(x.mean()), std=float(x.std()),
        trend_slope=stat["trend_slope"], trend_tstat=stat["trend_tstat"],
        adf_pvalue=stat["adf_pvalue"],
        acf_peak_lag=stat["acf_peak_lag"], acf_peak_value=stat["acf_peak_value"],
        trend_conf_stat=stat["trend_conf_stat"],
        season_conf_stat=stat["season_conf_stat"],
        stat_conf_stat=stat["stat_conf_stat"],
        trend_conf_llm=llm["trend_conf_llm"],
        season_conf_llm=llm["season_conf_llm"],
        stat_conf_llm=llm["stat_conf_llm"],
        trend_conf_xc=_cross(stat["trend_conf_stat"], llm["trend_conf_llm"]),
        season_conf_xc=_cross(stat["season_conf_stat"], llm["season_conf_llm"]),
        stat_conf_xc=_cross(stat["stat_conf_stat"], llm["stat_conf_llm"]),
        llm_reason=llm["llm_reason"],
        outlier_count_z3=stat["outlier_count_z3"],
        variance_ratio=stat["variance_ratio"],
    )


if __name__ == "__main__":
    from research.utils.data_loader import load_series
    from research.utils.splitter import few_shot_split
    s, meta = load_series("ETTh1")
    sp = few_shot_split(s, N=20, H=96, seed=1)
    d = diagnose(sp.train, season_m=meta.season_m)
    print(json.dumps(asdict(d), ensure_ascii=False, indent=2))
