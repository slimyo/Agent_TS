"""P6.4b-9 / task #38 · 扩充时序特征到 25-30 维（直击 task #34 limitation）。

包含四组（共 ~28 维）：
  A. 基础统计（mean / std / skew / kurt / range）       5
  B. 时间动态（trend slope / ADF p / autocorr peak）    4
  C. 频域 (FFT peak / spectral entropy / dominant ratio) 4
  D. 复杂度 (permutation entropy / sample entropy / DFA / zero-crossing rate / extrema count) 5
  E. 元信息 (log_L / log_N / n_classes / class_balance / N_per_class) 5
  F. 离群 (outlier_count_z3 / variance_ratio)            2

总维度：~25-28（与 plan §17.7 设计对齐 + feedback 推荐量级）
"""
from __future__ import annotations

import math

import numpy as np


def _safe(x, default=0.0):
    return float(x) if np.isfinite(x) else float(default)


def basic_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64).flatten()
    mean = float(x.mean())
    std = float(x.std()) + 1e-9
    # skewness
    m3 = float(((x - mean) ** 3).mean())
    skew = _safe(m3 / std ** 3)
    # kurtosis (excess)
    m4 = float(((x - mean) ** 4).mean())
    kurt = _safe(m4 / std ** 4 - 3.0)
    return {
        "mean": mean,
        "std": std,
        "skew": skew,
        "kurt": kurt,
        "range_": float(x.max() - x.min()),
    }


def trend_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)
    if n < 3:
        return {"trend_slope": 0.0, "trend_tstat": 0.0, "adf_pvalue": 1.0, "acf_peak": 0.0}
    from scipy import stats
    res = stats.linregress(np.arange(n), x)
    try:
        from statsmodels.tsa.stattools import adfuller
        adf_p = float(adfuller(x, autolag="AIC")[1])
    except Exception:
        adf_p = 1.0
    # ACF peak (lag 2..min(n-2, 24))
    max_lag = min(n - 2, 24)
    if max_lag >= 2:
        from statsmodels.tsa.stattools import acf
        ac = acf(x, nlags=max_lag, fft=True)
        peak = float(np.max(np.abs(ac[2:])))
    else:
        peak = 0.0
    return {
        "trend_slope": _safe(res.slope),
        "trend_tstat": _safe(res.slope / (res.stderr + 1e-12)),
        "adf_pvalue": _safe(adf_p, 1.0),
        "acf_peak": _safe(peak),
    }


def freq_stats(x: np.ndarray) -> dict:
    """FFT 频谱特征：peak frequency、spectral entropy、主频能量占比。"""
    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)
    if n < 8:
        return {"spectral_peak_freq": 0.0, "spectral_entropy": 0.0,
                "dominant_freq_ratio": 0.0, "spectral_centroid": 0.0}
    # Detrend
    x_d = x - x.mean()
    fft = np.fft.rfft(x_d)
    power = np.abs(fft) ** 2
    if power.sum() < 1e-12:
        return {"spectral_peak_freq": 0.0, "spectral_entropy": 0.0,
                "dominant_freq_ratio": 0.0, "spectral_centroid": 0.0}
    p_norm = power / power.sum()
    freqs = np.fft.rfftfreq(n)
    # peak freq
    peak_idx = int(np.argmax(power))
    peak_freq = float(freqs[peak_idx])
    # spectral entropy
    p_safe = p_norm + 1e-12
    spec_ent = -float((p_safe * np.log(p_safe)).sum())
    spec_ent_norm = spec_ent / math.log(len(p_norm))  # 归一化到 [0, 1]
    # dominant frequency energy ratio (peak energy / total)
    dom_ratio = float(power[peak_idx] / power.sum())
    # spectral centroid
    centroid = float((freqs * p_norm).sum())
    return {
        "spectral_peak_freq": peak_freq,
        "spectral_entropy": spec_ent_norm,
        "dominant_freq_ratio": dom_ratio,
        "spectral_centroid": centroid,
    }


def complexity_stats(x: np.ndarray) -> dict:
    """复杂度 / 非线性特征：permutation entropy / zero-crossing rate / extrema count。"""
    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)
    if n < 5:
        return {"perm_entropy": 0.0, "zero_cross_rate": 0.0, "extrema_density": 0.0,
                "hurst_proxy": 0.5}

    # Permutation entropy (order=3)
    order = 3
    permutations = {}
    for i in range(n - order + 1):
        pat = tuple(np.argsort(x[i:i + order]).tolist())
        permutations[pat] = permutations.get(pat, 0) + 1
    total = sum(permutations.values())
    pe = 0.0
    for cnt in permutations.values():
        p = cnt / total
        pe -= p * math.log(p + 1e-12)
    pe_norm = pe / math.log(math.factorial(order))

    # Zero-crossing rate (relative to mean)
    centered = x - x.mean()
    zc = float(np.sum(centered[:-1] * centered[1:] < 0)) / (n - 1)

    # Extrema density: 1st-order local maxima + minima
    d = np.diff(x)
    sign_changes = np.diff(np.sign(d))
    extrema = float(np.sum(sign_changes != 0)) / (n - 2 if n > 2 else 1)

    # Hurst-like proxy: log(std_late / std_early) for split-half (already in fault_taxonomy)
    half = n // 2
    se = x[:half].std() + 1e-9
    sl = x[half:].std() + 1e-9
    hurst = _safe(0.5 + 0.5 * np.tanh(np.log(sl / se)))

    return {
        "perm_entropy": pe_norm,
        "zero_cross_rate": zc,
        "extrema_density": extrema,
        "hurst_proxy": hurst,
    }


def outlier_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) + 1e-9
    z = np.abs(x - med) / (1.4826 * mad)
    n_out = int((z > 3.0).sum())
    half = n // 2
    se = float(x[:half].std()) + 1e-9 if half >= 2 else 1e-9
    sl = float(x[half:].std()) + 1e-9 if (n - half) >= 2 else 1e-9
    var_ratio = sl / se
    return {
        "outlier_count_z3": n_out,
        "outlier_rate": n_out / max(n, 1),
        "variance_ratio": var_ratio,
    }


def industrial_stats(x: np.ndarray) -> dict:
    """Industrial-regime markers: smoothness / noise-floor / quantization / plateau / acf decay.

    Motivated by Wafer N=5 finding (industrial_case §4.19): B7v3 routes to Rocket
    while Euclid wins by +8pp. The 25-dim Curator features did not separate the
    low-noise smooth-signal regime where Euclid dominates.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)
    std = float(x.std()) + 1e-9
    if n < 5:
        return {"smoothness": 0.0, "noise_floor": 1.0, "quant_bits": 16.0,
                "plateau_ratio": 0.0, "acf_decay": 0.0}
    d = np.diff(x)
    abs_d_norm = float(np.mean(np.abs(d)) / std)
    smoothness = 1.0 / (1.0 + abs_d_norm)
    # Noise floor: std of 2nd-difference (high-pass approx) over std
    d2 = np.diff(d)
    noise_floor = float(d2.std() / std) if n > 2 else 1.0
    # Quantization: log2(distinct values) — low when signal is bit-quantized
    uniq = len(np.unique(np.round(x, 6)))
    quant_bits = float(math.log2(uniq + 1))
    # Plateau ratio: fraction of consecutive points with |diff| < 0.01*std
    plateau_ratio = float(np.mean(np.abs(d) < 0.01 * std))
    # ACF decay: how fast |acf(1)| - |acf(5)|
    try:
        from statsmodels.tsa.stattools import acf
        max_lag = min(n - 2, 5)
        ac = acf(x, nlags=max_lag, fft=True)
        if max_lag >= 5:
            acf_decay = float(abs(ac[1]) - abs(ac[5]))
        else: acf_decay = float(abs(ac[1]) - abs(ac[-1]))
    except Exception:
        acf_decay = 0.0
    return {
        "smoothness": _safe(smoothness),
        "noise_floor": _safe(noise_floor),
        "quant_bits": _safe(quant_bits),
        "plateau_ratio": _safe(plateau_ratio),
        "acf_decay": _safe(acf_decay),
    }


def extract_full_features(series: np.ndarray, meta: dict | None = None) -> dict:
    """单序列 -> 完整特征 dict（不含 meta_*）。meta 可选合并 meta 信息。"""
    out = {}
    out.update(basic_stats(series))
    out.update(trend_stats(series))
    out.update(freq_stats(series))
    out.update(complexity_stats(series))
    out.update(outlier_stats(series))
    out.update(industrial_stats(series))
    if meta:
        # 安全转 numeric
        if "L" in meta: out["meta_log_L"] = float(np.log1p(meta["L"]))
        if "n_classes" in meta: out["meta_n_classes"] = float(meta["n_classes"])
        if "N_per_class" in meta: out["meta_log_N"] = float(np.log1p(meta["N_per_class"]))
        if "class_balance" in meta: out["meta_class_balance"] = float(meta["class_balance"])
        if "N_total" in meta: out["meta_log_N_total"] = float(np.log1p(meta["N_total"]))
    return out


FEATURE_ORDER = [
    # basic 5
    "mean", "std", "skew", "kurt", "range_",
    # trend 4
    "trend_slope", "trend_tstat", "adf_pvalue", "acf_peak",
    # freq 4
    "spectral_peak_freq", "spectral_entropy", "dominant_freq_ratio", "spectral_centroid",
    # complexity 4
    "perm_entropy", "zero_cross_rate", "extrema_density", "hurst_proxy",
    # outlier 3
    "outlier_count_z3", "outlier_rate", "variance_ratio",
    # industrial 5 (task #66)
    "smoothness", "noise_floor", "quant_bits", "plateau_ratio", "acf_decay",
    # meta 5
    "meta_log_L", "meta_n_classes", "meta_log_N", "meta_class_balance", "meta_log_N_total",
]


def featurize_cell(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """整 cell 的特征向量（avg of per-sample feature + meta 信息），定长。"""
    from collections import Counter
    L = X_train.shape[1]
    classes = np.unique(y_train)
    C = len(classes)
    N_total = len(X_train)
    N_per_class = N_total / max(C, 1)
    counts = Counter(y_train.tolist())
    max_class_frac = max(counts.values()) / N_total
    class_balance = 1.0 - max_class_frac  # 1 - max_class_frac, 越大越平衡
    meta = {"L": L, "n_classes": C, "N_per_class": N_per_class,
            "class_balance": class_balance, "N_total": N_total}

    per_sample_feats = []
    for x in X_train:
        f = extract_full_features(x)
        per_sample_feats.append([_safe(f.get(k, 0.0)) for k in FEATURE_ORDER
                                 if not k.startswith("meta_")])
    avg = np.array(per_sample_feats).mean(axis=0)
    # 合并 meta
    meta_vec = []
    for k in FEATURE_ORDER:
        if k.startswith("meta_"):
            key = k.replace("meta_", "")
            if key == "log_L": meta_vec.append(float(np.log1p(L)))
            elif key == "n_classes": meta_vec.append(float(C))
            elif key == "log_N": meta_vec.append(float(np.log1p(N_per_class)))
            elif key == "class_balance": meta_vec.append(float(class_balance))
            elif key == "log_N_total": meta_vec.append(float(np.log1p(N_total)))
            else: meta_vec.append(0.0)
    full_vec = np.concatenate([avg, np.array(meta_vec, dtype=np.float64)])
    return full_vec.astype(np.float32)


def normalize_zscore(vec: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """对应 z-score 归一化。std 为零的维度返回 0。"""
    std_safe = np.where(std < 1e-9, 1.0, std)
    return ((vec - mean) / std_safe).astype(np.float32)


if __name__ == "__main__":
    from research.utils.ucr_loader import load_ucr_fewshot
    X_tr, y_tr, _, _ = load_ucr_fewshot("Coffee", n_per_class=5, seed=1)
    feat = featurize_cell(X_tr, y_tr)
    print(f"Coffee 5-shot: feature dim={len(feat)}, sample values:")
    for i, k in enumerate(FEATURE_ORDER):
        print(f"  [{i:2}] {k:25} = {feat[i]:.4f}")
