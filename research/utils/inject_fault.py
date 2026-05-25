"""P6.4 · 时序故障注入器（fault_taxonomy 的反向）。

注入 4 种 fault 在 base series 上生成 labeled 数据集：
  0 normal           — 不注入
  1 trend_break      — 中段加 +k*std 阶跃
  2 seasonal_break   — 周期内随机翻转部分点
  3 outlier_burst    — 插入 3-5 个 ±3-5σ 离群点

(variance_explode / stationarity_flip 与 trend_break / outlier 重合度高，先 4-class)
"""
from __future__ import annotations

import numpy as np


FAULT_LABELS = ["normal", "trend_break", "seasonal_break", "outlier_burst"]

# task #25 RCA 5-fault 注入（不含 normal，对齐 fault_taxonomy.FAULT_NAMES）
RCA_FAULT_LABELS = [
    "trend_break", "seasonal_flip", "variance_explode",
    "outlier_burst", "stationarity_flip",
]


def inject_normal(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return x.copy()


def inject_trend_break(x: np.ndarray, rng: np.random.Generator,
                        shift_std: float = 2.5) -> np.ndarray:
    y = x.copy()
    n = len(y)
    # 在 30%-70% 之间随机位置阶跃
    pos = rng.integers(int(n * 0.3), int(n * 0.7))
    shift = shift_std * float(np.std(y)) * (1 if rng.random() > 0.5 else -1)
    y[pos:] = y[pos:] + shift
    return y


def inject_seasonal_break(x: np.ndarray, rng: np.random.Generator,
                          season_m: int = 24) -> np.ndarray:
    y = x.copy()
    n = len(y)
    # 后半段每个周期内 reverse subsequence
    half = n // 2
    if season_m > 1 and half >= season_m * 2:
        for start in range(half, n - season_m, season_m):
            end = min(start + season_m, n)
            # Reverse within season
            y[start:end] = y[start:end][::-1]
    else:
        # fallback: 随机翻转后半序列
        y[half:] = y[half:][::-1]
    return y


def inject_outlier_burst(x: np.ndarray, rng: np.random.Generator,
                          n_outliers: int = 4, z: float = 4.0) -> np.ndarray:
    y = x.copy()
    n = len(y)
    s = float(np.std(y))
    positions = rng.choice(n, size=n_outliers, replace=False)
    for p in positions:
        sign = 1 if rng.random() > 0.5 else -1
        y[p] = float(np.mean(y)) + sign * z * s
    return y


INJECTORS = [inject_normal, inject_trend_break, inject_seasonal_break, inject_outlier_burst]


def inject_variance_explode(x: np.ndarray, rng: np.random.Generator,
                             ratio: float = 3.5) -> np.ndarray:
    """task #25 fault：后半段 std × ratio，模拟 volatility regime shift。"""
    y = x.copy().astype(np.float64)
    n = len(y)
    half = n // 2
    seg = y[half:]
    seg_mean = float(seg.mean())
    y[half:] = seg_mean + (seg - seg_mean) * ratio
    return y


def inject_stationarity_flip(x: np.ndarray, rng: np.random.Generator,
                              drift_scale: float = 4.0) -> np.ndarray:
    """task #25 fault：注入非线性 drift (二次 trend) 破坏平稳性。"""
    y = x.copy().astype(np.float64)
    n = len(y)
    t = np.linspace(0, 1, n)
    sign = 1 if rng.random() > 0.5 else -1
    drift = sign * drift_scale * float(np.std(y)) * (t ** 2)
    return y + drift


# 5-fault injector dict (no normal) for task #25
RCA_INJECTORS = {
    "trend_break": inject_trend_break,
    "seasonal_flip": inject_seasonal_break,  # alias
    "variance_explode": inject_variance_explode,
    "outlier_burst": inject_outlier_burst,
    "stationarity_flip": inject_stationarity_flip,
}


# ========================================================
# Task #43 · Out-of-Taxonomy (OOT) faults
# These do NOT match any of the 5 in-taxonomy fault types.
# B0-rule's `detect_faults()` can only output trend_break/
# seasonal_flip/variance_explode/outlier_burst/stationarity_flip
# → will be wrong by construction on OOT data.
# Agent has free-form NL output → can describe OOT faults.
# ========================================================

def inject_missing_data_gap(x: np.ndarray, rng: np.random.Generator,
                             gap_frac: float = 0.20) -> np.ndarray:
    """OOT-1: 随机连续段置 mean (模拟缺失数据 → 用均值补全)。"""
    y = x.copy().astype(np.float64)
    n = len(y)
    gap_len = int(n * gap_frac)
    start = rng.integers(int(n * 0.3), int(n * 0.7) - gap_len)
    y[start:start + gap_len] = float(np.mean(y))
    return y


def inject_heavy_noise_contamination(x: np.ndarray, rng: np.random.Generator,
                                      noise_scale: float = 2.5) -> np.ndarray:
    """OOT-2: 全序列加 heavy Gaussian noise (不是 outlier burst，而是 SNR drop)。"""
    y = x.copy().astype(np.float64)
    sigma = float(np.std(y)) * noise_scale
    return y + rng.normal(0, sigma, size=len(y))


def inject_mode_collapse(x: np.ndarray, rng: np.random.Generator,
                          collapse_frac: float = 0.40) -> np.ndarray:
    """OOT-3: 后半段塌缩到近常数 (信号丢失，与 variance_explode 反向)。"""
    y = x.copy().astype(np.float64)
    n = len(y)
    start = int(n * (1 - collapse_frac))
    y[start:] = float(np.mean(y[start:])) + rng.normal(0, 0.01 * float(np.std(y)), size=n - start)
    return y


def inject_frequency_modulation(x: np.ndarray, rng: np.random.Generator,
                                 freq_scale: float = 3.0) -> np.ndarray:
    """OOT-4: 后半段乘以 chirp 调频信号 (周期数变化)。"""
    y = x.copy().astype(np.float64)
    n = len(y)
    half = n // 2
    t = np.arange(n - half) / (n - half)
    chirp = np.sin(2 * np.pi * freq_scale * t * t)  # 频率随 t 加速
    y[half:] = y[half:] + chirp * float(np.std(y))
    return y


def inject_quantization(x: np.ndarray, rng: np.random.Generator,
                         n_levels: int = 4) -> np.ndarray:
    """OOT-5: 后半段量化到 n_levels 离散电平 (sensor degradation)。"""
    y = x.copy().astype(np.float64)
    n = len(y)
    half = n // 2
    seg = y[half:]
    lo, hi = seg.min(), seg.max()
    levels = np.linspace(lo, hi, n_levels)
    digit = np.digitize(seg, levels) - 1
    digit = np.clip(digit, 0, n_levels - 1)
    y[half:] = levels[digit]
    return y


OOT_INJECTORS = {
    "missing_data_gap":           inject_missing_data_gap,
    "heavy_noise_contamination":  inject_heavy_noise_contamination,
    "mode_collapse":              inject_mode_collapse,
    "frequency_modulation":       inject_frequency_modulation,
    "quantization":               inject_quantization,
}


# 每种 OOT fault 的描述 + 关键词（用于 LLM-as-judge / 关键词 F1 评估）
OOT_DESCRIPTIONS = {
    "missing_data_gap": {
        "description": "A continuous segment of the series is replaced by a constant (the segment mean), simulating missing data filled by mean imputation.",
        "keywords": ["missing", "gap", "constant", "imputation", "flat", "plateau", "缺失", "常数", "平段"],
    },
    "heavy_noise_contamination": {
        "description": "The entire series is contaminated by heavy Gaussian noise (signal-to-noise ratio drops dramatically). Differs from outlier burst in that ALL points are noisy, not just a few.",
        "keywords": ["noise", "SNR", "noisy", "contamination", "Gaussian", "random", "signal-to-noise", "噪声", "信噪比"],
    },
    "mode_collapse": {
        "description": "The latter portion of the series collapses to near-constant value (signal loss). Opposite of variance_explode.",
        "keywords": ["collapse", "constant", "flat", "low variance", "signal loss", "dead", "frozen", "塌缩", "丢失"],
    },
    "frequency_modulation": {
        "description": "The latter half of the series has a chirp-modulated frequency component added (period length changes within the segment).",
        "keywords": ["chirp", "frequency", "modulation", "period", "tempo", "non-stationary spectrum", "频率", "调频"],
    },
    "quantization": {
        "description": "The latter half is quantized to a small number of discrete levels (sensor bit-depth reduction or ADC failure).",
        "keywords": ["quantization", "discrete", "levels", "step", "staircase", "ADC", "bit", "量化", "离散", "阶梯"],
    },
}


def build_oot_rca_dataset(base_series: np.ndarray, window_len: int = 128,
                           n_per_class: int = 5, seed: int = 1,
                           season_m: int = 24) -> list[dict]:
    """Task #43 · 构造 OOT × n_per_class cells，GT = OOT fault label。"""
    rng = np.random.default_rng(seed)
    n_total = len(base_series) - window_len * 2
    cells = []
    for cls_name, injector in OOT_INJECTORS.items():
        for k in range(n_per_class):
            start = int(rng.integers(0, n_total))
            train = injector(base_series[start:start + window_len].astype(np.float64), rng)
            test = injector(
                base_series[start + window_len:start + window_len * 2].astype(np.float64),
                rng,
            )
            cells.append({
                "train": train, "test": test,
                "fault_label": cls_name,
                "description": OOT_DESCRIPTIONS[cls_name]["description"],
                "keywords": OOT_DESCRIPTIONS[cls_name]["keywords"],
                "seed_idx": k, "start_idx": start,
            })
    return cells


def build_rca_synthetic_dataset(base_series: np.ndarray, window_len: int = 128,
                                 n_per_class: int = 10, seed: int = 1,
                                 season_m: int = 24) -> list[dict]:
    """task #25 · 构造 5-fault × n_per_class 个 cell，clean GT。
    Returns list of dicts {train, test, fault_label, seed, start_idx}.
    每 cell：train = injected window，test = 后续 H 步真实序列（也注入相同 fault for consistency）。
    """
    rng = np.random.default_rng(seed)
    n_total = len(base_series) - window_len * 2  # 留 test 区
    cells = []
    for cls_name, injector in RCA_INJECTORS.items():
        for k in range(n_per_class):
            start = int(rng.integers(0, n_total))
            train_raw = base_series[start:start + window_len].astype(np.float64)
            test_raw = base_series[start + window_len:start + window_len * 2].astype(np.float64)
            if cls_name == "seasonal_flip":
                train = injector(train_raw, rng, season_m=season_m)
                test = injector(test_raw, rng, season_m=season_m)
            else:
                train = injector(train_raw, rng)
                test = injector(test_raw, rng)
            cells.append({
                "train": train, "test": test,
                "fault_label": cls_name,
                "seed_idx": k, "start_idx": start,
            })
    return cells


def build_synthetic_dataset(base_series: np.ndarray, window_len: int = 128,
                             n_per_class: int = 25, seed: int = 1,
                             season_m: int = 24
                             ) -> tuple[np.ndarray, np.ndarray]:
    """从 base series 切窗 + 注入 4-class fault，返回 (X [4*n, L], y [4*n])."""
    rng = np.random.default_rng(seed)
    n_total = len(base_series) - window_len
    if n_total < n_per_class * 4 * 2:
        raise ValueError(f"base series too short: need >= {n_per_class*4*2} + {window_len}")
    X, y = [], []
    for c, injector in enumerate(INJECTORS):
        for _ in range(n_per_class):
            start = int(rng.integers(0, n_total))
            window = base_series[start:start + window_len].astype(np.float64)
            if c == 2:
                w = injector(window, rng, season_m=season_m)
            else:
                w = injector(window, rng)
            X.append(w)
            y.append(c)
    return np.array(X, dtype=np.float32), np.array(y, dtype=int)


if __name__ == "__main__":
    from research.utils.data_loader import load_series
    s, meta = load_series("ETTh1")
    X, y = build_synthetic_dataset(s, window_len=96, n_per_class=10, seed=1,
                                    season_m=meta.season_m)
    print(f"synthetic dataset: X={X.shape}, y={y.shape}")
    for c in range(4):
        sub = X[y == c]
        print(f"  class {c} ({FAULT_LABELS[c]}): n={len(sub)}, "
              f"mean_std={sub.std(axis=1).mean():.3f}")
