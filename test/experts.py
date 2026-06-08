"""异构预测专家池（method8 落地到 test/）—— 结构互补、能力差异>>噪声。

每个专家：统一签名 `(train, val, H, season_m) -> forecast[H]`（与 forecaster_reflect.STRATEGY_FN 同约定）。
每个专家带 **model signature**：在 7 个结构维度上的擅长向量（见 STRUCT_DIMS），
用于 capability-affinity 路由（task signature · model signature = "谁擅长此场景"，feedbackm7 路线2）。

2026-06-08 扩库：6 → 12 专家（新增 damped_trend / holt_winters / mean_revert / changepoint /
harmonic / momentum），结构维 5 → 7（+meanrev, +shift）。覆盖更多 regime。
"""
from __future__ import annotations

import numpy as np

STRUCT_DIMS = ["trend", "seasonal", "ac1", "noise", "spike", "meanrev", "shift"]


def _ctx(train, val):
    return np.concatenate([train, val]) if len(val) else np.asarray(train, float)


# ---------- 专家实现 ---------- #

def trend_expert(train, val, H, season_m=1):
    c = _ctx(train, val); t = np.arange(len(c)); a, b = np.polyfit(t, c, 1)
    return a * (len(c) + np.arange(H)) + b


def damped_trend_expert(train, val, H, season_m=1, phi=0.85):
    """阻尼趋势（Holt damped）：趋势随步衰减，避免过度外推。擅长饱和增长。"""
    c = _ctx(train, val); t = np.arange(len(c)); a, b = np.polyfit(t, c, 1)
    last = a * (len(c) - 1) + b
    steps = np.cumsum(phi ** np.arange(1, H + 1))
    return last + a * steps


def momentum_expert(train, val, H, season_m=1, w=6):
    """局部动量：用最近 w 步斜率外推（短期趋势/速度）。"""
    c = _ctx(train, val); w = min(w, len(c) - 1)
    if w < 1:
        return np.full(H, c[-1])
    slope = (c[-1] - c[-1 - w]) / w
    return c[-1] + slope * (1 + np.arange(H))


def seasonal_expert(train, val, H, season_m=1):
    c = _ctx(train, val); m = season_m if season_m and season_m > 1 else 12
    m = min(m, len(c))
    return np.array([c[-m + (i % m)] for i in range(H)])


def harmonic_expert(train, val, H, season_m=1, K=3):
    """多谐波 Fourier 回归（比季节朴素更平滑、处理多周期）。"""
    c = _ctx(train, val); n = len(c); t = np.arange(n)
    m = season_m if season_m and season_m > 1 else 12
    feats = [np.ones(n)]
    for k in range(1, K + 1):
        feats += [np.cos(2 * np.pi * k * t / m), np.sin(2 * np.pi * k * t / m)]
    Xd = np.stack(feats, 1)
    from numpy.linalg import lstsq
    w, *_ = lstsq(Xd, c, rcond=None)
    tf = n + np.arange(H)
    ff = [np.ones(H)]
    for k in range(1, K + 1):
        ff += [np.cos(2 * np.pi * k * tf / m), np.sin(2 * np.pi * k * tf / m)]
    return np.stack(ff, 1) @ w


def holt_winters_expert(train, val, H, season_m=1):
    """趋势+季节相加（线性趋势 + 季节残差均值）。擅长 trend×seasonal 混合。"""
    c = _ctx(train, val); t = np.arange(len(c)); a, b = np.polyfit(t, c, 1)
    detr = c - (a * t + b)
    m = season_m if season_m and season_m > 1 else 12; m = min(m, len(c))
    seas = np.array([np.mean(detr[i::m]) for i in range(m)])
    fut = a * (len(c) + np.arange(H)) + b
    return fut + np.array([seas[(len(c) + i) % m] for i in range(H)])


def ar_expert(train, val, H, season_m=1, p=3):
    c = _ctx(train, val)
    if len(c) <= p + 2:
        return np.full(H, c[-1])
    from numpy.linalg import lstsq
    Xd = np.stack([c[i:len(c) - p + i] for i in range(p)], 1); y = c[p:]
    w, *_ = lstsq(Xd, y, rcond=None)
    buf = list(c[-p:]); out = []
    for _ in range(H):
        nx = float(np.dot(w, buf[-p:])); out.append(nx); buf.append(nx)
    return np.array(out)


def mean_revert_expert(train, val, H, season_m=1, halflife=6):
    """均值回复（OU 风格）：向长期均值按半衰期收敛。擅长平稳振荡。"""
    c = _ctx(train, val); mu = float(np.mean(c)); last = float(c[-1])
    rho = 0.5 ** (np.arange(1, H + 1) / max(1, halflife))
    return mu + (last - mu) * rho


def robust_expert(train, val, H, season_m=1):
    c = _ctx(train, val); return np.full(H, np.median(c[-min(12, len(c)):]))


def changepoint_expert(train, val, H, season_m=1):
    """突变点：检测最近一次 level shift，只用最后稳定段的均值外推。"""
    c = _ctx(train, val); n = len(c)
    if n < 8:
        return np.full(H, c[-1])
    # 扫描最佳二分点，取后段均值
    best, bi = -1, n // 2
    for i in range(n // 4, 3 * n // 4):
        d = abs(np.mean(c[i:]) - np.mean(c[:i]))
        if d > best:
            best, bi = d, i
    return np.full(H, np.mean(c[bi:]))


def spike_expert(train, val, H, season_m=1):
    c = _ctx(train, val); base = np.median(c)
    rate = float(np.mean(c > base + 3 * (np.std(c) + 1e-9)))
    return np.full(H, base + rate * (c.max() - base))


def base_naive(train, val, H, season_m=1):   # base 锚点：naive-drift
    c = _ctx(train, val)
    drift = (c[-1] - c[0]) / (len(c) - 1) if len(c) >= 2 else 0.0
    return c[-1] + drift * (1 + np.arange(H))


EXPERTS = {
    "trend": trend_expert, "damped_trend": damped_trend_expert, "momentum": momentum_expert,
    "seasonal": seasonal_expert, "harmonic": harmonic_expert, "holt_winters": holt_winters_expert,
    "ar": ar_expert, "mean_revert": mean_revert_expert, "robust": robust_expert,
    "changepoint": changepoint_expert, "spike": spike_expert, "base": base_naive,
}
BASE = "base"

# model signature：每专家在 7 结构维 [trend,seasonal,ac1,noise,spike,meanrev,shift] 上的擅长度 ∈[0,1]
MODEL_SIGNATURE = {
    "trend":        np.array([1.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.2]),
    "damped_trend": np.array([0.8, 0.0, 0.3, 0.1, 0.0, 0.3, 0.2]),
    "momentum":     np.array([0.7, 0.1, 0.4, 0.1, 0.0, 0.0, 0.3]),
    "seasonal":     np.array([0.0, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0]),
    "harmonic":     np.array([0.1, 0.9, 0.3, 0.0, 0.0, 0.0, 0.0]),
    "holt_winters": np.array([0.7, 0.8, 0.3, 0.0, 0.0, 0.0, 0.1]),
    "ar":           np.array([0.2, 0.2, 1.0, 0.2, 0.0, 0.4, 0.1]),
    "mean_revert":  np.array([0.0, 0.0, 0.2, 0.2, 0.0, 1.0, 0.0]),
    "robust":       np.array([0.0, 0.0, 0.0, 1.0, 0.3, 0.3, 0.2]),
    "changepoint":  np.array([0.2, 0.0, 0.1, 0.2, 0.2, 0.2, 1.0]),
    "spike":        np.array([0.0, 0.0, 0.0, 0.2, 1.0, 0.0, 0.2]),
    "base":         np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]),   # 通用兜底
}


def task_signature(ctx):
    """从序列抽 7 维结构特征（部署可得）。与 MODEL_SIGNATURE 同序。"""
    ctx = np.asarray(ctx, float); n = len(ctx); t = np.arange(n)
    fit = np.polyval(np.polyfit(t, ctx, 1), t)
    ss = np.sum((ctx - ctx.mean()) ** 2) + 1e-9
    trend = float(max(0.0, 1.0 - np.sum((ctx - fit) ** 2) / ss))

    def ac(k):
        if k >= n:
            return 0.0
        a, b = ctx[:-k] - ctx.mean(), ctx[k:] - ctx.mean()
        d = np.sqrt((a ** 2).sum() * (b ** 2).sum())
        return float((a * b).sum() / d) if d > 0 else 0.0
    seas = abs(ac(12)); ac1 = abs(ac(1))
    noise = float(np.clip(np.std(np.diff(ctx)) / (np.std(ctx) + 1e-9), 0, 1))
    mad = np.median(np.abs(ctx - np.median(ctx))) + 1e-9
    spike = float(np.mean(np.abs(ctx - np.median(ctx)) > 5 * mad))
    # 均值回复：去趋势后 lag-1 自相关为负的强度
    detr = ctx - fit
    meanrev = float(max(0.0, -(np.corrcoef(detr[:-1], detr[1:])[0, 1] if n > 2 else 0.0)))
    # level shift：前后半段均值差 / 标准差
    shift = float(np.clip(abs(np.mean(ctx[n // 2:]) - np.mean(ctx[:n // 2])) / (np.std(ctx) + 1e-9), 0, 1))
    return np.array([trend, seas, ac1, noise, spike, meanrev, shift])


def regime_tag(ctx):
    """部署可得的粗 regime 标签 = task signature argmax（用于在线 bandit 的上下文分组）。"""
    return STRUCT_DIMS[int(np.argmax(task_signature(ctx)))]


def affinity_prior(ctx, scale=0.5):
    """手工 capability-affinity 先验：{expert: τ·s 归一}。小池可用；大池易失配（用 learned 版）。"""
    tau = task_signature(ctx)
    raw = {e: float(np.dot(tau, MODEL_SIGNATURE[e])) for e in EXPERTS}
    mx = max(raw.values()) + 1e-9
    return {e: scale * (raw[e] - raw[BASE]) / mx for e in EXPERTS}


def build_learned_capability(calib_stream, H=12, season_m=12):
    """**学出**的 capability profile（feedbackm7 路线2 正解，可扩展到任意池大小）：
    在一段校准历史上跑全专家，记录每 (regime_tag, expert) 的平均 reward(相对 base)。
    返回 {regime_tag: {expert: mean_reward}}。部署时按检测到的 regime 查表当先验。
    校准数据与评测分离 → 无泄漏。"""
    import collections
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for (train, test) in calib_stream:
        train = np.asarray(train, float); test = np.asarray(test, float)
        tag = regime_tag(train)
        def mae(e):
            try:
                p = np.asarray(EXPERTS[e](train, np.array([]), H, season_m), float)[:H]
                return float(np.mean(np.abs(p - test[:H])))
            except Exception:
                return float(np.mean(np.abs(test[:H])))
        bm = mae(BASE)
        for e in EXPERTS:
            acc[tag][e].append((bm - mae(e)) / (bm + 1e-9))
    return {tag: {e: float(np.mean(v)) for e, v in d.items()} for tag, d in acc.items()}
