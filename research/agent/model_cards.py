"""Phase 5.0 step1 · 策略库的 Model Cards（对应 plan §十五.15.2.①）。

每个策略一张结构化能力卡，供反思层在 prompt 中注入，让 LLM 不靠
"预训练泛泛印象"而是基于显式先验做根因推理。

字段定义：
  - class:           方法类别（一句话）
  - assumes:         数据假设（list）
  - strengths:       擅长场景（list）
  - weaknesses:      不擅长场景（list）
  - typical_failure: 一句话失败模式（"什么时候 MAE 会大"）

新增策略只需在此处加一项；反思 prompt 自动渲染。
"""
from __future__ import annotations

MODEL_CARDS: dict[str, dict] = {
    "naive_drift": {
        "class": "linear extrapolation (last - first slope)",
        "assumes": ["global linear trend", "no seasonality", "iid noise"],
        "strengths": ["short-horizon trend extrapolation", "zero parameters", "robust to N=5-10"],
        "weaknesses": ["nonlinear trends (curvature)", "seasonal patterns", "structural breaks at end"],
        "typical_failure": "high MAE when series has curvature or periodic component; "
                           "end-noise can corrupt slope estimate",
    },
    "naive_seasonal": {
        "class": "seasonal naive (tile last full season)",
        "assumes": ["fixed known period m", "no trend or trend already removed"],
        "strengths": ["pure seasonal series", "very short training (1-2 full periods)"],
        "weaknesses": ["strong trend", "irregular / drifting period", "multiple seasonal cycles"],
        "typical_failure": "trend contamination causes phase-shift errors; "
                           "if training < 2 periods the tile is biased",
    },
    "arima_ets": {
        "class": "auto ARIMA + ETS (AIC selects model family)",
        "assumes": ["stationary after differencing", "parametric (p,d,q)+(P,D,Q) structure"],
        "strengths": ["short-to-medium series N=30-500", "trend + seasonal decomposition",
                      "clear interpretable parameters"],
        "weaknesses": ["very short series N<20 (over-differencing risk)", "abrupt level shifts",
                       "long seasonality with insufficient cycles"],
        "typical_failure": "high parameter-estimation variance on tiny N, causing erratic drift; "
                           "fails on regime change",
    },
    "chronos": {
        "class": "pretrained zero-shot transformer (Amazon Chronos-T5-small, 60M)",
        "assumes": ["context length ≤ model max (512)", "patterns similar to pretraining corpus"],
        "strengths": ["complex nonlinear patterns", "multiple seasonalities", "no parameter fit",
                      "very stable on N=10-50 (low variance)"],
        "weaknesses": ["domain shift from pretraining (medical, financial unique scales)",
                       "black-box (no interpretability)", "limited improvement when N large"],
        "typical_failure": "out-of-distribution sequences; constant or weird-scale series",
    },
    "llmtime": {
        "class": "LLM in-context forecasting (digits-as-tokens)",
        "assumes": ["LLM tokenizer handles numeric tokens", "patterns expressible in pretraining"],
        "strengths": ["very small N=10-30 (LLM world-knowledge transfer)",
                      "no model fit", "captures broad textual / commonsense priors"],
        "weaknesses": ["large N>50 (context bloat + cost)", "non-deterministic across seeds",
                       "domain-specific scales not in pretraining"],
        "typical_failure": "LLM hallucinates plausible-but-wrong continuations on "
                           "atypical scales (e.g., temperatures in unusual units)",
    },
    # 以下为 feedback (2026-05) 增补的 SOTA TSFM 候选 ─────────────────
    "chronos2": {
        "class": "Chronos-2 (Amazon, 2025-10) — T5-based zero-shot multivariate TSFM with covariate support",
        "assumes": ["context patterns within pretrained distribution",
                    "univariate or multivariate input (we use univariate slice)"],
        "strengths": ["improved zero-shot accuracy over Chronos-Small",
                      "21-quantile probabilistic output",
                      "designed for cold-start including covariates"],
        "weaknesses": ["larger model than Chronos-Small (slower CPU inference)",
                       "still subject to pretraining-distribution shift on niche domains"],
        "typical_failure": "very-out-of-distribution series (e.g., synthetic ramps), "
                           "or scales not seen during pretraining",
    },
    "chronos_bolt": {
        "class": "Chronos-Bolt (Amazon, 2024-12) — patched-tokenization T5, 5-15x faster than Chronos-Small",
        "assumes": ["context patterns within pretrained distribution"],
        "strengths": ["highest throughput in Chronos family",
                      "9-quantile probabilistic output",
                      "strong baseline for short-context CPU inference"],
        "weaknesses": ["smaller quantile resolution vs Chronos-2",
                       "domain-shift sensitivity similar to Chronos-Small"],
        "typical_failure": "atypical seasonal periods not seen in pretraining corpus",
    },
    "timesfm2": {
        "class": "TimesFM 2.0 (Google, ICLR 2025) — decoder-only TSFM with multivariate + covariate support",
        "assumes": ["univariate or multivariate input", "context within model max length"],
        "strengths": ["state-of-the-art on many zero-shot benchmarks (Monash, ETT, etc.)",
                      "decoder architecture suits autoregressive forecasting"],
        "weaknesses": ["JAX-based, heavier installation footprint",
                       "less efficient on very short contexts (N<20)"],
        "typical_failure": "very-short-context tasks where pre-training prior dominates over series-specific signal",
    },
    "moirai": {
        "class": "Moirai (Salesforce, ICML 2024) — universal masked Transformer TSFM",
        "assumes": ["univariate or multivariate input", "patch-based tokenization"],
        "strengths": ["unified architecture handles any number of variates via masking",
                      "strong domain generalization (LOTSA pretraining corpus)"],
        "weaknesses": ["sensitive to patch-size choice for very short series",
                       "uni2ts dependency stack"],
        "typical_failure": "non-stationary series where masking patterns mismatch the natural period",
    },
}


def render_cards_block(names: list[str] | None = None) -> str:
    """把指定策略卡片渲染成 prompt 友好的简洁文本块。"""
    keys = names or list(MODEL_CARDS.keys())
    lines = []
    for k in keys:
        c = MODEL_CARDS.get(k)
        if not c:
            continue
        lines.append(f"- **{k}** ({c['class']})")
        lines.append(f"    assumes: {', '.join(c['assumes'])}")
        lines.append(f"    strengths: {', '.join(c['strengths'])}")
        lines.append(f"    weaknesses: {', '.join(c['weaknesses'])}")
        lines.append(f"    typical_failure: {c['typical_failure']}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(render_cards_block(["naive_drift", "chronos"]))
