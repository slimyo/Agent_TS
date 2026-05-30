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
    "moirai2": {
        "class": "Moirai 2.0 R-small (Salesforce, 2025) — distilled 11M-param Moirai successor",
        "assumes": ["univariate or multivariate input", "transformers<4.46 (tsci-py312 env)"],
        "strengths": ["11M params CPU-feasible (~0.7s inference)",
                      "drop-in upgrade over Moirai 1; better long-horizon"],
        "weaknesses": ["env split (tsci-py312) due to uni2ts 2.0 pinning",
                       "small capacity vs newer 8B-class TSFMs"],
        "typical_failure": "novel domain way outside LOTSA coverage; ultra-long context (>4096)",
    },
    "tirex": {
        "class": "TiRex (NX-AI, 2025) — xLSTM-based zero-shot TSFM, 128M params",
        "assumes": ["univariate input", "lookback ≤ model max"],
        "strengths": ["recurrent xLSTM avoids quadratic context cost",
                      "drop-in chronos2 API (9 quantiles vs C2's 21)",
                      "strong on financial / observability niches (Exchange -35%, ECL -16% vs C2)"],
        "weaknesses": ["catastrophic on Weather (+486%) — meteorology domain mismatch",
                       "smaller quantile resolution than C2"],
        "typical_failure": "high-variance industrial / climate signals where C2's saturation regime dominates",
    },
    "toto": {
        "class": "Toto Open Base 1.0 (DataDog, 2025) — observability-tuned 151M TSFM",
        "assumes": ["univariate observability-like series (counts, rates, percentiles)"],
        "strengths": ["dominates on electricity / observability proxies (ECL -43.9% vs C2) ⭐",
                      "tuned on DataDog production metrics — clean niche specialist"],
        "weaknesses": ["catastrophic outside observability (Weather +671%, ILI +52%, ETTh1 +110%)",
                       "torch 2.7 upgrade required for main env"],
        "typical_failure": "any non-observability domain — the model was never meant to be generic",
    },
    "toto2": {
        "class": "Toto Open Base 2.0 4m (DataDog, 2026) — successor to Toto 1.0, gluonts predictor API",
        "assumes": ["univariate input", "transformers<4.46 + tsci-py312 env"],
        "strengths": ["4M params, ~1.2s inference",
                      "expected to inherit observability strength of v1.0"],
        "weaknesses": ["new API surface (gluonts predictor)",
                       "domain partition same as v1 (likely)"],
        "typical_failure": "non-observability domains; behaviorally similar to v1.0 — niche specialist",
    },
    "time_moe": {
        "class": "Time-MoE 50M (Maple728, ICLR 2025) — Mixture-of-Experts decoder TSFM",
        "assumes": ["univariate input", "transformers~=4.40.1 (uses deprecated past_key_values.seen_tokens)"],
        "strengths": ["sparse activation → small effective compute",
                      "5s on RTX 5070 Ti (remote tsci-remote-tx440)",
                      "200M variant also available at same repo prefix"],
        "weaknesses": ["env pinning to transformers 4.40.1 — incompatible with Timer-S1 env",
                       "modeling_*.py reliance on removed API"],
        "typical_failure": "any env upgrade above transformers 4.45 → AttributeError on DynamicCache",
    },
    "sundial": {
        "class": "Sundial 128m (Tsinghua THUML, 2025) — autoregressive flow-matching TSFM",
        "assumes": ["univariate input", "transformers~=4.40.1 (same legacy API as Time-MoE)",
                    "L=1024 lookback recommended"],
        "strengths": ["~3s on RTX 5070 Ti (remote tsci-remote-tx440)",
                      "20-sample probabilistic output via flow matching",
                      "lightweight (128M params)"],
        "weaknesses": ["env pinning (transformers 4.40.1)",
                       "no head-to-head with C2 yet (待 remote sweep)"],
        "typical_failure": "unknown — pending sweep; expect saturation regime overlap with C2",
    },
    "timer": {
        "class": "Timer-S1 (ByteDance, 2025) — 8.3B-param MoE decoder TSFM, 0.75B activated/token",
        "assumes": ["univariate input", "transformers~=4.57.1 (tsci-remote env on remote GPU)",
                    "GPU ≥16GB VRAM (Blackwell sm_120 verified on RTX 5070 Ti)"],
        "strengths": ["state-of-the-art on GIFT-Eval — strongest medium/long-horizon TSFM cited",
                      "native 9-quantile zero-shot forecasts ([0.1..0.9])",
                      "11520 max context length"],
        "weaknesses": ["8.3B params → infeasible on local 6GB GPU",
                       "remote-only deployment (cross-env routing dispatcher needed)",
                       "use_cache=False required when sharing 16GB card"],
        "typical_failure": "OOM if VRAM <16GB or KV cache enabled with H>256",
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
