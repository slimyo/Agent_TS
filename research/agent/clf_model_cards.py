"""P6.4b-3 / task #32 · 分类策略 Model Cards。

参照 `agent/model_cards.py` (forecasting) 的设计，为每个分类器写：
  - class: 简短描述
  - assumes: 数据假设
  - strengths: 强项
  - weaknesses: 弱点
  - typical_failure: 典型失败模式

这些 Cards 用于 classification_planner LLM 提示中，让 LLM 引用先验做选择。

来源依据：
  - 公开文献（Bagnall 2017 / Dempster 2020 Rocket / Goswami 2024 MOMENT）
  - 我们 task #26 UCR 210-cell 实测数据（finish §3.1.32 winner-per-cell）
"""
from __future__ import annotations


CLF_MODEL_CARDS: dict[str, dict] = {
    "dtw_1nn": {
        "class": "1-NN classifier with Dynamic Time Warping distance",
        "assumes": [
            "global shape alignment is class-discriminative",
            "time warping (phase shift, speed variation) is meaningful",
            "series of similar length across train/test",
        ],
        "strengths": [
            "classical TSC benchmark (Bagnall 2017 baseline)",
            "handles phase shift / temporal misalignment",
            "long-cycle / periodic signals",
        ],
        "weaknesses": [
            "O(L²) computation (slow for L > 1000)",
            "noisy series (DTW can warp into noise)",
            "very short series N=3-5 (insufficient neighbor diversity)",
        ],
        "typical_failure":
            "high-frequency oscillation tasks where local pattern not phase-sensitive; "
            "image-outline silhouettes where global shape > local alignment",
        "ucr_evidence":
            "wins Coffee 10-shot (1.000 tie with Rocket); "
            "loses BeetleFly/BirdChicken (silhouette → MOMENT pretraining wins)",
        # v2 决策字段（task #40）
        "min_samples_per_class": 1,
        "multiclass_support": True,
        "max_sequence_length": 1000,        # O(L²) 实际可行上限
        "cost_level": "high",                # O(L² * N) per query
        "preprocessing_requirements": ["none"],
    },
    "euclid_1nn": {
        "class": "1-NN classifier with point-wise Euclidean distance",
        "assumes": [
            "pointwise alignment correct (no warping needed)",
            "amplitude differences class-discriminative",
            "low noise",
        ],
        "strengths": [
            "extremely fast O(L · N_train)",
            "sometimes wins when alignment already correct (e.g., ECG200 N=3-shot)",
            "no hyperparameters",
        ],
        "weaknesses": [
            "any phase shift breaks it",
            "amplitude scaling differences across classes",
            "high-frequency noise",
        ],
        "typical_failure":
            "phase-shifted series, varying-length series, noisy industrial data",
        "ucr_evidence":
            "wins ECG200 3-shot (0.815) where signals already aligned; "
            "loses TwoLeadECG (0.59 vs Rocket 0.97)",
        "min_samples_per_class": 1,
        "multiclass_support": True,
        "max_sequence_length": 10000,
        "cost_level": "low",
        "preprocessing_requirements": ["none"],
    },
    "rocket": {
        "class": "Rocket (random convolutional kernel transform, Dempster 2020 NeurIPS)",
        "assumes": [
            "fine-grained temporal patterns discriminative",
            "random kernels (10,000 by default) sample useful feature space",
            "linear classifier on transformed features",
        ],
        "strengths": [
            "UCR archive SOTA (winner 7 of 15 our settings, mean 87.5%)",
            "fast inference after fit (~3s per cell)",
            "robust to N=3-shot",
            "handles diverse pattern types via kernel diversity",
        ],
        "weaknesses": [
            "shape-based / image-outline tasks (random kernels miss global morphology)",
            "extreme small N where linear classifier underfits",
            "very long sequences (memory)",
        ],
        "typical_failure":
            "high-level shape tasks (BeetleFly: 82.5% vs MOMENT 92.5%; "
            "BirdChicken: 67.5% vs MOMENT 80%)",
        "ucr_evidence":
            "winner 7/15 cells; peak TwoLeadECG 10-shot 99.8%; "
            "weakest on image-outline data",
        "min_samples_per_class": 2,
        "multiclass_support": True,
        "max_sequence_length": 20000,
        "cost_level": "medium",              # 10k kernels transform
        "preprocessing_requirements": ["z-normalization per series"],
    },
    "moment_1nn": {
        "class": "1-NN classifier on MOMENT-1-small TSFM embeddings (Goswami 2024)",
        "assumes": [
            "TSFM pretraining captures high-level morphological features",
            "cosine similarity in 512-dim embedding space is class-meaningful",
            "test series within distribution of pretraining corpus",
        ],
        "strengths": [
            "shape-based / morphological tasks (BeetleFly, BirdChicken)",
            "ECG morphology (peak at ECG200 10-shot 86.5%)",
            "transfer from pretrained representations",
        ],
        "weaknesses": [
            "fine-grained local patterns (TwoLeadECG: 81.5% vs Rocket 99.8%)",
            "very short series < 100 length (pad-and-mask degrades)",
            "OOD domains (spectroscopy, financial scales)",
        ],
        "typical_failure":
            "TwoLeadECG, Coffee 3-shot where fine local discrimination matters; "
            "OOD domains where pretraining distribution mismatches",
        "ucr_evidence":
            "winner 3/15 (BeetleFly N=10, BirdChicken N=5, ECG200 N=10); "
            "mean 81.9% (rank 2 overall)",
        "min_samples_per_class": 1,
        "multiclass_support": True,
        "max_sequence_length": 4096,        # MOMENT-1-small context max
        "cost_level": "medium",              # 38M params CPU
        "preprocessing_requirements": ["pad to 512", "input_mask"],
    },
    "moment_logreg": {
        "class": "Logistic regression on MOMENT-1-small TSFM embeddings",
        "assumes": [
            "class boundaries linearly separable in TSFM embedding space",
            "sufficient samples to fit linear model (≥3 per class)",
            "TSFM features generalize",
        ],
        "strengths": [
            "multi-class with linear structure",
            "robust to noise compared to 1-NN",
            "fast inference",
        ],
        "weaknesses": [
            "extreme few-shot N < 3 (linear fit unstable)",
            "non-linear class structure",
            "high-dim with small N → overfit despite regularization",
        ],
        "typical_failure":
            "N=3 settings where logistic fit is shaky",
        "ucr_evidence":
            "winner 3/15 (BeetleFly N=3/5, BirdChicken N=3); "
            "mean 81.7% similar to 1-NN variant",
        "min_samples_per_class": 3,         # logistic fit 需要 ≥3
        "multiclass_support": True,
        "max_sequence_length": 4096,
        "cost_level": "medium",
        "preprocessing_requirements": ["pad to 512", "input_mask"],
    },
    "llm_direct": {
        "class": "LLM in-context learning on raw series summary",
        "assumes": [
            "LLM can extract discriminative features from numerical summaries",
            "training examples in context window",
        ],
        "strengths": [
            "zero training; pure prompt",
            "natural language explanation as side product",
        ],
        "weaknesses": [
            "raw numbers without diagnosis are not discriminative for LLM",
            "TSC accuracy lags classical methods by 30+ pp (mean 52.7%)",
            "non-deterministic across seeds",
        ],
        "typical_failure":
            "general TSC tasks where class labels are domain-specific patterns "
            "rather than statistical concepts; "
            "Curator-less LLM 'collapses to single class' on RCA (B1 baseline 0% R1)",
        "ucr_evidence":
            "winner 0/15 cells; mean 52.7%; routinely tied or worse than B6 Agent",
        "min_samples_per_class": 1,
        "multiclass_support": True,
        "max_sequence_length": 500,         # 受 LLM context 限
        "cost_level": "high",                # 每 query 1 LLM call
        "preprocessing_requirements": ["compact text summary"],
    },
    # === Round 2 (task #69) library expansion: sktime-backed alternatives ===
    "minirocket": {
        "class": "MiniRocket (Dempster 2021) — deterministic kernels variant of Rocket",
        "assumes": [
            "fixed set of 10,000 minimal binary kernels",
            "linear classifier on PPV-transformed features",
            "z-normalization per series",
        ],
        "strengths": [
            "complementary to Rocket on image-outline / shape data",
            "BeetleFly N=5 +20pp over Rocket (smoke test)",
            "deterministic — no random seed sensitivity",
        ],
        "weaknesses": [
            "slower fit than Rocket (~6s vs 1s at N=5)",
            "occasional Wafer-style industrial regression vs Rocket",
        ],
        "typical_failure": "smooth-persistent industrial signals where Euclid 1-NN dominates",
        "ucr_evidence": "BeetleFly N=5: 95% vs Rocket 75% (+20pp); Wafer N=5: 73.5% vs Rocket 81.5% (-8pp)",
        "min_samples_per_class": 2,
        "multiclass_support": True,
        "max_sequence_length": 10000,
        "cost_level": "medium",
        "preprocessing_requirements": ["z-normalization per series"],
    },
    "weasel": {
        "class": "WEASEL (Schäfer 2017) — dictionary-based symbolic representation classifier",
        "assumes": [
            "recurring discrete motifs encode class identity",
            "SFA word histograms discriminative",
            "logistic regression on bag-of-words",
        ],
        "strengths": [
            "Wafer industrial fault N=5: 87% vs Rocket 81.5% (+5.5pp)",
            "fast at small N (<1s)",
            "different inductive bias from kernel / distance methods",
        ],
        "weaknesses": [
            "memory-heavy at long sequences",
            "can over-fit on small N (alphabet-size sensitivity)",
        ],
        "typical_failure": "very-smooth non-symbolic signals lacking discrete motifs",
        "ucr_evidence": "Coffee N=5: 100% (tied); Wafer N=5: 87% (+5.5pp vs Rocket)",
        "min_samples_per_class": 3,
        "multiclass_support": True,
        "max_sequence_length": 5000,
        "cost_level": "low",
        "preprocessing_requirements": ["none"],
    },
    "catch22": {
        "class": "Catch22 (Lubba 2019) — 22 canonical features + RandomForest",
        "assumes": [
            "hand-engineered feature set (selected by mutual information) captures task",
            "Random Forest exploits feature interactions",
        ],
        "strengths": [
            "robust across diverse domains",
            "interpretable feature contributions",
            "moderate cost (~2s at N=5)",
        ],
        "weaknesses": [
            "ceiling lower than kernel methods on subtle patterns",
            "feature engineering may miss task-specific signals",
        ],
        "typical_failure": "tasks requiring local pattern detection at fine scale",
        "ucr_evidence": "BeetleFly N=5: 85% (+10pp Rocket); Wafer N=5: 83.5%; Coffee N=5: 100%",
        "min_samples_per_class": 2,
        "multiclass_support": True,
        "max_sequence_length": 10000,
        "cost_level": "low",
        "preprocessing_requirements": ["none"],
    },
}


def render_clf_cards_block(names: list[str] | None = None) -> str:
    """生成可贴 LLM prompt 的卡片块。"""
    keys = names or list(CLF_MODEL_CARDS.keys())
    out = []
    for k in keys:
        c = CLF_MODEL_CARDS.get(k)
        if not c:
            continue
        out.append(f"## {k}")
        out.append(f"- class: {c['class']}")
        out.append(f"- assumes: {'; '.join(c['assumes'])}")
        out.append(f"- strengths: {'; '.join(c['strengths'])}")
        out.append(f"- weaknesses: {'; '.join(c['weaknesses'])}")
        out.append(f"- typical_failure: {c['typical_failure']}")
        if "ucr_evidence" in c:
            out.append(f"- empirical (our UCR 15-cell study): {c['ucr_evidence']}")
        out.append("")
    return "\n".join(out)


def card_summary_one_line(name: str) -> str:
    """一行摘要：用于 quick context in routing prompt。"""
    c = CLF_MODEL_CARDS.get(name)
    if not c:
        return f"{name}: (no card)"
    return (f"{name}: {c['class'].split('(')[0].strip()}; "
            f"strong on: {c['strengths'][0]}; "
            f"weak on: {c['weaknesses'][0]}")


if __name__ == "__main__":
    print("=== render_clf_cards_block (all 6) ===\n")
    print(render_clf_cards_block())
    print("\n=== one-line summaries ===")
    for name in CLF_MODEL_CARDS:
        print(f"  {card_summary_one_line(name)}")
