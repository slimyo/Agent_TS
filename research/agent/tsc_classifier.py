"""TaskB · LLM-based 分类器 (B5 LLM-direct + B6 Agent)。

B5 LLM-direct: 给 LLM K 个 (series, label) + query series → 输出 class
B6 Agent:      给 LLM K 个 (diagnosis_features, label) + query.diagnosis → 输出 class
"""
from __future__ import annotations

import json
import re

import numpy as np

from research.agent.curator_uq import diagnose
from research.utils.llm import chat_cached


# ---------- prompts ---------- #

B5_PROMPT = """\
你是少样本时序分类专家。给你 K={K} 个有标签的训练样本和 1 个 query 序列，请输出 query 的 class。

候选 class: {classes}

【训练样本】
{train_examples}

【Query 序列】
{query_series}

输出 JSON（仅 JSON）：
{{
  "class": <one of {classes}>,
  "confidence": <0-1 float>,
  "reason": "<1 句话>"
}}
"""

B6_AGENT_PROMPT = """\
你是少样本时序分类专家。每个样本已被 Curator 转成 10 维诊断特征。基于诊断相似度做类比推理。

候选 class: {classes}

【训练样本（10-dim 诊断 + label）】
{train_diag}

【Query 诊断】
{query_diag}

任务：根据 query 诊断与训练样本诊断的相似度（趋势/季节/平稳性等），选最匹配的 class。

输出 JSON（仅 JSON）：
{{
  "class": <one of {classes}>,
  "confidence": <0-1 float>,
  "supporting_neighbors": [<sample idx of top-2 closest>, ...],
  "reason": "<1-2 句话，引用具体诊断特征>"
}}
"""


# ---------- helpers ---------- #

def _summarize_series(x: np.ndarray, n_show: int = 6) -> str:
    """把 series 压缩成可读字符串：mean/std + first/last n_show points。"""
    return (f"mean={x.mean():.3f}, std={x.std():.3f}, "
            f"first={[f'{v:.2f}' for v in x[:n_show]]}, "
            f"last={[f'{v:.2f}' for v in x[-n_show:]]}")


def _diag_to_dict(d) -> dict:
    """把 Diagnosis 转 readable dict for LLM prompt."""
    return {
        "n": d.n, "mean": round(d.mean, 3), "std": round(d.std, 3),
        "trend_slope": round(d.trend_slope, 4), "trend_tstat": round(d.trend_tstat, 2),
        "adf_pvalue": round(d.adf_pvalue, 3),
        "acf_peak_lag": d.acf_peak_lag, "acf_peak_value": round(d.acf_peak_value, 3),
        "trend_conf": d.trend_conf_xc, "season_conf": d.season_conf_xc,
        "stat_conf": d.stat_conf_xc,
    }


def _parse_class(text: str, classes: list) -> tuple:
    """从 LLM 输出提取 class + confidence。返回 (cls, conf)。"""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        d = json.loads(text)
        cls = d.get("class", classes[0])
        conf = float(d.get("confidence", 0.5))
        # 兼容 LLM 给字符串："1" / "class_1"
        try:
            cls_int = int(cls)
            if cls_int in classes:
                return cls_int, conf
        except Exception:
            pass
        # 字符串匹配
        for c in classes:
            if str(c) == str(cls):
                return c, conf
        return classes[0], 0.0
    except Exception:
        # 正则兜底
        m = re.search(r'"?class"?\s*[:=]\s*"?(-?\d+)"?', text)
        if m:
            v = int(m.group(1))
            if v in classes:
                return v, 0.5
        return classes[0], 0.0


# ---------- B5 LLM-direct ---------- #

def b5_llm_direct(X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray, llm_model: str | None = None) -> np.ndarray:
    classes = sorted(set(y_train.tolist()))
    K = len(X_train)
    train_lines = "\n".join(
        f"  [{i}] label={y_train[i]}: {_summarize_series(X_train[i])}"
        for i in range(K)
    )
    preds = []
    for q in X_test:
        prompt = B5_PROMPT.format(
            K=K, classes=classes,
            train_examples=train_lines,
            query_series=_summarize_series(q),
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            resp = chat_cached(messages, model=llm_model) if llm_model else chat_cached(messages)
            cls, _ = _parse_class(resp, classes)
        except Exception:
            cls = classes[0]
        preds.append(cls)
    return np.array(preds)


# ---------- B6 Agent (diagnosis + ICL) ---------- #

def b6_agent(X_train: np.ndarray, y_train: np.ndarray,
             X_test: np.ndarray, season_m: int = 1,
             llm_model: str | None = None) -> np.ndarray:
    classes = sorted(set(y_train.tolist()))
    # 1) 给每个 training series 算 Curator 诊断
    train_diags = [diagnose(x, season_m=season_m) for x in X_train]
    train_lines = "\n".join(
        f"  [{i}] label={y_train[i]} diagnosis={_diag_to_dict(d)}"
        for i, d in enumerate(train_diags)
    )

    preds = []
    for q in X_test:
        q_diag = diagnose(q, season_m=season_m)
        prompt = B6_AGENT_PROMPT.format(
            classes=classes,
            train_diag=train_lines,
            query_diag=_diag_to_dict(q_diag),
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            resp = chat_cached(messages, model=llm_model) if llm_model else chat_cached(messages)
            cls, _ = _parse_class(resp, classes)
        except Exception:
            cls = classes[0]
        preds.append(cls)
    return np.array(preds)


if __name__ == "__main__":
    from research.utils.ucr_loader import load_ucr_fewshot
    X_tr, y_tr, X_te, y_te = load_ucr_fewshot("Coffee", n_per_class=5, seed=1)
    # Sub-sample test set for quick smoke (avoid 28 LLM calls × 2)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X_te), size=8, replace=False)
    X_te_s, y_te_s = X_te[idx], y_te[idx]
    print(f"Coffee 5-shot, 8 test samples: classes={sorted(set(y_tr.tolist()))}")
    yp5 = b5_llm_direct(X_tr, y_tr, X_te_s)
    yp6 = b6_agent(X_tr, y_tr, X_te_s, season_m=24)
    print(f"B5 LLM-direct: acc={float((yp5==y_te_s).mean()):.3f}  preds={yp5.tolist()}  gt={y_te_s.tolist()}")
    print(f"B6 Agent:      acc={float((yp6==y_te_s).mean()):.3f}  preds={yp6.tolist()}")
