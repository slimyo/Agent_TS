"""
Demo Step 3 — Planner Agent
对应 TSci 论文第 3 节：基于 Curator 的输出 C = {Q, V, A}，从模型库 M 中
选出候选子集 Mp，对每个候选做小规模超参搜索，选出 top-k 调好参的模型。

流程：
  load curator_state.json + 重建清洗后序列
  → 划分 train/val/test
  → LLM (glm-4.7-flash) 看 Q+A，从模型库选 2-3 个候选 + 写理由
  → 对每个候选做随机搜索 (max 5 配置)，按 val MAPE 挑最优
  → 写 planner_state.json：{candidates: [{model, best_params, val_mape, reason}]}

模型库（4 个不同家族，故意不引入 GPU/神经模型）：
  - naive_seasonal: 周季节性朴素法（基线兜底）
  - arima:        ARIMA(p,d,q)，线性经典
  - holt_winters: Holt-Winters 三参数指数平滑（含加性季节）
  - ridge_lag:    Ridge 回归 + 滞后特征 + 周内虚拟变量
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.linear_model import Ridge
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


HERE = Path(__file__).parent

# ---------- 0. 复用前两步的工具 ----------

def _load_module(filename: str, alias: str):
    if alias in sys.modules:
        return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(alias, HERE / filename)
    m = importlib.util.module_from_spec(spec)
    sys.modules[alias] = m  # 必须先注册，dataclass/typing 会反查 sys.modules
    spec.loader.exec_module(m)
    return m


step1 = _load_module("01_curator_minimal.py", "step1")
step2 = _load_module("02_curator_visual.py", "step2")


# ---------- 1. 数据准备：复现 Step 2 的清洗输出 + 切分 ----------

H_TEST = 14   # 测试步长
H_VAL = 14    # 验证步长
PERIOD = 7


def prepare_data() -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    raw = step2.make_synthetic_series()
    q = step1.compute_quality_vector(raw)
    # 直接用 Step 2 已经验证过的策略，不再重复调 LLM
    strategy = {"missing_strategy": "linear_interpolation",
                "outlier_strategy": "clip_iqr"}
    cleaned = step1.apply_strategy(raw, strategy)

    # 时间序列严格按时间切分：train | val | test
    test = cleaned.iloc[-H_TEST:]
    val = cleaned.iloc[-(H_TEST + H_VAL):-H_TEST]
    train = cleaned.iloc[:-(H_TEST + H_VAL)]
    return cleaned, train, val, test


# ---------- 2. 模型库 M ----------
# 每个 fitter 函数签名：(train, params) -> 一个能 .forecast(h) 的对象的"封装"

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    eps = 1e-8
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


# --- Naive seasonal ---
def fit_naive_seasonal(train: pd.Series, params: dict):
    period = params.get("period", PERIOD)
    last_cycle = train.iloc[-period:].values

    def forecast(h: int) -> np.ndarray:
        return np.array([last_cycle[i % period] for i in range(h)])

    return forecast


# --- ARIMA ---
def fit_arima(train: pd.Series, params: dict):
    order = params.get("order", (1, 1, 1))
    model = ARIMA(train.values, order=order).fit()

    def forecast(h: int) -> np.ndarray:
        return np.asarray(model.forecast(steps=h))

    return forecast


# --- Holt-Winters ---
def fit_hw(train: pd.Series, params: dict):
    trend = params.get("trend", "add")
    damped = params.get("damped_trend", False)
    seasonal = "add"
    model = ExponentialSmoothing(
        train.values,
        trend=trend,
        damped_trend=damped,
        seasonal=seasonal,
        seasonal_periods=PERIOD,
        initialization_method="estimated",
    ).fit(optimized=True)

    def forecast(h: int) -> np.ndarray:
        return np.asarray(model.forecast(steps=h))

    return forecast


# --- Ridge with lag features ---
def _make_lag_design(s: pd.Series, lags=(1, 2, 3, 7, 14, 21)) -> tuple[np.ndarray, np.ndarray, list]:
    df = pd.DataFrame({"y": s.values}, index=s.index)
    for k in lags:
        df[f"lag_{k}"] = df["y"].shift(k)
    df["dow"] = df.index.dayofweek
    df = df.dropna()
    dow_dummies = pd.get_dummies(df["dow"], prefix="dow", drop_first=True)
    X = pd.concat([df.drop(columns=["y", "dow"]), dow_dummies], axis=1).astype(float)
    return X.values, df["y"].values, list(X.columns)


def fit_ridge(train: pd.Series, params: dict):
    alpha = params.get("alpha", 1.0)
    lags = params.get("lags", (1, 2, 3, 7, 14, 21))
    X, y, cols = _make_lag_design(train, lags)
    model = Ridge(alpha=alpha).fit(X, y)

    history = train.copy()

    def forecast(h: int) -> np.ndarray:
        preds = []
        cur = history.copy()
        for _ in range(h):
            next_idx = cur.index[-1] + pd.Timedelta(days=1)
            cur.loc[next_idx] = np.nan
            X_step, _, _ = _make_lag_design(cur, lags)
            yhat = model.predict(X_step[-1:])[0]
            cur.iloc[-1] = yhat
            preds.append(yhat)
        return np.array(preds)

    return forecast


MODEL_LIBRARY: dict[str, dict] = {
    "naive_seasonal": {
        "fit": fit_naive_seasonal,
        "desc": "周季节朴素法：用上一周相同位置的值作为预测；适合强周期、弱趋势的兜底基线。",
        "param_grid": [{"period": PERIOD}],
    },
    "arima": {
        "fit": fit_arima,
        "desc": "ARIMA(p,d,q)：经典线性时序模型，差分处理趋势，适合平稳/弱季节性序列。",
        "param_grid": [
            {"order": (1, 1, 1)},
            {"order": (2, 1, 1)},
            {"order": (1, 1, 2)},
            {"order": (2, 1, 2)},
        ],
    },
    "holt_winters": {
        "fit": fit_hw,
        "desc": "Holt-Winters 三参数指数平滑（加性季节）：显式建模水平+趋势+季节性，适合明显趋势+季节性的序列。",
        "param_grid": [
            {"trend": "add", "damped_trend": False},
            {"trend": "add", "damped_trend": True},
            {"trend": None, "damped_trend": False},
        ],
    },
    "ridge_lag": {
        "fit": fit_ridge,
        "desc": "Ridge 回归 + 滞后特征 + 周内虚拟变量：把时序当回归问题，能捕捉非线性趋势的同时正则化防过拟合。",
        "param_grid": [
            {"alpha": 0.1},
            {"alpha": 1.0},
            {"alpha": 10.0},
        ],
    },
}


# ---------- 3. LLM 选候选模型（论文里的 r_i） ----------

PLANNER_SYSTEM = """你是时序模型选型专家。给定 Curator 的诊断结果 (Q, A) 和模型库描述，
从模型库中选 **2 到 3 个** 最有可能效果好的模型，并为每个写一句中文理由（必须引用 Q 或 A 的具体字段）。

仅输出严格 JSON:
{
  "selected": ["model_name_1", "model_name_2", ...],
  "reason_per_model": {"model_name_1": "...", "model_name_2": "..."}
}
"selected" 中的名字必须是模型库的 key。不要输出 JSON 之外的任何文字。"""


def llm_pick_candidates(q: dict, a: dict) -> dict:
    library_brief = {name: cfg["desc"] for name, cfg in MODEL_LIBRARY.items()}
    user_msg = (
        "Curator 输出：\n"
        f"Q = {json.dumps(q, ensure_ascii=False)}\n"
        f"A = {json.dumps(a, ensure_ascii=False)}\n\n"
        "模型库（key → 一句话描述）：\n"
        f"{json.dumps(library_brief, ensure_ascii=False, indent=2)}"
    )

    client, model = step1.make_client()
    print(f"[规划模型: {model}]")

    messages = [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": user_msg},
    ]

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
        return json.loads(m.group(0))


# ---------- 4. 超参搜索：对每个候选在 val 上挑最优 ----------

@dataclass
class CandidateResult:
    model: str
    best_params: dict
    val_mape: float
    reason: str
    all_trials: list   # [(params, val_mape)]


def hp_search(name: str, train: pd.Series, val: pd.Series,
              reason: str) -> CandidateResult | None:
    cfg = MODEL_LIBRARY[name]
    fit_fn: Callable = cfg["fit"]
    grid = cfg["param_grid"]

    trials = []
    for params in grid:
        try:
            forecaster = fit_fn(train, params)
            yhat = forecaster(len(val))
            score = mape(val.values, yhat)
            trials.append((params, score))
        except Exception as e:
            trials.append((params, float("inf")))
            print(f"  [{name}] params={params} 拟合失败: {e}")

    if not trials or all(np.isinf(s) for _, s in trials):
        return None

    best_params, best_score = min(trials, key=lambda x: x[1])
    return CandidateResult(
        model=name,
        best_params=best_params,
        val_mape=round(best_score, 4),
        reason=reason,
        all_trials=[(p, round(s, 4)) for p, s in trials],
    )


# ---------- 5. 主流程 ----------

def main() -> None:
    print("=== Step 1: 加载 Curator 状态 ===")
    state_path = HERE / "curator_state.json"
    if not state_path.exists():
        raise FileNotFoundError("先跑 02_curator_visual.py 生成 curator_state.json")
    curator_state = json.loads(state_path.read_text())
    Q, A = curator_state["Q"], curator_state["A"]
    print(f"Q.n={Q['n']}, A={A}\n")

    print("=== Step 2: 准备数据切分 ===")
    cleaned, train, val, test = prepare_data()
    print(f"train={len(train)}  val={len(val)}  test={len(test)}\n")

    print("=== Step 3: LLM 选候选模型 ===")
    pick = llm_pick_candidates(Q, A)
    print(json.dumps(pick, ensure_ascii=False, indent=2), "\n")

    selected = [m for m in pick.get("selected", []) if m in MODEL_LIBRARY]
    if not selected:
        print("⚠️  LLM 没选出合法模型，回退到全库搜索")
        selected = list(MODEL_LIBRARY.keys())
    reasons = pick.get("reason_per_model", {})

    print("=== Step 4: 候选模型超参搜索 ===")
    results: list[CandidateResult] = []
    for name in selected:
        print(f"\n[搜索 {name}]")
        r = hp_search(name, train, val,
                      reason=reasons.get(name, "(no reason from LLM)"))
        if r is None:
            print(f"  {name} 全部配置失败，跳过")
            continue
        for p, s in r.all_trials:
            print(f"  params={p}  val_MAPE={s:.3f}")
        print(f"  → 最优: {r.best_params}  val_MAPE={r.val_mape}")
        results.append(r)

    # 按 val MAPE 排序
    results.sort(key=lambda r: r.val_mape)

    print("\n=== Step 5: 写 planner_state.json ===")
    out = {
        "task": {
            "horizon_test": H_TEST,
            "horizon_val": H_VAL,
            "period": PERIOD,
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
        },
        "candidates": [asdict(r) for r in results],
    }
    out_path = HERE / "planner_state.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    print(f"已保存 → {out_path}")

    print("\n=== 候选 top-k ===")
    for r in results:
        print(f"  {r.model:18s}  val_MAPE={r.val_mape:6.3f}  params={r.best_params}")


if __name__ == "__main__":
    main()
