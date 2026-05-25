"""B2 · ARIMA + ETS 自动选择（plan §三 B2）。

流程：
  1. 在 train 上分别拟合 auto_arima(seasonal=True) 和 ExponentialSmoothing(Holt-Winters)
  2. 比较两者的 AIC，取较低者
  3. refit 在 train+val 上，预测 H 步

少样本鲁棒性细节：
  - season_m 若 > len(train)//2 则关闭季节性（避免 m 太大无法拟合）
  - auto_arima 出错时回退到 ARIMA(1,1,1)
  - ExponentialSmoothing 拟合不到时回退到 SES（Simple Exp Smoothing）
"""
from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore")


def _try_arima(y: np.ndarray, m: int):
    import pmdarima as pm
    try:
        model = pm.auto_arima(
            y,
            seasonal=(m > 1),
            m=m if m > 1 else 1,
            suppress_warnings=True,
            error_action="ignore",
            stepwise=True,
            max_p=3, max_q=3, max_P=1, max_Q=1, max_d=2, max_D=1,
        )
        return model, float(model.aic())
    except Exception:
        try:
            model = pm.ARIMA(order=(1, 1, 1), suppress_warnings=True).fit(y)
            return model, float(model.aic())
        except Exception:
            return None, float("inf")


def _try_ets(y: np.ndarray, m: int):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
    seasonal = "add" if (m > 1 and len(y) >= 2 * m) else None
    trend = "add" if len(y) >= 4 else None
    try:
        model = ExponentialSmoothing(
            y, trend=trend, seasonal=seasonal,
            seasonal_periods=m if seasonal else None,
            initialization_method="estimated",
        ).fit(optimized=True, disp=False)
        return model, float(model.aic)
    except Exception:
        try:
            model = SimpleExpSmoothing(y, initialization_method="estimated").fit(disp=False)
            return model, float(model.aic)
        except Exception:
            return None, float("inf")


def _forecast(model, kind: str, H: int) -> np.ndarray:
    if kind == "arima":
        return np.asarray(model.predict(n_periods=H), dtype=np.float64)
    return np.asarray(model.forecast(H), dtype=np.float64)


def predict(train: np.ndarray, val: np.ndarray, H: int,
            seed: int = 42, season_m: int = 1, **_) -> np.ndarray:
    # 关闭过大的季节性
    m = season_m if (season_m > 1 and len(train) >= 2 * season_m) else 1

    # 阶段一：用 train 拟合，按 AIC 选模型族
    arima_m, arima_aic = _try_arima(train, m)
    ets_m,   ets_aic   = _try_ets(train, m)

    if arima_aic == float("inf") and ets_aic == float("inf"):
        # 全失败：退到全局均值
        return np.full(H, float(train.mean()))

    choice = "arima" if arima_aic <= ets_aic else "ets"

    # 阶段二：refit 在 train+val 上，再产出 H 步
    y_all = np.concatenate([train, val])
    if choice == "arima":
        refit, _ = _try_arima(y_all, m)
        if refit is None:
            refit = arima_m
            # 注意：原 model 只见过 train，需要扩展 H + len(val) 步丢掉前 len(val)
            full = _forecast(refit, "arima", H + len(val))
            return full[len(val):]
        return _forecast(refit, "arima", H)
    else:
        refit, _ = _try_ets(y_all, m)
        if refit is None:
            refit = ets_m
            full = _forecast(refit, "ets", H + len(val))
            return full[len(val):]
        return _forecast(refit, "ets", H)
