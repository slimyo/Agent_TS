"""评估指标（对应 plan §4.3）。

MASE 定义：
  MASE = MAE(y_true, y_pred) / MAE_naive_in_sample
  其中 MAE_naive_in_sample = mean(|y_train[t] - y_train[t-m]|)，
  m 为季节周期；若训练长度不足 m+1，退化到 m=1（一阶差分）。
"""
from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    num = np.abs(y_true - y_pred)
    den = (np.abs(y_true) + np.abs(y_pred)) / 2 + eps
    return float(np.mean(num / den) * 100)


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, m: int = 1) -> float:
    if len(y_train) <= m:
        m = 1
    scale = float(np.mean(np.abs(y_train[m:] - y_train[:-m])))
    if scale < 1e-12:
        return float("inf")
    return mae(y_true, y_pred) / scale


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                y_train: np.ndarray, season_m: int = 1) -> dict[str, float]:
    return {
        "mae":   mae(y_true, y_pred),
        "mse":   mse(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "mase":  mase(y_true, y_pred, y_train, m=season_m),
    }
