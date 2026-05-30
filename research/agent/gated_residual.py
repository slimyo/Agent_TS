"""task #67 · Gated Residual Forecaster (feedback 架构 #2).

ŷ = ŷ_C2 + g(d) · Δ(d)

  - ŷ_C2: Chronos-2 point forecast
  - Δ(d): learned residual / bias predictor (LightGBM regressor on cell features)
  - g(d): learned gate in [0,1] (logistic regressor)

Training data: cells with (C2_pred, y_true) pairs. Target:
  Δ_target = mean(y_true - C2_pred)  per cell  (single scalar bias)
  g_target = 1 if applying Δ reduces MAE else 0  per cell

Theoretical guarantee: at g→0 it equals C2; only deviates when gate fires.
Loss-aware training: jointly minimize MAE(ŷ, y) on held-out cells.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class GatedResidualHead:
    delta_model: object       # bias predictor
    gate_model: object        # binary classifier
    feature_mean: np.ndarray
    feature_std: np.ndarray
    feature_names: list

    def predict(self, c2_pred: np.ndarray, features: np.ndarray,
                tau: float = 0.5, scale: float = None) -> tuple[np.ndarray, float, float]:
        """Apply gated residual correction.

        scale: inference-time normalization (history std). Required for cross-scale.
               If None, falls back to std(c2_pred).
        """
        if features is None or not np.all(np.isfinite(features)):
            return c2_pred, 0.0, 0.0
        if scale is None or scale <= 0:
            scale = float(np.std(c2_pred)) + 1e-6
        x = (features - self.feature_mean) / (self.feature_std + 1e-9)
        x = x.reshape(1, -1)
        # Predict normalized bias, then scale back
        bias_norm = float(self.delta_model.predict(x)[0])
        bias_abs = bias_norm * scale
        if hasattr(self.gate_model, "predict_proba"):
            gprob = float(self.gate_model.predict_proba(x)[0, 1])
        else:
            gprob = float(self.gate_model.predict(x)[0])
        # v2 conservative: heavy shrinkage by 0.3 + require gate AND large normalized bias
        SHRINK = 0.3
        BIAS_NORM_MIN = 0.1   # only fire if normalized bias predicts >10% of scale
        if gprob >= tau and abs(bias_norm) >= BIAS_NORM_MIN:
            corrected = c2_pred + SHRINK * bias_abs
            return corrected, gprob, bias_abs
        return c2_pred, gprob, bias_abs


def featurize_history(train_history: np.ndarray, c2_pred: np.ndarray) -> np.ndarray:
    """Extract per-cell features for residual prediction.

    Combines:
      - 25 series_features on train_history (last L points)
      - 5 C2-pred-derived features (mean / std / trend / range / pred-vs-history-shift)
    """
    from research.utils.series_features import extract_full_features, FEATURE_ORDER
    # Limit history to last 256 points (memory/cpu)
    hist = train_history[-256:] if len(train_history) > 256 else train_history
    f = extract_full_features(hist)
    base_vec = [f.get(k, 0.0) for k in FEATURE_ORDER if not k.startswith("meta_")]
    # C2-pred features
    pred = np.asarray(c2_pred).flatten()
    hist_mean = float(np.mean(hist))
    hist_std = float(np.std(hist)) + 1e-9
    pred_mean = float(np.mean(pred))
    pred_std = float(np.std(pred))
    pred_trend = float(np.polyfit(np.arange(len(pred)), pred, 1)[0]) if len(pred) > 1 else 0.0
    pred_range = float(pred.max() - pred.min())
    shift = (pred_mean - hist_mean) / hist_std       # how far C2 predicts to drift
    rel_std = pred_std / hist_std                     # relative spread
    c2_feats = [pred_mean, pred_std, pred_trend, pred_range, shift, rel_std]
    return np.array(base_vec + c2_feats, dtype=np.float64)


def train_gated_residual(cells: list[dict]) -> GatedResidualHead:
    """Train Δ and g heads on a list of cell dicts.

    cells: each cell has {features, c2_pred (np.array H), y_true (np.array H),
                         history (np.array, optional)}.

    Cross-scale generalization: train on NORMALIZED bias = mean(y - C2) / scale
    where scale = std(history) (or std(c2_pred) fallback).
    At inference, multiply predicted normalized bias by inference-time scale.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingRegressor
    X = np.array([c["features"] for c in cells])
    # Per-cell scale: use history std if available; else C2 pred std
    scales = []
    for c in cells:
        if "history" in c and len(c["history"]) > 1:
            s = float(np.std(c["history"])) + 1e-6
        else:
            s = float(np.std(c["c2_pred"])) + 1e-6
        scales.append(s)
    scales = np.array(scales)
    abs_biases = np.array([float(np.mean(c["y_true"] - c["c2_pred"])) for c in cells])
    biases_norm = abs_biases / scales       # scale-invariant target
    c2_maes = np.array([float(np.mean(np.abs(c["y_true"] - c["c2_pred"]))) for c in cells])
    helps = np.array([
        1 if np.mean(np.abs(c["y_true"] - c["c2_pred"] - abs_biases[i])) < c2_maes[i] else 0
        for i, c in enumerate(cells)
    ])
    # Use normalized target
    biases = biases_norm

    # Standardize features
    mean = X.mean(0)
    std = X.std(0) + 1e-9
    X_z = (X - mean) / std

    # Train Δ (regression): predict per-cell bias
    delta_model = GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=0)
    delta_model.fit(X_z, biases)

    # Train g (classifier): predict whether bias helps
    if len(np.unique(helps)) > 1:
        gate_model = LogisticRegression(C=1.0, max_iter=1000)
        gate_model.fit(X_z, helps)
    else:
        # Fallback: constant gate
        class _ConstGate:
            def __init__(self, p): self.p = p
            def predict_proba(self, X): return np.tile([1 - self.p, self.p], (len(X), 1))
        gate_model = _ConstGate(float(helps.mean()))

    return GatedResidualHead(
        delta_model=delta_model, gate_model=gate_model,
        feature_mean=mean, feature_std=std,
        feature_names=list(range(X.shape[1]))
    )


def evaluate_lodo(cells: list[dict], tau: float = 0.5) -> dict:
    """Leave-one-dataset-out CV: train on all other datasets, evaluate on held-out."""
    datasets = sorted(set(c["dataset"] for c in cells))
    rows = []
    for held in datasets:
        train_cells = [c for c in cells if c["dataset"] != held]
        test_cells = [c for c in cells if c["dataset"] == held]
        if len(train_cells) < 5 or not test_cells:
            continue
        head = train_gated_residual(train_cells)
        for c in test_cells:
            mae_c2 = float(np.mean(np.abs(c["y_true"] - c["c2_pred"])))
            inf_scale = float(np.std(c["history"])) + 1e-6 if "history" in c else None
            corrected, gprob, bias = head.predict(
                np.asarray(c["c2_pred"]),
                np.asarray(c["features"]),
                tau=tau, scale=inf_scale,
            )
            mae_gr = float(np.mean(np.abs(c["y_true"] - corrected)))
            rows.append({
                "dataset": c["dataset"], "N": c.get("N"), "seed": c.get("seed"),
                "mae_c2": mae_c2, "mae_gr": mae_gr,
                "delta": mae_gr - mae_c2, "gate_prob": gprob, "bias_pred": bias,
            })
    return {"per_cell": rows,
            "mean_c2_mae": float(np.mean([r["mae_c2"] for r in rows])),
            "mean_gr_mae": float(np.mean([r["mae_gr"] for r in rows])),
            "n_helped": sum(r["mae_gr"] < r["mae_c2"] for r in rows),
            "n_hurt": sum(r["mae_gr"] > r["mae_c2"] for r in rows),
            "n_tied": sum(r["mae_gr"] == r["mae_c2"] for r in rows),
            "n_total": len(rows)}
