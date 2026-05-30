"""Round 7 P0-2 · Safe predict wrapper with shape + finite checks.

Wraps any predict_fn to enforce:
    1. Output length == H
    2. Output finite (no NaN / Inf)
    3. Output magnitude not pathological (>1e8 × history std)

On violation: classifies failure → records to ReliabilityTracker → optionally
falls back to a safe predictor (default = naive_drift).

Used by adaptive runtime to close the gap exposed by Round 6 ILI failures
(shape mismatch (24,) vs (10,), etc).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np

from research.agent.reliability_priors import get_tracker


@dataclass
class SafePredictResult:
    pred: np.ndarray
    chosen_model: str            # what actually produced the pred
    fallback_used: bool
    failure_type: Optional[str]  # if intervention happened
    raw_pred_shape: Optional[tuple]
    notes: str = ""


def _is_pathological(pred: np.ndarray, history: np.ndarray,
                      scale_mult: float = 1e8) -> bool:
    """Detect output magnitude wildly larger than input scale."""
    if pred.size == 0: return True
    h_std = float(np.std(history)) + 1e-9
    max_abs = float(np.max(np.abs(pred)))
    return max_abs > scale_mult * h_std


def safe_predict(
    model_name: str,
    predict_fn: Callable[[str], np.ndarray],
    H: int,
    history: np.ndarray,
    fallback_model: str = "naive_drift",
    fallback_predict_fn: Optional[Callable[[str], np.ndarray]] = None,
    register_outcome: bool = True,
) -> SafePredictResult:
    """Run model_name; validate; fallback if needed.

    Args:
        model_name: primary model to try
        predict_fn: callable (model_name) → ndarray
        H: required output length
        history: train series (for scale check)
        fallback_model: which model to use if primary fails
        fallback_predict_fn: defaults to predict_fn
        register_outcome: if True, record success/failure to ReliabilityTracker
    """
    tracker = get_tracker() if register_outcome else None
    if fallback_predict_fn is None:
        fallback_predict_fn = predict_fn

    # ─── 1. Try primary ─────────────────────────────────────────────
    raw_shape = None
    try:
        pred = np.asarray(predict_fn(model_name), dtype=np.float64)
        raw_shape = pred.shape
        # length check
        if pred.size != H:
            ftype = "shape_mismatch"
            notes = f"expected len {H}, got {pred.size}"
        elif not np.all(np.isfinite(pred)):
            ftype = "outlier_corruption"
            notes = "NaN/Inf in output"
        elif _is_pathological(pred, history):
            ftype = "outlier_corruption"
            notes = (f"output max_abs={float(np.max(np.abs(pred))):.2e} "
                     f"vs hist_std={float(np.std(history)):.2e}")
        else:
            # SUCCESS
            if tracker: tracker.record_outcome(model_name, success=True)
            return SafePredictResult(pred=pred, chosen_model=model_name,
                                      fallback_used=False, failure_type=None,
                                      raw_pred_shape=raw_shape)
    except Exception as e:
        ftype = _classify_exception(e)
        notes = f"{type(e).__name__}: {str(e)[:120]}"

    # ─── 2. Failure registered ──────────────────────────────────────
    if tracker:
        tracker.record_outcome(model_name, success=False, error_type=ftype)

    # ─── 3. Fallback ────────────────────────────────────────────────
    try:
        pred_fb = np.asarray(fallback_predict_fn(fallback_model),
                              dtype=np.float64)
        if pred_fb.size != H:
            # last resort: pad / trim
            if pred_fb.size > H:
                pred_fb = pred_fb[:H]
            else:
                pred_fb = np.concatenate([
                    pred_fb,
                    np.full(H - pred_fb.size,
                            pred_fb[-1] if pred_fb.size else 0.0)])
        if not np.all(np.isfinite(pred_fb)):
            pred_fb = np.where(np.isfinite(pred_fb), pred_fb, 0.0)
        if tracker: tracker.record_outcome(fallback_model, success=True)
        return SafePredictResult(
            pred=pred_fb, chosen_model=fallback_model,
            fallback_used=True, failure_type=ftype,
            raw_pred_shape=raw_shape,
            notes=f"primary={model_name} failed ({notes}); fallback={fallback_model}",
        )
    except Exception as e2:
        # absolute last resort: zero-pred
        pred_zero = np.zeros(H, dtype=np.float64)
        if tracker:
            tracker.record_outcome(fallback_model, success=False,
                                    error_type=_classify_exception(e2))
        return SafePredictResult(
            pred=pred_zero, chosen_model="<zero>",
            fallback_used=True, failure_type=f"all-fallback-failed:{ftype}",
            raw_pred_shape=raw_shape,
            notes=f"primary AND fallback failed; emitting zeros",
        )


def _classify_exception(e: Exception) -> str:
    s = str(e).lower()
    cn = type(e).__name__.lower()
    if "modulenotfound" in cn or "importerror" in cn:
        return "load_error"
    if "ssl" in s or "connection" in s or "timeout" in s:
        return "load_error"   # treat network as load fail
    if "out of memory" in s or "cuda" in s and "memory" in s:
        return "oom"
    if "shape" in s or "broadcast" in s:
        return "shape_mismatch"
    if "nan" in s or "inf" in s:
        return "outlier_corruption"
    return "unknown"


if __name__ == "__main__":
    print("=" * 60)
    print("Round 7 P0-2 · Safe predict smoke")
    print("=" * 60)
    from research.agent.reliability_priors import reset_tracker, get_tracker
    reset_tracker()

    H = 12
    history = np.sin(np.arange(100) * 0.1)

    # ─── Test 1: 正常预测
    def ok_fn(m):
        if m == "good": return np.linspace(0, 1, H)
        if m == "naive_drift": return np.zeros(H)
        raise RuntimeError("unknown")
    r1 = safe_predict("good", ok_fn, H, history, "naive_drift", ok_fn)
    print(f"\n[Test 1] success: chosen={r1.chosen_model}  fallback={r1.fallback_used}")
    print(f"  tracker(good): {get_tracker().health('good')}")

    # ─── Test 2: 长度错误
    def bad_shape_fn(m):
        if m == "bad_shape": return np.array([1.0, 2.0])   # wrong length
        if m == "naive_drift": return np.zeros(H)
        raise RuntimeError("unknown")
    r2 = safe_predict("bad_shape", bad_shape_fn, H, history,
                       "naive_drift", bad_shape_fn)
    print(f"\n[Test 2] shape_mismatch: chosen={r2.chosen_model}  "
          f"fallback={r2.fallback_used}  ftype={r2.failure_type}")
    print(f"  notes: {r2.notes[:100]}")
    print(f"  tracker(bad_shape): {get_tracker().health('bad_shape')}")

    # ─── Test 3: ModuleNotFoundError
    def missing_module_fn(m):
        if m == "moirai": raise ModuleNotFoundError("No module named 'moirai'")
        if m == "naive_drift": return np.zeros(H)
        raise RuntimeError("unknown")
    r3 = safe_predict("moirai", missing_module_fn, H, history,
                       "naive_drift", missing_module_fn)
    print(f"\n[Test 3] ModuleNotFound: chosen={r3.chosen_model}  "
          f"fallback={r3.fallback_used}  ftype={r3.failure_type}")
    print(f"  tracker(moirai): {get_tracker().health('moirai')}")

    # ─── Test 4: NaN output
    def nan_fn(m):
        if m == "broken": return np.full(H, np.nan)
        if m == "naive_drift": return np.zeros(H)
        raise RuntimeError("unknown")
    r4 = safe_predict("broken", nan_fn, H, history, "naive_drift", nan_fn)
    print(f"\n[Test 4] NaN output: chosen={r4.chosen_model}  "
          f"fallback={r4.fallback_used}  ftype={r4.failure_type}")
    print(f"  tracker(broken): {get_tracker().health('broken')}")

    # ─── Test 5: 3次 consecutive failure → trips circuit breaker
    print(f"\n[Test 5] After 3 fails on 'moirai', breaker should be open")
    for _ in range(2):
        safe_predict("moirai", missing_module_fn, H, history,
                      "naive_drift", missing_module_fn)
    print(f"  is_open(moirai): {get_tracker().is_open('moirai')}")
