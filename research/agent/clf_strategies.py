"""P6.4b-2 / task #31 · 统一分类器策略接口。

参照 `forecaster_reflect.STRATEGY_FN` 的设计，把 TSC baselines 包装成
统一签名 `predict(X_train, y_train, X_test, **kwargs) -> np.ndarray`，
注册到 CLF_STRATEGY_FN 字典，供 classification_planner 调用。

可加 kwargs（统一签名约定）：
  - season_m: int = 1   （季节周期）
  - llm_model: str | None = None   （LLM 模型 override）
  - num_kernels: int = 1000        （Rocket）

约定：
  - 输入 X 形状 [N, L]（N samples, L length），y 形状 [N]
  - 输出 y_pred 形状 [N_test]
  - 不抛异常（内部 try/except，失败时返回 majority class）
"""
from __future__ import annotations

from typing import Callable

import numpy as np


# ---------- adapter ---------- #

def _adapter(fn: Callable, **fixed_kwargs):
    """把固定 kwargs 绑定，剩 (X_tr, y_tr, X_te) 三正参数 + 额外可变 kwargs。"""
    def wrapped(X_train, y_train, X_test, **kw):
        merged = {**fixed_kwargs, **kw}
        # 过滤掉目标 fn 不接受的 kwarg
        import inspect
        sig = inspect.signature(fn)
        accept = set(sig.parameters.keys())
        passed = {k: v for k, v in merged.items() if k in accept}
        return fn(X_train, y_train, X_test, **passed)
    wrapped.__name__ = fn.__name__
    return wrapped


def _safe(fn: Callable):
    """失败时回退到 majority-class 预测，避免单点失败拖死整个 planner。"""
    def wrapped(X_train, y_train, X_test, **kw):
        try:
            return fn(X_train, y_train, X_test, **kw)
        except Exception as e:
            print(f"[clf_strategies] {fn.__name__} FAILED: {e!r}, falling back to majority")
            from collections import Counter
            major = Counter(y_train.tolist()).most_common(1)[0][0]
            return np.full(len(X_test), major, dtype=type(y_train[0]))
    wrapped.__name__ = fn.__name__
    return wrapped


# ---------- lazy classifier imports ---------- #

def _dtw_1nn(X_train, y_train, X_test, **_):
    from research.baseline.tsc_classical import b1_knn_dtw
    return b1_knn_dtw(X_train, y_train, X_test)


def _euclid_1nn(X_train, y_train, X_test, **_):
    from research.baseline.tsc_classical import b2_knn_euclid
    return b2_knn_euclid(X_train, y_train, X_test)


def _rocket(X_train, y_train, X_test, num_kernels=1000, **_):
    from research.baseline.tsc_classical import b3_rocket
    return b3_rocket(X_train, y_train, X_test, num_kernels=num_kernels)


def _moment_1nn(X_train, y_train, X_test, **_):
    from research.baseline.moment_classifier import classify_1nn
    return classify_1nn(X_train, y_train, X_test)


def _moment_logreg(X_train, y_train, X_test, **_):
    from research.baseline.moment_classifier import classify_logreg
    return classify_logreg(X_train, y_train, X_test)


def _llm_direct(X_train, y_train, X_test, llm_model=None, **_):
    from research.agent.tsc_classifier import b5_llm_direct
    return b5_llm_direct(X_train, y_train, X_test, llm_model=llm_model)


def _minirocket(X_train, y_train, X_test, num_kernels=10000, **_):
    from research.baseline.tsc_classical import b5_minirocket
    return b5_minirocket(X_train, y_train, X_test, num_kernels=num_kernels)


def _weasel(X_train, y_train, X_test, **_):
    from research.baseline.tsc_classical import b6_weasel
    return b6_weasel(X_train, y_train, X_test)


def _catch22(X_train, y_train, X_test, **_):
    from research.baseline.tsc_classical import b7_catch22
    return b7_catch22(X_train, y_train, X_test)


def _mantis_1nn(X_train, y_train, X_test, **_):
    from research.baseline.mantis_classifier import classify_1nn
    return classify_1nn(X_train, y_train, X_test)


def _mantis_lr(X_train, y_train, X_test, **_):
    from research.baseline.mantis_classifier import classify_logreg
    return classify_logreg(X_train, y_train, X_test)


# ---------- 统一策略池 ---------- #

CLF_STRATEGY_FN: dict[str, Callable] = {
    "dtw_1nn":       _safe(_dtw_1nn),
    "euclid_1nn":    _safe(_euclid_1nn),
    "rocket":        _safe(_rocket),
    "moment_1nn":    _safe(_moment_1nn),
    "moment_logreg": _safe(_moment_logreg),
    "llm_direct":    _safe(_llm_direct),
    "minirocket":    _safe(_minirocket),
    "weasel":        _safe(_weasel),
    "catch22":       _safe(_catch22),
    "mantis_1nn":    _safe(_mantis_1nn),
    "mantis_lr":     _safe(_mantis_lr),
}


# 默认 fallback（Rocket 已实证 UCR 上 mean 87.5% 单一最强）
DEFAULT_CLASSIFIER = "rocket"


def list_strategies() -> list[str]:
    return list(CLF_STRATEGY_FN.keys())


def predict_with(strategy: str, X_train, y_train, X_test, **kwargs):
    """对外统一调用入口。"""
    if strategy not in CLF_STRATEGY_FN:
        raise ValueError(f"unknown classifier: {strategy}. Available: {list_strategies()}")
    return CLF_STRATEGY_FN[strategy](X_train, y_train, X_test, **kwargs)


if __name__ == "__main__":
    # Smoke: Coffee 5-shot all strategies
    from research.utils.ucr_loader import load_ucr_fewshot
    X_tr, y_tr, X_te, y_te = load_ucr_fewshot("Coffee", n_per_class=5, seed=1)
    print(f"Coffee 5-shot: X_tr={X_tr.shape}, X_te={X_te.shape}")
    print(f"\nAvailable strategies: {list_strategies()}")
    print(f"Default: {DEFAULT_CLASSIFIER}\n")
    print(f"{'strategy':18}  {'acc':>6}")
    for name in list_strategies():
        if name == "llm_direct":
            # LLM 太慢，smoke 用小 test 子集
            import numpy as np
            idx = np.array([0, 1, 2, 3])
            yp = predict_with(name, X_tr, y_tr, X_te[idx])
            acc = float((yp == y_te[idx]).mean())
            print(f"  {name:18}  {acc:>6.3f} (subset n=4)")
        else:
            yp = predict_with(name, X_tr, y_tr, X_te)
            acc = float((yp == y_te).mean())
            print(f"  {name:18}  {acc:>6.3f}")
