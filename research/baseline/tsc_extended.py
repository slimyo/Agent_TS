"""扩展分类基线（sktime 原生）—— baseline.md 末尾「可扩展分类模型」类别一 + 类别四。

统一签名 `predict(X_train, y_train, X_test, **kw) -> y_pred`，输入兼容：
  - 2D `[N, L]`（单变量；UEA channel-flatten 后也是 2D）
  - 3D `[N, C, L]`（原生多变量；类别四用）
内部统一转成 sktime 需要的 `[N, C, L]`（单变量则 C=1）。

类别一（随机卷积核 / 区间，与 Rocket 互补，UCR/UEA 前列）：
  multirocket, arsenal, drcif
类别四（原生多变量，不做 channel-flatten）：
  muse, rocket_mv, cif_mv
全部 sktime 0.40 原生、纯 CPU、无额外大权重 → 本地 `tsci` env 即可跑。
"""
from __future__ import annotations

import numpy as np


def _to3d(X) -> np.ndarray:
    """[N,L]->[N,1,L]; [N,C,L]->原样。sktime 分类器吃 [n_inst, n_ch, n_time]。"""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 2:
        return X[:, None, :]
    if X.ndim == 3:
        return X
    raise ValueError(f"unexpected X.ndim={X.ndim}")


def _sk_run(make_clf, X_train, y_train, X_test):
    clf = make_clf()
    clf.fit(_to3d(X_train), np.asarray(y_train))
    return np.asarray(clf.predict(_to3d(X_test)))


# ---------- 类别一：经典 SOTA（卷积核 / 区间） ---------- #

def multirocket(X_train, y_train, X_test, num_kernels=6250, **_):
    """MultiRocket：Rocket 的强力扩展（多池化），UCR 常优于 Rocket。"""
    from sktime.classification.kernel_based import RocketClassifier
    return _sk_run(lambda: RocketClassifier(
        rocket_transform="multirocket", num_kernels=num_kernels, n_jobs=-1), X_train, y_train, X_test)


def arsenal(X_train, y_train, X_test, num_kernels=2000, n_estimators=25, **_):
    """Arsenal：Rocket 集成（带置信度），比单 Rocket 更稳。"""
    from sktime.classification.kernel_based import Arsenal
    return _sk_run(lambda: Arsenal(
        num_kernels=num_kernels, n_estimators=n_estimators, n_jobs=-1), X_train, y_train, X_test)


def drcif(X_train, y_train, X_test, n_estimators=50, **_):
    """DrCIF：随机区间 + catch22/统计特征森林，长序列/高噪声强，与卷积核互补。"""
    from sktime.classification.interval_based import DrCIF
    return _sk_run(lambda: DrCIF(n_estimators=n_estimators, n_jobs=-1), X_train, y_train, X_test)


# ---------- 类别四：原生多变量（输入 [N,C,L]，不 flatten） ---------- #

def muse(X_train, y_train, X_test, **_):
    """MUSE：WEASEL 的多变量扩展（符号化词袋），UEA 顶级经典法。"""
    from sktime.classification.dictionary_based import MUSE
    return _sk_run(lambda: MUSE(), X_train, y_train, X_test)


def rocket_mv(X_train, y_train, X_test, num_kernels=10000, **_):
    """多变量 Rocket（原生处理多通道，非 channel-flatten）。"""
    from sktime.classification.kernel_based import RocketClassifier
    return _sk_run(lambda: RocketClassifier(num_kernels=num_kernels, n_jobs=-1), X_train, y_train, X_test)


def cif_mv(X_train, y_train, X_test, n_estimators=50, **_):
    """CanonicalIntervalForest：原生多变量区间森林（catch22 + 统计）。"""
    from sktime.classification.interval_based import CanonicalIntervalForest
    return _sk_run(lambda: CanonicalIntervalForest(n_estimators=n_estimators, n_jobs=-1), X_train, y_train, X_test)


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    Xtr = rng.standard_normal((12, 60)).astype(np.float32); Xtr[:6] += 0.7
    ytr = np.array([0] * 6 + [1] * 6); Xte = rng.standard_normal((4, 60)).astype(np.float32)
    for nm, fn in [("multirocket", multirocket), ("arsenal", arsenal), ("drcif", drcif),
                   ("muse", muse), ("rocket_mv", rocket_mv), ("cif_mv", cif_mv)]:
        try:
            p = fn(Xtr, ytr, Xte); print(f"{nm:14} OK pred={list(p)}")
        except Exception as e:
            print(f"{nm:14} FAIL {type(e).__name__}: {str(e)[:80]}")
