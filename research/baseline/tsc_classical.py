"""TaskB · 经典 TSC baselines (B1-B3) + MOMENT (B4 via baseline.moment_classifier).

B1: 1-NN + DTW       — TSC 经典强 baseline (Bagnall 2017)
B2: 1-NN + Euclidean — sanity baseline
B3: MiniRocket / Rocket — Dempster 2020 NeurIPS, kernel-based SOTA
"""
from __future__ import annotations

import numpy as np


def b1_knn_dtw(X_train: np.ndarray, y_train: np.ndarray,
               X_test: np.ndarray) -> np.ndarray:
    """1-NN with DTW distance."""
    from dtaidistance import dtw
    preds = []
    for x_te in X_test:
        dists = np.array([dtw.distance(x_te.astype(np.float64),
                                        x_tr.astype(np.float64))
                          for x_tr in X_train])
        preds.append(y_train[dists.argmin()])
    return np.array(preds)


def b2_knn_euclid(X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray) -> np.ndarray:
    """1-NN with Euclidean distance."""
    diff = X_test[:, None, :] - X_train[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=-1))
    return y_train[dists.argmin(axis=1)]


def b3_rocket(X_train: np.ndarray, y_train: np.ndarray,
              X_test: np.ndarray, num_kernels: int = 1000) -> np.ndarray:
    """Rocket / MiniRocket via sktime."""
    from sktime.classification.kernel_based import RocketClassifier
    # sktime expects (n_instances, n_channels, length) or pd.DataFrame
    clf = RocketClassifier(num_kernels=num_kernels, random_state=0)
    X_tr_3d = X_train[:, None, :]
    X_te_3d = X_test[:, None, :]
    clf.fit(X_tr_3d, y_train)
    return clf.predict(X_te_3d)


if __name__ == "__main__":
    from research.utils.ucr_loader import load_ucr_fewshot
    X_tr, y_tr, X_te, y_te = load_ucr_fewshot("Coffee", n_per_class=5, seed=1)
    print(f"Coffee 5-shot: X_tr={X_tr.shape}, X_te={X_te.shape}")
    for name, fn in [("B1 1-NN DTW", b1_knn_dtw),
                     ("B2 1-NN Euclid", b2_knn_euclid),
                     ("B3 Rocket", b3_rocket)]:
        y_p = fn(X_tr, y_tr, X_te)
        acc = float((y_p == y_te).mean())
        print(f"{name:20}: {acc:.3f}")
