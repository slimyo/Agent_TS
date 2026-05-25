"""B4 baseline · MOMENT (Goswami et al. 2024) 作为 TaskB 分类器。

策略：MOMENT-1-small (38M, CPU 友好) zero-shot embedding → 1-NN / Logistic 分类。
论文价值：feedback §四"较高"优先级 TSFM；TaskB 强 baseline。
"""
from __future__ import annotations

import os

import numpy as np
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "0")  # 允许第一次下载

_MODEL = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_INPUT_LEN = 512  # MOMENT 原生输入长度


def _get_model():
    global _MODEL
    if _MODEL is None:
        from momentfm import MOMENTPipeline
        _MODEL = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-small",
            model_kwargs={"task_name": "embedding", "n_channels": 1},
        )
        _MODEL.init()
        _MODEL.eval()
        _MODEL.to(_DEVICE)
    return _MODEL


def embed(series_batch: np.ndarray) -> np.ndarray:
    """series_batch [B, L] → embeddings [B, 512]。
    L 可变；自动 pad 到 512 (左 zero-pad)，input_mask 对应。
    """
    m = _get_model()
    B, L = series_batch.shape
    L_pad = max(L, _INPUT_LEN)
    x = torch.zeros(B, 1, L_pad, dtype=torch.float32)
    mask = torch.zeros(B, L_pad, dtype=torch.float32)
    if L >= _INPUT_LEN:
        x[:, 0, :] = torch.from_numpy(series_batch[:, -L_pad:].astype(np.float32))
        mask[:, :] = 1.0
    else:
        x[:, 0, -L:] = torch.from_numpy(series_batch.astype(np.float32))
        mask[:, -L:] = 1.0
    x = x.to(_DEVICE); mask = mask.to(_DEVICE)
    with torch.no_grad():
        out = m(x_enc=x, input_mask=mask)
    return out.embeddings.cpu().numpy()


def classify_1nn(X_train: np.ndarray, y_train: np.ndarray,
                 X_test: np.ndarray) -> np.ndarray:
    """1-NN on MOMENT embeddings (Euclidean)."""
    E_tr = embed(X_train)
    E_te = embed(X_test)
    # L2-normalize for cosine-equivalent
    E_tr = E_tr / (np.linalg.norm(E_tr, axis=1, keepdims=True) + 1e-9)
    E_te = E_te / (np.linalg.norm(E_te, axis=1, keepdims=True) + 1e-9)
    # Pairwise cosine sim
    sim = E_te @ E_tr.T  # [N_test, N_train]
    nn_idx = sim.argmax(axis=1)
    return y_train[nn_idx]


def classify_logreg(X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray, C: float = 1.0) -> np.ndarray:
    """Linear probe on MOMENT embeddings via logistic regression."""
    from sklearn.linear_model import LogisticRegression
    E_tr = embed(X_train)
    E_te = embed(X_test)
    clf = LogisticRegression(C=C, max_iter=1000, multi_class="auto")
    clf.fit(E_tr, y_train)
    return clf.predict(E_te)


if __name__ == "__main__":
    # Smoke test on UCR Coffee
    from research.utils.ucr_loader import load_ucr_fewshot
    X_tr, y_tr, X_te, y_te = load_ucr_fewshot("Coffee", n_per_class=5, seed=1)
    print(f"Coffee 5-shot: X_tr={X_tr.shape}, X_te={X_te.shape}, classes={set(y_tr.tolist())}")

    y_pred_1nn = classify_1nn(X_tr, y_tr, X_te)
    acc_1nn = float((y_pred_1nn == y_te).mean())
    print(f"MOMENT 1-NN accuracy: {acc_1nn:.3f}")

    y_pred_lr = classify_logreg(X_tr, y_tr, X_te)
    acc_lr = float((y_pred_lr == y_te).mean())
    print(f"MOMENT LogReg accuracy: {acc_lr:.3f}")
