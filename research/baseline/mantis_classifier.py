"""task #69 follow-up · MantisV2 (Huawei 2025) TSC baselines.

MantisV2 embeddings + downstream linear/kNN classifier. Two variants registered:
  - mantis_1nn: 1-NN on Mantis embeddings (analog to MOMENT 1-NN)
  - mantis_lr:  LogReg on Mantis embeddings (analog to MOMENT LogReg)

Weights: paris-noah/MantisV2 (HF). Lightweight ~4M params, CPU-feasible.
"""
from __future__ import annotations
import json
import numpy as np

_TRAINER = None


def _get_trainer():
    """Lazy-load MantisV2 + trainer (~4M params, CPU)."""
    global _TRAINER
    if _TRAINER is None:
        from mantis.architecture import MantisV2
        from mantis.trainer import MantisTrainer
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        cfg = json.load(open(hf_hub_download("paris-noah/MantisV2", "config.json")))
        init_keys = {"hidden_dim", "num_patches", "kernel_size", "scalar_scales",
                     "hidden_dim_scalar_enc", "epsilon_scalar_enc", "transf_depth",
                     "transf_num_heads", "transf_mlp_dim", "transf_dim_head",
                     "transf_dropout", "return_transf_layer", "output_token",
                     "device", "pre_training"}
        init_cfg = {k: v for k, v in cfg.items() if k in init_keys}
        init_cfg["device"] = "cpu"
        net = MantisV2(**init_cfg)
        sd = load_file(hf_hub_download("paris-noah/MantisV2", "model.safetensors"))
        net.load_state_dict(sd, strict=False)
        net.eval()
        _TRAINER = MantisTrainer(device="cpu", network=net)
    return _TRAINER


def _to_mantis_input(X: np.ndarray, target_len: int = 512) -> np.ndarray:
    """Reshape and pad/truncate UCR-style (N,L) → (N, 1, target_len) float32."""
    X = np.asarray(X, dtype=np.float32)
    if X.shape[1] < target_len:
        X = np.concatenate([X, np.zeros((X.shape[0], target_len - X.shape[1]), dtype=np.float32)], axis=1)
    else:
        X = X[:, :target_len]
    return X[:, None, :]


def _embed(X: np.ndarray) -> np.ndarray:
    tr = _get_trainer()
    feats = tr.transform(_to_mantis_input(X), three_dim=True)
    return feats.reshape(len(X), -1)


def classify_1nn(X_train: np.ndarray, y_train: np.ndarray,
                 X_test: np.ndarray) -> np.ndarray:
    from sklearn.neighbors import KNeighborsClassifier
    ftr, fte = _embed(X_train), _embed(X_test)
    return KNeighborsClassifier(n_neighbors=1).fit(ftr, y_train).predict(fte)


def classify_logreg(X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    ftr, fte = _embed(X_train), _embed(X_test)
    return LogisticRegression(max_iter=1000, n_jobs=-1).fit(ftr, y_train).predict(fte)


if __name__ == "__main__":
    from research.utils.ucr_loader import load_ucr_fewshot
    import time
    for ds in ["Coffee", "Wafer", "BirdChicken"]:
        X_tr, y_tr, X_te, y_te = load_ucr_fewshot(ds, n_per_class=5, seed=1)
        X_te, y_te = X_te[:200], y_te[:200]
        t0 = time.time()
        yp = classify_logreg(X_tr, y_tr, X_te)
        print(f"  {ds:12} LR  acc={(yp==y_te).mean():.3f} ({time.time()-t0:.1f}s)")
