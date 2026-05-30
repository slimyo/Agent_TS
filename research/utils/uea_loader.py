"""task #44 / P6.4e · UEA Multivariate Archive loader。

UEA archive 30 个 multivariate TSC 数据集（Bagnall 2018）。
镜像同 UCR：https://timeseriesclassification.com/aeon-toolkit/{name}.zip

每个 dataset 目录：
  <name>/<name>_TRAIN.ts  (sktime .ts format, multivariate)
  <name>/<name>_TEST.ts

我们读 .ts 格式：第一行 header 包含 @dimensions @serieslength @classlabel etc.
"""
from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np

UEA_DATASETS_RECOMMENDED = {
    # 少样本友好（Train ≤ 200，length ≤ 1500 适合 DTW）
    "AtrialFibrillation":  {"channels": 2, "classes": 3, "length": 640, "train": 15},
    "BasicMotions":        {"channels": 6, "classes": 4, "length": 100, "train": 40},
    "Cricket":             {"channels": 6, "classes": 12, "length": 1197, "train": 108},
    "ERing":               {"channels": 4, "classes": 6, "length": 65, "train": 30},
    "Handwriting":         {"channels": 3, "classes": 26, "length": 152, "train": 150},
    "Libras":              {"channels": 2, "classes": 15, "length": 45, "train": 180},
    "UWaveGestureLibrary": {"channels": 3, "classes": 8, "length": 315, "train": 120},
    # 中等规模
    "ArticularyWordRecognition": {"channels": 9, "classes": 25, "length": 144, "train": 275},
    "Epilepsy":            {"channels": 3, "classes": 4, "length": 206, "train": 137},
    "NATOPS":              {"channels": 24, "classes": 6, "length": 51, "train": 180},
    "RacketSports":        {"channels": 6, "classes": 4, "length": 30, "train": 151},
    "HandMovementDirection": {"channels": 10, "classes": 4, "length": 400, "train": 160},
    "FingerMovements":     {"channels": 28, "classes": 2, "length": 50, "train": 316},
    "Heartbeat":           {"channels": 61, "classes": 2, "length": 405, "train": 204},
    # task #48 扩充（中等规模）
    "SelfRegulationSCP1":  {"channels": 6, "classes": 2, "length": 896, "train": 268},
    "SelfRegulationSCP2":  {"channels": 7, "classes": 2, "length": 1152, "train": 200},
    "MotorImagery":        {"channels": 64, "classes": 2, "length": 3000, "train": 278},  # 跳 DTW
    "StandWalkJump":       {"channels": 4, "classes": 3, "length": 2500, "train": 12},   # 跳 DTW
    "DuckDuckGeese":       {"channels": 1345, "classes": 5, "length": 270, "train": 50},
    "LSST":                {"channels": 6, "classes": 14, "length": 36, "train": 2459},  # 大池，subset
}

UEA_SKIP_DTW_IF_LENGTH = 500  # task #48 v2 (2026-05-27): Cricket L=1197 单 cell 16.6h, 降阈值

UEA_MIRROR_BASE = "https://timeseriesclassification.com/aeon-toolkit/{name}.zip"
DATA_DIR = Path(__file__).resolve().parents[1] / "datasets" / "uea"


def _ensure_dataset(name: str) -> Path:
    d = DATA_DIR / name
    train_ts = d / f"{name}_TRAIN.ts"
    test_ts = d / f"{name}_TEST.ts"
    if train_ts.exists() and test_ts.exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    url = UEA_MIRROR_BASE.format(name=name)
    print(f"downloading UEA {name} from {url} ...")
    try:
        zip_path = d / f"{name}.zip"
        urllib.request.urlretrieve(url, str(zip_path))
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(d)
        zip_path.unlink()
        if not train_ts.exists():
            for p in d.rglob(f"{name}_TRAIN*.ts"):
                p.rename(train_ts)
            for p in d.rglob(f"{name}_TEST*.ts"):
                p.rename(test_ts)
        if train_ts.exists():
            return d
    except Exception as e:
        print(f"  download failed: {e!r}")
    raise FileNotFoundError(f"Could not load {name}: missing {train_ts}")


def _parse_ts_file(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """读 sktime .ts 格式 multivariate 序列。
    每行 = "dim1_val1,dim1_val2,...,dim1_valL:dim2_val1,...,dim2_valL:...:label"
    返回 X [N, C, L]、 y [N]
    """
    series_list = []
    labels = []
    in_data = False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('@'):
                if line.lower().startswith('@data'):
                    in_data = True
                continue
            if not in_data:
                continue
            # data line: ch1:ch2:...:label  (or final field = label)
            parts = line.split(':')
            *dim_strs, label = parts
            dims = []
            for ds in dim_strs:
                vals = [float(x) for x in ds.split(',')]
                dims.append(vals)
            x = np.array(dims, dtype=np.float32)
            series_list.append(x)
            labels.append(label)
    # Pad/truncate to common length
    max_l = max(s.shape[1] for s in series_list)
    n_ch = series_list[0].shape[0]
    X = np.zeros((len(series_list), n_ch, max_l), dtype=np.float32)
    for i, s in enumerate(series_list):
        X[i, :, :s.shape[1]] = s
    y = np.array(labels)
    try:
        y = y.astype(int)
    except ValueError:
        try:
            y = y.astype(float).astype(int)
        except ValueError:
            # leave as string
            unique = sorted(set(labels))
            mapping = {l: i for i, l in enumerate(unique)}
            y = np.array([mapping[l] for l in labels], dtype=int)
    return X, y


def load_uea(name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """加载 UEA 数据集，返回 (X_train, y_train, X_test, y_test)，X 形状 [N, C, L]。"""
    d = _ensure_dataset(name)
    X_tr, y_tr = _parse_ts_file(d / f"{name}_TRAIN.ts")
    X_te, y_te = _parse_ts_file(d / f"{name}_TEST.ts")
    return X_tr, y_tr, X_te, y_te


def load_uea_fewshot(name: str, n_per_class: int, seed: int = 1):
    X_tr, y_tr, X_te, y_te = load_uea(name)
    rng = np.random.default_rng(seed)
    classes = np.unique(y_tr)
    pick = []
    for c in classes:
        idx = np.where(y_tr == c)[0]
        if len(idx) <= n_per_class:
            pick.extend(idx.tolist())
        else:
            pick.extend(rng.choice(idx, size=n_per_class, replace=False).tolist())
    pick = np.array(sorted(pick))
    return X_tr[pick], y_tr[pick], X_te, y_te


if __name__ == "__main__":
    import sys
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["BasicMotions", "ERing"]
    for name in targets:
        try:
            X_tr, y_tr, X_te, y_te = load_uea(name)
            print(f"{name}: train {X_tr.shape} (N, C, L) labels {sorted(set(y_tr.tolist()))}; test {X_te.shape}")
        except Exception as e:
            print(f"{name}: FAILED {e}")
