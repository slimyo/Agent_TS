"""P6.3 · UCR Time Series Classification archive loader。

数据来源：http://www.timeseriesclassification.com/aeon-toolkit/Archives/Univariate2018_arff.zip
镜像：https://raw.githubusercontent.com/sktime/sktime-datasets/main/datasets/Univariate_arff/

我们用 sktime 的 .ts 格式 mirror（更便携），或直接读 UCR 官方 .arff。

每个 dataset 目录结构 (假设)：
  research/datasets/ucr/Coffee/Coffee_TRAIN.tsv
  research/datasets/ucr/Coffee/Coffee_TEST.tsv

格式：每行 = "<class_label>\\t<v1>\\t<v2>\\t...<vL>"

少样本 N-shot 抽样：
  load_ucr_fewshot(name, n_per_class, seed) → (X_train, y_train, X_test, y_test)
"""
from __future__ import annotations

import gzip
import io
import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np

UCR_DATASETS_RECOMMENDED = {
    # 少样本友好（Train ≤ 100）
    "Coffee":       {"classes": 2, "length": 286, "train": 28,   "domain": "spectroscopy"},
    "ECG200":       {"classes": 2, "length": 96,  "train": 100,  "domain": "ecg"},
    "GunPoint":     {"classes": 2, "length": 150, "train": 50,   "domain": "motion"},
    "TwoLeadECG":   {"classes": 2, "length": 82,  "train": 23,   "domain": "ecg"},
    "BeetleFly":    {"classes": 2, "length": 512, "train": 20,   "domain": "image-outline"},
    "BirdChicken":  {"classes": 2, "length": 512, "train": 20,   "domain": "image-outline"},
    # 多分类
    "ECG5000":      {"classes": 5, "length": 140, "train": 500,  "domain": "ecg"},
    "Crop":         {"classes": 24,"length": 46,  "train": 7200, "domain": "remote-sensing"},
    # 工业故障
    "Wafer":        {"classes": 2, "length": 152, "train": 1000, "domain": "manufacturing"},
    "FordA":        {"classes": 2, "length": 500, "train": 3601, "domain": "manufacturing"},
    # 谷物 / 农作物 (TSFM 训练外)
    "Strawberry":   {"classes": 2, "length": 235, "train": 613,  "domain": "spectroscopy"},
}

# UCR Univariate 2018 镜像（aeon 维护）
UCR_MIRROR_BASE = "https://timeseriesclassification.com/aeon-toolkit/{name}.zip"
# Backup: sktime / GitHub user mirror
GITHUB_MIRROR_BASE = "https://raw.githubusercontent.com/ChangWeiTan/TS-Extrinsic-Regression/master/data/UCR/{name}/{file}"


DATA_DIR = Path(__file__).resolve().parents[1] / "datasets" / "ucr"


def _ensure_dataset(name: str) -> Path:
    """确保 dataset 文件在本地；缺则下载。返回数据目录路径。"""
    d = DATA_DIR / name
    train_tsv = d / f"{name}_TRAIN.tsv"
    test_tsv = d / f"{name}_TEST.tsv"
    if train_tsv.exists() and test_tsv.exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    # 尝试从 timeseriesclassification.com 下 zip
    url = UCR_MIRROR_BASE.format(name=name)
    print(f"downloading {name} from {url} ...")
    try:
        import zipfile
        zip_path = d / f"{name}.zip"
        urllib.request.urlretrieve(url, str(zip_path))
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(d)
        zip_path.unlink()
        # Some archives extract into subfolder
        if not train_tsv.exists():
            # search recursively
            for p in d.rglob(f"{name}_TRAIN*"):
                if p.suffix in (".tsv", ".txt"):
                    p.rename(train_tsv)
            for p in d.rglob(f"{name}_TEST*"):
                if p.suffix in (".tsv", ".txt"):
                    p.rename(test_tsv)
        if train_tsv.exists():
            return d
    except Exception as e:
        print(f"  primary mirror failed: {e!r}")

    raise FileNotFoundError(
        f"Could not download {name}. Please manually place "
        f"{name}_TRAIN.tsv and {name}_TEST.tsv in {d}/"
    )


def _load_tsv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """读 UCR tsv 格式：第一列 label，剩余列时序值。"""
    rows = []
    labels = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            labels.append(parts[0])
            rows.append([float(x) for x in parts[1:]])
    X = np.array(rows, dtype=np.float32)
    # labels 可能是 int 或 float string；先转 int 若可能
    y = np.array(labels)
    try:
        y = y.astype(int)
    except ValueError:
        try:
            y = y.astype(float).astype(int)
        except ValueError:
            pass
    return X, y


def load_ucr(name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """加载完整 train/test split。
    Returns: (X_train [N_train, L], y_train, X_test, y_test)
    """
    d = _ensure_dataset(name)
    X_tr, y_tr = _load_tsv(d / f"{name}_TRAIN.tsv")
    X_te, y_te = _load_tsv(d / f"{name}_TEST.tsv")
    return X_tr, y_tr, X_te, y_te


def load_ucr_fewshot(name: str, n_per_class: int, seed: int = 1
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """少样本抽样：每 class 取 n_per_class 个训练样本，test 不变。"""
    X_tr, y_tr, X_te, y_te = load_ucr(name)
    rng = np.random.default_rng(seed)
    classes = np.unique(y_tr)
    pick_idx = []
    for c in classes:
        idx = np.where(y_tr == c)[0]
        if len(idx) <= n_per_class:
            pick_idx.extend(idx.tolist())
        else:
            chosen = rng.choice(idx, size=n_per_class, replace=False)
            pick_idx.extend(chosen.tolist())
    pick_idx = np.array(sorted(pick_idx))
    return X_tr[pick_idx], y_tr[pick_idx], X_te, y_te


if __name__ == "__main__":
    # 依次试下载推荐数据集
    import sys
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["Coffee", "ECG200"]
    for name in targets:
        try:
            X_tr, y_tr, X_te, y_te = load_ucr(name)
            print(f"{name}: train {X_tr.shape} labels {sorted(set(y_tr.tolist()))}; "
                  f"test {X_te.shape}")
        except Exception as e:
            print(f"{name}: FAILED {e}")
