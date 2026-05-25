"""统一数据 loader。

输入：数据集名 + 目标列；输出：1-D numpy 时序 + 元信息（采样频率、季节周期 m）。
所有原始数据缓存到 research/datasets/raw/，git ignore。
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "datasets" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class SeriesMeta:
    name: str
    freq: str          # pandas freq alias
    season_m: int      # 主季节周期（用于 MASE / Naive seasonal）
    target_col: str


# 数据集元信息表 ----------------------------------------------------------
# 季节周期 m：ETTh* 是 1h 采样，日季节性 → 24；ILI 是周采样，年季节性 → 52。
REGISTRY: dict[str, SeriesMeta] = {
    "ETTh1":    SeriesMeta("ETTh1",    "h", 24, "OT"),
    "ETTh2":    SeriesMeta("ETTh2",    "h", 24, "OT"),
    # ECL (Electricity Consuming Load)：321 客户的小时级用电；取首列 MT_001 单变量
    "ECL":      SeriesMeta("ECL",      "h", 24, "MT_001"),
    # Exchange Rate：8 国汇率日采样；取首列（美元基准）作单变量
    "Exchange": SeriesMeta("Exchange", "D",  7, "rate_0"),
    # Weather：52,696 行 × 22 列，10min 采样；目标列 OT；日季节性 m = 24*6 = 144
    "Weather":  SeriesMeta("Weather",  "10min", 144, "OT"),
    # ILI（national_illness）：966 行周采样，目标列 OT，年季节性 m=52
    "ILI":      SeriesMeta("ILI",      "W", 52, "OT"),
}

URLS: dict[str, str] = {
    "ETTh1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "ETTh2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
    "ECL":      "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/electricity/electricity.txt.gz",
    "Exchange": "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/exchange_rate/exchange_rate.txt.gz",
    # Weather / ILI 走人工放置（公共镜像失效，2026-05 手动从 M4_ILI_Weather.zip 解压）
}


def _download(name: str) -> Path:
    """优先用本地已有文件（csv/txt.gz）；若没有且 name 在 URLS 里则下载。"""
    for suffix in (".csv", ".txt.gz"):
        fp = RAW_DIR / f"{name}{suffix}"
        if fp.exists():
            return fp
    if name not in URLS:
        raise FileNotFoundError(
            f"{name} 没有自动下载源，请把数据放到 {RAW_DIR}/{name}.csv 或 {name}.txt.gz"
        )
    url = URLS[name]
    suffix = ".txt.gz" if url.endswith(".txt.gz") else ".csv"
    fp = RAW_DIR / f"{name}{suffix}"
    print(f"[data_loader] downloading {name} from {url}")
    urllib.request.urlretrieve(url, fp)
    return fp


def _load_txt_gz(fp: Path, col_idx: int = 0) -> np.ndarray:
    """laiguokun mirror 格式：gz 压缩的逗号分隔数值矩阵，无表头。"""
    import gzip
    arr_rows = []
    with gzip.open(fp, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            arr_rows.append(float(line.split(",")[col_idx]))
    return np.asarray(arr_rows, dtype=np.float64)


def load_series(name: str, target_col: str | None = None) -> tuple[np.ndarray, SeriesMeta]:
    """加载单变量序列，返回 (ndarray, meta)。"""
    if name not in REGISTRY:
        raise KeyError(f"unknown dataset {name}; known={list(REGISTRY)}")
    meta = REGISTRY[name]
    target = target_col or meta.target_col
    fp = _download(name)
    if fp.suffix == ".gz":
        # laiguokun txt.gz：列名 MT_001 / rate_0 → 索引解析
        if target.startswith("MT_"):
            col_idx = int(target.split("_")[1]) - 1   # MT_001 → 0
        elif target.startswith("rate_"):
            col_idx = int(target.split("_")[1])
        else:
            col_idx = 0
        arr = _load_txt_gz(fp, col_idx=col_idx)
    else:
        df = pd.read_csv(fp)
        if target not in df.columns:
            target = df.columns[-1]
        arr = df[target].to_numpy(dtype=np.float64)
    return arr, meta


if __name__ == "__main__":
    for n in REGISTRY:
        s, m = load_series(n)
        print(f"{n}: len={len(s)}, mean={s.mean():.3f}, std={s.std():.3f}, meta={m}")
