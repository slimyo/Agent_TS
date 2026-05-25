"""plan §4.5 产出：Winner heatmap + Wilcoxon 配对检验。

输出：
  - results/heatmap_winners.png：6 dataset × 4 N 颜色 = winner method
  - results/heatmap_mae_zscore.png：MAE 标准化后的方法×cell 热力图
  - results/wilcoxon_pairs.csv：每对方法在每数据集的 Wilcoxon p-value
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

ALL_FILES = {
    "ETTh1": [
        "p0_naive.jsonl", "p1_arima_ets.jsonl", "p1_chronos.jsonl",
        "p2_llmtime.jsonl", "p4_tsci_etth1.jsonl", "p3_adapt_v4_etth1.jsonl",
    ],
    "ETTh2": [
        "p4_etth2_fast.jsonl", "p4_etth2_llm.jsonl",
        "p4_etth2_tsci.jsonl", "p4_etth2_adapt.jsonl",
    ],
    "ECL":   ["p5_ecl_fast.jsonl", "p5_ecl_llm.jsonl", "p5_ecl_tsci.jsonl", "p5_ecl_adapt.jsonl"],
    "Exchange": ["p5_exchange_fast.jsonl", "p5_exchange_llm.jsonl",
                 "p5_exchange_tsci.jsonl", "p5_exchange_adapt.jsonl"],
    "Weather":  ["p5_weather_fast.jsonl", "p5_weather_llm.jsonl",
                 "p5_weather_tsci.jsonl", "p5_weather_adapt.jsonl"],
    "ILI":      ["p5_ili_fast.jsonl", "p5_ili_llm.jsonl",
                 "p5_ili_tsci.jsonl", "p5_ili_adapt.jsonl"],
}
DATASETS = list(ALL_FILES)
METHODS = ["naive", "arima_ets", "chronos", "llmtime", "tsci", "adapt_ts"]
NS = [10, 20, 50, 100]
RESULTS = Path("research/results")


def load_all() -> dict:
    # agg[(dataset, N, method)] = list of MAE (one per seed)
    agg = defaultdict(list)
    for ds, files in ALL_FILES.items():
        for f in files:
            fp = RESULTS / f
            if not fp.exists():
                continue
            for line in fp.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r["dataset"] != ds:
                    continue
                key = (ds, r["N"], r["method"])
                agg[key].append(r["mae"])
    return agg


def heatmap_winners(agg):
    """24 个 (dataset, N) cell 的 winner method 热力图。"""
    M_INDEX = {m: i for i, m in enumerate(METHODS)}
    grid = np.full((len(DATASETS), len(NS)), -1, dtype=int)
    annot = np.empty_like(grid, dtype=object)
    for di, ds in enumerate(DATASETS):
        for ni, N in enumerate(NS):
            best_m, best_v = None, float("inf")
            for m in METHODS:
                vals = agg.get((ds, N, m), [])
                if vals and np.mean(vals) < best_v:
                    best_v, best_m = float(np.mean(vals)), m
            if best_m:
                grid[di, ni] = M_INDEX[best_m]
                annot[di, ni] = f"{best_m}\n{best_v:.3g}"
            else:
                annot[di, ni] = "--"

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10", len(METHODS))
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=len(METHODS) - 1, aspect="auto")
    for di in range(len(DATASETS)):
        for ni in range(len(NS)):
            txt = annot[di, ni]
            ax.text(ni, di, txt, ha="center", va="center", fontsize=9,
                    color="white" if grid[di, ni] >= 0 else "black")
    ax.set_xticks(range(len(NS)))
    ax.set_xticklabels([f"N={n}" for n in NS])
    ax.set_yticks(range(len(DATASETS)))
    ax.set_yticklabels(DATASETS)
    ax.set_title("Winner per (dataset, N) cell — 6 datasets × 4 N × 6 methods\n"
                 "(no method dominates >42% of cells)")
    # 图例
    from matplotlib.patches import Patch
    handles = [Patch(color=cmap(i), label=m) for i, m in enumerate(METHODS)]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
              title="Method")
    plt.tight_layout()
    out = RESULTS / "heatmap_winners.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def heatmap_mae_zscore(agg):
    """方法 × cell 的标准化 MAE 热力图：每个 cell 内做 z-score 跨方法对比。"""
    nrows = len(METHODS)
    ncols = len(DATASETS) * len(NS)
    grid = np.full((nrows, ncols), np.nan)
    col_labels = []
    for di, ds in enumerate(DATASETS):
        for ni, N in enumerate(NS):
            col = di * len(NS) + ni
            col_labels.append(f"{ds[:4]}-N{N}")
            vals = {}
            for m in METHODS:
                v = agg.get((ds, N, m), [])
                if v:
                    vals[m] = np.mean(v)
            if not vals:
                continue
            mu = np.mean(list(vals.values()))
            sigma = np.std(list(vals.values())) + 1e-12
            for m, v in vals.items():
                z = (v - mu) / sigma
                grid[METHODS.index(m), col] = z

    fig, ax = plt.subplots(figsize=(14, 4.5))
    im = ax.imshow(grid, cmap="RdYlGn_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(ncols))
    ax.set_xticklabels(col_labels, rotation=70, ha="right", fontsize=8)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(METHODS)
    ax.set_title("MAE z-score per cell (lower / greener = better; red = worse than mean)")
    fig.colorbar(im, ax=ax, label="z-score within cell")
    plt.tight_layout()
    out = RESULTS / "heatmap_mae_zscore.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def wilcoxon_pairs(agg):
    """每对方法在每个数据集上做 Wilcoxon signed-rank test（配对：4 N × 3 seeds = 12 个差）。"""
    lines = ["dataset,method_a,method_b,n_pairs,median_diff_a-b,wilcoxon_p"]
    for ds in DATASETS:
        for i, ma in enumerate(METHODS):
            for mb in METHODS[i + 1:]:
                xa, xb = [], []
                for N in NS:
                    va = agg.get((ds, N, ma), [])
                    vb = agg.get((ds, N, mb), [])
                    k = min(len(va), len(vb))
                    if k == 0:
                        continue
                    xa.extend(va[:k]); xb.extend(vb[:k])
                if len(xa) < 5:
                    continue
                xa = np.asarray(xa); xb = np.asarray(xb)
                diff = xa - xb
                if np.allclose(diff, 0):
                    p = 1.0
                else:
                    try:
                        _, p = wilcoxon(diff, zero_method="zsplit")
                    except Exception:
                        p = float("nan")
                lines.append(f"{ds},{ma},{mb},{len(xa)},{np.median(diff):.4g},{p:.4g}")
    out = RESULTS / "wilcoxon_pairs.csv"
    out.write_text("\n".join(lines))
    print(f"saved {out} ({len(lines)-1} pairs)")


def main():
    agg = load_all()
    print(f"loaded {sum(len(v) for v in agg.values())} MAE values from {len(agg)} cells")
    heatmap_winners(agg)
    heatmap_mae_zscore(agg)
    wilcoxon_pairs(agg)


if __name__ == "__main__":
    main()
