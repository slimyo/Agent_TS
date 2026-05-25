"""F1 · E2 置信度校准 CMR + Oracle 上界（plan §5.2 主结果表）。

对每个 (dataset, N, seed) cell：
  1. 重跑 curator_uq.diagnose 拿三路置信度（stat / llm / xc）
  2. 配上现有 AdaptTS 主实验的 test MAE
  3. 构建"置信度-误差"配对数据
  4. 对每路 conf 算 CMR
  5. 计算 Oracle 上界：用 test MAE 分位数倒推"理想三档"

输出：
  - results/e2_pairs.jsonl: 每条记录 = {dataset, N, seed, conf_stat/llm/xc × trend/season/stat, test_mae}
  - results/e2_cmr_table.csv: 5 路 × 3 维度的 CMR 表
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from research.agent.curator_uq import diagnose
from research.utils.data_loader import load_series
from research.utils.splitter import few_shot_split


ALL_FILES = {
    "ETTh1": "research/results/p3_adapt_v4_etth1.jsonl",
    "ETTh2": "research/results/p4_etth2_adapt.jsonl",
    "ECL":   "research/results/p5_ecl_adapt.jsonl",
    "Exchange": "research/results/p5_exchange_adapt.jsonl",
    "Weather":  "research/results/p5_weather_adapt.jsonl",
    "ILI":      "research/results/p5_ili_adapt.jsonl",
}
DATASETS = list(ALL_FILES)
NS = [10, 20, 50, 100]
SEEDS = [1, 42, 123]
CONF_ORDER = {"low": 0, "mid": 1, "high": 2}
DIMS = ["trend", "season", "stat"]
SOURCES = ["stat", "llm", "xc"]


def load_adapt_mae() -> dict:
    """(ds, N, seed) -> test_mae"""
    out = {}
    for ds, f in ALL_FILES.items():
        if not Path(f).exists():
            continue
        for line in Path(f).read_text().splitlines():
            r = json.loads(line)
            out[(ds, r["N"], r["seed"])] = r["mae"]
    return out


def collect_diagnoses(mae_lookup: dict) -> list[dict]:
    """对每 cell 重跑 diagnose（curator_uq 调 LLM 但走 cache，秒级）。"""
    pairs = []
    for ds in DATASETS:
        try:
            series, meta = load_series(ds)
        except Exception as e:
            print(f"skip {ds}: {e}")
            continue
        # ILI H=24 其他 H=96，切割窗口需对齐
        H = 24 if ds == "ILI" else 96
        for N in NS:
            for seed in SEEDS:
                if (ds, N, seed) not in mae_lookup:
                    continue
                try:
                    sp = few_shot_split(series, N=N, H=H, seed=seed)
                    diag = diagnose(sp.train, season_m=meta.season_m)
                    pair = {
                        "dataset": ds, "N": N, "seed": seed,
                        "test_mae": mae_lookup[(ds, N, seed)],
                        "trend_stat": diag.trend_conf_stat,
                        "season_stat": diag.season_conf_stat,
                        "stat_stat":  diag.stat_conf_stat,
                        "trend_llm":  diag.trend_conf_llm,
                        "season_llm": diag.season_conf_llm,
                        "stat_llm":   diag.stat_conf_llm,
                        "trend_xc":   diag.trend_conf_xc,
                        "season_xc":  diag.season_conf_xc,
                        "stat_xc":    diag.stat_conf_xc,
                    }
                    pairs.append(pair)
                except Exception as e:
                    print(f"  {ds} N={N} seed={seed} fail: {e}")
    return pairs


def zscore_per_dataset(pairs: list[dict]) -> None:
    """in-place 标准化：每个数据集 z-score MAE → 跨数据集可比。"""
    by_ds = defaultdict(list)
    for p in pairs: by_ds[p["dataset"]].append(p)
    for ds, ps in by_ds.items():
        maes = np.array([p["test_mae"] for p in ps], dtype=np.float64)
        mu, sigma = float(np.mean(maes)), float(np.std(maes)) + 1e-12
        for p in ps:
            p["mae_z"] = (p["test_mae"] - mu) / sigma


def bucket_mae(pairs: list[dict], dim: str, source: str, use_zscore: bool = True) -> dict[str, list[float]]:
    """{conf_label: [mae_or_zscore...]}; conf_label ∈ {low,mid,high}"""
    key = f"{dim}_{source}"
    metric_key = "mae_z" if use_zscore else "test_mae"
    bucket = defaultdict(list)
    for p in pairs:
        c = p[key]
        bucket[c].append(p[metric_key])
    return dict(bucket)


def cmr_from_buckets(bucket: dict[str, list[float]]) -> float:
    """CMR = (满足相邻"高 MAE < 中 < 低 MAE"的对数) / 总相邻对数。
    阶 = low(0) → mid(1) → high(2)；高 conf 应对应低 MAE。"""
    sizes = {c: len(bucket.get(c, [])) for c in ["low", "mid", "high"]}
    # 只在至少有 2 个非空桶时计算
    present = [c for c in ["low", "mid", "high"] if sizes[c] >= 1]
    if len(present) < 2:
        return float("nan")
    means = {c: float(np.mean(bucket[c])) for c in present}
    # 相邻桶对：low<mid, mid<high, low<high；
    pairs_checked = []
    pairs_satisfied = 0
    order = ["low", "mid", "high"]
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            a, b = order[i], order[j]
            if a in means and b in means:
                pairs_checked.append((a, b))
                # 期望 mean[a] > mean[b]（低 conf 应有更大 MAE）
                if means[a] > means[b]:
                    pairs_satisfied += 1
    if not pairs_checked:
        return float("nan")
    return pairs_satisfied / len(pairs_checked)


def oracle_label(pairs: list[dict]) -> list[str]:
    """按 mae_z（per-dataset 标准化后）三分位赋"理想 conf"标签。"""
    zs = [p["mae_z"] for p in pairs]
    q33, q67 = np.percentile(zs, [33.33, 66.67])
    labels = []
    for p in pairs:
        m = p["mae_z"]
        if m <= q33: labels.append("high")
        elif m <= q67: labels.append("mid")
        else: labels.append("low")
    return labels


def main():
    print("loading adapt_ts test MAEs ...")
    mae_lookup = load_adapt_mae()
    print(f"  {len(mae_lookup)} adapt cells available")

    print("collecting diagnoses for all cells ...")
    pairs = collect_diagnoses(mae_lookup)
    print(f"  {len(pairs)} (cell, diagnosis) pairs")

    # 标准化（per-dataset z-score）后再算 CMR，避免跨数据集量纲污染
    zscore_per_dataset(pairs)

    # 落 jsonl
    fp = Path("research/results/e2_pairs.jsonl")
    fp.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in pairs))
    print(f"saved {fp}")

    # CMR 主表：4 行（stat/llm/xc/Oracle）× 3 列（trend/season/stat dim）+ avg
    rows = []
    rows.append("source,trend_CMR,season_CMR,stat_CMR,avg_CMR,n_pairs")
    for src in SOURCES:
        cmrs = []
        for dim in DIMS:
            bucket = bucket_mae(pairs, dim, src)
            cmrs.append(cmr_from_buckets(bucket))
        avg = float(np.nanmean(cmrs))
        rows.append(f"{src},{cmrs[0]:.3f},{cmrs[1]:.3f},{cmrs[2]:.3f},{avg:.3f},{len(pairs)}")

    # Oracle 上界：用 test MAE 分位数生成 "理想 conf" 后计算 CMR（应 = 1.0）
    oracle_labels = oracle_label(pairs)
    for dim in DIMS:
        bucket = defaultdict(list)
        for p, lbl in zip(pairs, oracle_labels):
            bucket[lbl].append(p["test_mae"])
        cmr_o = cmr_from_buckets(dict(bucket))
        # oracle 各维度独立赋值方式都一样，cmr 相同
    bucket = defaultdict(list)
    for p, lbl in zip(pairs, oracle_labels):
        bucket[lbl].append(p["test_mae"])
    oracle_cmr = cmr_from_buckets(dict(bucket))
    rows.append(f"oracle,{oracle_cmr:.3f},{oracle_cmr:.3f},{oracle_cmr:.3f},{oracle_cmr:.3f},{len(pairs)}")

    out = Path("research/results/e2_cmr_table.csv")
    out.write_text("\n".join(rows))
    print(f"\nsaved {out}")
    print("\n=== E2 CMR 主表 ===")
    for r in rows:
        print("  " + r)

    # 桶内统计：每 source × dim × conf 的 (n, mean_MAE)
    print("\n=== Per-bucket MAE statistics (xc source only) ===")
    for dim in DIMS:
        bucket = bucket_mae(pairs, dim, "xc")
        print(f"  dim={dim}:")
        for c in ["low", "mid", "high"]:
            v = bucket.get(c, [])
            if v:
                print(f"    {c:>4}: n={len(v):>3}, mean_MAE={np.mean(v):>10.3f}, median={np.median(v):>10.3f}")


if __name__ == "__main__":
    main()
