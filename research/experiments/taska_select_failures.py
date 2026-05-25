"""P6.1 · 从现有 forecasting 结果筛选 catastrophic failure cells，自动标 ground-truth。

输入：所有 v10/v11/v12 + Chronos-2 baseline jsonl
输出：research/results/taska_failures.jsonl
   每行 = {dataset, N, seed, H, method, mae, chronos2_mae, mae_ratio,
           ground_truth: {primary_fault, secondary_faults, scores}}
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from research.utils.data_loader import load_series
from research.utils.fault_taxonomy import assign_ground_truth
from research.utils.splitter import few_shot_split


def load_rows(p):
    if not Path(p).exists():
        return []
    return [json.loads(l) for l in open(p)]


METHOD_FILES = {
    "v10":  ["research/results/p10_adapt_v10_n10.jsonl",
             "research/results/p9_adapt_v9.jsonl"],
    "v11":  ["research/results/p11_phaseA_populate.jsonl",
             "research/results/p11_adapt_weather_ili.jsonl"],
    "v12":  ["research/results/p12_adapt_v12.jsonl",
             "research/results/p12_adapt_weather_ili.jsonl"],
    "chronos2": ["research/results/f4_chronos2.jsonl",
                 "research/results/f4_bolt_c2_ecl_exchange.jsonl",
                 "research/results/f4_bolt_c2_weather.jsonl",
                 "research/results/f4_bolt_c2_ili.jsonl"],
}


def build_lookup(files, method_filter=None):
    """(dataset, N, seed) -> row."""
    d = {}
    for f in files:
        for r in load_rows(f):
            if method_filter and r.get("method") != method_filter:
                continue
            if "N" in r and "seed" in r:
                if r["N"] == 10:
                    pass  # keep
            key = (r["dataset"], r["N"], r["seed"])
            d[key] = r
    return d


def main():
    out = Path("research/results/taska_failures.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Build Chronos-2 baseline lookup
    c2_lookup = build_lookup(METHOD_FILES["chronos2"], "chronos2")
    print(f"Chronos-2 cells: {len(c2_lookup)}")

    # Build AdaptTS variant lookups
    variant_lookups = {}
    for v in ["v10", "v11", "v12"]:
        variant_lookups[v] = build_lookup(METHOD_FILES[v])

    # Compute mae_ratio = adapt_mae / chronos2_mae for each (variant, cell)
    # 选 ratio 最高的 30 个 cells（最 catastrophic）
    candidates = []
    seen_cells = set()
    for v, lookup in variant_lookups.items():
        for key, r in lookup.items():
            if key not in c2_lookup:
                continue
            c2_mae = c2_lookup[key]["mae"]
            adapt_mae = r["mae"]
            if c2_mae < 1e-9:
                continue
            ratio = adapt_mae / c2_mae
            candidates.append({
                "variant": v, "dataset": key[0], "N": key[1], "seed": key[2],
                "H": r["H"], "adapt_mae": adapt_mae, "chronos2_mae": c2_mae,
                "mae_ratio": ratio,
            })

    # Sort by ratio descending, keep top 50, then dedup by (dataset, N, seed) keeping highest variant
    candidates.sort(key=lambda x: -x["mae_ratio"])
    dedup = {}
    for c in candidates:
        key = (c["dataset"], c["N"], c["seed"])
        if key not in dedup:
            dedup[key] = c
    top30 = list(dedup.values())[:30]

    print(f"\nTop-30 catastrophic failure cells (by mae/chronos2_mae ratio):")
    print(f'{"variant":7} {"dataset":9} {"N":>4} {"seed":>5}  {"adapt_mae":>10}  {"c2_mae":>10}  ratio')

    # Assign ground truth for each
    n_written = 0
    with out.open("w") as fh:
        for c in top30:
            series, meta = load_series(c["dataset"])
            sp = few_shot_split(series, N=c["N"], H=c["H"], seed=c["seed"])
            gt = assign_ground_truth(sp.train, sp.test, season_m=meta.season_m)
            row = {**c, "season_m": meta.season_m, "ground_truth": gt}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1
            print(f'{c["variant"]:7} {c["dataset"]:9} {c["N"]:4} {c["seed"]:5}  '
                  f'{c["adapt_mae"]:10.4f}  {c["chronos2_mae"]:10.4f}  '
                  f'{c["mae_ratio"]:5.2f}  ←  primary={gt["primary_fault"]} '
                  f'sec={gt["secondary_faults"]}')

    print(f"\nwrote {n_written} cells to {out}")
    # Distribution of ground truth
    from collections import Counter
    primary_counts = Counter()
    for c in top30:
        series, meta = load_series(c["dataset"])
        sp = few_shot_split(series, N=c["N"], H=c["H"], seed=c["seed"])
        gt = assign_ground_truth(sp.train, sp.test, season_m=meta.season_m)
        primary_counts[gt["primary_fault"]] += 1
    print(f"\nPrimary fault distribution: {dict(primary_counts)}")


if __name__ == "__main__":
    main()
