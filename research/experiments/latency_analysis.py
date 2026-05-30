"""task #61 / P0-G · Aggregate wall_time per method/cell across all sweeps.

输出 table 答 feedback 实验 #3："几十倍复杂度 +0.89pp 是否值得？"
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_jsonl(p):
    if not Path(p).exists(): return []
    out = []
    for line in open(p):
        try: out.append(json.loads(line))
        except Exception: pass
    return out


def main():
    files = {
        "Forecasting": [
            "research/results/f4_chronos2.jsonl",
            "research/results/f4_chronos_bolt.jsonl",
            "research/results/p9_adapt_v9.jsonl",
            "research/results/p11_phaseA_populate.jsonl",
            "research/results/p12_adapt_v12.jsonl",
            "research/results/p13_phaseB.jsonl",
        ],
        "TSC": [
            "research/results/taskb_ucr.jsonl",
            "research/results/taskb_router_ucr.jsonl",
            "research/results/taskb_router_v3_ucr.jsonl",
        ],
        "UEA": ["research/results/taskb_uea_full.jsonl"],
    }

    print(f"{'Domain':12} {'Method':22} {'mean(s)':>10} {'median':>10} {'min':>8} {'max':>8} {'n':>5}")
    print("-" * 78)

    domain_summary = {}
    for domain, fps in files.items():
        rows = []
        for fp in fps:
            rows.extend(load_jsonl(fp))
        by_method = defaultdict(list)
        for r in rows:
            m = r.get("method")
            wt = r.get("wall_time")
            if m and wt is not None:
                by_method[m].append(wt)
        domain_summary[domain] = by_method
        for m in sorted(by_method):
            vals = by_method[m]
            print(f"{domain:12} {m:22} {np.mean(vals):>10.2f} {np.median(vals):>10.2f} "
                  f"{min(vals):>8.2f} {max(vals):>8.2f} {len(vals):>5}")
        print()

    # Save as JSON
    out = {}
    for d, by_m in domain_summary.items():
        out[d] = {m: {"mean_s": float(np.mean(v)), "median_s": float(np.median(v)),
                       "min_s": float(min(v)), "max_s": float(max(v)),
                       "n": len(v)}
                   for m, v in by_m.items()}
    with open("research/results/latency_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved → research/results/latency_summary.json")


if __name__ == "__main__":
    main()
