"""全量测试 v2 —— 双信号解耦决策 + 全量模型库(10) + 全量数据集(22, LODO)。

彻底改进版（research Round 12 / F-R12.5）：决策从"单 CV margin/trust"升级为
"conformal_safe AND saturation_headroom"双信号解耦门。在 oracle 库（22 数据集 × 10 分类器）
上做全量 LODO，对比 base / oracle，并扫两个阈值看损害控制 vs 获利的权衡。

用法：python -m test.run_full_test_v2
输出：test/results_signal_router.jsonl + 控制台摘要。
"""
from __future__ import annotations

import json
from pathlib import Path

from test.signal_router import (
    domain_of, load_oracle_library, run_signal_router, summarize,
)

OUT = Path(__file__).resolve().parent


def main():
    cells = load_oracle_library()
    nclf = len({c for v in cells.values() for c in v})
    print(f"=== 全量库：{len(cells)} cells / {len(set(k[0] for k in cells))} datasets / {nclf} classifiers ===",
          flush=True)

    ndom = {}
    for k in cells:
        ndom[domain_of(k[0])] = ndom.get(domain_of(k[0]), 0) + 1
    print(f"    域分布: {ndom}", flush=True)

    # 主配置：F-R12.5 双信号门（trust≥0.5 避险 AND headroom≥0.5 获利）
    print("\n--- 主决策：双信号解耦 (trust≥0.5 AND headroom≥0.5) ---", flush=True)
    recs = run_signal_router(cells, trust_tau=0.5, head_tau=0.5)
    s = summarize(recs)
    for k, v in s.items():
        print(f"  {k:22}: {v}", flush=True)
    # 分域看（UCR 单变量 / UEA 多变量）
    print("  --- 分域 ---", flush=True)
    for dom in ("UCR", "UEA"):
        sub = [r for r in recs if domain_of(r["ds"]) == dom]
        if sub:
            ss = summarize(sub)
            print(f"  [{dom}] sys={ss['system_acc']}% base={ss['base_acc']}% "
                  f"vs_base={ss['system_vs_base_pp']:+.2f}pp dev={ss['n_deviations']} "
                  f"n={ss['n_cells']}/{ss['n_datasets']}ds", flush=True)

    # 阈值扫描：展示"避险 vs 获利"权衡（reviewer 关心的 risk-coverage）
    print("\n--- 阈值扫描（trust_tau, head_tau）→ vs_base / 偏离数 / safe-dev ---", flush=True)
    grid = []
    for tt in [0.0, 0.5, 0.7]:
        for ht in [0.0, 0.5, 0.7]:
            r = run_signal_router(cells, trust_tau=tt, head_tau=ht)
            ss = summarize(r)
            grid.append({"trust_tau": tt, "head_tau": ht, **ss})
            print(f"  trust≥{tt} head≥{ht}: vs_base={ss['system_vs_base_pp']:+.2f}pp "
                  f"dev={ss['n_deviations']} safe={ss['safe_deviation_rate']} "
                  f"sys={ss['system_acc']}%", flush=True)

    # ---- 激进策略对比（不追求 pp，看 oracle-capture / 命中率）----
    # 按 method 结论：oracle 头寸真实(5.95pp)、winner 不可预测 → 保守门收敛守 base(+0.00pp)。
    # 这里展示"主动出手"的激进策略：明知逐-cell 抓不准也去抓，价值看捕获/命中而非 +pp。
    print("\n--- 激进策略对比（不追求 pp）：vs_base / 偏离率 / oracle命中 / 头寸捕获 ---", flush=True)
    aggr = {}
    configs = [
        ("conservative(双信号门)", dict(policy="conservative", trust_tau=0.5, head_tau=0.5)),
        ("aggressive(头寸≥0.5)", dict(policy="aggressive", head_tau=0.5)),
        ("aggressive(头寸≥0.3)★", dict(policy="aggressive", head_tau=0.3)),
        ("aggressive(头寸≥0.15)", dict(policy="aggressive", head_tau=0.15)),
        ("aggressive-belief(始终偏离)", dict(policy="aggressive-belief")),
    ]
    aggr_recs = {}
    for name, kw in configs:
        r = run_signal_router(cells, **kw)
        ss = summarize(r)
        aggr[name] = ss; aggr_recs[name] = r
        print(f"  {name:30} vs_base={ss['system_vs_base_pp']:+.2f}pp  dev率={ss['deviation_rate']:.2f}  "
              f"oracle命中={ss['oracle_hit_rate']}  头寸捕获={ss['headroom_capture_rate']}  "
              f"安全偏离={ss['safe_deviation_rate']}", flush=True)
    print(f"  [参照] oracle 上界 vs_base = +{s['oracle_headroom_pp']:.2f}pp（全部偏离且每格选对才能达到）", flush=True)
    print("  读法（method F-R12.x）：存在一个**激进甜区**——丢掉 conformal 避险门、只在高头寸(≥0.3)格主动出手，"
          "\n        净 +0.15pp（偏离 7%、安全率 0.69），是保守门(+0.00)拿不到的；但更激进(头寸≥0.15→washes out，"
          "\n        始终偏离→−2.46pp)就崩——因 winner 不可预测，激进收益薄且对阈值敏感。"
          "\n        '不追求pp'的真义：用激进换 oracle 头寸的主动覆盖(命中率/捕获率)，pp 只是副产物。", flush=True)

    out = OUT / "results_signal_router.jsonl"
    with out.open("w") as fh:
        fh.write(json.dumps({"_summary_main": s}, ensure_ascii=False) + "\n")
        fh.write(json.dumps({"_grid": grid}, ensure_ascii=False) + "\n")
        fh.write(json.dumps({"_aggressive": aggr}, ensure_ascii=False) + "\n")
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
