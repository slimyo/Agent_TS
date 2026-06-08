"""结构专家 × 真实数据 —— 填补 2×2 的缺格，隔离"池结构 vs 数据真假"这个变量。

2×2：
  合成数据 × 结构专家   = +14%   (m24/run_hetero)
  真实数据 × 通用 TSFM  = −6.6%  (run_realcap，F-R13.5)
  真实数据 × 结构专家   = ???    ← 本实验（决定性）
若 real×结构 为正 → 杠杆确是**池的结构互补性**（与数据真假无关）；
若为负 → 真实少样本数据本身缺可路由结构（瓶颈在数据，不在池）。

做法：真实序列(ETT/ECL/Weather/Exchange/ILI/ETTm) → 多窗少样本切片(N≥50，给足上下文检测 regime)
     → **真实执行** 12 结构专家 → MAE → learned-capability(LODO by dataset，无泄漏) → 在线路由。
输出：test/results_realstruct.jsonl
"""
from __future__ import annotations

import json
import collections
from pathlib import Path

import numpy as np

from test.experts import EXPERTS, BASE, regime_tag
from test.online_router import TrustAwareOnlineRouter

OUT = Path(__file__).resolve().parent
DATASETS = ["ETTh1", "ETTh2", "ECL", "Exchange", "Weather", "ILI", "ETTm1", "ETTm2"]
NS = [50, 100]          # 给足上下文（解 F-R13.5 的少样本退化混淆）
N_WIN = 12              # 每 (ds,N) 切 12 个随机窗 → 足够 cell 做 LODO
H = 24


def build_real_library():
    """真实序列上跑 12 结构专家 → {(ds,N,win): {expert: -MAE/base_MAE}}（goodness，base=−1）。"""
    from research.utils.data_loader import load_series
    from research.utils.splitter import few_shot_split
    cells, tags = {}, {}
    for ds in DATASETS:
        try:
            s, meta = load_series(ds)
        except Exception as e:
            print(f"  skip {ds}: {e!r}", flush=True); continue
        m = meta.season_m
        for N in NS:
            for w in range(N_WIN):
                try:
                    sp = few_shot_split(s, N=N, H=H, seed=1000 + w)
                except Exception:
                    continue
                tr, te = np.asarray(sp.train, float), np.asarray(sp.test, float)
                def mae(e):
                    try:
                        p = np.asarray(EXPERTS[e](tr, np.array([]), H, m), float)[:H]
                        return float(np.mean(np.abs(p - te[:H])))
                    except Exception:
                        return float(np.mean(np.abs(te[:H])))
                bm = mae(BASE)
                if bm <= 0:
                    continue
                cells[(ds, N, w)] = {e: -mae(e) / bm for e in EXPERTS}
                tags[(ds, N, w)] = regime_tag(tr)
    return cells, tags


def learned_cap_LODO(cells, tags, held):
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for k, v in cells.items():
        if k[0] == held:
            continue
        bg = v[BASE]
        for m, g in v.items():
            if m != BASE:
                acc[tags[k]][m].append(g - bg)
    return {t: {m: float(np.mean(r)) for m, r in d.items()} for t, d in acc.items()}


def run(cells, tags, mode="online", seed=0):
    models = list(EXPERTS.keys())
    datasets = sorted({k[0] for k in cells})
    out = []
    for held in datasets:
        cap = learned_cap_LODO(cells, tags, held)
        router = TrustAwareOnlineRouter(models, BASE, cap_prior={m: 0.0 for m in models},
                                        discount=0.9, seed=seed)
        for k in sorted([k for k in cells if k[0] == held], key=lambda x: (x[1], x[2])):
            v = cells[k]; tag = tags[k]
            if mode == "base":
                chosen = BASE
            else:
                router.cap = {m: cap.get(tag, {}).get(m, 0.0) for m in models}
                chosen = router.act(tag, models, topk=5)
            reward = v.get(chosen, v[BASE]) - v[BASE]
            if mode == "online":
                router.update(tag, chosen, reward)
            out.append(reward)
    return float(np.mean(out)) * 100


def main():
    print("=== 结构专家 × 真实数据（填补 2×2 缺格）===", flush=True)
    cells, tags = build_real_library()
    print(f"  {len(cells)} cells / {len(set(k[0] for k in cells))} 真实域 / {len(EXPERTS)} 结构专家", flush=True)
    print(f"  regime 分布: {dict(collections.Counter(tags.values()))}", flush=True)
    summ = {}
    for mode in ["static", "online"]:
        summ[mode] = round(float(np.mean([run(cells, tags, mode, sd) for sd in range(5)])), 2)
    summ["oracle"] = round(float(np.mean([(max(v.values()) - v[BASE]) for v in cells.values()])) * 100, 2)
    print(f"  learned-cap static(只先验)  vs_base = {summ['static']:+.2f}%relMAE", flush=True)
    print(f"  online(学出先验+反馈)        vs_base = {summ['online']:+.2f}%relMAE", flush=True)
    print(f"  oracle-per-cell 上界         vs_base = +{summ['oracle']:.2f}%relMAE", flush=True)
    print("\n2×2 汇总：", flush=True)
    print("  合成×结构 +14% | 真实×通用TSFM −6.6% | **真实×结构 见上** | →", flush=True)
    print("  若真实×结构为正 = 杠杆是'结构互补性'(真数据也成立)；若负 = 真少样本数据缺可路由结构。", flush=True)
    (OUT / "results_realstruct.jsonl").write_text(json.dumps(
        {"summary": summ, "regime_dist": dict(collections.Counter(tags.values())),
         "n_cells": len(cells)}, ensure_ascii=False, indent=2))
    print(f"\nwrote {OUT/'results_realstruct.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
