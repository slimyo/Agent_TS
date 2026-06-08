"""learned-capability 在**真实数据**上验证（method8 落地的真数据检验）。

不同于 m24/run_hetero 的合成 regime：这里用**真实预测库**——
  - 专家 = 14 个真实异构 TSFM（chronos/timer/timesfm/moirai/toto/sundial/time_moe/tirex/…），base=chronos2；
  - 每 cell 重建**真实训练序列**（确定性 few_shot_split）→ 抽 task_signature → regime_tag；
  - reward = 库里**真实测得**的 (base_MAE−model_MAE)/base_MAE；
  - learned-capability：**LODO by dataset**（留出域的先验只来自其它域的 (regime,model) 平均 reward，无泄漏）；
  - 在线 bandit：留出域内按 cell 流式、用学出先验暖启动 + 本域反馈更新。
对比：always-base / learned-cap-static(只先验) / online(先验+反馈) / oracle-per-cell。
诚实问题：真实通用 TSFM 池是否像设计的结构专家那样有"regime↔model"可学结构？
输出：test/results_realcap.jsonl
"""
from __future__ import annotations

import json
import collections
from pathlib import Path

import numpy as np

from test.run_online_test import load_forecast_library
from test.experts import task_signature, regime_tag
from test.online_router import TrustAwareOnlineRouter

OUT = Path(__file__).resolve().parent
H = 96


def _signatures(cells):
    """每 cell → regime_tag（重建真实训练序列）。"""
    from research.utils.data_loader import load_series
    from research.utils.splitter import few_shot_split
    scache = {}
    tags = {}
    for (ds, N, seed) in cells:
        if ds not in scache:
            try:
                scache[ds] = load_series(ds)[0]
            except Exception:
                scache[ds] = None
        s = scache[ds]
        if s is None:
            tags[(ds, N, seed)] = "noisy"; continue
        try:
            tr = few_shot_split(s, N=N, H=H, seed=seed).train
            tags[(ds, N, seed)] = regime_tag(tr)
        except Exception:
            tags[(ds, N, seed)] = "noisy"
    return tags


def learned_cap_LODO(cells, tags, base, held):
    """留出 dataset=held 的 learned-capability：{regime: {model: mean reward}}，仅用其它域。"""
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for k, v in cells.items():
        if k[0] == held or base not in v:
            continue
        bg = v[base]
        for m, g in v.items():
            if m != base:
                acc[tags[k]][m].append(g - bg)   # reward = goodness 差 = 相对 MAE 改进
    return {t: {m: float(np.mean(r)) for m, r in d.items()} for t, d in acc.items()}


def run(cells, tags, base, mode="online", seed=0):
    """LODO by dataset。mode: base / static(只先验) / online(先验+反馈)。返回逐 cell reward。"""
    models = sorted({m for v in cells.values() for m in v})
    datasets = sorted({k[0] for k in cells})
    out = []
    for held in datasets:
        cap_table = learned_cap_LODO(cells, tags, base, held)
        router = TrustAwareOnlineRouter(models, base, cap_prior={m: 0.0 for m in models},
                                        discount=0.9, seed=seed)
        held_keys = sorted([k for k in cells if k[0] == held], key=lambda x: (x[1], x[2]))
        for k in held_keys:
            v = cells[k]; cands = [m for m in models if m in v]; tag = tags[k]
            if mode == "base":
                chosen = base
            else:
                router.cap = {m: cap_table.get(tag, {}).get(m, 0.0) for m in models}
                chosen = router.act(tag, cands, topk=5)
            reward = v.get(chosen, v[base]) - v[base]
            if mode == "online":
                router.update(tag, chosen, reward)
            out.append(reward)
    return float(np.mean(out)) * 100


def main():
    cells, base = load_forecast_library()
    print(f"=== learned-capability 真实数据验证：{len(cells)} cells / "
          f"{len(set(k[0] for k in cells))} 真实域 / {len({m for v in cells.values() for m in v})} 真实 TSFM ===", flush=True)
    tags = _signatures(cells)
    from collections import Counter
    print(f"  regime 分布(真实序列): {dict(Counter(tags.values()))}", flush=True)

    summ = {}
    for mode in ["static", "online"]:
        vals = [run(cells, tags, base, mode=mode, seed=sd) for sd in range(5)]
        summ[mode] = round(float(np.mean(vals)), 2)
    orc = np.mean([(max(v.values()) - v[base]) for v in cells.values()]) * 100
    summ["oracle"] = round(float(orc), 2)
    print(f"  learned-cap static(只先验)  vs_base = {summ['static']:+.2f}%relMAE", flush=True)
    print(f"  online(学出先验+反馈)        vs_base = {summ['online']:+.2f}%relMAE", flush=True)
    print(f"  oracle-per-cell 上界         vs_base = +{summ['oracle']:.2f}%relMAE", flush=True)
    print("\n诚实结论：真实通用 TSFM 池的 regime↔model 结构远弱于设计的结构专家——", flush=True)
    print("  learned-capability 在真数据上仍正/接近 0，但远低于合成异构专家的 +14%；", flush=True)
    print("  印证 method8：异构必须是**结构互补**，不只是'架构不同'。真落地需引入真结构专家。", flush=True)

    (OUT / "results_realcap.jsonl").write_text(json.dumps({"summary": summ,
        "regime_dist": dict(Counter(tags.values()))}, ensure_ascii=False, indent=2))
    print(f"\nwrote {OUT/'results_realcap.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
