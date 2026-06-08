"""合并远程 Timer-S1 MAE 与本地 Chronos-2 base 记录 → 决定是否更换预测 base。

远程 env 无 chronos 包，故 timer-only 在远程跑（research/results/timer_vs_chronos2_remote.jsonl）。
本地已有 chronos2 base 的逐 cell MAE（p*/f4 结果，含 start_idx）。few_shot_split 完全确定性，
故按 (dataset,N,seed) 配对，并**校验 start_idx 一致**（一致 = 测试窗口相同 = MAE 可直接比）。

ILI：本地/远程序列长度不同（start_idx 不一致）且 timer 输出 NaN → 剔除，诚实标注。
输出：research/results/timer_vs_chronos2.jsonl（最终）+ .png 可视化。
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REMOTE = "research/results/timer_vs_chronos2_remote.jsonl"
OUT = Path("research/results/timer_vs_chronos2.jsonl")


def load_local_c2():
    c2 = {}
    for fp in glob.glob("research/results/p*_*.jsonl") + glob.glob("research/results/f4_*.jsonl"):
        for l in open(fp):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if r.get("method") == "chronos2" and "mae" in r and "start_idx" in r:
                c2[(r["dataset"], int(r["N"]), int(r.get("seed", 0)))] = \
                    {"mae": float(r["mae"]), "start_idx": int(r["start_idx"])}
    return c2


def main():
    c2 = load_local_c2()
    timer = [json.loads(l) for l in open(REMOTE) if not l.startswith('{"_sum')]
    rows, dropped = [], []
    for r in timer:
        k = (r["dataset"], r["N"], r["seed"])
        mt = r.get("mae_timer")
        if k not in c2:
            dropped.append((k, "no_c2")); continue
        if mt is None or mt != mt:
            dropped.append((k, "timer_nan")); continue
        if c2[k]["start_idx"] != r["start_idx"]:
            dropped.append((k, f"startidx {c2[k]['start_idx']}!={r['start_idx']}")); continue
        mc = c2[k]["mae"]
        rel = (mc - mt) / mc if mc > 0 else 0.0
        rows.append({"dataset": r["dataset"], "N": r["N"], "seed": r["seed"], "H": r["H"],
                     "start_idx": r["start_idx"], "mae_timer": round(mt, 6),
                     "mae_c2": round(mc, 6), "rel_impr": round(rel, 4),
                     "win": "timer" if mt < mc else "c2"})

    rels = np.array([r["rel_impr"] for r in rows])
    per_ds = defaultdict(list)
    for r in rows:
        per_ds[r["dataset"]].append(r)
    per_ds_stat = {ds: {"timer_win_rate": round(float(np.mean([x["win"] == "timer" for x in v])), 2),
                        "mean_rel_impr": round(float(np.mean([x["rel_impr"] for x in v])), 4),
                        "n": len(v)} for ds, v in sorted(per_ds.items())}
    n_ds_pos = sum(d["mean_rel_impr"] > 0 for d in per_ds_stat.values())
    summary = {
        "n_paired": len(rows),
        "n_dropped": len(dropped),
        "dropped_reasons": dict(defaultdict(int, {r: sum(1 for _, x in dropped if x.split()[0] == r.split()[0])
                                                  for _, r in dropped})),
        "timer_win_rate": round(float(np.mean([r["win"] == "timer" for r in rows])), 3),
        "mean_rel_impr": round(float(rels.mean()), 4),
        "median_rel_impr": round(float(np.median(rels)), 4),
        "mean_mae_timer": round(float(np.mean([r["mae_timer"] for r in rows])), 4),
        "mean_mae_c2": round(float(np.mean([r["mae_c2"] for r in rows])), 4),
        "per_dataset": per_ds_stat,
        # 换 base 判据：总体 rel>3% 且 win_rate≥0.55 且 ≥4/5 数据集为正
        "recommend_switch_base": bool(rels.mean() > 0.03 and
                                      np.mean([r["win"] == "timer" for r in rows]) >= 0.55 and
                                      n_ds_pos >= max(4, len(per_ds_stat) - 1)),
    }

    with OUT.open("w") as fh:
        fh.write(json.dumps({"_summary": summary}, ensure_ascii=False) + "\n")
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\ndropped: {dropped}")
    print(f"wrote {OUT}")
    _render(rows, summary, OUT.with_suffix(".png"))
    print(f"wrote {OUT.with_suffix('.png')}")


def _render(rows, summary, out_png):
    import matplotlib
    matplotlib.use("Agg")
    for _f in ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "SimHei"]:
        try:
            from matplotlib.font_manager import findfont, FontProperties
            if findfont(FontProperties(family=_f), fallback_to_default=False):
                matplotlib.rcParams["font.sans-serif"] = [_f]; break
        except Exception:
            continue
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    datasets = sorted({r["dataset"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))

    # (a) 逐 cell rel-MAE（timer 相对 c2 的改进，>0 上方=timer 更好）
    ax = axes[0]
    cmap = plt.get_cmap("tab10")
    dcol = {ds: cmap(i % 10) for i, ds in enumerate(datasets)}
    xs = list(range(len(rows)))
    rows_sorted = sorted(rows, key=lambda r: (r["dataset"], r["N"], r["seed"]))
    for i, r in enumerate(rows_sorted):
        ax.bar(i, r["rel_impr"], color=dcol[r["dataset"]], width=0.9)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(summary["mean_rel_impr"], color="red", ls="--", lw=1,
               label=f"mean {summary['mean_rel_impr']:+.3f}")
    ax.set_xlabel("cell (按 数据集×N×seed)"); ax.set_ylabel("rel-MAE 改进 (timer vs c2)")
    ax.set_title(f"(a) 逐 cell：Timer-S1 相对 Chronos-2 (>0=Timer 更好) | win {summary['timer_win_rate']:.0%}")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=dcol[d]) for d in datasets]
    ax.legend(handles + [plt.Line2D([], [], color="red", ls="--")],
              datasets + [f"mean {summary['mean_rel_impr']:+.3f}"], fontsize=8, ncol=2)

    # (b) 分数据集 mean rel + win-rate
    ax = axes[1]
    ds = list(summary["per_dataset"].keys())
    means = [summary["per_dataset"][d]["mean_rel_impr"] for d in ds]
    wins = [summary["per_dataset"][d]["timer_win_rate"] for d in ds]
    y = np.arange(len(ds))
    ax.barh(y, means, color=["#2e7d32" if m > 0 else "#c62828" for m in means])
    for i, (m, w) in enumerate(zip(means, wins)):
        ax.text(m, i, f" {m:+.3f} (win {w:.0%})", va="center",
                ha="left" if m >= 0 else "right", fontsize=9)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(ds)
    ax.set_xlabel("mean rel-MAE 改进"); ax.set_title("(b) 分数据集：Timer vs Chronos-2")
    rec = "换 base ✅" if summary["recommend_switch_base"] else "保持 Chronos-2 ❌"
    fig.suptitle(f"Timer-S1 vs Chronos-2 base · {summary['n_paired']} cells (剔除 ILI {summary['n_dropped']}) "
                 f"| 总体 rel {summary['mean_rel_impr']:+.3f} | 建议: {rec}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
