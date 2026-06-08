"""三大任务（分类 / 预测 / 检测）统一决策相图 —— 全量库。

对每个任务，在其 per-cell per-model 性能库上做 LODO 双信号解耦决策（与 test/signal_router 同口径）：
  - saturation = 1 − 归一(预测 oracle gap)   高=base 已近 oracle
  - trust      = conformal（belief 对 winner 的可信度在已见分布内的分位）
  - headroom   = 归一(预测 gap)               高=还有头寸
  - 决策律     = deviate iff trust≥0.5 AND headroom≥0.5（否则守 base）
  - a*(事后最优) = 偏离到 belief-argmax 是否真的 ≥ base
每 cell 记录 (saturation, trust, chosen_model, a*)；渲染 3 任务并排相图 PNG。

任务库：
  分类   = test/signal_router.load_oracle_library()（38 数据集 × 22 分类器，base=rocket）
  预测   = research/results/p*_*.jsonl + f4_*.jsonl（72 cell，base=chronos2，MAE 越小越好）
  检测   = research/results/taskc_synth4class.jsonl（base=rocket）
输出：research/results/m22_threetask_phase.{jsonl,png}
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test"))


# ---------- 三个任务的 cell 库 ---------- #

def lib_classification():
    import signal_router as S
    cells = S.load_oracle_library()   # {(ds,N,seed): {clf: acc}}
    return cells, "rocket", True


def lib_detection():
    from collections import defaultdict
    import signal_router as S
    by = defaultdict(dict)
    for name in ["taskc_synth4class", "taskc_newlib"]:   # 旧 7 + 新增 22 全量库
        fp = ROOT / "research" / "results" / f"{name}.jsonl"
        if not fp.exists():
            continue
        for l in fp.read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            clf = S.METHOD_TO_CLF.get(r["method"])
            a = r.get("acc")
            if clf and a is not None and a == a:
                by[(r["dataset"], r["N_per_class"], r["seed"])][clf] = float(a)
    cells = {k: v for k, v in by.items() if "rocket" in v and len(v) >= 4}
    return cells, "rocket", True


def lib_forecasting():
    cell = defaultdict(dict)
    cand = {"chronos2", "chronos_bolt", "chronos", "arima_ets", "naive", "llmtime",
            # 扩充 TSFM（taska_tsfm sweep）：moirai2 全量；timer/sundial/time_moe/timesfm2/tirex/toto/toto2
            # 若 taska_tsfm.jsonl 内有其非空行也会自动并入
            "moirai2", "timer", "sundial", "time_moe", "timesfm2", "tirex", "toto", "toto2"}
    files = (glob.glob(str(ROOT / "research/results/p*_*.jsonl")) +
             glob.glob(str(ROOT / "research/results/f4_*.jsonl")) +
             glob.glob(str(ROOT / "research/results/taska_tsfm*.jsonl")))
    for fp in files:
        for l in open(fp):
            try:
                r = json.loads(l)
            except Exception:
                continue
            mae = r.get("mae")
            if mae is not None and r.get("method") in cand and "dataset" in r and "N" in r:
                cell[(r["dataset"], int(r["N"]), int(r.get("seed", 0)))][r["method"]] = float(mae)
    cells = {k: v for k, v in cell.items() if "chronos2" in v and len(v) >= 4}
    return cells, "chronos2", False   # MAE 越小越好


# ---------- 通用 LODO 双信号决策（higher_better 适配 MAE） ---------- #

def run_phase(cells, base, higher_better):
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LogisticRegression

    keys = sorted(cells.keys())
    models = sorted({m for v in cells.values() for m in v})
    bi = models.index(base)
    n = len(keys)
    good = np.full((n, len(models)), np.nan)   # goodness: 越大越好
    for i, k in enumerate(keys):
        for mi, m in enumerate(models):
            if m in cells[k]:
                good[i, mi] = cells[k][m] if higher_better else -cells[k][m]
    # 特征：N、候选数、base 绝对水平、logN
    X = np.array([[float(k[1]), float((~np.isnan(good[i])).sum()),
                   float(good[i, bi]), float(np.log1p(k[1]))] for i, k in enumerate(keys)])
    win = np.zeros(n, dtype=int); gap = np.zeros(n)
    for i in range(n):
        valid = np.where(~np.isnan(good[i]))[0]
        win[i] = int(valid[np.argmax(good[i][valid])])
        gap[i] = float(good[i][valid].max() - good[i, bi])
    ds_of = [k[0] for k in keys]; datasets = sorted(set(ds_of))

    recs = []
    for held in datasets:
        tr = np.array([i for i in range(n) if ds_of[i] != held])
        te = np.array([i for i in range(n) if ds_of[i] == held])
        if len(tr) == 0 or len(te) == 0 or len(set(win[tr].tolist())) < 2:
            for i in te:
                recs.append(_mk(keys[i], models, good[i], bi, bi, 0.0, 0.0, "no-train"))
            continue
        sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        belief = LogisticRegression(max_iter=2000).fit(Xtr, win[tr]); bc = list(belief.classes_)
        head = RandomForestRegressor(n_estimators=200, max_depth=4, random_state=0).fit(Xtr, gap[tr])
        gmin, gmax = float(gap[tr].min()), float(gap[tr].max()); gspan = (gmax - gmin) or 1.0
        perm = np.random.default_rng(0).permutation(len(tr)); cal = tr[perm[len(tr) // 2:]]
        Pcal = belief.predict_proba(sc.transform(X[cal]))
        cal_alpha = sorted(1.0 - (Pcal[r, bc.index(win[g])] if win[g] in bc else 0.0)
                           for r, g in enumerate(cal)) or [1.0]
        cal_alpha = np.array(cal_alpha)
        Pte = belief.predict_proba(Xte); hp = head.predict(Xte)
        for r, i in enumerate(te):
            p = Pte[r]; order = [bc[j] for j in np.argsort(-p)]
            prop = next((c for c in order if c != bi), bi)
            trust = float((cal_alpha >= (1.0 - float(p.max()))).mean())
            headroom = float(np.clip((hp[r] - gmin) / gspan, 0, 1))
            chosen = prop if (trust >= 0.5 and headroom >= 0.5 and prop != bi) else bi
            recs.append(_mk(keys[i], models, good[i], bi, prop, trust, headroom,
                            "deviate" if chosen != bi else "commit", chosen=chosen))
    return recs, models, bi


def _mk(key, models, grow, bi, prop, trust, headroom, mode, chosen=None):
    if chosen is None:
        chosen = bi
    base_g = grow[bi]
    dev_g = grow[prop] if not np.isnan(grow[prop]) else base_g
    a_star = "deviate" if (prop != bi and dev_g > base_g) else "commit"
    oracle_winner = models[int(np.nanargmax(grow))]   # 该 cell 事后最优模型（展示全库真实参与）
    sat = 1.0 - float(np.clip(headroom, 0, 1))   # 高=饱和
    return {"ds": key[0], "N": key[1], "seed": key[2], "saturation": round(sat, 4),
            "trust": round(trust, 4), "headroom": round(headroom, 4),
            "chosen_model": models[chosen], "prop_model": models[prop],
            "oracle_winner": oracle_winner,
            "a_star": a_star, "deviated": bool(chosen != bi), "best_is_base": bool(prop == bi),
            "chosen_g": float(grow[chosen]) if not np.isnan(grow[chosen]) else float(base_g),
            "base_g": float(base_g)}


def summarize(recs, higher_better):
    dev = [r for r in recs if r["deviated"]]
    if higher_better:   # 分类/检测：acc 百分点
        vs = float(np.mean([(r["chosen_g"] - r["base_g"]) for r in recs]) * 100)
        unit = "pp"
    else:               # 预测：逐 cell 相对 MAE 改进（g=-MAE）
        rels = [((-r["base_g"]) - (-r["chosen_g"])) / max(1e-9, -r["base_g"]) for r in recs]
        vs = float(np.mean(rels) * 100)
        unit = "%relMAE"
    return {"n_cells": len(recs), "n_dev": len(dev),
            "vs_base": round(vs, 3), "unit": unit}


def render(task_recs, out_png):
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
    from matplotlib.patches import Rectangle

    # 全局模型配色（按所有任务的 oracle_winner 并集，保证一致+展示全库参与）
    all_models = sorted({r["oracle_winner"] for _, recs, _, _ in task_recs for r in recs})
    cmap = plt.get_cmap("tab20"); cmap2 = plt.get_cmap("tab20b")
    mcol = {m: (cmap(i) if i < 20 else cmap2(i - 20)) for i, m in enumerate(all_models)}

    fig, axes = plt.subplots(1, 3, figsize=(22, 6.6))
    for ax, (task, recs, summ, sat_frac) in zip(axes, task_recs):
        ax.add_patch(Rectangle((0, 0.5), 0.5, 0.5, fc="#e8f5e9", ec="none", zorder=0))
        ax.text(0.02, 0.97, "rule: deviate\n(trust≥.5 & sat≤.5)", fontsize=7, va="top", color="#2e7d32")
        # 每个 cell 上色 = 该 cell 事后最优模型（oracle_winner）→ 直观展示全库 22 模型都在参与
        for r in recs:
            mk = "o" if r["a_star"] == "deviate" else "x"
            ax.scatter(r["saturation"], r["trust"], c=[mcol.get(r["oracle_winner"], "gray")],
                       marker=mk, s=40, edgecolors="k", linewidths=0.25, alpha=0.8, zorder=3)
        ax.axvline(0.5, color="gray", ls="--", lw=0.8); ax.axhline(0.5, color="gray", ls="--", lw=0.8)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("saturation (高=base 近 oracle)"); ax.set_ylabel("trust (conformal)")
        nwin = len({r["oracle_winner"] for r in recs})
        ax.set_title(f"{task}\n{len(recs)}cell · oracle由{nwin}个模型分摊 · 系统偏离={summ['n_dev']} vs_base={summ['vs_base']:+.3f}{summ['unit']}",
                     fontsize=10)
    # 统一图例（全库模型 → 颜色），放最右
    handles = [plt.Line2D([], [], marker="s", ls="", mfc=mcol[m], mec="k", label=m) for m in all_models]
    fig.legend(handles=handles, fontsize=7, loc="center right", ncol=1, title="oracle-winner 模型(全库)",
               bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("三大任务统一决策相图（全量库 LODO）· 点色=该cell事后最优模型(oracle-winner) · "
                 "○=偏离最优 ×=守base最优 · 决策仍守base(+0.00pp)：头寸真实但逐-cell不可预测", fontsize=11)
    fig.tight_layout(rect=[0, 0, 0.9, 0.95]); fig.savefig(out_png, dpi=130); plt.close(fig)


def main():
    out = ROOT / "research" / "results" / "m22_threetask_phase.jsonl"
    tasks = [("分类 Classification", lib_classification),
             ("预测 Forecasting", lib_forecasting),
             ("检测 Detection", lib_detection)]
    task_recs = []
    allrows = []
    for name, fn in tasks:
        cells, base, hb = fn()
        recs, models, bi = run_phase(cells, base, hb)
        summ = summarize(recs, hb)
        sat_frac = float(np.mean([r["saturation"] >= 0.5 for r in recs]))
        print(f"=== {name} === base={base} cells={len(cells)} models={len(models)} | "
              f"{summ} | high-sat frac={sat_frac:.2f}", flush=True)
        task_recs.append((name, recs, summ, sat_frac))
        for r in recs:
            allrows.append({"task": name, **r})
    with out.open("w") as fh:
        for r in allrows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    render(task_recs, out.with_suffix(".png"))
    print(f"\nwrote {out} + {out.with_suffix('.png')}", flush=True)


if __name__ == "__main__":
    main()
