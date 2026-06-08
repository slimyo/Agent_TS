"""E2 (method6 §1B) · 决策相图 —— (saturation × trust) → 最优动作。

主线 B：决策不再是 argmax+margin，而是显式策略 π 读 (saturation, trust, belief-shape)。
本实验对每个 cell：
  1. 算 saturation s = ĝ(z)（LODO 回归预测 oracle gap，归一）
  2. 算 trust t = conformal 可信度（E1 F-R11.6 证它最强）
  3. 标"事后最优动作" a*（commit-base / deviate）—— 用真实 outcome（仅离线标签）
然后：
  A. 在 (s, t) 平面上看 a* 的分布 → 画相图边界 + 决策树纯度（边界清晰度）
  B. 训一个 π(s,t,shape)→a，比 LODO regret vs 各固定策略
  C. 跨任务（10-clf 分类 vs forecasting）边界是否一致

复用：m13 的 belief-head/epistemic/conformal；m10.build_dataset；m12 的 gap 回归思想（内联）。
默认 10-clf 非饱和库（M10_LIBPLUS=1 M10_EXPANDED=1），机制此时是"活的"。
输出：results/m14_phase_diagram.jsonl
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("M10_EXPANDED", "1")
os.environ.setdefault("M10_LIBPLUS", "1")

from research.experiments.m10_learned_belief import build_dataset, CLF_ORDER, ROCKET_I
from research.experiments.m13_trust_vs_confidence import _fit_heads, _head_probs

ACTIONS = ["commit-base", "deviate"]   # E2 先做二元（method6 5 动作的核心切分）


def _shape(bbar):
    p = np.clip(bbar, 1e-12, 1)
    ent = float(-(p * np.log(p)).sum())
    srt = np.sort(p)[::-1]
    return {"entropy": ent, "gini": float(1 - (p ** 2).sum()),
            "top_gap": float(srt[0] - srt[1]) if len(srt) > 1 else float(srt[0]),
            "tail": float(srt[2:].sum()) if len(srt) > 2 else 0.0}


def build_states():
    """每 cell → (saturation, trust, shape, a*, regret_if_deviate)。全 LODO。"""
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor

    X, A, info = build_dataset()
    n = len(info)
    win = np.full(n, -1, dtype=int)
    gap = np.zeros(n)        # oracle - base（真实，作 saturation 回归标签）
    for i, a in enumerate(A):
        v = np.where(~np.isnan(a))[0]
        win[i] = int(v[np.argmax(a[v])])
        base = a[ROCKET_I] if not np.isnan(a[ROCKET_I]) else 0.0
        gap[i] = float(a[v].max() - base)
    datasets = sorted(set(it["ds"] for it in info))

    states = []
    for held in datasets:
        tr = np.array([j for j, it in enumerate(info) if it["ds"] != held])
        te = np.array([j for j, it in enumerate(info) if it["ds"] == held])
        if len(tr) == 0 or len(te) == 0 or len(set(win[tr].tolist())) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        Ztr, Zte = sc.transform(X[tr]), sc.transform(X[te])
        win_tr = win[tr]
        # saturation 回归（gap）+ 归一
        sat_reg = RandomForestRegressor(n_estimators=200, max_depth=4, random_state=0).fit(Ztr, gap[tr])
        gmin, gmax = float(gap[tr].min()), float(gap[tr].max())
        gspan = (gmax - gmin) or 1.0
        # belief heads + conformal calib（用 tr 的一半做 calib）
        m = len(tr); perm = np.random.default_rng(0).permutation(m)
        fit_i, cal_i = perm[: m // 2], perm[m // 2:]
        heads = _fit_heads(Ztr[fit_i], win_tr[fit_i]) or _fit_heads(Ztr, win_tr)
        cal_alpha = []
        for li in cal_i:
            P = _head_probs(heads, Ztr[li]); cal_alpha.append(1.0 - P.mean(0).max())
        cal_alpha = np.sort(np.array(cal_alpha)) if len(cal_alpha) else np.array([1.0])

        for li, j in enumerate(te):
            P = _head_probs(heads, Zte[li]); bbar = P.mean(0)
            # saturation：1 − 归一预测gap（gap 大→不饱和→saturation 低）。这里 sat = 预测"还有多少头寸"取反
            pred_gap = float(sat_reg.predict(Zte[li].reshape(1, -1))[0])
            sat = 1.0 - float(np.clip((pred_gap - gmin) / gspan, 0, 1))   # 高=饱和
            # trust (conformal)
            a_new = 1.0 - bbar.max()
            trust = float((cal_alpha >= a_new).mean())
            # 事后最优动作：偏离(argmax非base)是否真的 ≥ base
            accs = A[j]
            base = accs[ROCKET_I] if not np.isnan(accs[ROCKET_I]) else 0.0
            best_i = int(np.argmax(bbar))
            dev_model_acc = accs[best_i] if not np.isnan(accs[best_i]) else base
            a_star = "deviate" if (best_i != ROCKET_I and dev_model_acc > base) else "commit-base"
            regret_dev = float(base - dev_model_acc)   # 偏离的后悔（>0 表示偏离亏）
            prop_model = CLF_ORDER[best_i]              # belief 提议偏离到的模型
            # 部署规则策略（rule-phase）下"实际会用哪个模型"：高 trust + 低 saturation 才偏离
            rule_deviate = (trust >= 0.5 and sat <= 0.5 and best_i != ROCKET_I)
            chosen_model = prop_model if rule_deviate else CLF_ORDER[ROCKET_I]
            states.append({
                "ds": info[j]["ds"], "N": info[j]["N"], "seed": info[j]["seed"],
                "saturation": round(sat, 4), "trust": round(trust, 4),
                **{f"shape_{k}": round(v, 4) for k, v in _shape(bbar).items()},
                "a_star": a_star,
                "base_acc": round(base, 4),
                "dev_acc": round(float(dev_model_acc), 4),
                "best_is_base": best_i == ROCKET_I,
                "prop_model": prop_model,        # belief argmax（提议偏离目标）
                "chosen_model": chosen_model,    # rule-phase 策略实际选用的模型
            })
    return states


def phase_grid(states, n_bins=4):
    """在 (saturation × trust) 网格上统计 deviate 是最优的比例。"""
    grid = {}
    for r in states:
        if r["best_is_base"]:
            continue   # belief 都选 base，无偏离决策可言
        si = min(n_bins - 1, int(r["saturation"] * n_bins))
        ti = min(n_bins - 1, int(r["trust"] * n_bins))
        grid.setdefault((si, ti), []).append(1 if r["a_star"] == "deviate" else 0)
    return {k: (sum(v) / len(v), len(v)) for k, v in grid.items()}


def eval_policy(states):
    """对比固定策略 vs 学习版 π（LODO），用 regret（vs 事后最优动作）。"""
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    cand = [r for r in states if not r["best_is_base"]]   # 只在 belief 想偏离的 cell 上决策
    if len(cand) < 10:
        return {}
    feat_keys = ["saturation", "trust", "shape_entropy", "shape_gini", "shape_top_gap", "shape_tail"]
    datasets = sorted(set(r["ds"] for r in cand))

    def realized(action, r):
        return r["dev_acc"] if action == "deviate" else r["base_acc"]

    # 固定策略基线
    res = {}
    for name, act in [("always-commit", lambda r: "commit-base"),
                      ("always-deviate", lambda r: "deviate")]:
        acc = np.mean([realized(act(r), r) for r in cand])
        res[name] = round(float(acc), 4)
    # 规则相图：trust 高 & saturation 低 → deviate
    def rule(r):
        return "deviate" if (r["trust"] >= 0.5 and r["saturation"] <= 0.5) else "commit-base"
    res["rule-phase"] = round(float(np.mean([realized(rule(r), r) for r in cand])), 4)
    # 学习版 π（LODO 决策树，便于读边界）
    preds = {}
    boundary_purity = []
    for held in datasets:
        tr = [r for r in cand if r["ds"] != held]
        te = [r for r in cand if r["ds"] == held]
        if len(tr) < 8 or not te or len(set(r["a_star"] for r in tr)) < 2:
            for r in te:
                preds[id(r)] = "commit-base"
            continue
        Xtr = np.array([[r[k] for k in feat_keys] for r in tr])
        ytr = [r["a_star"] for r in tr]
        sc = StandardScaler().fit(Xtr)
        dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=0).fit(sc.transform(Xtr), ytr)
        boundary_purity.append(float(dt.score(sc.transform(Xtr), ytr)))
        Xte = sc.transform(np.array([[r[k] for k in feat_keys] for r in te]))
        for r, p in zip(te, dt.predict(Xte)):
            preds[id(r)] = p
    res["learned-pi"] = round(float(np.mean([realized(preds[id(r)], r) for r in cand])), 4)
    res["oracle-action"] = round(float(np.mean([realized(r["a_star"], r) for r in cand])), 4)
    res["n_decision_cells"] = len(cand)
    res["deviate_optimal_frac"] = round(float(np.mean([r["a_star"] == "deviate" for r in cand])), 3)
    res["boundary_train_purity"] = round(float(np.mean(boundary_purity)), 3) if boundary_purity else None
    return res


def render_phase_png(states, grid, res, out_png):
    """可视化决策相图：(a) (saturation×trust) 平面散点(事后最优动作 + rule-phase 选用模型)；
    (b) (saturation×trust) 网格热力(P(deviate 最优))；(c) 每个 cell 的 chosen_model 矩阵(行=数据集)。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    for _f in ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "Source Han Sans SC", "SimHei"]:
        try:
            from matplotlib.font_manager import findfont, FontProperties
            if findfont(FontProperties(family=_f), fallback_to_default=False):
                matplotlib.rcParams["font.sans-serif"] = [_f]; break
        except Exception:
            continue
    matplotlib.rcParams["axes.unicode_minus"] = False

    cand = [r for r in states if not r["best_is_base"]]   # belief 想偏离的决策 cell
    # 颜色：每个被用到的模型一个颜色
    models = sorted({r["chosen_model"] for r in states} | {r["prop_model"] for r in cand})
    cmap = plt.get_cmap("tab10")
    mcol = {m: cmap(i % 10) for i, m in enumerate(models)}

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.2))

    # ---- (a) 相图散点：x=saturation y=trust，颜色=rule-phase chosen_model，形状=事后最优动作 ----
    ax = axes[0]
    ax.add_patch(Rectangle((0, 0.5), 0.5, 0.5, fc="#e8f5e9", ec="none", zorder=0))  # deviate 区
    ax.text(0.02, 0.96, "rule: deviate\n(trust≥.5 & sat≤.5)", fontsize=8, va="top", color="#2e7d32")
    for r in cand:
        m = ("o" if r["a_star"] == "deviate" else "x")
        ax.scatter(r["saturation"], r["trust"], c=[mcol[r["chosen_model"]]],
                   marker=m, s=46, edgecolors="k", linewidths=0.4, alpha=0.85, zorder=3)
    ax.axvline(0.5, color="gray", ls="--", lw=0.8); ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("saturation (高=base 已近 oracle)"); ax.set_ylabel("trust (conformal)")
    ax.set_title("(a) 决策相图 · 点色=rule选用模型, ○=偏离最优 ×=守base最优")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    handles = [plt.Line2D([], [], marker="o", ls="", mfc=mcol[m], mec="k", label=m) for m in models]
    ax.legend(handles=handles, fontsize=7, loc="lower right", title="chosen_model")

    # ---- (b) (saturation×trust) 网格热力：P(deviate 最优) ----
    ax = axes[1]
    nb = 4
    M = np.full((nb, nb), np.nan)
    for (si, ti), (f, nn) in grid.items():
        M[si, ti] = f
    im = ax.imshow(M, origin="lower", cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for (si, ti), (f, nn) in grid.items():
        ax.text(ti, si, f"{f:.2f}\nn={nn}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(nb)); ax.set_yticks(range(nb))
    ax.set_xlabel("trust bin (低→高)"); ax.set_ylabel("saturation bin (低→高)")
    ax.set_title("(b) P(deviate 是事后最优 | belief 想偏离)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ---- (c) 每 cell 的 chosen_model 矩阵：行=数据集, 列=cell(N×seed) ----
    ax = axes[2]
    datasets = sorted({r["ds"] for r in states})
    midx = {m: i for i, m in enumerate(models)}
    # 每个数据集的 cell 排序（N,seed）
    by_ds = defaultdict(list)
    for r in sorted(states, key=lambda r: (r["ds"], r["N"], r["seed"])):
        by_ds[r["ds"]].append(r)
    maxc = max(len(v) for v in by_ds.values())
    grid_m = np.full((len(datasets), maxc), np.nan)
    for di, ds in enumerate(datasets):
        for ci, r in enumerate(by_ds[ds]):
            grid_m[di, ci] = midx[r["chosen_model"]]
    from matplotlib.colors import ListedColormap, BoundaryNorm
    listed = ListedColormap([mcol[m] for m in models])
    norm = BoundaryNorm(np.arange(-0.5, len(models) + 0.5), len(models))
    ax.imshow(grid_m, aspect="auto", cmap=listed, norm=norm)
    ax.set_yticks(range(len(datasets))); ax.set_yticklabels(datasets, fontsize=8)
    ax.set_xlabel("cell (按 N×seed 排序)"); ax.set_title("(c) 每个 cell rule-phase 实际选用的模型")
    handles = [Rectangle((0, 0), 1, 1, fc=mcol[m]) for m in models]
    ax.legend(handles, models, fontsize=7, loc="upper right", ncol=2, title="chosen_model")

    n_dev = sum(r["chosen_model"] != "rocket" for r in states)
    fig.suptitle(f"m14 决策相图 | {len(states)} cells, {len(cand)} 决策cell | "
                 f"rule-phase 偏离 {n_dev} 次 | learned-π {res.get('learned-pi','?')} vs "
                 f"always-commit {res.get('always-commit','?')} (oracle {res.get('oracle-action','?')})",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main():
    out = Path("research/results/m14_phase_diagram.jsonl")
    print("=== E2 · Decision Phase Diagram (LODO, 10-clf un-saturated lib) ===", flush=True)
    states = build_states()
    print(f"states: {len(states)} cells", flush=True)

    grid = phase_grid(states, n_bins=4)
    print("\n(saturation_bin, trust_bin) → P(deviate optimal | belief wants deviate), n:", flush=True)
    for si in range(4):
        row = []
        for ti in range(4):
            if (si, ti) in grid:
                f, nn = grid[(si, ti)]
                row.append(f"{f:.2f}({nn})")
            else:
                row.append("  -  ")
        print(f"  sat[{si}] " + " | ".join(row), flush=True)
    print("  (行=saturation 0→3 低到高, 列=trust 0→3 低到高)", flush=True)

    res = eval_policy(states)
    print("\n=== policy realized acc on decision cells ===", flush=True)
    for k in ["n_decision_cells", "deviate_optimal_frac", "always-commit", "always-deviate",
              "rule-phase", "learned-pi", "oracle-action", "boundary_train_purity"]:
        if k in res:
            print(f"  {k:22}: {res[k]}", flush=True)

    with out.open("w") as fh:
        fh.write(json.dumps({"_summary": res}, ensure_ascii=False) + "\n")
        fh.write(json.dumps({"_grid": {f"{k[0]},{k[1]}": v for k, v in grid.items()}}, ensure_ascii=False) + "\n")
        for r in states:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {out}", flush=True)

    out_png = out.with_suffix(".png")
    try:
        render_phase_png(states, grid, res, out_png)
        print(f"wrote {out_png}", flush=True)
    except Exception as e:
        print(f"[warn] render PNG failed: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
