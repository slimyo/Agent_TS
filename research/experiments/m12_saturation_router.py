"""M12 · Saturation Detection + selective routing（method5，回应 feedback_m4 方向1）。

核心问题从"选哪个模型"→"该不该选"。对每个 cell：
    z = 轻量元特征
    ĝ_φ(z) = 预测 oracle gap（LODO 回归）
    gap>τ → ROUTE（用 CV-winner 偏离 base）；gap≤τ → ABSTAIN（用 base）

跨三任务验证 saturation 是否可检测：
  - Forecasting（未饱和：chronos2 仅 25% cell 是 oracle）→ 检测器应多 route
  - Classification（UCR 饱和：rocket 71%）→ 检测器应多 abstain
  - Detection（synth 4-class 饱和：rocket 75%）→ 应多 abstain

诚实：leave-one-dataset-out。gap 标签 = 历史 cell 真实 oracle gap（离线经验，合法）。
metric（采纳 reviewer #93）：regret-to-oracle / safe-deviation-rate / abstain-accuracy /
  对照 always-route vs always-abstain(=base) vs saturation-gated。

输出：results/m12_saturation_router.jsonl
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------- 1) 收集三任务 per-cell per-method 表现 ----------

def _forecasting_cells():
    """返回 {(ds,N,seed): {method: -MAE}}（取负，统一成"越大越好"）。"""
    cell = defaultdict(dict)
    files = (glob.glob("research/results/p*_*.jsonl")
             + glob.glob("research/results/f4_*.jsonl")
             + glob.glob("research/results/*vs_c2*.jsonl"))
    for fp in files:
        for l in open(fp):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if "mae" in r and "method" in r and "dataset" in r and "N" in r:
                cell[(r["dataset"], r["N"], r.get("seed", 0))][r["method"]] = -float(r["mae"])
    return {k: v for k, v in cell.items() if len(v) >= 3}, "chronos2"


def _clf_cells():
    cell = defaultdict(dict)
    M = {"B1_dtw", "B2_euclid", "B3_rocket", "B4a_moment_1nn", "B4b_moment_lr"}
    for fp in ["research/results/taskb_ucr.jsonl",
               "research/results/taskb_extended_ucr.jsonl",
               "research/results/taskb_expand_ucr.jsonl"]:
        if not Path(fp).exists():
            continue
        for l in open(fp):
            r = json.loads(l)
            if r["method"] in M:
                cell[(r["dataset"], r["N_per_class"], r["seed"])][r["method"]] = float(r["acc"])
    return {k: v for k, v in cell.items() if len(v) >= 3}, "B3_rocket"


def _detect_cells():
    cell = defaultdict(dict)
    M = {"B1_dtw", "B2_euclid", "B3_rocket", "B4a_moment_1nn", "B4b_moment_lr"}
    for l in open("research/results/taskc_synth4class.jsonl"):
        r = json.loads(l)
        if r["method"] in M:
            cell[(r["dataset"], r["N_per_class"], r["seed"])][r["method"]] = float(r["acc"])
    return {k: v for k, v in cell.items() if len(v) >= 3}, "B3_rocket"


# ---------- 2) LODO saturation regression + selective routing ----------
# 元特征在 run_task 内构建：[N, n_methods, base_perf, log(N)]——均为 deployment 可得量
# （base 一定会跑出 base_perf；N/方法数已知）。不含任何"哪个模型赢"的 test 信息。

def run_task(task_name, cells, base_method, tau_grid):
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor

    keys = sorted(cells.keys())
    # per-cell: base perf, oracle perf, best non-base, gap, feature
    rows = []
    for k in keys:
        v = cells[k]
        if base_method not in v:
            continue
        base_p = v[base_method]
        oracle_p = max(v.values())
        others = {m: p for m, p in v.items() if m != base_method}
        best_other_m = max(others, key=others.get) if others else base_method
        best_other_p = others.get(best_other_m, base_p)
        gap = oracle_p - base_p                      # 标签：oracle gap（≥0）
        ds, N, seed = k
        # 元特征：N、方法数、base 绝对水平、方法间分散度（deployment-proxy：用 base 水平+N）
        spread = float(np.std(list(v.values())))     # 历史 cell 才知；作 label-side 仅训练用 → 不进特征
        feat = [float(N), float(len(v)), float(base_p), float(np.log1p(N))]
        rows.append({"key": k, "ds": ds, "base_p": base_p, "oracle_p": oracle_p,
                     "best_other_m": best_other_m, "best_other_p": best_other_p,
                     "gap": gap, "feat": feat})
    datasets = sorted(set(r["ds"] for r in rows))

    # 全 cell 的 per-method 表（用于 LODO 选 deployable 路由备选）
    cell_v = cells

    # LODO：预测 gap + 选 deployable route-alt（训练集上均值最优的非 base 方法，无 test peek）
    preds = {}
    route_alt = {}   # key -> 部署可用的路由备选方法名（held-out 不可见自身）
    for held in datasets:
        tr = [r for r in rows if r["ds"] != held]
        te = [r for r in rows if r["ds"] == held]
        if not tr or not te:
            continue
        Xtr = np.array([r["feat"] for r in tr]); ytr = np.array([r["gap"] for r in tr])
        sc = StandardScaler().fit(Xtr)
        reg = RandomForestRegressor(n_estimators=200, max_depth=4, random_state=0)
        reg.fit(sc.transform(Xtr), ytr)
        Xte = sc.transform(np.array([r["feat"] for r in te]))
        for r, p in zip(te, reg.predict(Xte)):
            preds[r["key"]] = float(p)
        # deployable route alternative = 训练 ds 上均值最优的非 base 方法
        msum = defaultdict(list)
        for r in tr:
            for m, pp in cell_v[r["key"]].items():
                if m != base_method:
                    msum[m].append(pp)
        if msum:
            alt = max(msum, key=lambda m: np.mean(msum[m]))
            for r in te:
                route_alt[r["key"]] = alt

    # gap 预测质量：相关 + 排序 AUC（高真实 gap 是否被预测排前）
    yk = [r for r in rows if r["key"] in preds]
    gt = np.array([r["gap"] for r in yk]); pr = np.array([preds[r["key"]] for r in yk])
    corr = float(np.corrcoef(gt, pr)[0, 1]) if len(set(pr.tolist())) > 1 else float("nan")

    # 选择性路由评估（多 τ）
    def eval_tau(tau):
        reg_sum = sel_sum = base_sum = ora_sum = 0.0
        n = 0; n_route = 0; safe_dev = 0; n_dev = 0
        for r in yk:
            g = preds[r["key"]]
            alt_m = route_alt.get(r["key"])
            alt_p = cell_v[r["key"]].get(alt_m, r["base_p"]) if alt_m else r["base_p"]
            if g > tau and alt_m is not None:  # ROUTE → deployable 备选（LODO 训练集均值最优非 base，无 test peek）
                chosen = alt_p; routed = True
            else:
                chosen = r["base_p"]; routed = False
            sel_sum += chosen; base_sum += r["base_p"]; ora_sum += r["oracle_p"]
            reg_sum += (r["oracle_p"] - chosen); n += 1
            if routed:
                n_route += 1; n_dev += 1
                if chosen >= r["base_p"]:
                    safe_dev += 1
        return {
            "tau": round(tau, 4), "n": n,
            "selected": round(sel_sum / n, 4), "base": round(base_sum / n, 4),
            "oracle": round(ora_sum / n, 4),
            "regret_to_oracle": round((ora_sum - sel_sum) / n, 4),
            "vs_base": round((sel_sum - base_sum) / n, 4),
            "route_rate": round(n_route / n, 3),
            "safe_dev_rate": round(safe_dev / n_dev, 3) if n_dev else float("nan"),
        }

    return {
        "task": task_name, "n_cells": len(yk), "n_datasets": len(datasets),
        "base": base_method, "gap_pred_corr": round(corr, 3),
        "mean_true_gap": round(float(gt.mean()), 4),
        "base_is_oracle_frac": round(float(np.mean([abs(r["gap"]) < 1e-9 for r in yk])), 3),
        "tau_sweep": [eval_tau(t) for t in tau_grid],
        # 对照：always-route(τ=-inf) / always-abstain(τ=+inf)
        "always_route": eval_tau(-1e9), "always_abstain": eval_tau(1e9),
    }


def main():
    out = Path("research/results/m12_saturation_router.jsonl")
    tasks = [
        ("forecasting", *_forecasting_cells()),
        ("classification", *_clf_cells()),
        ("detection", *_detect_cells()),
    ]
    # τ 网格：forecasting gap 是相对 MAE（0~0.x），clf/detect 是 acc pp（0~0.x）；统一小网格
    results = []
    print("=== M12 Saturation Detection · cross-task LODO ===", flush=True)
    for name, cells, base in tasks:
        # 任务自适应 τ 网格（gap 量纲不同）
        all_gap = sorted(max(v.values()) - v[base] for v in cells.values() if base in v)
        if not all_gap:
            print(f"{name}: no base cells", flush=True); continue
        hi = all_gap[int(len(all_gap) * 0.75)]
        grid = [0.0, hi * 0.25, hi * 0.5, hi * 0.75, hi] if hi > 0 else [0.0]
        res = run_task(name, cells, base, grid)
        results.append(res)
        print(f"\n[{name}] cells={res['n_cells']} ds={res['n_datasets']} base={base} "
              f"base_is_oracle={res['base_is_oracle_frac']:.0%} mean_gap={res['mean_true_gap']:.4f} "
              f"gap_pred_corr={res['gap_pred_corr']}", flush=True)
        ar, aa = res["always_route"], res["always_abstain"]
        print(f"  always-ROUTE   : vs_base={ar['vs_base']:+.4f} regret={ar['regret_to_oracle']:.4f} "
              f"route_rate={ar['route_rate']} safe_dev={ar['safe_dev_rate']}", flush=True)
        print(f"  always-ABSTAIN : vs_base={aa['vs_base']:+.4f} regret={aa['regret_to_oracle']:.4f}", flush=True)
        for s in res["tau_sweep"]:
            print(f"  gated τ={s['tau']:.4f}: vs_base={s['vs_base']:+.4f} regret={s['regret_to_oracle']:.4f} "
                  f"route_rate={s['route_rate']} safe_dev={s['safe_dev_rate']}", flush=True)
    with out.open("w") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
