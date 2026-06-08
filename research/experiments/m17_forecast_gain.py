"""E5 (method7 §2, #102) · Forecasting 主战场 —— 未饱和域测 trust+gain 显著性。

feedback_m6 #1：预测域未饱和（chronos2 仅 25% cell 是 oracle），正确偏离样本多，
能真正测出 trust/gain 的统计显著性（分类域被 0.4pp 天花板卡死）。

数据：72 forecasting cell（6 数据集 × N × seed），8 个方法的 per-cell MAE（既有结果聚合）。
- base = chronos2；候选 = {chronos_bolt, arima_ets, llmtime, naive, ...}
- gain(z, m) = (base_MAE − m_MAE)/base_MAE   （相对 MAE 降幅，越大越好）
- z = 训练序列 hand 特征（部署可得，无 test）
- trust = bagged gain 回归的分歧（epistemic）反向归一
- 决策 = deviate iff pred_gain ≥ τ（LODO）

对比：always-base / pred-gain-gate / oracle。问句：未饱和域 gain model 能否带来**正向** rel-MAE？
输出：results/m17_forecast_gain.jsonl
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from research.utils.series_features import extract_full_features, FEATURE_ORDER
from research.utils.data_loader import load_series
from research.utils.splitter import few_shot_split

BASE = "chronos2"
NONMETA = [k for k in FEATURE_ORDER if not k.startswith("meta_")]


def _collect():
    cell = defaultdict(dict)
    for fp in glob.glob("research/results/p*_*.jsonl") + glob.glob("research/results/f4_*.jsonl"):
        for l in open(fp):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if "mae" in r and "method" in r and "dataset" in r and "N" in r:
                cell[(r["dataset"], int(r["N"]), int(r.get("seed", 0)))][r["method"]] = float(r["mae"])
    return {k: v for k, v in cell.items() if BASE in v and len(v) >= 4}


def _feat(series, N, seed, H=96):
    sp = few_shot_split(series, N=N, H=H, seed=seed)
    f = extract_full_features(np.asarray(sp.train, dtype=np.float64))
    return np.array([float(f.get(k, 0.0)) for k in NONMETA] + [np.log1p(N)], dtype=np.float64)


def build():
    cells = _collect()
    # candidate methods present widely（排 adapt_ts/tsci 这种本身就是路由器，避免循环）
    cand = ["chronos_bolt", "arima_ets", "llmtime", "naive", "chronos"]
    series_cache = {}
    rows = []
    for (ds, N, seed), v in sorted(cells.items()):
        if ds not in series_cache:
            try:
                series_cache[ds] = load_series(ds)[0]
            except Exception:
                series_cache[ds] = None
        s = series_cache[ds]
        if s is None:
            continue
        try:
            z = _feat(s, N, seed)
        except Exception:
            continue
        base_mae = v[BASE]
        # 每候选相对 gain
        cg = {m: (base_mae - v[m]) / (base_mae + 1e-9) for m in cand if m in v}
        if not cg:
            continue
        best_m = max(cg, key=cg.get)
        rows.append({"ds": ds, "N": N, "seed": seed, "z": z.tolist(),
                     "base_mae": base_mae, "cand_gain": cg,
                     "cand_mae": {m: v[m] for m in cg},
                     "best_cand": best_m, "best_gain": cg[best_m]})
    return rows, cand


def run():
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor

    rows, cand = build()
    datasets = sorted(set(r["ds"] for r in rows))
    Z = np.array([r["z"] for r in rows])
    recs = []
    gp_all, gt_all = [], []
    for held in datasets:
        tri = [i for i, r in enumerate(rows) if r["ds"] != held]
        tei = [i for i, r in enumerate(rows) if r["ds"] == held]
        if not tri or not tei:
            continue
        sc = StandardScaler().fit(Z[tri]); Ztr, Zte = sc.transform(Z[tri]), sc.transform(Z[tei])
        # per-候选 gain 回归 + bagged for trust（用不同子样本估分歧）
        regs, bags = {}, {}
        for m in cand:
            xy = [(k, rows[k]["cand_gain"][m]) for k in tri if m in rows[k]["cand_gain"]]
            if len(xy) < 8:
                continue
            idx = [k for k, _ in xy]; yv = np.array([g for _, g in xy])
            Xk = sc.transform(Z[idx])
            regs[m] = RandomForestRegressor(n_estimators=200, max_depth=4, random_state=0).fit(Xk, yv)
            bags[m] = [RandomForestRegressor(n_estimators=60, max_depth=4, random_state=b).fit(
                Xk[np.random.default_rng(b).integers(0, len(Xk), len(Xk))],
                yv[np.random.default_rng(b).integers(0, len(yv), len(yv))]) for b in range(8)]
        for local, i in enumerate(tei):
            r = rows[i]
            # 对每候选预测 gain，取 pred 最优者作 proposal
            preds = {m: float(regs[m].predict(Zte[local].reshape(1, -1))[0]) for m in regs}
            if not preds:
                continue
            prop = max(preds, key=preds.get)
            pg = preds[prop]
            # trust = bag 分歧反向（分歧小→可信）
            bag_preds = np.array([b.predict(Zte[local].reshape(1, -1))[0] for b in bags[prop]])
            disagree = float(bag_preds.std())
            tg = r["cand_gain"].get(prop, -1.0)        # 该 proposal 的真实 gain
            gp_all.append(pg); gt_all.append(tg)
            recs.append({"ds": r["ds"], "N": r["N"], "seed": r["seed"], "prop": prop,
                         "pred_gain": round(pg, 4), "true_gain": round(tg, 4),
                         "disagree": round(disagree, 4),
                         "base_mae": r["base_mae"], "prop_mae": r["cand_mae"][prop],
                         "best_gain": round(r["best_gain"], 4)})
    return recs, np.array(gp_all), np.array(gt_all)


def _corr(x, y):
    if len(x) < 3 or len(set(x.tolist())) < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def analyze(recs, gp, gt):
    n = len(recs)
    out = {"n_cells": n, "base_is_best_frac": round(float(np.mean([r["best_gain"] <= 0 for r in recs])), 3),
           "gain_pred_corr": round(_corr(gp, gt), 3),
           "mean_true_best_gain": round(float(np.mean([r["best_gain"] for r in recs])), 4)}
    # 策略：相对 MAE 改善 vs base（>0 = 比 chronos2 好）
    def rel(action_mae, r):
        return (r["base_mae"] - action_mae) / (r["base_mae"] + 1e-9)
    out["always-base"] = 0.0
    out["always-prop"] = round(float(np.mean([rel(r["prop_mae"], r) for r in recs])), 4)
    for tg in [0.0, 0.05, 0.1]:
        out[f"gain-gate(g>{tg})"] = round(float(np.mean([
            rel(r["prop_mae"] if r["pred_gain"] > tg else r["base_mae"], r) for r in recs])), 4)
    out["oracle-rel"] = round(float(np.mean([max(0.0, r["best_gain"]) for r in recs])), 4)
    # safe-deviation-rate at g>0.05
    dev = [r for r in recs if r["pred_gain"] > 0.05]
    safe = sum(1 for r in dev if r["true_gain"] >= 0)
    out["gate_calls"] = len(dev)
    out["safe_dev_rate"] = round(safe / len(dev), 3) if dev else float("nan")
    return out


def main():
    out = Path("research/results/m17_forecast_gain.jsonl")
    print("=== E5 · Forecasting Gain (LODO, 6 datasets) ===", flush=True)
    recs, gp, gt = run()
    res = analyze(recs, gp, gt)
    for k in ["n_cells", "base_is_best_frac", "gain_pred_corr", "mean_true_best_gain",
              "always-base", "always-prop", "gain-gate(g>0.0)", "gain-gate(g>0.05)",
              "gain-gate(g>0.1)", "oracle-rel", "gate_calls", "safe_dev_rate"]:
        print(f"  {k:20}: {res.get(k)}", flush=True)
    with out.open("w") as fh:
        fh.write(json.dumps({"_summary": res}, ensure_ascii=False) + "\n")
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
