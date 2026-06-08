"""SignalRouter v2 —— 双信号解耦决策（research Round 12 / F-R12.5 落地）。

彻底改进 test/ 的决策方法。核心结论（finish7 F-R12.5）：
  "避险"和"获利"是两个需不同信号的子问题，不能用单一 trust/confidence：
    - 避险（偏离会不会变差）→ conformal nonconformity（AUC 0.79）
    - 获利（偏离能不能净赚）→ saturation/gain 预测（AUC 0.71）
  决策律：**deviate iff  conformal_safe(z)  AND  saturation_headroom(z)**；否则守 base。

与旧 RouterAgent 的区别：
  旧 = 单 CV margin（+ 可选单 trust 门）；
  新 = 两个独立、各取所长的信号 AND 门 + belief 提候选（proposal）。

诚实：全 leave-one-dataset-out；两个信号都只用其它域历史 outcome 训练；无 test 泄漏。
本模块在 **oracle 库**（per-cell per-classifier acc）上做 LODO 决策，全量 10 分类器 + 全量数据集。
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# 全量分类器库（method/CLF 名映射，与 research oracle 库 schema 对齐）
METHOD_TO_CLF = {
    "B1_dtw": "dtw_1nn", "B2_euclid": "euclid_1nn", "B3_rocket": "rocket",
    "B4a_moment_1nn": "moment_1nn", "B4b_moment_lr": "moment_logreg",
    "B7_catch22": "catch22", "B8_mantis_1nn": "mantis_1nn", "B9_mantis_lr": "mantis_lr",
    "B10_minirocket": "minirocket", "B11_weasel": "weasel",
    # --- 扩展库（baseline.md §1.1 新增 12 个；taskb_newlib sweep）---
    "E1_multirocket": "multirocket", "E2_arsenal": "arsenal", "E3_drcif": "drcif",
    "E4_fcn": "fcn", "E5_resnet": "resnet", "E6_inceptiontime": "inceptiontime",
    "E7_chronos2_emb": "chronos2_emb", "E8_timesfm_emb": "timesfm_emb", "E9_timer_emb": "timer_emb",
    "E10_muse": "muse", "E11_rocket_mv": "rocket_mv", "E12_cif_mv": "cif_mv",
}
CLF_ORDER = ["rocket", "moment_1nn", "moment_logreg", "dtw_1nn", "euclid_1nn",
             "catch22", "mantis_1nn", "mantis_lr", "minirocket", "weasel",
             # 扩展库
             "multirocket", "arsenal", "drcif", "fcn", "resnet", "inceptiontime",
             "chronos2_emb", "timesfm_emb", "timer_emb", "muse", "rocket_mv", "cif_mv"]
ROCKET_I = CLF_ORDER.index("rocket")
BASE = "rocket"
# 全量 oracle 库（合并多 sweep）：
#   UCR 24 = 22(原) + FordA/FordB；UEA 14 多变量（rocket/euclid/dtw + channel-flatten 7）
#   taskb_newlib = 扩展库 12 分类器（capped test 150，见 taskb_newlib_sweep）
SWEEPS = ["taskb_ucr", "taskb_extended_ucr", "taskb_expand_ucr", "taskb_libplus_ucr",
          "taskb_ford_fulllib",                       # +UCR FordA/FordB（全量 10-clf）
          "taskb_uea_full", "taskb_uea_libplus",      # +UEA 14 多变量（库 3→10）
          "taskb_newlib"]                             # +扩展库 12 分类器（4 类）


def load_oracle_library():
    """返回 {(ds,N,seed): {clf: acc}}，合并全量 sweep。"""
    by = defaultdict(dict)
    for name in SWEEPS:
        fp = ROOT / "research" / "results" / f"{name}.jsonl"
        if not fp.exists():
            continue
        for l in fp.read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            clf = METHOD_TO_CLF.get(r["method"])
            if clf and not (isinstance(r["acc"], float) and r["acc"] != r["acc"]):
                by[(r["dataset"], r["N_per_class"], r["seed"])][clf] = float(r["acc"])
    return {k: v for k, v in by.items() if BASE in v and len(v) >= 8}


def _featurize_cells(cells):
    """每 cell 的轻量特征（部署可得、无 test）：N、类数代理、base 绝对水平、候选数。
    注：这是系统级元特征（不含'哪个模型赢'）。"""
    keys = sorted(cells.keys())
    X = []
    for k in keys:
        ds, N, seed = k
        v = cells[k]
        X.append([float(N), float(len(v)), float(v[BASE]), float(np.log1p(N))])
    return keys, np.array(X)


def run_signal_router(cells, trust_tau=0.5, head_tau=0.0, policy="conservative"):
    """全量 LODO 决策。policy 控制激进度：
      - conservative（默认/双信号解耦）：deviate iff trust≥trust_tau AND headroom≥head_tau —— 损害控制、收敛守 base。
      - aggressive：**丢弃 conformal 安全门**，只要预测有头寸(headroom≥head_tau)就偏离到 belief 提议候选。
        哲学（按 method 结论）：oracle 头寸真实(5.95pp)、winner 不可预测——既然抓不准就**主动出手抓**，
        不追求 vs-base pp，转而最大化"尝试捕获 oracle 头寸"；价值看 oracle-capture / 命中率，不看 +pp。
      - aggressive-belief：始终路由到 belief-argmax 提议候选（最大激进，无视任何门）。
      - explore（F-R12.6 信息增益）：高头寸 OR 低 trust 区主动偏离，把"未知未知"当探索机会。
    返回逐 cell 记录。"""
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LogisticRegression

    keys, X = _featurize_cells(cells)
    n = len(keys)
    # 标签：oracle winner（proposal/belief 用）、gap（saturation/headroom 用）
    win = np.zeros(n, dtype=int); gap = np.zeros(n)
    acc_mat = np.full((n, len(CLF_ORDER)), np.nan)
    for i, k in enumerate(keys):
        v = cells[k]
        for ci, c in enumerate(CLF_ORDER):
            if c in v:
                acc_mat[i, ci] = v[c]
        valid = np.where(~np.isnan(acc_mat[i]))[0]
        win[i] = int(valid[np.argmax(acc_mat[i][valid])])
        gap[i] = float(acc_mat[i][valid].max() - v[BASE])
    ds_of = [k[0] for k in keys]
    datasets = sorted(set(ds_of))

    recs = []
    for held in datasets:
        tr = np.array([i for i in range(n) if ds_of[i] != held])
        te = np.array([i for i in range(n) if ds_of[i] == held])
        if len(tr) == 0 or len(te) == 0 or len(set(win[tr].tolist())) < 2:
            for i in te:
                recs.append(_rec(keys[i], cells[keys[i]], BASE, "no-train", 0, 0, oracle_i=win[i]))
            continue
        sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        # proposal: belief = LogisticRegression 预测 winner（提候选）
        belief = LogisticRegression(max_iter=2000).fit(Xtr, win[tr])
        bclasses = list(belief.classes_)
        # 获利信号: saturation-headroom = gap 回归
        head = RandomForestRegressor(n_estimators=200, max_depth=4, random_state=0).fit(Xtr, gap[tr])
        gmin, gmax = float(gap[tr].min()), float(gap[tr].max()); gspan = (gmax - gmin) or 1.0
        # 避险信号: conformal —— 训练域 split 算 nonconformity 分位
        perm = np.random.default_rng(0).permutation(len(tr))
        cal = tr[perm[len(tr) // 2:]]
        # 用 belief 对 cal 的 P(winner) 算 nonconformity（1 - P(true winner)）
        Pcal = belief.predict_proba(sc.transform(X[cal]))
        cal_alpha = []
        for r_i, gi in enumerate(cal):
            wi = win[gi]
            p = Pcal[r_i, bclasses.index(wi)] if wi in bclasses else 0.0
            cal_alpha.append(1.0 - p)
        cal_alpha = np.sort(np.array(cal_alpha)) if cal_alpha else np.array([1.0])

        Pte = belief.predict_proba(Xte)
        head_pred = head.predict(Xte)
        for r_i, i in enumerate(te):
            p = Pte[r_i]
            order = [bclasses[j] for j in np.argsort(-p)]
            prop = next((c for c in order if c != ROCKET_I), ROCKET_I)   # 提议非 base 候选
            # 避险信号: conformal trust（提议候选的可信度 = 这点在不在已见分布内）
            a_new = 1.0 - float(p.max())
            trust = float((cal_alpha >= a_new).mean())
            # 获利信号: 归一 headroom（预测 gap 越大越有头寸）
            headroom = float(np.clip((head_pred[r_i] - gmin) / gspan, 0, 1))
            # 决策：按 policy 选激进度
            has_prop = prop != ROCKET_I
            if policy == "conservative":
                deviate = has_prop and trust >= trust_tau and headroom >= head_tau
                mode_tag = "safe&headroom"
            elif policy == "aggressive":
                deviate = has_prop and headroom >= head_tau          # 丢 trust 门，只看头寸
                mode_tag = "aggr-headroom"
            elif policy == "aggressive-belief":
                deviate = has_prop                                   # 始终听 belief 提议
                mode_tag = "aggr-belief"
            elif policy == "explore":
                deviate = has_prop and (headroom >= head_tau or trust < trust_tau)
                mode_tag = "explore"
            else:
                deviate = has_prop and trust >= trust_tau and headroom >= head_tau
                mode_tag = "safe&headroom"
            if deviate:
                chosen = CLF_ORDER[prop]; mode = f"deviate({mode_tag})"
            else:
                chosen = BASE
                mode = "commit-base(" + ("low-trust" if trust < trust_tau else "no-headroom") + ")"
            recs.append(_rec(keys[i], cells[keys[i]], chosen, mode, trust, headroom, oracle_i=win[i]))
    return recs


def _rec(key, accs, chosen, mode, trust, headroom, oracle_i=None):
    ds, N, seed = key
    base = accs.get(BASE, 0.0); oracle = max(accs.values())
    ca = accs.get(chosen, base)
    oracle_clf = CLF_ORDER[oracle_i] if oracle_i is not None and oracle_i < len(CLF_ORDER) else None
    return {"ds": ds, "N_per_class": N, "seed": seed, "chosen": chosen, "mode": mode,
            "trust": round(float(trust), 3), "headroom": round(float(headroom), 3),
            "acc": round(ca, 4), "base_acc": round(base, 4), "oracle_acc": round(oracle, 4),
            "deviated": chosen != BASE,
            "correct_deviation": (chosen != BASE) and ca >= base,
            "hit_oracle": (chosen == oracle_clf)}


UEA_DATASETS = {"BasicMotions", "ERing", "AtrialFibrillation", "Cricket", "Handwriting",
                "Libras", "UWaveGestureLibrary", "ArticularyWordRecognition", "Epilepsy",
                "NATOPS", "RacketSports", "HandMovementDirection", "FingerMovements", "Heartbeat"}


def domain_of(ds):
    return "UEA" if ds in UEA_DATASETS else "UCR"


def summarize(recs):
    n = len(recs)
    sys_a = np.mean([r["acc"] for r in recs]) * 100
    base_a = np.mean([r["base_acc"] for r in recs]) * 100
    orc_a = np.mean([r["oracle_acc"] for r in recs]) * 100
    dev = [r for r in recs if r["deviated"]]
    safe = sum(1 for r in dev if r["acc"] >= r["base_acc"])
    hit = sum(1 for r in dev if r.get("hit_oracle"))
    headroom_pp = orc_a - base_a                      # 可达 oracle 头寸
    captured_pp = sys_a - base_a                      # 实际相对 base 捕获(可负)
    # oracle-capture：在偏离的 cell 上，捕获了多少该 cell 的可达头寸
    dev_head = sum((r["oracle_acc"] - r["base_acc"]) for r in dev)
    dev_cap = sum((r["acc"] - r["base_acc"]) for r in dev)
    return {"n_cells": n, "n_datasets": len(set(r["ds"] for r in recs)),
            "system_acc": round(sys_a, 2), "base_acc": round(base_a, 2), "oracle_acc": round(orc_a, 2),
            "system_vs_base_pp": round(captured_pp, 2),
            "regret_to_oracle_pp": round(orc_a - sys_a, 2),
            "oracle_headroom_pp": round(headroom_pp, 2),
            "headroom_capture_rate": round(captured_pp / headroom_pp, 3) if headroom_pp > 1e-9 else float("nan"),
            "n_deviations": len(dev), "deviation_rate": round(len(dev) / n, 3),
            "oracle_hit_rate": round(hit / len(dev), 3) if dev else float("nan"),
            "safe_deviation_rate": round(safe / len(dev), 3) if dev else float("nan"),
            "dev_cell_capture_rate": round(dev_cap / dev_head, 3) if dev_head > 1e-9 else float("nan")}
