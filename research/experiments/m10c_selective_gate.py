"""#86c · Selective-deviation gate（直击 F-R9.2：belief 强度是反指标）。

F-R9.2 实测：偏离 default 时 belief 在**错时反而更高**（0.79>0.51），raw belief 强度做门控会
误偏离。F-R9.5 又证伪了"换更强 embedding"。→ 本轮不换模型，改两件事：

  (1) Bagged belief：K 个 bootstrap belief head 的 softmax 平均，压掉 F-R9.5 的尖锐过拟合方差。
  (2) Calibrated selective gate：训一个二级分类器 g(belief-shape) → P(本次偏离是对的)，
      **只在 g 的学习概率 > 阈值时才偏离 rocket**，而不是信 raw belief 强度。

诚实性 = **nested LODO**（关键）：
  外层 LODO 留出 D_test；训练集 = 其余数据集。
  在训练集上再做**内层 LODO** 生成 out-of-fold (belief-shape, 偏离对/错) 对 → 训 gate。
  → gate 的训练标签从不来自 D_test，belief head 也从不预测自己训过的 cell。无泄漏。

对照：Rocket-alone / M10 raw-belief(τ0.3) / M10c gated。
输出：results/m10c_selective_gate.jsonl
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from research.experiments.m10_learned_belief import (
    build_dataset, CLF_ORDER, ROCKET_I, _shape_feats,
)

N_BAG = 25          # bagging heads
GATE_THRESHOLDS = [0.5, 0.6, 0.7]


def _fit_bagged_belief(Xz_tr, win_tr, classes, n_bag=N_BAG, seed0=0):
    """K 个 bootstrap LogisticRegression，返回 predict→平均 softmax 的闭包。"""
    from sklearn.linear_model import LogisticRegression
    heads = []
    rng = np.random.default_rng(seed0)
    n = len(Xz_tr)
    for b in range(n_bag):
        idx = rng.integers(0, n, n)               # bootstrap resample
        yb = win_tr[idx]
        if len(set(yb.tolist())) < 2:
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xz_tr[idx], yb)
        heads.append((clf, list(clf.classes_)))
    if not heads:
        return None

    def predict_belief(Xz):
        acc = np.zeros((len(Xz), len(CLF_ORDER)))
        for clf, cls in heads:
            p = clf.predict_proba(Xz)
            for ci, c in enumerate(cls):
                acc[:, c] += p[:, ci]
        acc /= len(heads)
        return acc
    return predict_belief


def _gate_features(belief_row):
    """gate 输入：belief 形状 + top/rocket 强度（全是分布形状量，不含原始 z）。"""
    b = belief_row
    s = _shape_feats(b)
    best_i = int(np.argmax(b))
    return [
        s["entropy"], s["gini"], s["top1_top2_gap"], s["tail_mass"],
        float(b[best_i]), float(b[ROCKET_I]), float(b[best_i] - b[ROCKET_I]),
    ]


def _inner_lodo_gate_data(info, X, win, train_ds):
    """内层 LODO：在 train_ds 上生成 out-of-fold gate 训练数据。
    只对"argmax 偏离 rocket"的 cell 产生样本（label=偏离是否正确）。"""
    from sklearn.preprocessing import StandardScaler
    Gx, Gy = [], []
    idx_train = [j for j, it in enumerate(info) if it["ds"] in train_ds]
    for held in train_ds:
        inner_tr = [j for j in idx_train if info[j]["ds"] != held]
        inner_te = [j for j in idx_train if info[j]["ds"] == held]
        if not inner_te or not inner_tr:
            continue
        if len(set(win[inner_tr].tolist())) < 2:
            continue
        sc = StandardScaler().fit(X[inner_tr])
        pb = _fit_bagged_belief(sc.transform(X[inner_tr]), win[inner_tr],
                                None, seed0=hash(held) % 10000)
        if pb is None:
            continue
        B = pb(sc.transform(X[inner_te]))
        for row, j in enumerate(inner_te):
            b = B[row]
            best_i = int(np.argmax(b))
            if best_i == ROCKET_I:
                continue                      # 不偏离 → gate 无需介入
            accs = info[j]["accs"]
            correct = int(accs.get(CLF_ORDER[best_i], 0.0) >= accs.get("rocket", 0.0))
            Gx.append(_gate_features(b)); Gy.append(correct)
    return np.array(Gx), np.array(Gy)


def run(threshold=0.6):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    X, A, info = build_dataset()
    datasets = sorted(set(i["ds"] for i in info))
    win = np.full(len(info), -1, dtype=int)
    for i, a in enumerate(A):
        valid = np.where(~np.isnan(a))[0]
        win[i] = int(valid[np.argmax(a[valid])])

    records = []
    for D_test in datasets:
        train_ds = [d for d in datasets if d != D_test]
        te = [j for j, it in enumerate(info) if it["ds"] == D_test]
        tr = [j for j, it in enumerate(info) if it["ds"] != D_test]
        if not te or not tr or len(set(win[tr].tolist())) < 2:
            for j in te:
                records.append(_rec(info[j], "rocket", win[j], 0.0, False, gated=True))
            continue

        # gate 训练数据（内层 LODO，无 D_test）
        Gx, Gy = _inner_lodo_gate_data(info, X, win, train_ds)
        gate = None
        if len(Gx) >= 6 and len(set(Gy.tolist())) == 2:
            gsc = StandardScaler().fit(Gx)
            gate = LogisticRegression(max_iter=2000, C=1.0)
            gate.fit(gsc.transform(Gx), Gy)
            gate = (gate, gsc)

        # 最终 belief head（全 train）
        sc = StandardScaler().fit(X[tr])
        pb = _fit_bagged_belief(sc.transform(X[tr]), win[tr], None, seed0=7)
        B = pb(sc.transform(X[te]))
        for row, j in enumerate(te):
            b = B[row]
            best_i = int(np.argmax(b))
            chosen = "rocket"
            p_correct = 1.0
            if best_i != ROCKET_I:
                if gate is None:
                    chosen = "rocket"          # 无 gate → 保守退回
                else:
                    g, gsc = gate
                    p_correct = float(g.predict_proba(gsc.transform([_gate_features(b)]))[0, 1])
                    chosen = CLF_ORDER[best_i] if p_correct >= threshold else "rocket"
            records.append(_rec(info[j], chosen, win[j], p_correct, chosen != "rocket", gated=True))
    return records


def _rec(it, chosen, oracle_idx, p_correct, deviated, gated):
    accs = it["accs"]
    return {
        "ds": it["ds"], "N": it["N"], "seed": it["seed"], "chosen": chosen,
        "chosen_acc": float(accs.get(chosen, 0.0)),
        "rocket_acc": float(accs.get("rocket", 0.0)),
        "oracle_acc": float(max(accs.values())),
        "oracle_clf": CLF_ORDER[oracle_idx] if oracle_idx >= 0 else None,
        "gate_p_correct": round(p_correct, 3),
        "deviated": bool(deviated),
        "correct_deviation": bool(deviated and accs.get(chosen, 0.0) >= accs.get("rocket", 0.0)),
    }


def summarize(records, threshold):
    n = len(records)
    sel = sum(r["chosen_acc"] for r in records) / n * 100
    roc = sum(r["rocket_acc"] for r in records) / n * 100
    ora = sum(r["oracle_acc"] for r in records) / n * 100
    dev = [r for r in records if r["deviated"]]
    dev_ok = [r for r in dev if r["correct_deviation"]]
    return {
        "threshold": threshold, "n": n,
        "gated_acc": round(sel, 2), "rocket_acc": round(roc, 2), "oracle_acc": round(ora, 2),
        "vs_rocket_pp": round(sel - roc, 2), "regret_pp": round(ora - sel, 2),
        "n_deviations": len(dev),
        "deviation_precision": round(len(dev_ok) / len(dev), 3) if dev else float("nan"),
    }


def main():
    out = Path("research/results/m10c_selective_gate.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    print("=== M10c Selective-Deviation Gate · nested LODO UCR-10 ===", flush=True)
    all_summ, best = [], None
    for th in GATE_THRESHOLDS:
        recs = run(threshold=th)
        s = summarize(recs, th)
        all_summ.append(s)
        if th == 0.6:
            best = recs
        print(f"threshold={th}: gated={s['gated_acc']}% (vs Rocket {s['rocket_acc']}% "
              f"= {s['vs_rocket_pp']:+}pp) oracle={s['oracle_acc']}% regret={s['regret_pp']}pp "
              f"| dev={s['n_deviations']} prec={s['deviation_precision']}", flush=True)
    with out.open("w") as fh:
        for s in all_summ:
            fh.write(json.dumps({"_summary": s}, ensure_ascii=False) + "\n")
        for r in best:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    # per-dataset for threshold 0.6
    byds = defaultdict(list)
    for r in best:
        byds[r["ds"]].append(r)
    print("\n=== per-dataset (threshold=0.6) ===", flush=True)
    for ds in sorted(byds):
        rs = byds[ds]; nn = len(rs)
        roc = sum(x["rocket_acc"] for x in rs) / nn * 100
        gat = sum(x["chosen_acc"] for x in rs) / nn * 100
        nd = sum(x["deviated"] for x in rs)
        print(f"  {ds:12} n={nn} rocket={roc:.1f} gated={gat:.1f} ({gat-roc:+.1f}) dev={nd}", flush=True)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
