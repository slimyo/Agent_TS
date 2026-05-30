"""P0-A · 论文 main comparisons 的 Wilcoxon signed-rank 检验。

对每对 (method_A, method_B)，per-cell paired difference → Wilcoxon。
报告 p-value、median Δ、effect size r。
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def load_jsonl(p):
    if not Path(p).exists(): return []
    return [json.loads(l) for l in open(p)]


def cell_acc(rows, method, key_fn=None):
    """Return {cell_key: acc/mae} for a method."""
    out = {}
    for r in rows:
        if method and r.get("method") != method: continue
        k = key_fn(r) if key_fn else (r["dataset"], r["N"], r["seed"])
        out[k] = r.get("mae", r.get("acc"))
    return out


def paired_test(name_a: str, a_dict: dict, name_b: str, b_dict: dict,
                higher_is_better: bool = True):
    """A vs B paired Wilcoxon."""
    shared = sorted(a_dict.keys() & b_dict.keys())
    if len(shared) < 5:
        print(f"  [{name_a} vs {name_b}] only {len(shared)} paired cells, skip")
        return
    diffs = np.array([a_dict[k] - b_dict[k] for k in shared])
    median_diff = float(np.median(diffs))
    mean_diff = float(np.mean(diffs))
    try:
        stat, p = wilcoxon(diffs, zero_method="pratt")
    except ValueError:
        p = 1.0; stat = 0.0
    # Effect size r = Z / sqrt(N)
    from scipy.stats import norm
    n = len(diffs)
    z = float(stat / np.sqrt(n * (n + 1) * (2 * n + 1) / 6)) if stat else 0.0
    r = abs(z) / np.sqrt(n) if n else 0.0
    direction = "↑" if median_diff > 0 else ("↓" if median_diff < 0 else "=")
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"  [{name_a} vs {name_b}] n={n} median Δ={median_diff:+.4f} {direction}  "
          f"mean Δ={mean_diff:+.4f}  W={stat:.1f}  p={p:.4f} {sig}  r={r:.3f}")
    return {"name_a": name_a, "name_b": name_b, "n": n,
            "median_diff": median_diff, "mean_diff": mean_diff,
            "p_value": p, "effect_size_r": r, "significance": sig}


def main():
    out_records = []

    # ============ 1) Forecasting MAE: v11 / v12 / v13 / v10 vs Chronos-2 ============
    print("\n=== §4 Forecasting: v* vs Chronos-2 (MAE) ===")
    c2_4 = load_jsonl("research/results/f4_chronos2.jsonl")
    c2_ecl = load_jsonl("research/results/f4_bolt_c2_ecl_exchange.jsonl")
    c2_wi = load_jsonl("research/results/f4_bolt_c2_weather.jsonl") + load_jsonl("research/results/f4_bolt_c2_ili.jsonl")
    c2_all = c2_4 + [r for r in c2_ecl if r.get("method")=="chronos2"] + [r for r in c2_wi if r.get("method")=="chronos2"]
    c2_dict = cell_acc(c2_all, method=None)

    v11 = load_jsonl("research/results/p11_phaseA_populate.jsonl") + load_jsonl("research/results/p11_adapt_weather_ili.jsonl")
    v11_dict = cell_acc(v11, method=None)
    out_records.append(paired_test("v11", v11_dict, "Chronos-2", c2_dict, higher_is_better=False))

    v10_n10 = load_jsonl("research/results/p10_adapt_v10_n10.jsonl")
    v9 = load_jsonl("research/results/p9_adapt_v9.jsonl")
    v10_dict = {**cell_acc(v10_n10, None), **{k:v for k,v in cell_acc(v9, None).items() if k[1] != 10}}
    out_records.append(paired_test("v10", v10_dict, "Chronos-2", c2_dict, higher_is_better=False))

    v12 = load_jsonl("research/results/p12_adapt_v12.jsonl")
    v12_dict = cell_acc(v12, None)
    out_records.append(paired_test("v12", v12_dict, "Chronos-2", c2_dict, higher_is_better=False))

    # ============ 2) Forecasting CRPS: v11 vs Chronos-2 ============
    print("\n=== §4.9 Forecasting CRPS: v11/v12 vs Chronos-2 ===")
    c2_crps = load_jsonl("research/results/a3_prob_metrics.jsonl")
    c2_crps_dict = {(r["dataset"], r["N"], r["seed"]): r["crps"] for r in c2_crps if r.get("method")=="chronos2"}
    # For v11: equals C2 → trivial; v12 should differ
    v11_crps = c2_crps_dict.copy()  # v11/v13 always = C2 on parity cells
    # Compute v10/v12 CRPS = MAE on deviation cells, c2_crps on parity cells
    # Quick: use v12 mae as crps for deviation, c2_crps for parity
    v12_crps_dict = {}
    for k, mae_ in v12_dict.items():
        c2_mae = c2_dict.get(k)
        if c2_mae is None: continue
        if abs(mae_ - c2_mae) / max(c2_mae, 1e-9) < 0.005:
            v12_crps_dict[k] = c2_crps_dict.get(k, mae_)
        else:
            v12_crps_dict[k] = mae_
    out_records.append(paired_test("v12 CRPS (proxy)", v12_crps_dict, "Chronos-2 CRPS", c2_crps_dict, higher_is_better=False))

    # ============ 3) TSC UCR-5: B7v3 vs Rocket ============
    print("\n=== §4.5/4.6 TSC: B7v3 / B7v1 vs Rocket ===")
    ucr_rows = load_jsonl("research/results/taskb_ucr.jsonl")
    rocket_dict = {(r["dataset"], r["N_per_class"], r["seed"]): r["acc"]
                    for r in ucr_rows if r["method"] == "B3_rocket"}
    b7v3_rows = load_jsonl("research/results/taskb_router_v3_ucr.jsonl")
    b7v3_dict = {(r["dataset"], r["N_per_class"], r["seed"]): r["acc"] for r in b7v3_rows}
    out_records.append(paired_test("B7v3 Router", b7v3_dict, "Rocket alone", rocket_dict))

    b7v1_rows = load_jsonl("research/results/taskb_router_ucr.jsonl")
    b7v1_dict = {(r["dataset"], r["N_per_class"], r["seed"]): r["acc"] for r in b7v1_rows}
    out_records.append(paired_test("B7v1 Router", b7v1_dict, "Rocket alone", rocket_dict))

    # B6 direct from taskb_ucr
    b6_dict = {(r["dataset"], r["N_per_class"], r["seed"]): r["acc"]
                for r in ucr_rows if r["method"] == "B6_agent"}
    out_records.append(paired_test("B6 Direct", b6_dict, "Rocket alone", rocket_dict))

    # ============ 4) TSC less-saturated extended ============
    print("\n=== §4.6 TSC less-saturated (B7v3 vs Rocket) ===")
    ext_rows = load_jsonl("research/results/taskb_extended_ucr.jsonl")
    rocket_ext = {(r["dataset"], r["N_per_class"], r["seed"]): r["acc"]
                   for r in ext_rows if r["method"] == "B3_rocket"}
    b7v3_ext = {(r["dataset"], r["N_per_class"], r["seed"]): r["acc"]
                 for r in ext_rows if r["method"] == "B7v3_router"}
    out_records.append(paired_test("B7v3 (extended)", b7v3_ext, "Rocket (extended)", rocket_ext))

    # ============ 5) RCA: Agent v1 vs B0-rule vs B1 LLM-direct ============
    print("\n=== §4.7 RCA natural failures (30 cells) ===")
    rca_v1 = load_jsonl("research/results/taska_rca_predictions_v1.jsonl")
    b0_rule = load_jsonl("research/results/taska_b0_rule_predictions.jsonl")
    # Binary correct/wrong
    def make_correct_dict(rows, pred_key, gt_key="ground_truth"):
        d = {}
        for r in rows:
            gt = r[gt_key]["primary_fault"]
            pred = r[pred_key]["primary_fault"]
            d[r["cell_id"]] = 1 if pred == gt else 0
        return d
    agent_v1_correct = make_correct_dict(rca_v1, "agent_pred")
    b1_correct = make_correct_dict(rca_v1, "b1_pred")
    b0_correct = make_correct_dict(b0_rule, "b0_pred")
    out_records.append(paired_test("Agent v1", agent_v1_correct, "B1 LLM-direct", b1_correct))
    out_records.append(paired_test("Agent v1", agent_v1_correct, "B0-rule", b0_correct))

    # ============ 6) RCA synthetic clean GT ============
    print("\n=== §4.7 RCA synthetic clean GT (50 cells) ===")
    synth = load_jsonl("research/results/taska_synthetic_rca.jsonl")
    syn_agent = {r["cell_id"]: 1 if r["agent_pred"]["primary_fault"] == r["ground_truth"]["primary_fault"] else 0
                  for r in synth}
    syn_b0 = {r["cell_id"]: 1 if r["b0_pred"]["primary_fault"] == r["ground_truth"]["primary_fault"] else 0
               for r in synth}
    syn_b1 = {r["cell_id"]: 1 if r["b1_pred"]["primary_fault"] == r["ground_truth"]["primary_fault"] else 0
               for r in synth}
    out_records.append(paired_test("Agent (clean GT)", syn_agent, "B0-rule (clean GT)", syn_b0))
    out_records.append(paired_test("Agent (clean GT)", syn_agent, "B1 (clean GT)", syn_b1))

    # ============ 7) OOT abstain head ============
    print("\n=== §4.7.3 OOT abstain head (50 OOT cells) ===")
    abst = load_jsonl("research/results/taska_abstain_eval.jsonl")
    oot_only = [r for r in abst if r["kind"] == "oot"]
    no_abs = {r["cell_id"]: 1 if r["agent_no_abstain"] == "out_of_taxonomy" else 0 for r in oot_only}
    w_abs = {r["cell_id"]: 1 if r["agent_w_abstain"] == "out_of_taxonomy" else 0 for r in oot_only}
    out_records.append(paired_test("Agent + abstain", w_abs, "Agent no abstain", no_abs))

    # ============ 8) Learned margin L1 vs heuristic ============
    # Already evaluated in agent/learned_margin.py LODO CV
    print("\n=== §4.5 Learned margin L1 vs heuristic (oracle gap closure) ===")
    print("  (5-fold CV: L1=0.8597 vs heuristic=0.8548, n=56 cells)")
    print("  Wilcoxon test on per-cell improvements available via learned_margin.py LODO output")

    # Save records
    out_path = Path("research/results/stat_tests.json")
    with out_path.open("w") as fh:
        json.dump(out_records, fh, indent=2, ensure_ascii=False)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
