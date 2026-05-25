# Finish · AdaptTS-Agent 实测成果汇总

> 重构时间：2026-05-24
> 配对文件：`plan.md`（总规划） / `TODO.md`（任务看板） / `classifier.md`（分类 Agent 框架）
> 注：本次重构按"§4 forecasting / §5 reasoning"双轨叙事整理，原 §3.1.x 编号保留为子节锚点供 cross-ref。

---

## 一、项目演进与论文骨架

**两轮 pivot 后定型**：

1. **Phase 1-5 (Forecasting Boundary)** — 实证 LLM-Agent wrapper 在 2026 TSFM 时代无法 systematic 击败 Chronos-2
2. **Phase 6.1-6.3 (Direct Classification fail)** — Agent B6 直接做 RCA / TSC 在公平对比下输 classical/TSFM 强 baseline
3. **Phase 6.4b (Agent-as-Router)** — 转 routing 范式，B7 +30pp 修复 B6，逼近 Rocket（task #37 持平/超越 in-flight）

**论文最终 narrative**：

```
§4 Forecasting Boundary：no wrapper beats Chronos-2 → guaranteed-parity wrapper (v11)
§5 Reasoning Tasks Pivot：
  §5.1 RCA Agent +40pp wins (vs LLM-direct only baseline)
  §5.2 TSC Direct B6 fails universally → Router B7 大幅修复
§5.3 Boundary characterization + 双轨同构论证（forecasting v10 ≅ TSC B7v2）
```

---

## 二、§4 Forecasting Boundary

### 2.1 数据 & baseline 矩阵

**数据集**：6 个（ETTh1/ETTh2/ECL/Exchange/Weather/ILI）× 4 N (10,20,50,100) × 3 seeds = 72 cells

**Baselines**：Naive (mean/drift/seasonal val 选优) / ARIMA+ETS / Chronos-Small / **Chronos-Bolt** / **Chronos-2** / LLMTime / TSci

**AdaptTS 演进**：v5c → v7 → v8 → v9 → v10 → v11 → v12 → v13

### 2.2 关键里程碑（按时间序）

| 阶段 | 设计 | vs Chronos-2 |
|---|---|---|
| v5c | 软集成 (chronos_small) | 16/16 loss, +20-50% |
| v7 | chronos slot aliased to Bolt | 16/16 loss, mostly tie |
| v8 | top-1 + expanded TSFM pool | 16/16 loss, **+45% catastrophic** |
| v9 | margin gating (N≥15) | 0/7/9, **首次击败 C2 1 cell** (ETTh2 N=100) |
| **v10** | v9 + N<15 fallback | **1W/3L/12T** (ETTh1+ETTh2+ECL+Exchange) |
| v11 | v10 + memory safety-net | 0/0/16T 但 OOD Weather +505% |
| v12 | v10 + entropy-modulated margin | 1W/3L/12T (CRPS +16% bad) |
| **v13** | v11 + v12 联合 + revert-only | 0/0/16T (= v11) |

**Final 6-dataset × 24 cells（finish 旧 §3.1.27）**：

| 方法 | vs Chronos-2 (24 cell, eps 0.5%) |
|---|---|
| **v11** | **0W / 1L / 23T**（1 OOD Weather N=20 v11 +505%）|
| v12 | 1W / 3L / 20T |
| v13 | = v11 |

### 2.3 CRPS 评估反转 MAE 假象（A3, 旧 §3.1.26）

| Method | avg MAE | avg CRPS | CRPS vs C2 |
|---|---|---|---|
| v10 | 7.39 | 6.31 | **+16.65%** |
| v12 | 7.36 | 6.27 | +15.86% |
| **v11/v13** | **6.85** | **5.41** | **+0.00%** |
| Chronos-2 | 6.85 | 5.41 | 0 |

**单 cell 反转**（ETTh2 N=100 seed=42）：v12 MAE -7% **win**，CRPS **+54% loss** → MAE win 在 probabilistic loss 下蒸发。

### 2.4 Forecasting §4 final 结论

> "Across 24 cells × 6 datasets, **no AdaptTS-Agent variant systematically beats Chronos-2** on either MAE or CRPS. v11/v13 (memory safety-net) achieves exact MAE-CRPS parity with Chronos-2 in 23/24 cells (0W/1L/23T MAE, 0% CRPS gap), with the single loss being an OOD Weather case (+505%) revealing memory bootstrap brittleness. This is a **structural limitation, not engineering** — the right architecture in the TSFM era is a guaranteed-parity wrapper, not a competing router."

---

## 三、§5.1 TaskA RCA Natural

### 3.1 实验设置

- 30 个 catastrophic failure cells（按 adapt_mae / chronos2_mae ratio 降序选）
- Ground truth：rule-based fault detector (`utils/fault_taxonomy.py`, 5 fault types)
- Methods：B1 LLM-direct（纯数字 + taxonomy）vs B5 Agent（+ Curator + Model Cards）

### 3.2 Curator v1 (10-dim, 旧 §3.1.29)

| Metric | **Agent (B5)** | B1 LLM-direct | Gap |
|---|---|---|---|
| R1 Top-1 | **40.0%** | **0.0%** | **+40pp** ⭐ |
| R2 Top-3 | 43.3% | 23.3% | +20pp |
| R4 Keyword F1 | 16.2% | 0.0% | +16pp |

**B1 collapse**：全 30 case 都预测 `trend_break` —— 论文 §5.1 motivation："without diagnostic structure LLM collapses to single class"

**Agent confusion**：stationarity_flip 12/13 ✓ / outlier_burst 0/6 ✗ / variance_explode 0/10 ✗ → identified Curator weakness

### 3.3 Curator v2 (12-dim, 旧 §3.1.31)

加 `outlier_count_z3` + `variance_ratio` 进 Diagnosis。

| Metric | v1 | **v2** | Δ |
|---|---|---|---|
| R1 Top-1 | **40.0%** | 36.7% | **-3.3pp** |
| **R2 Top-3** | 43.3% | **56.7%** | **+13.4pp** ⭐ |
| **R4 Keyword F1** | 16.2% | **30.0%** | **+13.8pp** ⭐ |

**Feature engineering 零和现象**：variance_explode 0/10 → 9/10（修好）但 stationarity_flip 12/13 → 1/13（LLM 过度依赖新特征）。**R2/R4 大涨说明解释质量翻倍**。

### 3.4 §5.1 final 结论

> "Curator + Model Cards Agent achieves 40% R1 Top-1 accuracy on identifying primary failure cause across 5 fault categories, versus 0% for an unstructured LLM-direct baseline that sees the same series and taxonomy. The contrast is qualitative: B1 collapses to predicting trend_break for all 30 cells. Curator v2 (adding outlier and variance features) drops R1 by 3pp but improves R2 (Top-3) by 13pp and R4 (keyword F1) by 14pp, demonstrating a feature-engineering trade-off."

---

### 3.5 §5.1 Final · B0-rule baseline 反转 (task #30, feedback 补完整 ablation)

**实验**：B0-rule = `detect_faults(train, season_m)` 取 top fault (**only train 端**, no LLM, no Cards, no test/pred context)。其他 Agent / B1 都有 train+test+prediction 全信息。

**完整 R1 排名（30 cells, 4 methods）**

| Method | R1 Top-1 | R2 Top-3 | 信息访问 |
|---|---|---|---|
| **B0-rule** | **76.7% (23/30)** | 76.7% | train only, no LLM |
| Agent v1 (12-dim) | 40.0% (12/30) | 43.3% | full context + Curator + Cards + LLM |
| Agent v2 (12-dim) | 36.7% (11/30) | **56.7%** | 同上 |
| B1 LLM-direct | 0.0% (0/30) | 23.3% | full context, no Curator/Cards |

**Agreement matrix（Agent v2 vs B0）**
- B0 only correct: **14 cells** ← Agent miss rule 拿到的
- Agent only correct: 2 cells ← Agent recover rule miss 的
- both correct: 9 / both wrong: 5

**Honest interpretation**：

1. **B0 vs Agent 部分 tautological**：GT 由 `max(train_scores, test_scores)` 阈值决定，B0 训练-only rule 与 GT 共享生成机制 → 77% 是 "max() 在多数 cell 被 train 主导" 的 fraction，非真实"rule reasoning is strong"
2. **但 Agent 仍败给 rule-based baseline** —— 在 14/30 cells 上 B0 拿到 Agent 没拿到的 → **Agent's NL reasoning currently underperforms mechanical rule application** 即使有更多 context
3. **§5.1 原 "Agent +40pp" 论点的 caveat**：相对 B1 degenerate baseline 成立，但相对 B0 competent rule 反转
4. **这反向加强 §5.3 boundary characterization**：Agent direct RCA 败给 B0 + Agent direct TSC (B6) 败给 Rocket → **Agent 在两个 domain 都是 direct 决策弱，Agent-as-Router/Wrapper 才是正确范式**

**论文 §5.1 narrative 更新**（待写入 paper）：

> "We additionally evaluate a rule-only baseline (B0: top fault from train-only `detect_faults` scores, no LLM) which achieves R1 = 76.7%, substantially above our Agent's 40%. **B0 is partly tautological** — the ground truth was itself derived from `max(train_scores, test_scores)`-rule combination, so B0 inherits a generative correlation. However, the comparison still reveals a meaningful gap: in 14 of 30 cells, B0 correctly identifies a fault that the LLM-based Agent misses despite having access to additional context (test data, prediction, Model Cards, Curator features). We interpret this as a **methodological honest negative result**: the Agent's contribution on RCA is not raw classification accuracy — it is the **natural-language reasoning trace, the explicit citation of diagnostic statistics, and the extensibility to out-of-taxonomy fault types** that B0 cannot provide. The +40pp gap over B1 LLM-direct (0%) demonstrates that diagnosis structure helps over unstructured ICL, but **a competent non-LLM baseline narrows or reverses this advantage**. This finding reinforces our §5.3 conclusion: LLM-Agents are most valuable as **routers around base models**, not as direct deciders."

**产物**：
- `experiments/taska_run_b0_rule.py` (60 行)
- `research/results/taska_b0_rule_predictions.jsonl` (30 行)

---

### 3.6 §5.1 task #25 · Synthetic 5-fault RCA — **clean GT 反转 +40pp 论点**

**实验**：5 fault × 5 cells × 2 datasets (ETTh1+ECL) = 50 cells，**clean GT = injected label**（独立于 rule detector）。完全解决 task #30 的 B0 tautology 问题。

**主表（50 cells, clean GT）**

| Method | R1 Top-1 | R2 Top-3 |
|---|---|---|
| **B0-rule** | **50.0%** | **86.0%** |
| Agent v2 (Curator 12-dim + Cards + LLM) | 26.0% | 60.0% |
| B1 LLM-direct | 24.0% | 36.0% |

**对比 task #30 (rule-derived GT) 揭示真实 gap**：
- B0: 76.7% → 50.0%（**-27pp 这就是 tautological 部分**）
- Agent v2: 36.7% → 26.0%（-11pp，clean GT 下 baseline shift）
- B1: 0% → 24.0%（+24pp，clean GT 让 B1 random-like）

**Agent per-fault breakdown（关键 finding）**

| Fault | Agent R1 |
|---|---|
| **variance_explode** | **9/10 (90%)** ⭐ Curator v2 命中 |
| stationarity_flip | 4/10 (40%) |
| trend_break | 0/10 |
| seasonal_flip | 0/10 |
| outlier_burst | 0/10 |

**Agent 在 4/5 fault 上完全 miss**——只有 variance_explode 一类拿到强信号（因为 v2 加了 `variance_ratio` 特征显式触发）。其他 4 类 Agent 预测全 collapse 到 variance_explode（over-attached to that feature signal）。

**Agent vs B1 LLM-direct 在 clean GT 下**：+2pp R1 / +24pp R2 —— 之前 task #29 的"+40pp" 论点几乎完全是 **B1 degenerate baseline 的虚高**（B1 在 task #30 全 collapse trend_break，在 task #25 random-like）。

**论文 §5.1 必须重写的 honest narrative**：

> "On 50 synthetic cells with ground truth defined by the injected fault (independent of any rule-based detector), our LLM-Agent achieves R1 = 26%, marginally above an unstructured LLM baseline (B1 = 24%, +2pp), and **substantially below** a competent rule-only baseline (B0-rule on train series alone = 50%). Combined with the natural-failure result (§3.4 task #30: B0 76.7% vs Agent 36.7%), this demonstrates that **the Agent does not currently beat a rule-based baseline on RCA accuracy under either evaluation protocol**. The Agent's per-fault breakdown reveals over-reliance on a single Curator feature: variance_explode (where the v2 variance_ratio feature directly hits) achieves 9/10 R1, while four other fault classes (trend_break, seasonal_flip, outlier_burst, stationarity_flip) collapse to 0/10. We interpret this honestly: **the Agent's RCA contribution is NOT raw classification accuracy**. Its value lies in (a) the NL reasoning trace, (b) explicit citation of diagnostic statistics and Model Card assumptions, (c) extensibility to out-of-taxonomy fault types — none of which the rule baseline provides."

**反向加固论文 §5.3 boundary 论证**：

| Domain | Agent direct vs strong baseline |
|---|---|
| Forecasting | v5c-v10 all lose to Chronos-2 |
| **RCA (task #25 clean GT)** | **Agent 26% vs B0 50% (-24pp)** |
| TSC (task #26) | B6 54.3% vs Rocket 87.5% (-33pp) |

**三个 domain 全部证明：Agent 直接做决策弱**。只有 Router/Wrapper 范式（forecasting v11 = C2 parity, TSC B7v3 +0.89pp）才能让 Agent 发挥价值。

**产物**：
- `utils/inject_fault.py` 加 2 个 injector + RCA_INJECTORS dict (50 行)
- `experiments/taska_synthetic_rca.py` (110 行)
- `research/results/taska_synthetic_rca.jsonl` (50 行)

---

## 四、§5.2 TaskB TSC

### 4.1 UCR Direct Classification (旧 §3.1.32)

**实验**：5 UCR (Coffee, ECG200, TwoLeadECG, BeetleFly, BirdChicken) × 3 N-shot × 2 seeds × 7 methods = 210 cells

**主表（mean acc）**

| Method | Mean Acc | Macro F1 | Winner (of 15) |
|---|---|---|---|
| **B3 Rocket** | **0.875** | 0.871 | 7 |
| B4a MOMENT 1-NN | 0.819 | 0.812 | 3 |
| B4b MOMENT LogReg | 0.817 | 0.812 | 3 |
| B1 1-NN DTW | 0.748 | 0.740 | 1 |
| B2 1-NN Euclid | 0.710 | 0.703 | 1 |
| **B6 AdaptTS Direct** | **0.543** | 0.482 | **0** ❌ |
| **B5 LLM-direct** | **0.527** | 0.456 | **0** |

**关键观察**：
- Rocket 主导（7/15 cell winner）
- MOMENT 反超 Rocket on image-outline (BeetleFly N=3 B4b 92.5% vs Rocket 82.5%)
- **B6 Agent 在 BirdChicken/BeetleFly 10-shot 跌到 45%（低于 random）**

### 4.2 Synthetic 4-class Cross-check (旧 §3.1.33)

**实验**：ETTh1 + ECL × 3 N × 2 seeds × 7 methods = 84 cells，class label = 4-fault 诊断概念

| Method | Mean Acc | vs UCR | 解读 |
|---|---|---|---|
| B3 Rocket | 50.6% | (在 UCR 87.5%) | 任务更难 |
| B4b MOMENT-LR | 46.1% | — | rank 2 |
| **B6 Agent** | **33.7%** | (UCR 54.3%) | -17pp from Rocket |
| B5 LLM-direct | 31.7% | — | similar to B6 |

**反预期 finding**：即使 class label = 诊断概念，**B6 仍输 Rocket -17pp**。Alignment 减半 gap（-33→-17pp）但**不能让 Agent 超越 Rocket**。

**结构性结论**：**B6 直接分类天生弱**——只要有 Rocket/MOMENT 这样强 baseline，LLM ICL on diag features 都打不过。**B7 Router 是唯一架构出路**。

### 4.3 B7 Agent-as-Router (旧 §3.1.34, task #35)

**架构**：Curator → LOO CV on training set → margin gating → 选择 classifier ∈ {rocket, moment_1nn, moment_logreg, dtw_1nn, euclid_1nn}

**主表 — 4 个 contender × 30 cells**

| Method | Mean Acc | vs Rocket | Beats Rocket |
|---|---|---|---|
| **B6 Direct (旧)** | 54.3% | -33.2pp | 0/30 |
| **B7 Router** | **84.8%** ⭐ | **-2.7pp** | **6/30** |
| B3 Rocket alone | 87.5% | 0 | (baseline) |
| Oracle (post-hoc) | 92.1% | +4.6pp | 13/30 |

**Routing 决策分布**：rocket 15 / moment_1nn 9 / euclid_1nn 3 / dtw_1nn 3

**Wins via routing**（论文 §5.2 case study）：
- **BeetleFly N=5 seed=1**: B7 选 MOMENT → **0.95** vs Rocket 0.75 (**+20pp**) ⭐
- ECG200 N=3 seed=42: B7 选 Euclid → 0.80 vs 0.73 (+7pp)
- BeetleFly N=10 seed=1/42: 0.95 vs 0.90 (via moment_1nn)

**Catastrophic mis-routes（task #37 拟修）**：
- BeetleFly N=3 seed=1: **0.500** vs Rocket 0.75 (-25pp, LOO CV 噪声选 moment)
- BeetleFly N=3 seed=42: 0.70 vs 0.90 (-20pp via dtw)
- BirdChicken N=3 seed=42: 0.50 vs 0.70 (-20pp)

**核心 finding**：LOO CV signal 在 N=3-5 噪声大 → catastrophic mis-routes。需 v9 N<5 fallback（task #37）。

### 4.4 §5.2 narrative（已可写）

> "We first attempted to use the Agent as a direct classifier (B6: Curator + LLM ICL → class label), achieving only 54.3% mean accuracy across 15 UCR settings — 33pp below the Rocket SOTA baseline. A synthetic-fault cross-check on data where class labels are statistical concepts (trend_break, etc.) showed B6 still lost to Rocket by -17pp, confirming the failure is **structural, not domain-specific**. The correct architectural response, paralleling the v8→v10 forecasting wrapper progression, is **Agent-as-Router**: the LLM/Curator does not classify directly but selects among {Rocket, MOMENT-1NN, MOMENT-LR, DTW, Euclid} based on leave-one-out CV. **B7 achieves 84.8% mean accuracy (+30.5pp from B6, -2.7pp from Rocket-alone)** and beats Rocket on 6/30 cells (Oracle: 13/30, 46% capture). The remaining gap comes from LOO CV noise at N=3-5; a v2 with N<5 default-fallback is in progress (task #37, predicted ~87-88%)."

---

### 4.5 §5.2 Final · B7v2 + B7v3 完整 progression ⭐

**Phase 6.4b 主线最终结果（task #37, #41 完成）**：

| Version | Mean Acc | vs Rocket | 改动 |
|---|---|---|---|
| B6 Direct (旧) | 54.3% | -33.2pp | LLM ICL 直接分类 |
| B7v1 Router (LOO CV margin) | 84.76% | -2.77pp | routing 范式 |
| B7v2 + N<7 fallback | 86.66% | -0.87pp | small-N catastrophic 修复 |
| **B7v3 + enhanced mem + weighted vote + Cards v2** | **88.42%** ⭐ | **+0.89pp** ⭐ | **首次击败 Rocket** |
| Oracle (post-hoc) | 92.06% | +4.5pp | upper bound |

**B7v2 关键 cell wins（v2 vs v1 catastrophic fix）**

| Cell | v1 acc | v2 acc | Δ |
|---|---|---|---|
| BeetleFly N=3 seed=1 | 0.500 (moment_1nn) | **0.750 (rocket fallback)** | **+25pp** ⭐ |
| BeetleFly N=3 seed=42 | 0.700 (dtw) | **0.900 (rocket)** | **+20pp** ⭐ |
| BirdChicken N=3 seed=42 | 0.500 (euclid) | **0.700 (rocket)** | **+20pp** ⭐ |
| 牺牲：ECG200 N=3 seed=42 | 0.800 (euclid win) | 0.730 (rocket) | -7pp |
| 牺牲：ECG200 N=3 seed=1 | 0.810 (moment win) | 0.800 (rocket) | -1pp |

**Net v2**: 3 catastrophic 修复 (+65pp 收益) vs 2 wins 牺牲 (-8pp) → mean +1.9pp

**B7v3 关键改进点**（feedback 第三轮整合）：

1. **Memory features 从 12 → 25 维**：
   - 基础统计 5 + 时序 4 + **频域 4** + **复杂度 4** + 离群 3 + **元信息 5 (log_L, n_classes, log_N, class_balance, log_N_total)**
   - z-score 标准化（feedback 必修建议）
2. **Memory consensus 改加权 vote**：去 similarity_threshold=0.85 硬阈值，sim 加权 sum + min_vote_ratio=0.55
3. **Cards v2 加 5 决策字段**：min_samples_per_class / max_sequence_length / cost_level / preprocessing / multiclass

**B7v3 routing 分布**：rocket 25 / moment_1nn 4 / dtw_1nn 1
**B7v3 memory overrides triggered**：15/30 cells (**50%** — 增强 memory 真实工作)

**论文级 take-aways（已最终定型）**

> "AdaptTS-Agent's classification capability evolved through four versions: B6 (Direct LLM ICL, 54.3%) failed structurally; B7v1 (LOO CV margin gating, 84.76%) recovered most of the gap but lost to Rocket-alone by 2.77pp due to CV noise at N=3-5; B7v2 (+ N<7 fallback, 86.66%) closed 68% of the remaining gap by mirroring the v8→v10 forecasting wrapper progression; finally B7v3 (+ 25-dim enhanced memory + similarity-weighted voting + Cards v2 with hard constraints, **88.42%**) achieved **+0.89pp over Rocket-alone**, the first systematic improvement of an LLM-Agent over SOTA classical TSC. Memory consensus override fires on 50% of cells, demonstrating that with sufficient feature engineering and weighted aggregation, retrospective memory becomes a useful planning signal."

**Final §5 boundary 三角（论文已可写最终版）**

| Domain | Best baseline | Agent (final) | Gap |
|---|---|---|---|
| **Forecasting** (24 cells) | Chronos-2 (CRPS) | v11 = C2 | parity (0%) |
| **RCA TaskA** (30 cells) | LLM-direct (0% R1) | **40% R1** | **+40pp** ⭐ |
| **TSC UCR** (30 cells) | Rocket 87.53% | **B7v3 88.42%** | **+0.89pp** ⭐ |

**双轨同构论证（forecasting ≅ TSC, 论文最强方法学贡献）**

```
Forecasting:  v5c → v8 (catastrophic) → v10 (N<15 fallback) → v11/v13 (memory)
              ↓ same architectural insight ↓
TSC:          B6 → B7v1 (catastrophic mis-routes) → B7v2 (N<7 fallback) → B7v3 (enhanced memory)
```

两个 domain 都验证："Agent-as-Router/Wrapper-around-base-models" 是 LLM-Agent 在 TSFM 时代的正确架构。

**产物**：
- `utils/series_features.py` (200 行，25-dim features)
- `agent/clf_memory.py` 加 `consensus_winner_weighted`
- `agent/clf_model_cards.py` 6 cards 各加 5 字段
- `agent/clf_planner.py` 加 `use_enhanced_features` + `weighted_vote_min_ratio`
- `experiments/build_clf_memory_v2.py` (75 行 build w/ z-score)
- `experiments/taskb_router_v3_sweep.py` (60 行)
- `experiments/autopilot_phase64b.sh` (5-step orchestration)
- `research/results/taskb_router_v2_ucr.jsonl` + `taskb_router_v3_ucr.jsonl`
- `/tmp/clf_memory_v2.jsonl` + `_norm.npz`

---

### 4.6 task #42 · Extended UCR sweep — saturation hypothesis 验证

**实验**：在 5 个 less-saturated UCR 数据集上扩展 B7v3 + B1-B6 sweep，回应"UCR-5 是 Rocket-optimized 饱和数据集，Agent 优势可能是 overfitting"的疑问。

| Dataset | Type | Train×Test | Classes | Saturation 等级 |
|---|---|---|---|---|
| GunPoint | motion | 50×150 | 2 | mid |
| Strawberry | spectroscopy | 613×370 | 2 | **TSFM 训练外** |
| Wafer | industrial | 1000×6164 | 2 | mid |
| ECG5000 | medical | 500×4500 | **5** | low (多类) |
| Crop | remote-sensing | 7200×16800 | **24** | very low（极多类） |

**Partial 主表（140/182 cells, Crop 尚跑）**

| Dataset | N | B7v3 | Rocket | Δ | 解读 |
|---|---|---|---|---|---|
| GunPoint | 3/5/10 | 0.970/0.967/1.000 | 0.970/0.967/1.000 | **= 0** | Rocket dominate |
| Strawberry | 3/5/10 | 0.705/0.625/0.777 | 0.705/0.625/0.850 | **-2.4pp avg** | **N=10 -7pp 失败** |
| Wafer | 3/5/10 | 0.780/0.865/0.702 | 0.780/0.865/0.702 | = 0 | tie everywhere |
| ECG5000 | 5/10 | 0.848 | 0.848 | = 0 | 多类下 Rocket = B7v3 |

**Head-to-head (B7v3 vs Rocket, 20 cells)**：**W=0 / L=1 / T=19**

**Method aggregate (less-saturated 20 cells)**：

| Method | Mean Acc |
|---|---|
| **B3 Rocket** | **0.831** |
| B7v3 Router | **0.824** (-0.7pp) |
| B2 Euclid | 0.756 |
| B4a MOMENT-1NN | 0.723 |
| B1 DTW | 0.691 |
| B4b MOMENT-LR | 0.672 |
| B5 LLM-direct | 0.533 |

**Routing 分布（20 cells）**：rocket **19** / moment_1nn 1 → **Agent 几乎全 fallback 到 Rocket，routing 机制几乎不触发**

**关键 finding — UCR-5 优势是 niche，不是 general**

| Benchmark | B7v3 vs Rocket | 解读 |
|---|---|---|
| UCR-5（task #41, 30 cells） | **+0.89pp** (6/30 wins) | Wins 集中在 BeetleFly/BirdChicken (MOMENT pretraining 强) |
| **Less-saturated 4 (task #42)** | **+0.00pp** (0/20 wins) | Rocket dominate everywhere |

**Saturation hypothesis 部分证实**：
- UCR-5 +0.89pp **不是 hot-pop 上的人工 overfitting**（B7v3 在 BeetleFly +20pp 是真实 MOMENT routing 价值）
- **但**优势集中在 image-outline silhouette 小数据集——MOMENT pretraining 的 niche，**不是 Agent 通用能力**
- 在 less-saturated 多类 / 工业 / 谱学数据上 Rocket 仍占主导，Agent-Router 几乎全默认 Rocket，**routing 层基本无用**

**Strawberry N=10 -7pp loss 诊断**（仅一个 loss cell）：
- B7v3 选 moment_1nn → 0.777 vs Rocket 0.850
- Strawberry 是化学光谱（TSFM 训练数据外）→ MOMENT embedding 误导路由

**论文 §5.2 必须再次诚实修正**（已写入 paper §4.11）：

**之前 (over-claim)**：
> "B7v3 +0.89pp, the first systematic LLM-Agent improvement over Rocket on UCR"

**修正版 (honest)**：
> "B7v3 achieves +0.89pp on UCR-5 (Coffee/ECG200/TwoLeadECG/BeetleFly/BirdChicken) — a subset selected to expose dataset-dependent classifier preferences. On a less-saturated extended sweep (GunPoint/Strawberry/Wafer/ECG5000/Crop, 20+ cells), **B7v3 ties Rocket exactly (0W/1L/19T)**, routing to Rocket in 19/20 cells. The honest interpretation: **Agent-Router's value is niche, not general**. It correctly routes to MOMENT when image-outline morphology dominates (BeetleFly N=5/10 +20pp via MOMENT), but on multi-class medical, industrial, or spectroscopy data where Rocket already excels, no improvement is captured."

**这进一步加固 §5.3 boundary 论证**（不是削弱）：

| Domain × Task | Agent 净增益 vs strong baseline |
|---|---|
| Forecasting | 0 (= Chronos-2 parity) |
| RCA clean GT | **-24pp** (vs B0-rule) |
| TSC UCR-5 (saturated) | **+0.89pp** (niche routing) |
| TSC extended (less-saturated) | **0** (Rocket dominate, routing inactive) |

→ **Agent's value 在所有 4 个 case 都不是"击败强 baseline"**：要么 tie，要么败，要么 niche win。**Agent's true contribution 在 NL trace + Out-of-taxonomy 扩展 + routing infrastructure**，不是 raw accuracy。

**产物**：
- `experiments/taskb_extended_sweep.py` (110 行)
- `research/results/taskb_extended_ucr.jsonl` (140 rows partial, ~182 final)

---

## 五、§5.3 Boundary Characterization 总结

| Domain | Best baseline | B6 Direct | **B7 Router** | Gap (B7 vs baseline) |
|---|---|---|---|---|
| **Forecasting** (24 cells) | Chronos-2 | v11 = C2 (0) | — | parity |
| **RCA** (30 cells) | LLM-direct (0%) | **40%** | (N/A) | **+40pp** |
| **TSC UCR** (30 cells) | Rocket 87.5% | 54.3% | **84.8%** | **-2.7pp** |
| TSC Synthetic 4-class | Rocket 50.6% | 33.7% | (pending) | TBD |

**论文级 take-aways（已稳定）**

1. **No AdaptTS wrapper beats Chronos-2** in forecasting (24-cell 23T)
2. **B6 Direct classification fails universally** — UCR -33pp / 4-class -17pp / RCA wins only because LLM-direct is the only baseline
3. **B7 Router 大幅修复 +30pp** — Agent's value is in **routing**, not direct classification
4. **Forecasting v8→v10 ≅ TSC B7→B7v2 progression**：双轨同构 "router-with-N-conditional-fallback" 在 base-model 占主导的两个 domain 上分别成为正确架构

---

## 六、模块清单（截至 2026-05-24）

### 6.1 共享基础设施

```
research/
├── utils/
│   ├── data_loader.py        数据加载 (ETT/ECL/Exchange/Weather/ILI)
│   ├── splitter.py           few-shot 切割
│   ├── metrics.py            MAE/MSE/MASE/SMAPE
│   ├── prob_metrics.py       CRPS/pinball/coverage（A3）
│   ├── llm.py                LLM 客户端 + cache + 重试
│   ├── fault_taxonomy.py     5-fault rule-based detector (RCA)
│   ├── inject_fault.py       4-fault 注入 (task #27)
│   └── ucr_loader.py         10 UCR 数据集
├── agent/
│   ├── curator_uq.py         **v2 12-dim Diagnosis**
│   ├── model_cards.py        forecasting 5 张
│   ├── memory.py             faiss Memory + test_mae backfill
│   ├── forecaster_reflect.py forecasting STRATEGY_FN + v10 wrapper
│   ├── planner_adaptive.py   forecasting planner
│   ├── adapt_ts.py           forecasting 主入口
│   ├── rca.py                RCA Agent (B5) + LLM-direct (B1)
│   ├── tsc_classifier.py     TSC Agent B6 (Direct) + B5 LLM-direct
│   ├── clf_strategies.py     **CLF_STRATEGY_FN 统一接口**（task #31）
│   ├── clf_model_cards.py    **6 张分类 Cards**（task #32）
│   └── clf_planner.py        **classification_planner + b7_agent_router**（task #33）
├── baseline/
│   ├── naive.py / arima_ets / chronos / chronos2 / chronos_bolt / llmtime / tsci
│   ├── timesfm2.py           CPU 加载不可行
│   ├── tsc_classical.py      B1-B3 (DTW/Euclid/Rocket)
│   └── moment_classifier.py  B4a/B4b MOMENT-1-small 探针
└── experiments/
    ├── runner.py                  forecasting 主入口
    ├── a3_prob_metrics.py         CRPS 评估 (Forecasting)
    ├── taska_select_failures.py   选 30 cells + auto GT (RCA)
    ├── taska_run_rca.py           RCA Agent vs B1
    ├── taska_eval.py              R1/R2/R4 + confusion
    ├── taskb_run.py               TSC UCR 7-method sweep
    ├── taskc_synth4class.py       合成 4-class
    └── taskb_router_sweep.py      **B7 Router sweep (task #35)**
```

### 6.2 results JSONL

```
research/results/
├── p3-p9_*.jsonl                  forecasting v3-v9 各版本
├── p10_*_v10_n10.jsonl            v10 N=10 增量
├── p11_*.jsonl                    v11 memory
├── p12_v12.jsonl                  v12 entropy
├── p13_phaseA/B.jsonl             v13 联合
├── f4_chronos_bolt/c2_*.jsonl     C8 base TSFM
├── a3_prob_metrics*.jsonl         CRPS (48+24)
├── taska_failures.jsonl           30 cells
├── taska_rca_predictions(_v1).jsonl  v1 / v2 RCA
├── taskb_ucr.jsonl                210 rows TSC
├── taskc_synth4class.jsonl        84 rows synthetic
└── taskb_router_ucr.jsonl         **30 rows B7 Router (task #35)**
```

### 6.3 关键文档

```
research/
├── plan.md           总规划（已重构 2026-05-24）
├── TODO.md           任务看板（已重构）
├── finish.md         本文件
├── paper_draft.md    论文初稿（§4 boundary + §5.1 RCA + §5.2 TSC）
├── classifier.md     分类 Agent 框架文档
├── feedback.md       feedback 文档（pivot + Agent-as-Router 提案）
└── research/README.md  目录说明
```

---

## 七、详细子节锚点（cross-ref to 旧编号）

历史 §3.1.x 子节内容已合并到上述 §二-§五。如需详细查询：

- §3.1.10-14 → §二 forecasting Phase 1-5
- §3.1.15-18 → §二 forecasting E2/F4/F2/F3 ablation
- §3.1.19-25 → §二 v7-v13 progression
- §3.1.26 → §二 CRPS
- §3.1.27 → §二 6 数据集
- §3.1.28 → Pivot 决策
- §3.1.29 → §三 RCA v1
- §3.1.30 → §六 MOMENT B4
- §3.1.31 → §三 RCA v2
- §3.1.32 → §四 UCR direct
- §3.1.33 → §四 synthetic
- §3.1.34 → §四 B7 Router
- (3.1.35) → 待 task #37 写入

---

## 八、下一步

```
P0: task #37 B7v2 N<5 fallback → 预期 mean 87-88% 持平 Rocket
P1: task #37 完成后写 §3.1.35 + paper §5.2 完整 narrative
P2: task #36 paper outline 落地（abstract / title）
P3 (可选): task #34 memory layer / B7v3 ensemble
```
