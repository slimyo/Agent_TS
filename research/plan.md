# Plan · AdaptTS-Agent 项目主规划

> 重构时间：2026-05-24
> 配对文件：`TODO.md`（任务看板）/ `finish.md`（实测结果）/ `classifier.md`（分类 Agent 框架）
> 注：本文件经过 2 轮 pivot 后重写。原 §一-§十五 forecasting plan 的实测数据汇总见 `finish.md`；本文档保留当前直接相关的设计。

---

## 0. Executive Summary

**项目目标**：在 2026 TSFM 时代，定位 LLM-Agent 在时序任务中能贡献的位置。

**最终论文骨架（论文标题待 task #37/36 后定）**

```
§1 Introduction
   - 2026 TSFM (Chronos-2 等) 主导 forecasting → LLM Agent 价值何在？

§2 Related Work
   - ATSF, TSFM, Few-shot TSC, RCA

§3 Method
   - Curator 12-dim 诊断 + Model Cards + Memory (forecasting & TSC)
   - Forecasting wrapper: v10 confidence-gated wrapper
   - TSC: Agent-as-Router (B7) 而非 direct classifier

§4 Forecasting Boundary (Establishing the Wall)
   - 4.1-4.8: v5c→v13 progression，CRPS 评估
   - Final result: v11/v13 = Chronos-2 on 24 cells (0W/1L/23T MAE, 0% CRPS)

§5 Beyond Forecasting: Reasoning Tasks
   - 5.1 TaskA RCA: Agent +40pp vs LLM-direct
   - 5.2 TaskB TSC: Direct B6 -33pp / Router B7 -2.7pp / Router-v2 ~88% (predicted)
   - 5.3 Boundary characterization

§6 Discussion & Future Work
```

**核心 finding（论文级 take-aways）**

1. **No AdaptTS wrapper beats Chronos-2 on forecasting** (24-cell 23T MAE / 16T CRPS)
2. **Agent direct classification fails universally** (UCR -33pp / synthetic 4-class -17pp / 即使 statistical-aligned 任务也输 Rocket)
3. **Agent-as-Router 大幅修复 (+30pp from B6)** —— routing 才是 Agent 价值所在
4. **B6→B7→B7v2 progression** 完美对应 forecasting v8→v10 N-fallback 设计 → 论文最优雅双轨同构

---

## 1. 数据集与切割协议

### 1.1 Forecasting 数据集（§4 boundary）

| Dataset | Sampling | Length | H | Season m | 主要用途 |
|---|---|---|---|---|---|
| ETTh1 | 1h | 17,420 | 96 | 24 | 主表 |
| ETTh2 | 1h | 17,420 | 96 | 24 | 主表 |
| ECL (MT_001) | 1h | 26,304 | 96 | 24 | TSFM coverage 评估 |
| Exchange (rate_0) | 1d | 7,588 | 96 | 7 | low-coverage 数据 |
| Weather (OT) | 10min | 52,696 | 96 | 144 | OOD memory 测试 |
| ILI (OT) | 1w | 966 | 24 | 52 | 量纲极端 + 周采样 |

**Few-shot 协议**：N ∈ {10, 20, 50, 100} × 3 seeds (1, 42, 123)

### 1.2 Classification 数据集（§5 reasoning tasks）

**A. UCR Univariate Archive**（10 个，已下载 `research/datasets/ucr/`）

| Tier | Dataset | Train×Test | Classes | Length | Domain |
|---|---|---|---|---|---|
| 核心 | Coffee | 28×28 | 2 | 286 | spectroscopy |
| 核心 | ECG200 | 100×100 | 2 | 96 | ecg |
| 核心 | GunPoint | 50×150 | 2 | 150 | motion |
| 极少 | TwoLeadECG | 23×1139 | 2 | 82 | ecg |
| 极少 | BeetleFly | 20×20 | 2 | 512 | image |
| 极少 | BirdChicken | 20×20 | 2 | 512 | image |
| 多类 | ECG5000 | 500×4500 | 5 | 140 | ecg |
| 多类 | Crop | 7200×16800 | 24 | 46 | remote-sensing |
| 工业 | Wafer | 1000×6164 | 2 | 152 | manufacturing |
| 谱学 | Strawberry | 613×370 | 2 | 235 | spectroscopy |

**N-shot 协议**：N_per_class ∈ {3, 5, 10} × 2 seeds (1, 42)

**B. Synthetic 4-class fault**（在 ETTh1/ECL 上注入）

| Class | 注入方式 |
|---|---|
| 0 normal | 无注入 |
| 1 trend_break | 中段加 ±2.5σ 阶跃 |
| 2 seasonal_break | 后半段周期内 reverse subsequence |
| 3 outlier_burst | 插入 4 个 ±4σ 离群点 |

**C. RCA 自然失败 cells**（30 个，从 forecasting Phase 5 catastrophic 选）

---

## 2. Strategy Pools（两个 task 各自的策略池）

### 2.1 Forecasting 策略池（`agent/forecaster_reflect.STRATEGY_FN`）

`naive_drift, naive_seasonal, arima_ets, chronos (=bolt aliased), llmtime, chronos2, chronos_bolt`

### 2.2 Classification 策略池（`agent/clf_strategies.CLF_STRATEGY_FN`）

`rocket (DEFAULT), moment_1nn, moment_logreg, dtw_1nn, euclid_1nn, llm_direct`

### 2.3 Model Cards

- Forecasting: `agent/model_cards.py` 5 张（naive/arima/chronos/llmtime + chronos2/bolt 4 张增补）
- Classification: `agent/clf_model_cards.py` 6 张（含 UCR 15-cell 实证 evidence）

---

## 3. Evaluation Metrics

### 3.1 Forecasting

| Metric | 用途 |
|---|---|
| MAE | 主表 |
| MSE | 辅助 |
| SMAPE | scale-invariant |
| MASE | season-normalized |
| **CRPS** | 概率指标（A1 之后加，§4.8） |
| pinball q10/q50/q90 | 概率细粒度 |
| coverage_80 / width_80 | 校准 |

### 3.2 RCA (TaskA)

| Metric | 用途 |
|---|---|
| **R1 Top-1 acc** | primary fault 单选准确率 |
| **R2 Top-3 incl** | gt fault 是否进 top-3 |
| R3 LLM-as-Judge | semantic（待人工/GPT-4） |
| **R4 Keyword F1** | evidence text fault-keyword 命中 |
| R5 Cohen's κ | 人工一致性（暂未做） |

### 3.3 TSC (TaskB)

| Metric | 用途 |
|---|---|
| **Accuracy** | per-cell |
| **Macro F1** | 多类平衡 |
| Routing trace (B7) | 论文可解释性 contribution |
| Oracle gap | 上界差距 |

---

## 4. Phase 计划

### 4.1 ✅ Phase 1-5 · Forecasting Boundary（已完成）

**核心实验**：6 数据集 × 4 N × 3 seeds = 72 cells，v5c→v13 progression

**Final result**：
- v11/v13 vs Chronos-2: **0W / 1L / 23T**（24 cells, eps=0.5%）
- CRPS: v11/v13 = C2 exactly (0%), v10/v12 +16% 输
- OOD safety-net failure (Weather N=20 v11 +505%)

**论文 §4 结论**：在 2026 TSFM 时代，**没有 forecasting wrapper 能 systematic 击败 Chronos-2**——v11 是 guaranteed-parity wrapper。

详 finish.md §3.1.10 - §3.1.28。

### 4.2 ✅ Phase 6.1 · TaskA RCA Natural（已完成）

**实验**：30 个 catastrophic failure cells × Agent (B5) vs LLM-direct (B1)

**Final result**：
- v1 Curator 10-dim: R1 40% / R2 43% / R4 16% (Agent vs B1 0%)
- v2 Curator 12-dim: R1 37% / R2 57% / R4 30% (trade-off finding)

**论文 §5.1 结论**：诊断特征 + Model Cards 让 LLM 在 statistical-label classification 上 +40pp 击败 unstructured LLM ICL。

详 finish.md §3.1.29 / §3.1.31。

### 4.3 ✅ Phase 6.3 · TaskB UCR（已完成）

**实验**：5 UCR 数据集 × 3 N-shot × 2 seeds × 7 methods = 210 cells

**Final result**：
- B3 Rocket 87.5% (winner 7/15)
- B4 MOMENT 81.8% (winner 6/15, BeetleFly/BirdChicken 反超)
- B6 Agent direct 54.3% (winner 0/15) ❌

**论文 §5.2 结论（旧）**：B6 直接分类失败 -33pp。

详 finish.md §3.1.32。

### 4.4 ✅ Phase 6.4a · Synthetic 4-class（已完成）

**实验**：ETTh1/ECL × 3 N × 2 seeds × 4-fault × 7 methods = 84 cells

**Final result**：B6 33.7% / Rocket 50.6% / -17pp **Agent 仍输 Rocket on statistical-aligned task**

**结论**：B6 直接分类**结构性弱**——alignment 不足以让 Agent 击败 Rocket。**Routing 才是出路**。

详 finish.md §3.1.33。

### 4.5b ⏳ Phase 6.4b 第二批（feedback 第三轮整合 — 2026-05-24 第三 batch）

**触发**：task #34 实测 avg-diag 在 UCR 跨数据集 sim 退化全 1.0 → memory consensus 不工作。feedback 直击此问题给出 4 项优化建议。

**4 个新增 task（已写入 TODO）**

| Task | 内容 | 设计要点 |
|---|---|---|
| #38 | Memory 25-30 维扩充 | 元信息 (L, C, N_per_class, class_balance) + 频域 (FFT peak, spectral entropy) + 复杂度 (perm entropy, DFA) + z-score 标准化 |
| #39 | 加权 vote consensus | 去 similarity_threshold=0.85 硬阈值，加权 sum + min_vote_ratio=0.6 |
| #40 | Cards v2 扩 5 字段 | min_samples_per_class / max_sequence_length / cost_level / preprocessing / multiclass_support |
| #41 | B7v3 完整集成 | #37 N-fallback + #38 enhanced mem + #39 weighted vote + #40 Cards v2 → 重跑 sweep |

**预期 B7v3 vs Rocket** ：85% → 88-89%（mem 真正参与决策 + Cards 给 LLM 硬约束）

### 4.5 🔄 Phase 6.4b · Agent-as-Router（进行中）

**已完成（5/8 主线）**：

- ✅ #31 CLF_STRATEGY_FN（`agent/clf_strategies.py`，6 策略统一接口）
- ✅ #32 6 张 clf Model Cards（`agent/clf_model_cards.py`，含 UCR evidence）
- ✅ #33 LOO/K-fold CV + classification_planner（`agent/clf_planner.py`，B7 入口）
- ✅ #35 B7 sweep on 30 cells UCR
- ✅ #27 Synthetic 4-class for boundary

**B7 Router 主结果（finish §3.1.34）**：
- **Mean Acc 84.8%** (vs B6 54.3%: **+30.5pp** / vs Rocket 87.5%: -2.7pp / vs Oracle 92.1%)
- Beats Rocket: 6/30 cells (Oracle 13/30 = 46% capture rate)
- Routing 分布: rocket 15 / moment 9 / euclid 3 / dtw 3
- 失败模式：LOO CV 在 N=3-5 上噪声大 → BeetleFly/BirdChicken N=3 误选 -20~25pp

**剩余（3/8 主线）**：
- ⏳ #37 **B7v2 N-fallback**（next）：N<5 强制 rocket default，类比 forecasting v8→v10
- ⏳ #34 分类 Memory layer
- ⏳ #36 论文 §5.2 重写

### 4.6 ⏳ Phase 7 · 论文整合与写作

待 #37 完成后启动。

---

## 5. 论文 boundary characterization 总结表

| Domain | Best baseline | Agent (B6 direct) | **Agent-Router (B7)** | Router gap |
|---|---|---|---|---|
| Forecasting (24 cells) | Chronos-2 | v11=C2 (0) | v11=C2 (0) | parity wrapper |
| **RCA TaskA** (30 cells) | LLM-direct (0%) | **40%** | (N/A) | **+40pp** ⭐ |
| **TSC UCR** (30 cells) | Rocket 87.5% | 54.3% | **84.8%** | **-2.7pp** |
| TSC Synthetic 4-class | Rocket 50.6% | 33.7% | (pending) | TBD |

**论文最干净的方法学贡献（已稳定）**：

> "**Agent direct classification fails universally**. Even on synthetic data where class labels are diagnostic concepts, B6 loses to Rocket -17pp. **The right architecture is Agent-as-Router**: Curator + LOO CV + Model Cards select among classifiers per cell, recovering +30pp from B6 direct and approaching parity with the SOTA Rocket baseline. **This exactly parallels the forecasting v8→v10 progression** where Agent-as-Wrapper around Chronos-2 became the right design, demonstrating a domain-invariant 'router/wrapper-around-base-models' principle for LLM-Agent systems in the TSFM era."

---

## 6. 风险预案

| 风险 | 现状 | 对策 |
|---|---|---|
| B7 仍输 Rocket | 实测 -2.7pp | **task #37 N-fallback**（预期持平/超过）|
| LOO CV 在 N<10 噪声大 | 实证 BeetleFly N=3 输 -25pp | N-conditional gate |
| TaskB Memory 设计未启用 | task #34 暂搁置 | optional contribution |
| 投稿目标 | ICLR/NeurIPS 2026 Workshop or KDD 2026 Applied | 待 #37 结果定 |

---

## 7. 立即下一步

```
P0 (现在): task #37 B7v2 + N<5 fallback + 重跑 sweep → 30 cells
P1 (跑完): 写 finish.md §3.1.35 + paper §5.2 完整 narrative
P2: task #36 paper §5.2 重写 + §5 boundary 升级 + abstract 起草
P3 (可选): task #34 memory layer / B7v3 ensemble
```
