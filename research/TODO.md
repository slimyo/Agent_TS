# TODO · 任务看板

> 重构时间：2026-05-24
> 配对文件：`plan.md`（总规划） / `finish.md`（实测结果） / `classifier.md`（分类 Agent 框架）

---

## 🎯 当前 sprint · Phase 6.4b Agent-as-Router

**核心 thesis**：Agent 直接分类已穷尽证伪（UCR -33pp / 4-class -17pp / RCA 仅胜 vs 弱 baseline）→ **B7 Router 是唯一可行架构**。当前 B7 已 +30pp 修复 B6 (54.3→84.8%)，仍 -2.7pp 输 Rocket。下一步 v9 N-fallback 类比 forecasting v8→v10 修 catastrophic mis-routes。

### 主线 task

| Task | 状态 | 一句话 | 产物 / Δ |
|---|---|---|---|
| **#37** | ✅ completed | B7v2 N<7 fallback | **86.66% (-0.87pp from Rocket)** ⭐ |
| #34 | ✅ completed | 分类 Memory layer | clf_memory.py + 揭示 avg-diag 退化 limitation |

**feedback 第三轮（Memory + Cards 优化，直击 #34 limitation）**：

| Task | 内容 | ROI | 顺序 |
|---|---|---|---|
| **#38** | Memory 特征扩 25-30 维 (元信息 L/C/N + 频域 FFT/spectral_entropy + 复杂度 PermutationEntropy/DFA) + z-score | ⭐⭐⭐ | P0 |
| **#39** | 加权 vote consensus_winner_weighted（去硬阈值 0.85，min_vote_ratio=0.6）| ⭐⭐⭐ | P0 |
| **#40** | Cards v2 加 5 决策字段 (min_samples_per_class / max_sequence_length / cost_level / preprocessing / multiclass) | ⭐⭐ | P1 |
| **#41** | ✅ B7v3 集成完成 → **88.42% (+0.89pp 击败 Rocket)** ⭐ | DONE |
| #36 | 论文 §5.2 重写 + abstract 起草 | ⭐⭐⭐ | 等 #41 结果 |

### Phase 6.4b 已完成

- ✅ #31 CLF_STRATEGY_FN 统一接口（6 策略 + safe wrapper, 110 行）
- ✅ #32 6 张分类 Model Cards（含 UCR 15-cell evidence, 160 行）
- ✅ #33 LOO/K-fold CV + classification_planner（220 行，B7 入口 b7_agent_router）
- ✅ #35 **B7 Router sweep on 30 cells**（finish §3.1.34）
  - **B7 84.8%** (+30.5pp from B6, -2.7pp from Rocket, Oracle 92.1%)
  - Beats Rocket 6/30; Routing: rocket 15 / moment 9 / euclid 3 / dtw 3
  - Failure: LOO CV 噪声在 N=3-5 → BeetleFly/BirdChicken N=3 误选 -20~25pp
- ✅ #30 B0-rule RCA baseline（合并到 #29 时一并完成）

---

## ✅ 已完成 Phase 1-6 时间线

### Phase 1-5 · Forecasting Boundary（已立 paper §4）

| Step | 内容 | 关键 finding |
|---|---|---|
| §3.1.1-9 | v5c 基线 + Phase 0-3 闭环 | 起点 |
| §3.1.10-14 | 6 数据集 × 6 方法 144 cells | no-method-dominates winner-take-all |
| §3.1.15 | E2 三路置信度 CMR | F1 论文 contribution |
| §3.1.16 | F3 v6 strategy promotion | N6 future work |
| §3.1.17/30 | F4 SOTA TSFM 扩充 (Chronos-2/Bolt) | Chronos-2 5/8 ETTh new SOTA |
| §3.1.18 | F2 跨 LLM 鲁棒性 | AdaptTS CV 2.7% vs LLMTime 22.6% |
| §3.1.19 | v7 chronos→bolt alias | 50× 加速 + 微改善 |
| §3.1.20 | v8 ECL/Exchange 反转 | Chronos-2 0W/8L |
| §3.1.21 | v9 margin gating | 0/7/9, 首次击败 C2 1 cell |
| §3.1.22 | v10 N<15 fallback | 1W/3L/12T = ETTh2 N=100 win |
| §3.1.23-25 | v11/v13 memory safety-net | 0W/0L/16T (= C2 with zero variance) |
| §3.1.26 | **A3 CRPS 评估** | v11/v13 +0% / v10/v12 +16% on probabilistic loss |
| §3.1.27 | 6 数据集 24-cell 完整 | v11 0W/1L/23T + Weather OOD +505% |
| §3.1.28 | Pivot 决策（feedback 第一轮） | 转 reasoning task |

**结论**：forecasting wrapper 上限 = Chronos-2，无 systematic 击败可能。

### Phase 6.1 · TaskA RCA Natural

| Step | 内容 | 关键 finding |
|---|---|---|
| #24 / §3.1.29 | v1 Curator 10-dim Agent vs LLM-direct | R1 **40% vs 0%** (+40pp), B1 全 30 cells collapse to trend_break |
| #29 / §3.1.31 | v2 Curator 12-dim (+outlier+variance) | R1 37% / **R2 +13pp** / **R4 +14pp** trade-off; variance_explode 0/10→9/10 但 stationarity_flip 12/13→1/13 |

### Phase 6.3 · TaskB UCR

| Step | 内容 | 关键 finding |
|---|---|---|
| #26 / §3.1.32 | 5 UCR × 3 N × 2 seeds × 7 methods = 210 cells | **B3 Rocket 87.5%**, B6 Agent 54.3% (-33pp), Agent 0/15 winner; MOMENT 反超 Rocket on BeetleFly/BirdChicken |
| #19 / §3.1.30 | C8 redirect: MOMENT B4 落地 (38M CPU 友好) | Coffee 5-shot 92.9% |

### Phase 6.4a · Synthetic 4-class

| Step | 内容 | 关键 finding |
|---|---|---|
| #27 / §3.1.33 | ETTh1/ECL × 4-fault × 7 methods = 84 cells | **B6 33.7% vs Rocket 50.6% (-17pp)**；alignment 减半 gap 但仍输 → **Agent direct 结构性弱** |

### Phase 6.4b · Agent-as-Router (current)

详见上方 "当前 sprint"。

---

## 🔮 Future work (论文 §5.4 / §6)

### Forecasting (已搁置)
- task #15 B6 Memory feature 加 entropy 维度
- task #17 B5 Curator dataset-name 语义先验 + multi-modal panel
- task #18 A2 多变量启用（Chronos-2 multivariate API）
- task #20 B7 walk-forward fold-consistency
- task #21 B4 反思层去留决策（论文写作期）
- task #22 C10 abstain 机制
- task #25 RCA synthetic (合成 fault 注入)
- task #28 论文双轨结构（已合并到 #36）

### Classification (开放)
- v3 Curator ensemble vote (v1+v2)
- TSC Memory layer (task #34, 案例检索)
- Counterfactual / NL explanation (TaskC 备选)
- 5-class / 24-class UCR (ECG5000 / Crop)
- TimesFM-2.0 / Moirai / Aurora GPU baselines

---

## 📊 整体 Task ID 索引

| Task | Subject | Status |
|---|---|---|
| #12 | v11 memory closed loop | ✅ |
| #13 | A1 TSFM entropy gating (v12) | ✅ |
| #14 | A3 CRPS / pinball / coverage | ✅ |
| #15 | B6 Memory feature 加 entropy | ⏳ |
| #16 | C9 Weather+ILI 6 数据集补完 | ✅ |
| #17 | B5 Curator dataset-name 先验 | ⏳ |
| #18 | A2 多变量 (Chronos-2) | ⏳ |
| #19 | C8 TSFM baseline (→ MOMENT 完成) | ✅ |
| #20 | B7 walk-forward fold-consistency | ⏳ |
| #21 | B4 反思层去留 | ⏳ |
| #22 | C10 abstain | ⏳ |
| #23 | v13 entropy+memory 联合 | ✅ |
| #24 | P6.1 RCA natural | ✅ |
| #25 | P6.2 RCA synthetic | ⏳ |
| #26 | P6.3 TaskB UCR | ✅ |
| #27 | P6.4 Synthetic 4-class | ✅ |
| #28 | P6.5 论文整理 | ⏳ |
| #29 | v2 Curator 12-dim | ✅ |
| #30 | B0-rule RCA baseline | ⏳（小，可一起做）|
| #31 | CLF_STRATEGY_FN 接口 | ✅ |
| #32 | clf Model Cards | ✅ |
| #33 | Classification Planner | ✅ |
| #34 | 分类 Memory layer | ⏳ |
| #35 | B7 Router sweep | ✅ |
| #36 | 论文 §5.2 重写 | ⏳ |
| **#37** | **B7v2 N-fallback (current)** | **🔄** |
