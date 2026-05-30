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

---

## 🆕 Learned Routing 4-Level 路径（feedback 第四轮：启发式 → learned 系统解）

> feedback 指出当前路由本质是手调启发式（20% margin / N<7 / 加权 memory），无法 auto-adapt 到新 TSFM/classifier。完整解决路径：

### Level 1 · Learned Margin（最快 POC, 1 天）
- [ ] **task #50** 用 (Curator features, optimal_margin) 训小回归头，替换 `margin=0.10` 常量
- **价值** ⭐⭐ / **风险** 仅修单超参，不解结构

### Level 2 · Meta-Router（已实现，需 v2 改进）
- [x] **task #49 v1 完成**：56 cells LODO CV label-match 44.6%，**selected_acc -1.3~-4pp vs rocket-alone**（class imbalance + 训练数据不足）
- [ ] **task #51 Meta-Router v2** 改进：
  - confidence-gated override（仅 prob>0.7 偏离 rocket）
  - 扩训练数据（合并 UEA + synthetic 4-class）
  - Regression-mode：predict per-classifier acc，不只 arg max
- **价值** ⭐⭐⭐ feedback 核心痛点解 / **工程** 2 天

### Level 3 · Contextual Bandit RL（中等, 1 周）
- [ ] **task #52** 每 cell 作 bandit context，classifier 作 arm，Thompson Sampling 在线学习
- **价值** ⭐⭐⭐ auto-adapt 新模型 / **风险** 需要 deployment loop 才好评估

### Level 4 · Meta-Learning via TSFM Transfer（最 ambitious, 2 周）
- [ ] **task #53** 用 MOMENT/Chronos-2 embedding 作 universal representation；synthetic 分布偏移数据 pre-train，target dataset few-shot fine-tune
- **价值** ⭐⭐⭐⭐ "Learned Time-Series Routing FM" 论文重定位 / **风险** 超出 deadline

**执行序**：现做 Level 2 v2（task #51）→ Level 1（task #50, polish margin）→ Level 3/4 进 paper §5.4 future work（不本期实现）

---

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

---

## 🆕 Round 8+ 自演化路线候选（接续 method3.md / finish3.md）

> 已完成（Round 7-8，截至 2026-05-30）：M1 Meta-bandit / M2 Model 淘汰 / M3 Empirical Bayes Prior strength / M4 Per-regime bandit decay / M7 Phase 1 Anomaly / **M8 Factor Attribution + Bayesian framing 修正**（详见 `method3.md` §2-§4 / §4.5 / §8 / §10 + `finish3.md` §0-§5 / §7）
>
> **M8（2026-05-30）直击 feedback 两条理论硬伤**：
> - ✅ **问题 1**（fake Bayesian）：`bayesian_router.py` docstring 改 framing 为 energy-based / posterior-inspired，论文不再 claim exact Bayesian
> - ✅ **问题 2**（factor explosion / 黑盒）：新增 `factor_log_contributions` + `attribute_decision`(LOFO) + `FactorAttributionAccumulator`(跨决策冗余检测)，**零新增运行时模块**，符合"收敛 abstraction 不堆模块"
> - 关键 finding F-R8.5/8.6：拆解精确重构 log_posterior(误差 0)；KL_drop/argmax_changed 比 Δmargin 更反映因果影响力；当前 6 forecasting factor 无 |corr|≥0.8 冗余对（redundancy_matrix 作未来加 factor 的自动护栏）
>
> **下次会话开始这里查**：按 effort × value 排序的剩余优化项，每项含设计要点 + 影响范围 + 依赖。

### 🟢 M5 · Memory importance sampling（中 effort / 中 value）

**问题**：当前 `memory_decay.py` 只做 exp time decay。老样本一律按时间衰减，但**高信息量样本**（router 强后悔 / 多模型分歧大 / outcome 远离 regime baseline）也被同等遗忘 —— 损失关键学习信号。

**设计要点**：
- 给 `memory_cases` 每条增加 `importance: float` 字段，写入时计算：
  - `disagreement = std(per_model_predicted_loss)` 高 → importance↑
  - `surprise = |outcome − μ_regime| / σ_regime` 高 → importance↑
  - `regret = outcome − min(outcomes_seen_for_chosen_in_regime)` 高 → importance↑
- 综合：`importance = clip(α·disagreement + β·surprise + γ·regret, 0.1, 5.0)`
- `compute_decay_weights` 改为 `weight = exp(−Δt / τ) · importance`
- 检索时按 `weight × similarity` 排序

**影响范围**：`memory_decay.py` / `memory.py` / `representation.py:RepresentationLikelihood` / `clf_memory.py`

**依赖**：无新依赖；可独立做

---

### 🟢 M6 · Curator 25-d → 15-18 维（小 effort / 低-中 value）

**问题**：`utils/series_features.py` 当前 25 维 hand feature 有冗余（feedback 前§4 表"Curator 特征较多但可能冗余"）。25 维冗余拖累 embedding cache hit + 增大 z 空间噪声。

**设计要点**：
- 用历史 `state.memory_cases.z` 跑 PCA / 互信息 / per-feature importance
- 目标维度 15-18 维
- 选拔策略：top-k by `|corr(feature_i, log(outcome))|`
- 验证：在 `g_real_demo` ETTh1 上跑新旧两版 embedding 比 MAE
- 留 `enable_pruned_features: bool` 旋钮做 ablation

**影响范围**：`utils/series_features.py` / `representation.py:HandFeatureEmbedding`

**依赖**：需要 ≥ 200 条已观察 telemetry 做 importance ranking

---

### 🟡 M7 Phase 2 · per-fault Memory（中 effort / 高 value）

**问题**：M7 Phase 1 的 `AnomalyTypePrior` 完全依赖手设规则。Phase 2 加 per-fault-type Memory，让系统从历史检测里学习"哪种故障在哪种序列特征下更常见"。

**设计要点**：
- 复用 `failure_memory.py` 的 `FailureCase` 结构，按 `fault_type` 而非 `model` 分桶
- 新增 `FaultTypeMemory.add(window, detected_type, ground_truth_type?)` 接口
- 当 detected_type ≠ ground_truth_type（人工标注 / 后期校正）时该样本权重大
- 检索：给定新 window，找 top-k 历史 case，按 `fault_type` 投票
- 与 `AnomalyTypePrior` 联合：`combined = α·rule_prior + β·memory_prior`
- 复用 M3 EB 自动学 α, β
- 注意：**不**引入 LLM；Phase 3 才有

**影响范围**：`anomaly.py` / 新文件 `agent/fault_type_memory.py`

**依赖**：M7 Phase 1（已完成）；可选 ground truth 数据

---

### 🟡 M7 Phase 3 · LLM RCA agent（大 effort / 高 value，可选模块）

**问题**：Phase 2 输出 `(fault_type, score)` 仍是分类标签；工业部署需要自然语言根因 + 建议介入。

**设计要点**：
- 新文件 `agent/rca_llm.py`，**默认 disabled**（可关闭模块）
- 输入：`(window, AnomalyResult, RouterState 摘要, optional 历史 Failure cases)`
- 输出：`RCAResponse(natural_language_diagnosis, suggested_actions, confidence)`
- LLM 通过 `utils/llm.py` 已有的 SiliconFlow / DashScope / DeepSeek 接口（用户偏好，**不用付费官方服务**）
- 必须支持 fallback：LLM 不可用时回到 Phase 2 输出
- E1 Action Layer 把 LLM `suggested_actions` 作为额外候选介入加权
- 实测：用 RCA-natural（task #24 已完成的数据）做 zero-shot evaluation

**影响范围**：`utils/llm.py` / 新文件 `agent/rca_llm.py` / `action_layer.py`

**依赖**：M7 Phase 2 / 用户 LLM 服务可用性 / 验证数据集（RCA-natural 已就位）

---

### 🟡 R6-E2 · forecaster_reflect ADAPTTS_ACTION=1 集成（小-中 effort / 高 value）

**问题**：当前 Action Layer 只在独立 demo 里跑（`g_real_demo.py`）；生产路径 `forecaster_reflect.forecast_with_reflection(...)` 仍只返回 prediction，不出 ActionDecision。

**设计要点**：
- 在 `forecaster_reflect.py` 加 env flag `ADAPTTS_ACTION=1`
- 启用时：拿到 ensemble pred 后，自动构造 `ForecastDist(mean, std)` 调 `decide_from_router`
- ActionDecision 写进 `ForecastTrace.action_decision: ActionDecision | None`
- 阈值来源：env `ADAPTTS_UPPER_THRESHOLD` / `ADAPTTS_LOWER_THRESHOLD` 或自动 95% 分位
- 不影响默认路径（flag off 时行为完全不变）

**影响范围**：`forecaster_reflect.py` 单文件

**依赖**：无；纯 wrapper

---

### 🟡 R6-E3 · Drift Engine 第 6 信号 · pred_residual_z 双向（小 effort / 中 value）

**问题**：F-R6.1 修复的 `pred_residual_z` 是 `|E[recent] − E[hist]|`（取绝对值）。但**方向性漂移**应该触发不同 action：变坏 → boost_exploration + lower_memory_trust（已有）；**变好** → tighten_decay + raise_memory_trust（保留新行为更激进）。

**设计要点**：
- 在 `drift_engine.py` 把 z-score 拆成 `pred_residual_z_signed`
- 新增 action `raise_memory_trust`（trust 1.0 上限不变，但允许从 0.3 恢复到 1.0）
- 新增 action `tighten_decay`（独立于 boost_exploration）

**影响范围**：`drift_engine.py` 单文件

**依赖**：F-R6.1 修复（已完成）

---

### 🔵 长尾候选（按需挑做）

- **MLE 替代 PAV 的 Calibration**：`calibration.py` 当前 isotonic monotone-pooling，可以换成 sklearn `IsotonicRegression` + 持久化曲线 + 重训机制
- **Telemetry 自动压缩**：`telemetry.py` 现在保留最近 2000 条；可加 Reservoir sampling 让长期存储有更均匀的时间分布
- **GMM 替代 KMeans 的 Regime**：method2 §11.5 提到的 soft assignment 升级（`soft_router.py` 已就位但未做 GMM 切换）
- **regime stale 自动 resurrect culled**：当前只在 drift 触发 boost_exploration 时 resurrect；regime_stale 单独触发后应同步 resurrect
- **Per-task RouterConfig 模板**：forecast / classification / anomaly 三套 sane defaults，避免用户每次手配

---

### 📐 优先级建议（effort × value 排序）

```
1. M6 Curator pruning            (小, ~半天)      快速 wins
2. R6-E2 ADAPTTS_ACTION=1         (小-中, 半天)    生产可用性
3. M5 Memory importance sampling  (中, 1天)        system 自演化深度
4. M7 Phase 2 per-fault Memory    (中, 1-2天)      开 Phase 2 路径
5. R6-E3 双向 drift signal        (小, 半天)       完善 B3
6. M7 Phase 3 LLM RCA agent       (大, 2-3天)      需要联调 LLM
```

**预算**：完成 1-3 约 2-2.5 天 → Round 9 收口；4-5 约 2-3 天 → Round 10；6 约 2-3 天 → Round 11。

---

### 📁 不会忘的文件 hook

- 本节维护在 `research/TODO.md` 文末（grep "Round 8+ 自演化路线候选" 就能定位）
- 关联：`method3.md`（方法）/ `finish3.md`（实测）/ `feedback.md`（原始动机表）
- 已完成项在 `finish3.md` §0-§5 + Findings F-R6.1 ~ F-R8.4 都登记过 → 可以回查
- **下次会话第一句问 "Round 8 下一步"** 时，先 grep 这里
