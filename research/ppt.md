# Agent + 时序 · 工业应用三大问题

> PPT 文字稿。目标听众：高校专家 + 研究所人员。
> 配套：`method.md`（技术方法）/ `finish-1.md`（实测）/ `README.md`（项目目标）

---

## Slide 1 · 总览

工业场景下时间序列分析的三大核心问题，可被 **multi-Agent + TSFM (Time Series Foundation Model) routing** 框架统一：

| 问题 | 工业例 | 经典做法 | Agent + TSFM |
|---|---|---|---|
| **预测** | 电力负荷、生产线产量、库存需求 | ARIMA / LSTM 单模型 | 多模型路由 + 概率集成 |
| **分类** | 故障类型识别、产品质量分级 | Rocket / DTW | 库扩张 + Bayesian 路由 |
| **机械问题判断** | 振动/电流异常根因、剩余寿命估计 | 阈值告警 | LLM Agent + 反事实推理 |

**核心 thesis**：三个问题本质同构 — 都是 `p(model | series_regime, history)` 的概率决策。一套框架统一解决。

---

## Slide 2 · 问题一 · 时序预测 (Forecasting)

### 背景

- 工业需要 **小样本 + 多变量 + 长 horizon** 零样本预测：新产线、新工艺、传感器接入即用
- 单一 TSFM 在不同数据域饱和度差异极大 — Weather 上 Chronos-2 SOTA，Exchange/ECL 上落后 niche specialist 35-44%
- 工业部署对 **CRPS（概率质量）** 比 MAE 更敏感（决策需要置信区间）

### 方法

1. **多模型 library**：Chronos-2 + TiRex (xLSTM, 金融 niche) + Toto (observability niche) + Time-MoE + Sundial + Timer-S1 + TimesFM-2 + Moirai + Moirai2，每个模型有 **能力卡 (Model Card)** 描述假设/强项/弱项
2. **Curator Agent** 做诊断：N、季节性、平稳性、频域熵 → 25-d 特征向量
3. **Bayesian Routing Framework**：
   - 静态 prior π_k = 1/CRPS_val 归一
   - 后验 p(M_k | x) ∝ exp(-CV_loss / σ²) · π_k
   - L0/L1/L2 三层分层：快通道 single-model → 集成 → softmax 混合
4. **Memory layer**：反事实存储 + 多样性检索（避免 default-collapse）
5. **Online adaptation**：Thompson Sampling contextual bandit，部署中持续 update belief

### 参考范式

| 工业实践 | 学术 SOTA |
|---|---|
| GE Predix / Siemens MindSphere（ARIMA + 季节分解）| Chronos / Chronos-2 (Amazon 2024-25) — T5 类 TSFM |
| Amazon SageMaker DeepAR（自回归 RNN）| TimesFM-2.0 (Google ICLR 2025) — decoder-only TSFM |
| Uber Orbit / Prophet（Bayesian 状态空间）| Moirai / Moirai 2.0 (Salesforce ICML 2024) — masked Transformer |
| | TiRex (NX-AI 2025) — xLSTM zero-shot |
| | GIFT-Eval benchmark：Timer-S1 (ByteDance 8.3B MoE) 当前榜首 |

**我们的定位**：
- **不训练新模型**，做 **router over library**
- 比 single-TSFM 路线更鲁棒：oracle gain +5.24%（已实测）
- 比 hand-coded ensemble (Uber Orbit) 更原理化：Bayesian posterior 替线性池

### 难点 / 缺陷分析

- **TSFM Saturation Hypothesis (F1)**：当 base model 与 ground truth 协方差 → 1，所有 wrapper 改进上界仅 **2.43%** (实测)
- **库扩张边际收益递减 (F2)**：5-way oracle == 3-way oracle (-5.24%)，加无关 niche 的 TSFM 零边际贡献
- **Niche 互补性 >> 模型数量 (F3)**：选型应先问 training corpus 与目标域 KL 距离，后问参数量
- **Soft mixture 在 saturation 域 NEGATIVE (F6)**：softmax 集成不优于 hard top-1
- **CV instability at low N (F7)**：N≤5 时 LOO CV 与 test winner Spearman ρ<0.3 → 必须 fallback 到 robust default

---

## Slide 3 · 问题二 · 时序分类 (TSC)

### 背景

- 工业场景：故障类型识别（轴承故障 7 类 / 电网攻击多类）、产品分级（半导体 wafer 良品/次品）
- **Few-shot 是常态**：新产线只有 N=3~10 个故障样本，模型必须冷启动
- 多模态信号（振动 + 电流 + 温度）通常用单变量分类器并行处理

### 方法

1. **11-classifier library**：Rocket / MiniRocket / WEASEL (NEW SOTA +2.7pp) / Catch22 / MOMENT / Mantis / DTW / Euclidean / LLM-direct
2. **Planner Agent** 跑 LOO/K-fold CV → 各分类器估计 acc
3. **Memory + 反事实**：每 case 存 `{all_clf_accs: {clf: test_acc}}` 完整字典（非只存 winner），avoid hindsight bias
4. **1/CRPS-style weighted vote**：每邻居为**全部** classifier 按 `sim × 1/(1-acc+ε)` 投票
5. **Domain prior**：工业振动域（acf_decay 高 + 离散水平少）自动 boost Euclidean (Wafer N=5 实测 0.965，超 Rocket 0.815)
6. **B7 Router**：CV + Memory + Domain prior 三路融合 → **+0.89pp 击败 Rocket** (现 SOTA)

### 参考范式

| 工业实践 | 学术 SOTA |
|---|---|
| NI LabVIEW / OSIsoft PI-AF（阈值 + 状态机）| Rocket / MiniRocket (Dempster 2020-21) — 随机卷积 + ridge |
| MATLAB Predictive Maintenance Toolbox（FFT + SVM）| HIVE-COTE 2.0 (Middlehurst 2021) — 5 分类器 ensemble |
| Edge ML on PLC（Decision Tree / kNN）| WEASEL (Schäfer 2017) — 实测 **NEW SOTA +2.7pp** |
| | MOMENT / Mantis (CMU 2024 / Paris 2026) — TSFM embedding + linear probe |
| | UCR / UEA archive — 标准对比集 30+ datasets |

**我们的定位**：
- 11-classifier library 全收 + Bayesian Router
- 击败 Rocket SOTA **+0.89pp** (B7v3，我们)
- 跨 task 复用同一 Router framework（forecasting / TSC 一套接口）

### 难点 / 缺陷分析

- **N<7 catastrophic (F7)**：LOO CV 给 BeetleFly N=3 -25pp、BirdChicken N=3 -20pp → 必须 hard fallback
- **多 classifier ensemble NEGATIVE (F6)**：所有 β ∈ {1,3,5,10,20,50,100} soft router 均 -3pp~-4.4pp 输 Rocket-alone
- **Memory hindsight bias (F5)**：朴素 top-K 检索自我强化 default 单峰 → memory 永远不 record 反例
- **跨数据集 generalization gap**：Meta-Router LODO CV cell label-match 仅 44.6% → online learning 远期方向

---

## Slide 4 · 问题三 · 根据时序判断机械可能的问题 (Anomaly + RCA)

### 背景

- 工业核心刚需：从振动 / 电流 / 温度时序中**实时识别**轴承磨损、转子失衡、齿轮断齿、绝缘老化等故障
- 不只 "是否异常"，更要 **根因 (RCA)**：故障类型 + 严重度 + 剩余寿命
- LLM Agent 可融合**多模态 (image + numeric + 维修记录文本)** 的天然优势

### 方法

1. **Curator Agent** 输出 12-d 诊断（v2 含 outlier / variance）：trend / season / stat × 3 置信源
2. **Rule baseline (B0)**：滑窗方差、IQR outlier、ADF 检验 → 兜底 deterministic 结论
3. **LLM Agent (RCA)**：基于 Curator 输出 + Model Cards + 历史 case retrieval → 多步推理输出根因 + 置信度
4. **A3 概率指标**：每个判断必须配 CRPS / coverage / width → 工程师可基于区间宽度决定是否需要人工干预
5. **Counterfactual memory**：存储 "如果选了别的 classifier 会怎样"，让 Agent 学到"什么时候不该相信自己"

### 参考范式

| 工业实践 | 学术 SOTA |
|---|---|
| GE Bently Nevada / SKF @ptitude（振动 FFT + 工程师手册）| Anomaly Transformer (ICLR 2022) — association discrepancy |
| Siemens SIMATIC Diagnostic（规则 + 设备 digital twin）| GPT4TS / Time-LLM (NeurIPS 2024) — LLM as zero-shot anomaly scorer |
| PdMA MCEMAX（电机电流签名分析 + 阈值）| Aurora / GraphRCA (2024) — 图模型 root cause analysis |
| NREL Wind Plant Monitoring（SCADA + 经验阈值）| HyperODE-RCA (2025) — 连续时间 ODE 根因 |
| | **ICLR 2026 质疑**：TSFM 在异常检测上未必优于简单基线 |

**我们的定位**：
- **rule baseline + LLM Agent 混合**（非纯神经路线）
- LLM 不做预测，做**结构化推理 + Model Card 解释**
- 反事实记忆 (all_clf_accs) 让 Agent "学会怀疑自己"

### 难点 / 缺陷分析

- **TSFM 异常检测争议**：ICLR 2026 paper 质疑 TSFM 未必优于规则基线 → 必须保留 rule baseline 参照
- **LLM "根本幻觉"**：在 RCA 上易编造不存在的故障模式 → 必须 retrieval-grounded 约束 + 工程师审核
- **小样本 + 不平衡**：故障样本远少于正常样本（极端 1:1000+）→ 阈值微调极敏感
- **多模态融合 cost**：视频 + 振动 + 文本同步对齐 latency 高 → 必须 **cost-aware routing**：ℒ = ForecastError + λ·(latency + VRAM + remote overhead)
- **持续运行 drift**：设备老化、季节温度、维护周期 → 必须 **online adaptation**（Thompson Sampling contextual bandit）

---

## Slide 5 · 统一的方法学骨架

### 架构概念图（AI 生成提示词）

```
A clean technical architecture diagram, isometric style, soft pastel colors,
white background, vector-art aesthetic. Top-to-bottom data flow:

LEFT-TO-RIGHT input rail at top: three icons representing
"raw time-series signals" — a vibration waveform, an electricity load curve,
and a temperature thermometer with timeline.

CENTER block "Curator Agent": a clipboard icon analyzing the series,
emitting a small feature vector (depicted as a row of colored cells labeled
"trend / season / stat / entropy / industrial").

BELOW Curator, a wide horizontal layer "Embedding f_φ(x) → z":
shows the series being mapped into a small dot cluster on a 2D
"regime manifold" (colored Voronoi regions K=6).

BELOW that, the central decision block "Bayesian Router":
- left subblock "Prior π_k(z)" (stacked horizontal bars per model)
- right subblock "Likelihood L_k(z, history)"
- merger arrow "log π + log L" feeding into
- a glowing center cube "Posterior p(M_k | z, h)"
- output arrow forks into 3 modes:
  ① argmax (single arrow)
  ② Thompson sample (multiple dashed exploratory arrows)
  ③ risk-min (arrow weighted by std)

RIGHT side: model library shelf with 12 model cards labeled
"Chronos-2 / TiRex / Toto / TimesFM-2 / Moirai / Time-MoE / Sundial /
Timer-S1 / Rocket / WEASEL / MOMENT / Mantis".
Selected model glows with a checkmark.

BOTTOM feedback loop: dashed orange arrow from prediction outcome
back to "Memory + Bandit State" cylinder, labeled "observe(z, chose, loss)".

Three downstream icons in a row at very bottom:
"Forecasting (chart with confidence band)",
"Classification (3 color buckets)",
"Mechanical Diagnosis (gear with warning triangle)".

Style: Linear-icons mixed with subtle gradients, modern paper-figure look,
white-and-cool-blue palette, minimal text, every arrow labeled,
high-contrast typography (no Chinese letters in image).
```

### 备选简化版

```
A minimalist three-tier diagram on white background, vector style:

Tier 1 (top): "Series x" — single horizontal time-series curve.
Tier 2 (mid): "Embedding → Regime → Bayesian Posterior":
  three connected pill-shaped boxes, labeled clearly,
  arrow labels: z, regime r, p(M_k | x).
Tier 3 (bot): "Decision + Observe":
  one box outputs to a model library shelf;
  a curved feedback arrow goes back to update Tier 2.

Colors: light blue boxes, dark blue arrows, gray library shelf,
orange feedback arrow. No clutter, paper-figure quality.
```

### 数据流（文字版）

```
Series
  ↓
Curator Agent: 诊断 + 25-d 特征向量
  ↓
Embedding f_φ(series) → z   ← Phase 4: frozen TSFM encoder (MOMENT)
  ↓
RegimeAssigner: k-means → regime label   ← 替代 dataset 名
  ↓
Bayesian Router:
  log π_k(z) + log L_k(z, history)
  decide ∈ {argmax, Thompson, risk_min}
  ↓
执行 + Observe outcome → update bandit state
```

---

## Slide 6 · 工业部署关键指标

| 指标 | 为什么重要 |
|---|---|
| **CRPS / pinball** | 工程决策需要置信区间，不只点估计 |
| **80% coverage** | 区间校准（覆盖率 = 区间宽度 / 真值落入比例）|
| **latency + VRAM** | 边缘部署约束（树莓派 4GB vs A100 40GB 模型库差 5×）|
| **online regret** | bandit 收敛速度，反映 drift 适应能力 |
| **interpretability** | Model Card 渲染到 LLM prompt → 决策可审计 |

---

## Slide 7 · 可行性

我们提出的 **"统一 Bayesian 时序路由框架"** 在数据、方法、工程、学术四个维度同时成立，**已在 6 个标准 benchmark + 14 UCR/UEA 数据集上完成可复现实证**。

### 数据 & 实证基础

- **基准覆盖**：ETTh1/h2 / ECL / Exchange / Weather / ILI 6 主流 forecasting 集，UCR/UEA 14 few-shot TSC 集
- **完整 sweep 体量**：已跑通 162-cell 多策略对比（3 policies × 6 datasets × 3 N × 3 seeds），5-way oracle 实测 -5.24% vs base
- **零样本**：所有方法均 **不重新训练**，仅 zero-shot TSFM + classical baseline + LLM Agent

### 方法 & 理论闭环

- **数学公式统一**：$\hat{M}(x, h) = \arg\max_k \log \pi_k(z) + \log L_k(z, h)$，其中 $z = f_\phi(x)$（frozen TSFM encoder）
- **每个组件理论可证伪**：6 priors + 2 likelihoods + 3 decision modes (argmax / Thompson / risk-min) 均可独立消融
- **NEGATIVE 发现亦可发表**：F1 TSFM Saturation Hypothesis (2.43% oracle ceiling)、F6 Soft Router 双轨 NEGATIVE、F2 库扩张收益递减 — 三个反例已具命题形式
- **跨任务统一**：同一 BayesianRouter 抽象贯通 forecasting + TSC（feedback Round 4 §八 "Universal Routing Framework"）

### 工程可复现性

- 完全开源 conda + Python 栈，5 个独立 env 隔离已 documented
- **CPU-first**：Chronos2 / Rocket / Mantis / arima 全 CPU 单 cell 亚秒级；GPU 仅 Timer-S1 等 8B 级 TSFM 需消费级 GPU (RTX 5070 Ti 16GB)
- 模块化：Curator / Embedding / RegimeAssigner / BayesianRouter / BanditState / Memory 全部正交可替换
- 持久化：BanditState save/load JSONL，跨 session 累积，online 自适应已闭环

### 学术贡献度（论文 readiness）

| 维度 | 已具备 |
|---|---|
| Method 节核心公式 | ✅ 单一概率决策规则 |
| Ablation 矩阵 | ✅ 8 组件 × 3 modes 全独立开关 |
| NEGATIVE finding 章节 | ✅ F1 / F2 / F6 三大反例 |
| 跨任务 generalization | ✅ forecasting / TSC 同框架实证 |
| 工业接地 | ✅ Cost-aware metric (latency + VRAM + env_penalty) 已落 |

---

## Slide 8 · 核心 thesis

> 工作的贡献**不在新 TSFM**，而在**把已有 TSFM library 重构为 principled adaptive inference system**。
> 系统层创新（Bayesian routing + counterfactual memory + contextual bandit + learned regime representation）的可证伪边界已经画清楚 — 可消融、可复现、可扩展到任意新模型/新任务而无需重训练。

每个组件都有 **学术论文支撑** + **工业级延迟实测** + **可消融**的工程实践。

---

## 附录 A · 实测关键数字（可作锚点）

| 数字 | 含义 |
|---|---|
| **-5.24%** | 5-way TSFM oracle gain vs Chronos-2 alone (34 cells) |
| **+2.7pp** | WEASEL aggregate over Rocket (TSC NEW SOTA, 8-classifier sweep) |
| **+0.89pp** | B7v3 击败 Rocket (TSC routing, 现 SOTA) |
| **82.4%** | regime cluster purity (K=8, hand25 embedding, 34 cells) |
| **2.43%** | wrapper-class 改进上界（TSH 实证） |
| **0.965** | Wafer N=5 Mantis-LR 实测（超 Rocket 0.815） |
| **162 cells** | 当前 sweep 体量 (3 policies × 6 datasets × 3 N × 3 seeds) |

## 附录 B · 12 个理论化 finding (F1-F12)

详见 `finish-1.md §0`，每条含 (a) 观察 (b) 实证出处 (c) 命题草稿 (d) 工程意义。

---

**End of ppt.md**
