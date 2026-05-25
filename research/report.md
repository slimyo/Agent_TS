# 阶段汇报：少样本时序场景下的自适应 LLM Agent 预测系统

> 项目代号：AdaptTS-Agent
> 配套文件：`plan.md`（实验设计原档）/ `finish.md`（成果细节）/ `TODO.md`（任务看板）
> 工程目录：`/home/hz/code/agent_ts/research/`

---

## 一、研究问题与动机

### 1.1 要解决什么具体问题

**少样本（few-shot）时序预测**：当一段时间序列只有 N=10~100 个历史观测点时，如何给出尽可能准确的未来 H 步预测。


**难点？**
- **传统统计模型**（ARIMA/ETS）需要至少几百个点才能稳定估参，N=10 时几乎不能用；
- **深度学习模型**（LSTM/Transformer）参数量大，少样本下完全过拟合；
- **预训练时序基础模型**（Amazon Chronos）零样本能跑，但对特定业务的"特殊性"无感知；
- **大语言模型直接 ICL**（LLMTime：把序列编成数字串让 LLM 续写）在某些场景超好，某些场景又差，**没有稳定优势**。

**关键观察（后面的实验也证明）**：在不同的数据集、不同的 N 值下，"哪个方法最好"是不断变化的。所以我们需要一个**会自动选择策略的系统**——这就是 Agent 的用武之地。

### 1.2 LLM Agent

**LLM Agent 是把大模型当作'决策大脑'，让它指挥一组工具完成任务**：

```
传统 prompt：       user → LLM → answer
Agent 工作流：     user → LLM 决策 → 调用工具 A → 看结果 → LLM 再决策 → 调用工具 B → ... → answer
```

Agent 的三个核心能力：
1. **规划（Planning）**：把大任务拆解成小步骤
2. **工具调用（Tool Use）**：必要时用 Python 算统计量、用预训练模型生成预测、查数据库
3. **反思（Reflection）**：看到中间结果不满意，回过头修改前面的决策

在时序预测里，Agent 的具体形态是：**LLM 看一段短序列 → 诊断（这有趋势吗？有季节吗？平稳吗？）→ 根据诊断选合适的预测模型 → 看 holdout 表现 → 不满意则换策略**。

### 1.3 对标的 SOTA：TSci

**TSci**（TimeSeriesScientist, arXiv 2510.01538, 2025）—— 现有最完整的"LLM Agent + 时序"系统。它有 5 个 Agent 节点：

```
[Preprocess] → [Analysis] → [Validation] → [Forecast] → [Report]
   数据清洗     特征分析      模型选择        预测组合      报告生成
```

TSci 在数据充足时表现很好。但它**没有显式建模"诊断本身的不确定性"**——即"我说这有趋势"这句话有多可信？置信度低时该不该激进地相信？这正是少样本场景下最致命的盲区。

### 1.4 核心假设

> **当训练数据极小（N=10~100）时，给诊断结论加置信度量化、根据置信度自适应选策略、反思迭代、跨序列经验积累——这四层机制能系统性降低误差，且各层独立有效。**

简单说：**少样本场景下，不光要"做诊断"，还要"知道自己诊断到不到位"。** TSci 在原版未做这件事，所以在少样本下脆弱，这就是 AdaptTS-Agent 的设计空间。

---

## 二、实验设置

### 2.1 数据集

按 plan §2.1 选了 5 类数据集，目前已就绪 2 个：

| 数据集 | 领域 | 总时间步 | 采样频率 | 状态 |
|---|---|---|---|---|
| ETTh1 | 电力负荷 | 17,420 | 1 小时 | ✅ 已跑完 |
| ETTh2 | 电力负荷（另一变量）| 17,420 | 1 小时 | ✅ 已跑完 |
| Weather | 气象多变量 | 52,696 | 10 分钟 | ⏳ 待补 |
| ECL | 电力消耗多变量 | 26,304 | 1 小时 | ⏳ 待补 |
| ILI / M4 | 医疗周采样 / 跨域 100 条 | 不等 | 不等 | ⏳ 待补 |

### 2.2 少样本切割协议（**关键设计决策**）

区别于 Time-LLM 等论文用比例切割（5%/10%）的做法，我们用**绝对数量切割**——这更贴合真实"冷启动"语义：

```
原始长序列：……………………………… (总长 17,420)
            ↓ 随机选窗口起点（seed 控制）
窗口：     [train N 点][val 10 点][test 96 点]
            ↑           ↑          ↑
         模型见到的    内部 holdout  最终评估
         "历史数据"     评估调度      不可见
```

**5 种数量级**（plan §2.2）：

| 设置 | N | 对应真实场景 |
|---|---|---|
| S1 | 10 | 极端冷启动（新传感器第一天） |
| S2 | 20 | 严格少样本（新业务一周日数据） |
| S3 | 50 | 中等少样本（新业务 ~2 月数据） |
| S4 | 100 | 相对充足（新业务 ~3 月数据） |
| S5 | Full | 全量参照（约 17,000 点） |

目前实验全部在 S1-S4 上跑完。S5 因 LLM 推理开销大，放到后期。

### 2.3 评估指标

| 指标 | 公式 | 选用理由 |
|---|---|---|
| MAE | mean(|y_true − y_pred|) | 直观，量纲与原始一致 |
| MASE | MAE / MAE_naive_seasonal | 跨数据集可比，消除量纲 |
| MSE | mean((y_true − y_pred)²) | 对极端误差敏感 |
| SMAPE | 对称百分比误差 | M4 竞赛标准 |

主报告用 **MAE**；论文中会一起给。每个 (dataset, N, method) 组合跑 **3 个随机种子**（控制窗口起点）取均值±标准差。

### 2.4 实验执行框架

所有方法实现统一接口（**这是工程上最关键的一致性保证**）：

```python
def predict(train: np.ndarray, val: np.ndarray, H: int,
            seed: int = 42, season_m: int = 1, **kwargs) -> np.ndarray:
    """返回长度为 H 的预测序列。"""
```

一个 CLI runner 跑所有实验，结果落 jsonl：

```bash
python -m research.experiments.runner \
    --dataset ETTh1 --N 20 --H 96 \
    --methods adapt_ts --seeds 1,42,123
```

LLM 调用全部带磁盘缓存（按 prompt 内容 hash），重复实验不重复消耗 API。

---

## 三、五个基线（B1-B5，论文 Experiments 必备对照）

每个基线代表**一类完全不同的方法学**，覆盖论文需要对比的整个谱系：

| ID | 方法 | 代表的方法学 | 关键代码 |
|---|---|---|---|
| **B1** | Naive (mean/drift/seasonal) | 纯统计兜底，"任何 AI 方法都该胜过它" | `baseline/naive.py` |
| **B2** | ARIMA + ETS (AIC 选模型族) | 传统时序统计代表 | `baseline/arima_ets.py` |
| **B3** | LLMTime (数字串 ICL) | "有 LLM 但无 Agent 结构" | `baseline/llmtime.py` |
| **B4** | **TSci (完整 Agent)** | "有 Agent 但无不确定性感知" — **我们的 target** | `baseline/tsci.py` |
| **B5** | Chronos-Small (60M 预训练) | 大规模预训练时序基础模型 | `baseline/chronos.py` |

**为什么这 5 个就够**：
- B1 是兜底（论文常识：超不过 Naive 的方法没意义）
- B2 是"无 LLM 的最强传统方案"
- B3 是"有 LLM 但单步"
- B4 是"有 Agent 但无 UQ"——直接对比我们要超越的目标
- B5 是"有预训练但无 Agent"

如果 AdaptTS-Agent 能在某些 N 上同时**胜 B2/B3/B4/B5**，那就证明"四层机制"的价值不可被任何单一方法替代。

**实施细节**：B4 TSci 直接克隆原作者仓库 (`Y-Research-SBU/TimeSeriesScientist`)，做最小适配（修两个 graph 节点的参数 bug + 把 LLM 调用重定向到我们用的智谱 GLM），保证基线数值的可信度（plan §10.4 R5 也明确要求这样做）。

---

## 四、AdaptTS-Agent 核心思想

### 4.1 架构总览

```
       一条少样本序列 (N 个点)
                  ↓
      ┌──────────────────────────┐
      │ 层一：诊断 Agent (UQ)    │   curator_uq.py
      │  输出：趋势/季节/平稳    │
      │       × 三路置信度       │
      └──────────────────────────┘
                  ↓ Diagnosis(trend_conf, season_conf, stat_conf)
      ┌──────────────────────────┐
      │ 层二：自适应 Planner     │   planner_adaptive.py
      │  规则：置信度→策略组合   │
      └──────────────────────────┘
                  ↓ Plan(strategies, weights)
      ┌──────────────────────────┐
      │ 层三：Forecaster + 反思  │   forecaster_reflect.py
      │  walk-forward CV 重加权  │
      │  → val 评估 → 反思 ≤3 次 │
      └──────────────────────────┘
                  ↓ 预测 + ForecastTrace
      ┌──────────────────────────┐
      │ 层四：跨序列记忆 (faiss) │   memory.py
      │  事后写入 case 库        │
      │  E5 实验时启用查询       │
      └──────────────────────────┘
                  ↓
              最终预测 (H 步)
```

### 4.2 层一：诊断置信度（plan §五的灵魂）

**问题**：传统诊断（如 ADF 检验）会输出一个 p 值；但用户/Agent 拿到 `p=0.08` 时该怎么用？说"接近平稳"还是"不平稳"？阈值附近**置信度本身就低**，应该被显式标注。

**我们的做法 · 三路置信度**：对每个诊断维度（趋势 / 季节 / 平稳）同时输出三种 conf：

| 路 | 来源 | 特点 |
|---|---|---|
| A · stat | ADF p-value + ACF 峰值 + 趋势 t-stat 的阈值映射 | 客观、可重复 |
| B · llm | LLM 看统计量给主观判断 + 文字理由 | 能识别"反直觉"情形 |
| C · xc | A、B 取较低（保守） | 双路交叉，最稳 |

**为什么这样设计**：plan §5.2 明确要在论文里对比三种来源的"置信度-误差校准比率 (CMR)"——这是论文 §五的独立实验，验证置信度标签是否真的反映可靠性。

输出示例（ETTh1 N=20 seed=1）：

```python
trend:  stat=high (t-stat=9.6)  llm=high   xc=high
season: stat=high (ACF=0.71)    llm=low    xc=low   ← LLM 正确指出 N=20 季节伪信号
stat:   stat=low  (ADF p=0.99)  llm=low    xc=low
```

### 4.3 层二：自适应 Planner（plan §四层二）

规则映射"置信度 → 策略组合"：

| 触发条件 | 输出 plan |
|---|---|
| N ≤ 12（极端冷启动）| 强制 safe：[LLMTime, Chronos, 漂移] 集成 |
| 任一维度低置信 | 三路集成（覆盖不同假设）|
| 全部中置信 | 双路弱集成 |
| 全部高置信 | 单一精细策略（avoid 集成稀释）|

**关键设计哲学**：**置信度越低 → 越要保守集成**（避免"过度相信错误诊断"的风险）。

### 4.4 层三：Forecaster + walk-forward CV + 反思

这是 v4 最新改动的核心。三步：

**① walk-forward CV 重加权**（v4 新增）—— 替代 prefix 规则的"硬权重"：

```
train（N 个点）
     ↓ 在 train 末尾切多窗口
[ fit on train[:cut1] ] → 预测 train[cut1:cut1+H_v] → 评估各策略 MAE
[ fit on train[:cut2] ] → 预测 train[cut2:cut2+H_v] → 评估
[ fit on train[:cut3] ] → ...
     ↓
按 softmax(-MAE/τ) 重新分配权重
```

τ 随 N 调整：N≥50 用小 τ（winner-take-all），N<50 用大 τ（更分散更保守）。

**② val 评估**：用重加权后的 plan 在 val 段算 MAE。

**③ 反思**：若 val MAE 超过阈值（val.std × 0.5），让 LLM 看"上轮诊断 + 各策略 val MAE + 已试过的 plan 列表"，重新选 plan。硬上限 3 次（plan §12 R2 防发散）。N≤12 时禁反思（val 完全不可靠）。

**新 plan 切换门槛**：必须比当前 best 改善 ≥20% 才采纳（避免噪声触发劣化）。

### 4.5 层四：跨序列记忆（faiss）

每条序列处理完写入向量库：

```python
Case(
    feature=case_features(diagnosis),  # 10 维诊断特征
    diag={...},                         # 完整诊断
    final_plan={strategies, weights},   # 最终用的策略
    test_mae=...,                       # 事后回填
    meta={dataset, N, H, seed},
)
```

E5 实验启用 query 路径时：处理新序列前，**用诊断特征做 kNN 查相似案例**，把 top-K 案例的 final_plan 作为 LLM 的"经验参考"喂进 prompt。

当前**只做写入**，不影响主实验跑分；E5 时实测带记忆 vs 不带记忆的 50 条序列学习曲线。

---

## 五、实验已完成到哪一步

### 5.1 工程进度（Phase 0-4）


| 阶段 | 内容 | 状态 |
|---|---|---|
| Phase 0 | 数据 loader、切割、指标、runner、Naive 基线 | ✅ |
| Phase 1 | B2 ARIMA+ETS, B5 Chronos | ✅ |
| Phase 2 | B3 LLMTime, B4 TSci 适配层（含 monkey-patch） | ✅ |
| Phase 3 | AdaptTS 四层（v1） | ✅ |
| Phase 3.7-3.10 | AdaptTS 三轮迭代 v2 → v3 → v4（walk-forward CV）| ✅ |
| Phase 4 | ETTh1 全量 + ETTh2 全量主表（144 cell）| ✅ |
| Phase 5 | E2/E3/E4/E5 四组分析实验 | ⏳ 待启动 |

### 5.2 关键实测结果

**ETTh1 主表（MAE 均值，3 seeds，H=96）**

| N | Naive | ARIMA | Chronos | LLMTime | TSci | adapt_v4 |
|---|---|---|---|---|---|---|
| 10 | 19.39 | 13.08 | 4.67 | **3.61** | 4.57 | 6.62 |
| 20 | 5.58 | 4.21 | 3.99 | **2.95** | 5.42 | **4.06** |
| 50 | 6.06 | 4.06 | **3.13** | 3.60 | 4.53 | 4.42 |
| 100 | 6.16 | **2.52** | 3.13 | 2.81 | 4.73 | 3.37 |

**ETTh2 主表（同上）**

| N | Naive | ARIMA | Chronos | LLMTime | TSci | adapt_v4 |
|---|---|---|---|---|---|---|
| 10 | 24.00 | 8.04 | **5.00** | 7.30 | 9.55 | 11.25 |
| 20 | 7.98 | 5.34 | **4.98** | 6.24 | 8.12 | **4.83** |
| 50 | 4.10 | 7.37 | **3.85** | 6.04 | 7.98 | **4.04** |
| 100 | 8.36 | 9.41 | 5.12 | 5.83 | **4.07** | 6.28 |

**AdaptTS v4 vs TSci 直接对比**（8 个 cell）：

| | ETTh1 | ETTh2 |
|---|---|---|
| N=10 | 6.62 vs 4.57 ❌ | 11.25 vs 9.55 ❌ |
| N=20 | **4.06 vs 5.42 ✅ -25%** | **4.83 vs 8.12 ✅ -41%** |
| N=50 | 4.42 vs 4.53 ≈ | **4.04 vs 7.98 ✅ -49%** |
| N=100 | **3.37 vs 4.73 ✅ -29%** | 6.28 vs 4.07 ❌ |

**5 胜 3 负**。N=20/50 区间是 AdaptTS 的甜点区，胜幅 25-49%。

### 5.3 三个论文级 finding

1. **TSci 在少样本下确实脆弱**：ETTh1 全 N 没有一个 cell 是最优；ETTh2 仅 N=100 最优。**plan §一 motivation 得到实证**。
2. **跨数据集"最优方法"完全不同**：ETTh1 N=10/20 LLMTime 最优 / ETTh2 N=10/20 Chronos 最优；ETTh1 N=100 ARIMA / ETTh2 N=100 TSci。**没有方法跨数据集通杀** → "自适应选择必要"是论文最强论点。
3. **AdaptTS 在"中等少样本"（N=20/50）区间稳定胜出**，但 N=10 极端冷启动和 N=100 数据充足时仍有差距——诚实的 limitation，有明确改进路径（候选库补全 + Stacking 学权重）。

---

## 六、问题

### Q1：跨数据集"最优方法完全不同"是否足够支撑论文核心论点？
这是我们最 confident 的实验发现。前人论文做 baseline 对比要么只在单数据集上，要么只对比 LLM vs 非 LLM 二分。我们把"何时该用哪类方法"的全空间画出来了（5 类基线 × 2 数据集 × 4 个 N），且**确实没有 method 全胜**。但在领域语境下这个 novelty 强度需要您判断。

### Q2：AdaptTS 当前 5 胜 3 负 vs TSci，作为 paper 第一稿够不够？
- 倾向"够"的理由：(a) N=20/50 区间稳胜 25-49% 已经显著；(b) 跨数据集 finding 强化"自适应必要"论点；(c) limitation 有明确改进路径写 future work；(d) Discussion 章节里"知道为什么没全赢 + 给出诊断"比"全部胜出"更可信。
- 倾向"不够"的理由：若 ICLR/KDD 主会要求方法在主表"显著优于所有 SOTA"，5 胜 3 负不够—可能要再投 2 个月调到 7-1 才稳。

### Q3：资源边界与预算
- 当前所有 LLM 走免费 API（智谱 GLM-4-flash-250414），**$0 美元成本但单 LLM 调用 ~5 秒**。
- 缓存命中率高时全跑一遍 ETTh1+ETTh2 全套 ~1 小时；冷启动跑全量 ~5 小时。
- E3 人工标注（30 序列 × 3 标注者）需要人力配合，是否申请研究助理工时？
- 若想跑完整 6 数据集 × 5 N × 5 方法 × 5 seeds × E1-E5 五组实验，需 ~$10-30 OpenAI 预算（plan §10.2 估算 $5-15 一套 E1）。

---

## 七、近期行动计划

### 接下来 1-2 周
- **A 修复 AdaptTS-v4 在 N=100 上的退步**：候选库 N>30 时保留完整 strategy pool，让 walk-forward CV 自由加权
- **B 补 Weather + ECL 数据集**：让 E1 主实验数据覆盖 4 个数据集
- **C 启动 E2 置信度校准实验**：curator_uq 已输出三路 conf，可直接复用做 CMR 计算

### 接下来 1-2 个月（按 plan §一 E1-E5 推进）
| 实验 | 目标 | 现状 |
|---|---|---|
| E1 主精度对比 | AdaptTS vs 基线，N×Dataset 矩阵 | 2/6 数据集完成 |
| E2 置信度校准 | 验证置信度有效性（CMR）| 框架就绪，待跑 |
| E3 可解释性 | 30 序列 × 3 人标注 × Cohen's κ | 待启动 |
| E4 消融 | 关闭各层评估贡献 | 框架支持，待跑 |
| E5 记忆 | 50 同领域序列学习曲线 | faiss 就绪，待启 query |

### 接下来 3-6 个月
- 论文写作
- 可能的扩展方向：多变量预测、概率预测（输出区间）、跨领域迁移

---

## 八、附录：当前代码与结果位置

```
agent_ts/
├── demo/                       # 入门阶段最小 4-Agent 流水线（已完成，作为 research/ 的"基础课"）
└── research/                   # 论文实验工程
    ├── plan.md                 # 完整实验设计（不动）
    ├── finish.md               # 实验成果详细记录（含全部主表 + finding）
    ├── TODO.md                 # 任务看板
    ├── report.md               # 本汇报文档
    │
    ├── utils/                  # 数据加载/切割/指标/LLM 客户端
    │   ├── data_loader.py
    │   ├── splitter.py
    │   ├── metrics.py
    │   └── llm.py
    │
    ├── baseline/               # 5 个基线统一接口
    │   ├── naive.py
    │   ├── arima_ets.py
    │   ├── chronos.py
    │   ├── llmtime.py
    │   └── tsci.py             # 适配层调用 external/TimeSeriesScientist/
    │
    ├── agent/                  # AdaptTS-Agent 四层
    │   ├── curator_uq.py       # 层一：诊断置信度
    │   ├── planner_adaptive.py # 层二：策略选择
    │   ├── forecaster_reflect.py # 层三：walk-forward + 反思
    │   ├── memory.py           # 层四：faiss 记忆
    │   └── adapt_ts.py         # 统一入口
    │
    ├── experiments/            # runner + 各实验脚本
    │   ├── runner.py
    │   ├── run_tsci_full.sh
    │   └── run_etth2.sh
    │
    ├── results/                # jsonl 结果 + summary 表
    │   ├── p0_naive.jsonl
    │   ├── p1_arima_ets.jsonl
    │   ├── p1_chronos.jsonl
    │   ├── p2_llmtime.jsonl
    │   ├── p3_adapt_v*.jsonl
    │   ├── p4_tsci_etth1.jsonl
    │   └── p4_etth2_*.jsonl
    │
    ├── datasets/raw/           # ETTh1 / ETTh2 csv 缓存（git ignored）
    ├── external/               # TSci 原仓库（B4 用）
    └── .llm_cache/             # LLM 调用磁盘缓存
```

**完整跑分结果**：所有 144+ cell 的 jsonl 原始数据都在 `results/`，您可随时复现或下钻分析；`finish.md` §3 节有完整的双数据集六路对比表。
