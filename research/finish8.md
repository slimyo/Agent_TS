# Finish v8 — Round 13 实测（Heterogeneous-Expert Routing）

> 版本：2026-06-08。方法见 `method8.md`；外部评审 `feedbackm7.md`；前置 `finish7.md`（F-R12.x）。
> 本文件承接 method8：用受控 regime benchmark 证明"winner 可预测性 = 资产池结构属性"。

---

## 0. 总判断

> **method8 thesis 成立（强证据）**：同一套决策机制、同样的 regime 数据，**只改资产池的结构异构度**——
> 异构专家池上 winner 可预测、路由捕获 88% oracle 头寸（+35%relMAE）；同质池上 winner≈不可预测、路由净亏。
> → method7 的"+0.00pp / gain 不可学"是**同质学术池的局部结论，非普适规律**。
> 决定选择性行动可达价值的是 **决策机制 × 资产池结构** 的乘积，单改任一不够。

---

## 1. E12 · 异构 vs 同质池对比（`m24_hetero_experts.py`）

**受控 benchmark**：合成 5 种主导结构序列（trend/seasonal/ar/noisy/spike）各 40 cell = 200 cell，每 cell 有 task signature。
两个池跑同一评测（5-fold：task-signature→winner 分类 + 路由 rel-MAE 捕获）：

| 池 | winner 预测 acc | 多数类基线 | 路由 vs_base | oracle 头寸 | 捕获率 |
|---|---|---|---|---|---|
| **异构专家池**（trend/seasonal/ar/robust/spike+base）| **0.590** | 0.300 | **+35.23%relMAE** | 40.14% | **0.878** |
| 同质池（5×MA 窗口变体 + base）| 0.385 | 0.340 | **−2.91%relMAE** | 9.36% | −0.311 |

**F-R13.1（核心）**：**winner 可预测性是资产池的结构属性**。结构异构专家（不同归纳偏置，能力差异>>噪声）
使 task-signature→winner 可学（acc 0.59 ≈ 2× 多数类），capability 路由捕获 **88%** 头寸；
同质调参变体使 winner 退化到≈多数类（0.385 vs 0.34），路由净亏（复现 method7 的 +0.00/负）。

**F-R13.2**：**头寸本身也由池结构决定**。异构池 oracle 头寸 40%relMAE（专家在各自 regime 上碾压），
同质池仅 9%（变体间差异小）。→ 不仅"能不能抓"、连"有没有得抓"都取决于池的结构互补性。

---

## 2. 对全研究链的意义（method4→8 收束）

| 阶段 | 结论 | 条件 |
|---|---|---|
| method4-6 | 决策机制四变量、trust 避险闭环 | — |
| method7 | 同质池上 gain 离线不可学、+0.00pp（正确弃权）| **同质学术池** |
| test/ 在线 | 反馈把 gain 变现，∝ 域级结构稳定性（分类 +2.83pp）| 同质池 + 反馈 |
| **method8** | **换结构异构池，winner 可预测、路由 +35%relMAE** | **异构专家池 + regime 数据** |

→ 完整论点：**"选择性行动能不能创造收益" = f(决策机制, 资产池结构, 反馈)**。
method7 钉死了"机制已到顶（避险）"，method8 钉死了"换池子才出获利"，test/ 钉死了"反馈是变现媒介"。三者缺一不可。

---

## 2b. E13 · 端到端落地（异构专家接进 test/ 4-agent 闭环，真实执行）

把异构专家池接进 `test/`：`experts.py`（**12 专家**+model signature+学出 capability）+ `pipelines.run_hetero_online_stream`（Curator→在线 bandit→真跑专家→反馈）+ `run_hetero_test.py`。9 regime 部署流、**真实执行模型**：

| 策略 | vs_base | 偏离率 |
|---|---|---|
| **learned-cap（学出先验+反馈）** | **+13.74%relMAE** | 0.45 |
| learned-cap（只学出先验）| +14.08%relMAE | 0.45 |
| online+手工 affinity | +13.55%relMAE | 0.65 |
| static 手工 affinity（不学）| **−24.94%relMAE** | 1.00 |
| online-no-prior（只反馈）| +3.10%relMAE | 0.39 |
| oracle 上界 | +46.78%relMAE | — |

**F-R13.3（杠杆随池结构转移）**：同质池（method7/test 分类）无 signature→model 映射 → **反馈是唯一杠杆**（+2.83pp，需域级稳定）；
异构池 **capability 结构匹配成主杠杆**（学出 +14%），反馈是纠偏/兜底。→ 换异构专家，选择性行动从 +0.00 真正变现，L1 base 仍兜底。

**F-R13.4（扩库/可扩展性，6→12 专家）**：**手工 capability signature 不随池扩展**——6 专家时手工 affinity +19%，12 专家时手调 12×7 失配→ **−24.94%**（净亏）。
**学出的 capability profile（校准历史记录每 (regime,expert) 平均 reward）才是正解**（feedbackm7 路线2 正确实现）：稳定 +14%、可扩展任意池。
且**反馈对坏先验有纠偏力**（手工 static −24.94% → 加反馈 +13.55%）。→ 工业落地：能力画像必须**数据驱动学出**，不能手工；反馈是抗错先验的保险。

## 2c. E14 · learned-capability 真实数据验证（`test/run_realcap_test.py`）—— 关键负结果

把 learned-capability 接到**真实预测库**：14 个真实异构 TSFM（chronos/timer/timesfm/moirai/toto/…，base=chronos2）、
6 真实域（ETT/ECL/Weather/Exchange/ILI）、每 cell 重建**真实训练序列**→task_signature→regime_tag、
reward=库里真实测得、learned-cap **LODO by dataset**（无泄漏）：

| 策略 | vs_base |
|---|---|
| learned-cap static（只先验）| **−6.61%relMAE** |
| online（学出先验+反馈）| −2.92%relMAE |
| oracle-per-cell | +23.93%relMAE |

regime 分布（真实少样本序列）：shift 41 / noise 16 / ac1 14 / **seasonal 仅 1**。

**F-R13.5（关键边界）**：**method8 在真实通用 TSFM 池上不成立**——learned-capability 净亏（−6.6%）、反馈只拉到 −2.9%。两个诚实原因：
1. **真实通用 TSFM 是"架构不同"，非"结构互补"**：它们都是通用预测器，无清晰 regime 专长 → regime↔model 映射不存在 → 学出的先验是噪声、misroute。
2. **少样本窗口退化 regime 检测**：N=10–100 训练窗 task_signature 失真（季节测不出，多数误判 shift/noise）。
→ **精确界定 method8 thesis**：成立条件是"**结构互补的专家**（各有清晰 regime 专长）+ **足够上下文检测 regime**"。
合成结构专家两者皆满足（+14%）；真实少样本 TSFM 两者皆不满足（−6.6%）。
→ "异构能救"= **结构互补的异构**，不是"架构不同"；真落地需引入**真正的结构专家**（physics/规则/统计），不能指望通用 TSFM 池。

## 2d. E15 · 决定性 2×2（结构专家 × 真实数据，`test/run_realstruct_test.py`）—— 重大诚实修正

为隔离"池结构 vs 数据真假"，把 **我的 12 结构专家**跑到**真实序列**（ETT/ECL/Weather/Exchange/ILI/ETTm，N≥50 给足上下文），
learned-cap LODO 路由。填满 2×2：

| | 结构专家 | 通用 TSFM |
|---|---|---|
| **合成 regime 数据** | **+14.08%** (E13) | — |
| **真实数据** | **−6.37%** (E15) | −6.6% (E14) |

真实×结构：static −6.37% / online −6.30% / **oracle 头寸 +38.66%**（头寸巨大但抓不到）。
regime 分布（真实少样本）：ac1 91 / shift 79 / noise 22 —— **完全没有 trend/seasonal 标签**。

**F-R13.6（决定性，修正 method8）**：**瓶颈不是池结构，是真实数据的 winner 从可观测 task-signature 不可预测**——
结构专家在真数据上也净亏（−6.37%），尽管 oracle 头寸 38.66%。即：
- method8 的 **+14% 是"合成 regime 数据"的属性**（regime 按构造可观测且与专家对齐），**不迁移到真数据**。
- 真实少样本序列被 regime 检测器塌成 ac1/shift/noise（无干净 trend/seasonal），tag 内 winner 不稳定 → 路由失败。
- **全研究链闭环回 F-R12.1**：真实 TS 的 per-cell winner 从离线特征不可预测——**换结构专家也不例外**。

→ **唯一在真数据上变现的是 per-domain 在线反馈**（分类 +2.83pp，domain=稳定单元）；
capability-routing 在真预测域（无稳定 task-signature→winner 映射）不成立。
**真正的工业杠杆 = 存在稳定单元（域/episode）+ 在线反馈，而非"换异构专家池"**。这是从合成到真数据最硬的一课。

## 3. 诚实边界

- m24 是**受控合成**证明（regime 可观测、专家与 regime 对齐），证的是"异构能让 winner 可预测"的存在性，
  **不等于工业数据**——真实场景 regime 可能不可观测、专家与场景未必对齐。
- winner acc 0.59 远非 1.0：异构也只是**部分**可预测（regime 边界含噪），但已足以让路由从负转 +35%。
- 下一步（真落地）：把 `test/online_router` 的 capability 先验接到**真异构专家**（真 physics/统计/深度模型）+ 真域 +
  真实部署反馈流，验证 F-R13.1 在非合成数据上成立。

---

## 4. Findings 索引 F-R13.x

| ID | 一句话 | 出处 |
|---|---|---|
| **F-R13.1** | winner 可预测性是资产池结构属性：异构池 acc0.59/路由 +35%relMAE(捕获88%)，同质池 ≈多数类/净亏 | E12 |
| **F-R13.2** | oracle 头寸也由池结构决定：异构 40% vs 同质 9% relMAE | E12 |
| **F-R13.3** | 端到端落地：异构池 capability 匹配成主杠杆+反馈兜底；杠杆随池结构从"反馈"(同质)转移到"结构匹配"(异构) | E13 |
| **F-R13.4** | 扩库 6→12 专家：**手工 capability signature 不随池扩展**(+19%→−25%)；**学出的 capability profile 才是正解**(稳定+14%、任意池可扩)；反馈对坏先验有纠偏力(−25%→+13.55%) | E13 |
| **F-R13.5（关键边界）** | **真实通用 TSFM 池上 learned-cap 净亏(−6.6%，反馈拉到−2.9%)**：真 TSFM 是"架构不同"非"结构互补"、少样本退化 regime 检测 | E14 |
| **F-R13.6（决定性修正）** | **结构专家在真数据上也净亏(−6.37%，oracle 头寸 38.66%)**：瓶颈是真数据 winner 从 task-signature 不可预测，非池结构；method8 +14% 是合成 regime 属性、不迁移；真数据唯一变现=per-domain 在线反馈(分类+2.83pp) | E15 |

---

**End of finish8.md** — 真异构专家 + 真域落地后追加 §5+。
