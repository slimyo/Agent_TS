# Method v7 — Trust-Aware Gain Routing → 双信号解耦的选择性行动决策

> 版本：2026-06-07（Round 12 完整版）。接续 `method6.md`（Round 11：决策机制 / trust 只避险）。
> **配套**：`feedback_m6.md`（v6 review·3 reviewer 一致）/ `finish7.md`（F-R12.x 实测）/
> `paper_mechanism.md`（机制论文）/ `advisor_report.md`（脉络）/ `baseline.md`（全量库）/ `plan.md §零`。
>
> method7 的设计起点是"补 gain model"，但**实验把它改写成了一个更强的命题**：选择性行动决策应当
> **解耦成两个用不同信号的子问题**，而 gain 的逐-cell 可学性是一个领域无关的负结果。本文件完整记录
> method7 的设计、框架、思想、理论与落地。

---

## 0. Thesis（一句话）

> Round 11 证明决策可拆成四变量、且 **trust ≈ P(action safe) ≠ gain ≈ E[Δreward]**（F-R11.7）。
> method7 原计划补 gain model 把决策升到 `if trust>τ1 AND gain>τ2`；
> **实验发现 gain 逐-cell 不可学（F-R12.1/12.2），但"避险"和"获利"分别有各自最优信号（F-R12.5）**——
> 于是 method7 的最终形态不是"trust+gain 双门"，而是 **双信号解耦决策**：
> **`deviate iff conformal_safe(z) ∧ saturation_headroom(z)`**——用对的信号干对的子问题，不求万能 confidence。
> 在全量 22 模型库下它诚实收敛到守 base（+0.00pp），这不是失败而是
> **"可达天花板 ≠ 逐-cell 可预测"** 下的正确弃权；若放宽避险门则进入一个薄而真实的**激进甜区**（+0.15pp）。

---

## 1. 框架：选择性行动的四变量分解（method7 核心结构）

把单一的"confidence/该不该偏离"拆成**四个语义不同、各自可测**的量。这是 method6→7 的骨架，也是
本研究最稳的正向贡献（feedback_m6 三 reviewer 一致认可）。

| 变量 | 定义 | 回答的子问题 | 估计器 | 状态 |
|---|---|---|---|---|
| **saturation** `ĝ(z)` | 预测 oracle−base gap（归一头寸） | 有没有可决策的余地（该不该进场） | gap 回归（RF/LODO） | 已闭环 |
| **belief / proposal** `b(M\|z)` | 预测哪个候选最可能最优 | 提哪个偏离候选 | softmax-CE 头 / proposal-net / LLM | 已闭环 |
| **trust** `≈P(safe)` | 这次偏离在不在"已见安全分布"内 | 偏离会不会变差（避险） | **conformal nonconformity** | 已闭环（F-R11.9 部署） |
| **gain** `≈E[Δreward]` | 这次偏离能净赚多少 | 偏离能不能赚（获利） | gain 回归（原计划主件） | **证不可学（F-R12.1）** |

**关键认知（F-R11.7 → F-R12.5）**：trust 是**风险估计器**，gain 是**效用估计器**，二者语义正交；
而且经测，**避险**这一维 conformal 信号最强、**获利**这一维 saturation 信号最强，把它们融成一个标量会互相稀释。

---

## 2. 设计演进：从"trust+gain AND 门"到"双信号解耦"

这是 method7 最重要的思想转折，记录"设计假设 → 实验证据 → 修正后的设计"。

```
原设计（method7 §1 初版，feedback_m6 建议）:
    decide = deviate  iff  trust(z) ≥ τ1  AND  gain(z, proposal) ≥ τ2
            （避险门）            （获利门，需 gain model）
                       │
        E4/E5 实测 gain 逐-cell 不可学（profit-AUC 0.47≈瞎猜，预测域 17% 头寸吃不到）
                       │
        E8 实测：避险/获利各有最优信号，且不是同一个，朴素融合稀释
                       ▼
最终设计（method7 终版）:
    decide = deviate  iff  conformal_safe(z)  AND  saturation_headroom(z)
            （避险=conformal，AUC0.79）   （获利优先级=saturation 头寸，AUC0.71）
    —— 不再训练一个"万能 gain model"，而是用 saturation 做"获利优先级排序"、conformal 做"避险闸"。
```

**为什么不是 gain model 而是 saturation 当获利信号**：gain（逐-cell 净增益）不可回归（F-R12.1），
但 saturation（该域/该 cell 有没有头寸）是更粗、更可学的标量——它排不动"赚多少"，但能排"哪里值得试"。
这就是把"获利"从"不可学的精细反事实"降级为"可学的头寸优先级"的设计折中。

---

## 3. 决策算子（最终形态）与激进度调节

### 3.1 双信号解耦决策（保守，损害控制）

```
对 query cell z:
  s      = saturation_headroom(z)        # 归一预测头寸，∈[0,1]，高=有余地
  prop   = belief_argmax_nonbase(z)      # 提议偏离候选
  t      = conformal_trust(z, prop)      # 偏离可信度，∈[0,1]，高=在已见安全分布内
  if  t ≥ τ_trust  AND  s ≥ τ_head:  a = deviate→prop
  else:                              a = commit→base
```
全 LODO；conformal 标定集来自其它域历史 outcome；无 test 泄漏。

### 3.2 激进度调节（policy 参数，"不追求 pp"时用）

按 method 结论，保守门在全量库收敛守 base（+0.00pp）。若**不追求 pp、要主动出手抓 oracle 头寸**，
`run_signal_router(cells, policy=...)` 提供一条**激进度拨盘**（`test/signal_router.py`）：

| policy | 决策律 | 行为 |
|---|---|---|
| `conservative` | `t≥τ AND s≥τ` | 守 base（双信号门）|
| `aggressive` | **丢 conformal 门**，`s≥τ_head` | 高头寸格主动偏离 |
| `aggressive-belief` | `prop≠base` | 始终听 belief 偏离（最激进）|
| `explore` | `s≥τ OR t<τ` | 未知未知区主动探索（F-R12.6 信息增益视角）|

**激进甜区（实测，全量 22 模型库 / 221 cell）**：

| 策略 | vs_base | 偏离率 | oracle 命中 | 安全偏离率 |
|---|---|---|---|---|
| conservative 双信号门 | +0.00pp | 0% | — | — |
| **aggressive 头寸≥0.3 ★** | **+0.15pp** | 7% | 0.125 | 0.69 |
| aggressive 头寸≥0.15 | −0.00pp | 42% | 0.106 | 0.61 |
| aggressive-belief 始终偏离 | −2.46pp | 100% | 0.068 | 0.48 |
| oracle 上界 | +5.95pp | — | — | — |

**读法**：丢掉避险门、只在高头寸(≥0.3)格主动出手 → 净 **+0.15pp**（保守门拿不到的真实正收益）；
但更激进就崩（≥0.15 washes out，始终偏离 −2.46pp）——winner 不可预测使激进收益**薄且对阈值敏感**。
**"不追求 pp"的真义**：用激进换 oracle 头寸的**主动覆盖**（命中率/捕获率），pp 是甜区副产物。

---

## 4. 全量库审计：头寸真实但逐-cell winner 不可预测（非循环核心）

method7 必须回答一个尖锐质疑：*"若库窄、数据集固定，某模型一饱和，'路由不如直接用它'就是同义反复——
+0.00pp 会不会只是窄库 artifact？"* 我们把分类库扩到 **全量 22 模型（全 228 cell 真跑完）** 来正面回应。

**① un-saturation 是真的（窄库 artifact 已修）**：

| 库规模 | rocket 即 oracle | oracle 头寸 |
|---|---|---|
| 5 分类器（旧 F-R9.7） | 71% | 1.88pp |
| **22 分类器 / 38 数据集（全覆盖）** | **24.4%** | **5.95pp**（中位 3.0pp） |

oracle 由 **21/22 个模型**分摊（rocket 仅 24% / muse / minirocket / rocket_mv / multirocket / fcn …），
22/38 数据集 top 非 rocket。含真·远程大 TSFM 嵌入（chronos2_emb / timesfm_emb / timer_emb 全 228 真实推理）。

**② 但 winner 逐-cell 不可预测（真瓶颈）**：用部署可得特征 LODO 预测"谁赢"：

| 预测器 | winner 命中 | 选预测赢家 vs base |
|---|---|---|
| 永远选 rocket | 0.204 | +0.00pp |
| 4 维系统特征 | 0.249 | −0.25pp |
| 30 维序列统计 | **0.113** | **−4.64pp**（更差）|

**头寸真实(5.95pp)、但照预测去抓 = 负收益**，特征越丰富越差（过拟合）。谁赢主要由少样本噪声/换种子驱动，
不是稳定"模型A↔数据类型B"映射。→ **+0.00pp 的正解 = "可达天花板 ≠ 逐-cell 可预测"，非窄库 artifact。**

---

## 5. 理论：选择性行动的 regret bound（#105）

把两大实证支柱（避险可学 / 获利不可学）连成一个命题。固定 proposal `z↦m(z)`，记 δ_i = score(m_i)−base 为
**带符号**偏离增益，门在 D={i: t(z_i)≥τ} 上偏离。实现超额 E(D)=Σ_{i∈D} δ_i；同 proposal 完美门的 oracle
在 D⁺={δ_i>0} 偏离得 G⁺=Σ max(δ_i,0)。

**命题（AUC 控制 regret）**：设避险信号 Trust-AUC=A。
- **A→1**：D→D⁺，regret→0（抓全正头寸、零泄漏）；
- **A=½**：E[E(D)] = coverage·Σδ_i = coverage·n·E[δ]，而饱和/强 base 库 **E[δ]<0** ⇒ 任何非零覆盖期望超额为**负**——
  正是实测 always-deviate −2.46pp / 决策 cell −5.3pp。

**推论（为何 commit-dominant 最优）**：捕获的**收益**项由 *seek-gain* AUC 决定（F-R12.1≈½），
*avoid-harm* 在部署级轻特征下也≈½（F-R12.8，仅富 epistemic 特征达 0.79）。两者≈½ ⇒ 唯一最优阈值是 coverage→0，
即 **commit to base**，regret-to-oracle = 不可约 oracle 头寸 G⁺/n（5.95pp，全量库），任何部署门都无法变现。∎(sketch)

> 一句话：**机制可达价值 = 避免的损害（avoid-harm AUC 控），永不是捕获的收益（seek-gain AUC≈瞎猜）。**
> 当避险信号够强（conformal 富特征 A=0.79）门可证把损害降到~0——正是 F-R11.9 部署结果（−0.42→+0.00pp，误偏离 2→0）。

---

## 6. 六个研究方向与实测（E4–E11，详见 finish7 F-R12.x）

| # | 方向 | 文件 | 结论 | Finding |
|---|---|---|---|---|
| 101 | **Gain Modeling**（最高优先）| `m16_gain_model.py` | gain 幅度弱可回归(corr0.40)、"是否获利"分类失败(AUC0.47)；双门精度翻倍仍被天花板锁死 | **F-R12.1** |
| 102 | Forecasting 主场 | `m17_forecast_gain.py` | 未饱和域 17% oracle 头寸也吃不到(gain corr0.18)；否证"转预测即显著" | **F-R12.2** |
| 102b | 换不换预测 base（E10）| `timer_vs_chronos2_*` | Timer-S1(8.3B) 不优于 Chronos-2（60cell −7.0% rel-MAE）→ base 强非参数量决定 | **F-R12.7** |
| 103 | Counterfactual/Portfolio | `m18_portfolio.py` | 组合输单 base（−0.14~−3.6pp）；饱和库无值得分散的互补结构 | **F-R12.3** |
| 104 | Proposal Net 替代 LLM | `m19_proposal_net.py` | 轻量 proposal-net ≈ LLM（profits 0.235 vs 0.267）→ proposal 角色可无-API 平替 | **F-R12.4** |
| 105 | Meta-trust / regret | `m20_meta_trust.py` `m23_trust_transfer.py` | 避险↔conformal(0.79)/获利↔saturation(0.71) **不同信号、融合稀释**；trust 迁移是**特征级非任务级** | **F-R12.5 / F-R12.8** |
| 106 | 探索性偏离 | `m21_explore.py` | 未知未知区大(28.8%/12pp)但离线探索净负；唯在线信息增益有价值 | **F-R12.6** |

**主线收敛**：M4 inversion → M5 saturation → M6 trust 只避险 → **M7 避险↔conformal / 获利↔saturation 双信号解耦**。

---

## 7. 落地（research → test/ 部署）

- **保守双信号门**已接入 `test/signal_router.py`（全量 22 分类 / 14 预测 TSFM / 22 检测，LODO），
  三任务全量库相图 `results/m22_threetask_phase.png`（按 oracle-winner 上色，证全库参与）。
- **激进档**（§3.2）作为 `policy` 参数同框架提供；`test/run_full_test_v2.py` 输出保守 vs 激进对比 + 阈值扫描 + oracle-capture 指标。
- 三任务实测：分类 +0.00pp / 检测 +0.94pp\*(n=12 噪声) / 预测 −1.6%relMAE；激进甜区分类 +0.15pp。
- 远程大模型经 `emb_clf_sweep.py`（顶层加载一次+重试，GPU，绝不写 majority 假值）真实远程推理（见 baseline.md §4）。

---

## 8. 诚实底线与开放条件（seek-gain 何时才可学）

**底线**：全 LODO；gain/trust/proposal 标签来自其它域历史 outcome；无 test 泄漏。oracle 天花板低、winner 不可预测——
目标从"刷 SOTA"改为"诚实刻画选择性行动能/不能知道什么"。

**当前结论的边界（质疑里对的部分）**：
1. 域仍是学术 UCR/UEA/ETT；"不可预测"是这些域+这些特征下的结论，工业盲测/在线域可能不同。
2. 库虽 22 模型但偏同质感知模型（kernel/distance/dict/deep 变体）；**真异构、结构互补的资产池**（不同物理先验）
   是否出现"可预测的稳定专长"= 开放问题。

**正路（要拿正收益靠换条件，非更复杂的离线 confidence）**：
- 未饱和域（forecasting 主场，已部分验证仍难）；
- 真互补资产池（非同质分类器）；
- **在线反馈**——把逐-cell winner 从离线猜测变成累积后验，唯一能让 F-R12.6 探索与 gain 预测变现的设置。

---

## 9. 文件地图

```
research/
├── method7.md                       # 本文件（设计/框架/思想/理论全集）
├── finish7.md                       # F-R12.1–12.8 实测
├── paper_mechanism.md               # 机制论文（含 §4.5 regret bound）
├── advisor_report.md                # §6 全量库审计 + +0.00pp 非循环解读
├── baseline.md                      # 全量模型库 / 数据集 / 远程流程
└── experiments/
    ├── m16_gain_model.py  m17_forecast_gain.py  m18_portfolio.py
    ├── m19_proposal_net.py  m20_meta_trust.py  m21_explore.py
    ├── m22_threetask_phase.py        # 三任务全量库决策相图（oracle-winner 上色）
    ├── m23_trust_transfer.py         # trust 跨域可迁移性（F-R12.8）
    ├── timer_vs_chronos2_sweep.py    # E10 换 base 实测（F-R12.7）
    ├── taskb_newlib_sweep.py / emb_clf_sweep.py   # 全量 22 分类器 sweep（含远程嵌入）
    └── taska_tsfm_sweep.py           # 全量 14 预测 TSFM sweep
test/
├── signal_router.py                 # 双信号解耦 + 激进 policy 拨盘 + oracle-capture 指标
└── run_full_test_v2.py              # 保守/激进/阈值扫描 全量 LODO
```

---

## 术语表（method7 增量）

| 术语 | 含义 |
|---|---|
| 双信号解耦 | 避险用 conformal、获利用 saturation，两个子问题两个信号，不合并成万能 confidence |
| gain 不可学 | 逐-cell 净增益 profit-AUC≈0.47≈瞎猜，跨饱和/非饱和、分类/预测一致（F-R12.1/12.2） |
| 可达天花板 ≠ 可预测 | oracle 头寸真实(5.95pp)但部署特征选不对 winner → +0.00pp 是正确弃权非 artifact |
| 激进甜区 | 丢避险门、只在高头寸格主动偏离 → +0.15pp；过度激进即崩（对阈值敏感）|
| trust 迁移是特征级 | conformal 避险力强依赖富 epistemic 特征；轻部署特征下同域弱、跨域≈瞎猜（F-R12.8）|
| selective-action regret bound | 超额受 avoid-harm AUC 控；获利项受 seek-gain AUC(≈½)控 → commit-dominant 最优（§5）|

---

**End of method7.md** — Round 12 设计与实测已完整收口；后续若开"在线反馈/异构资产池"新方向另起 method8。
