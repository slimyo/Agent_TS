# test/ 工业 Agent 系统 —— 设计思想与框架

> 版本：2026-06-07。本文件**事实记录** test/ 从"研究结论系统"升级为"工业收益系统"的设计思想、框架与依据。
> 上游：`research/method7.md`（机制设计）/ `research/feedbackm7.md`（工业评审）/ `research/finish7.md`（F-R12.x 实测）。
> 实现：`signal_router.py`（离线双信号 + 激进档）/ `online_router.py`（在线 bandit）/ `run_full_test_v2.py` / `run_online_test.py`。

---

## 1. 问题定义：强 base 旁的选择性行动

生产环境已有强 base（分类=Rocket / 预测=Chronos-2 / 检测=Rocket）。给定一个查询 cell（或一段部署），
agent 要回答"**该守 base，还是偏离到某候选模型**"。三大任务（forecasting / classification / detection）同构。

**价值三层（对标 feedbackm7，工业付费点在 L2/L3）**：

| 层 | 问题 | 机制 | 现状 |
|---|---|---|---|
| L1 | 避免犯错（Don't do stupid things）| conformal trust = 风险闸/OOD 检测 | ★★★★★ 成熟 |
| L2 | 自动选更优方案 | 在线 bandit 学该域 winner | ★★★★☆ 在线达成 +2.83pp |
| L3 | 持续创造增量收益 | observe→act→reward→update 闭环 | ★★★★☆ 学习曲线单调升 |

---

## 2. 思想主线：从"何时该守"到"何时值得动"

```
method4-7（离线、避险为主）             本系统（在线、获利闭环）
  belief 选谁 → trust 避险 → 守 base      choose → observe reward → update posterior
  「诚实判断 base 何时已足够」              「在反馈中学会该域该用谁」
  结论：离线最优 = 不动 (+0.00pp)          结果：反馈闭环 → 收敛 +2.83pp
```

**核心认知链（为什么这样设计）**：
1. **trust ≠ gain**（F-R11.7）：避险是风险估计，获利是效用估计，语义正交。
2. **避险↔conformal / 获利↔saturation**（F-R12.5）：两个子问题用两个不同信号，融合会稀释。
3. **gain 逐-cell 离线不可学**（F-R12.1，profit-AUC 0.47）：静态部署特征预测不了 winner。
4. **但 winner per-domain 可学**（本系统实测）：同域反馈累积下，per-dataset 最优模型一致(0.78)、+4.20pp 真实存在。
   → feedbackm7 的纠正成立：**"离线特征不可学" ≠ "本质不可学"**，瓶颈是**没有反馈**，不是 gain 不存在。

---

## 3. 系统框架（两层决策 + 四变量）

```
                 ┌────────────────────────── 每次部署 ──────────────────────────┐
  cell/episode → │  task signature z                                            │
                 │     │                                                         │
                 │     ├─ saturation ĝ(z)  : 有没有头寸（该不该进场）            │  ← 离线信号
                 │     ├─ belief/proposal   : 提哪个候选（capability 先验暖启动）│  ← 离线先验
                 │     ├─ trust(conformal)  : 偏离安不安全（L1 风险闸）          │  ← 离线/在线
                 │     └─ gain posterior    : 该域该模型的累积 reward（在线学）  │  ← 在线反馈★
                 │            │                                                  │
                 │   Thompson 采样(候选 shortlist=先验 top-k) + L1 护栏 → action │
                 │            │                                                  │
                 │   deploy → observe realized reward(acc−base) → update posterior（按 domain 持久化）
                 └──────────────────────────────────────────────────────────────┘
```

**四变量**（method7 框架，本系统继承并加在线维度）：
- `saturation`：预测 oracle−base 头寸，判"该不该进场"。
- `belief/proposal`：提偏离候选；用 **capability profile** 暖启动（模型在其它域的平均 reward，LODO）。
- `trust`：conformal nonconformity，L1 避险闸——不确定就守 base。
- `gain`：**唯一靠在线反馈学**的维度——per-(domain,model) reward 后验，Thompson 采样。

---

## 4. 三大工业缺陷 → 对策（feedbackm7）

| 缺陷 | feedbackm7 诊断 | 本系统对策 | 实现 |
|---|---|---|---|
| ① 完全离线无反馈 | winner 离线预测失败 ⇒ 误判"本质不可学" | **Contextual Bandit 反馈闭环**：observe→update | `online_router.py` Thompson + `update()` |
| ② 无状态/episode | LODO cell-level 把时序连续性抹掉 | **后验按 domain 持久化**；每域当部署 episode 重放 | `run_online_test.py` `_stream` + epochs |
| ③ 资产池同质 | 22 模型多但同范式，winner 被噪声淹没 | **capability profile 先验**做 task↔model 匹配 + 候选 shortlist | `build_capability_prior` + `act(topk)` |

**L1 始终保留**：conformal/安全边际护栏——明显负收益候选退场，不确定守 base，避免激进探索灾难。

---

## 5. 关键实证（支撑设计的事实，全量 22 模型库 / 38 域）

| 配置 | 冷启 vs_base | 收敛 vs_base | 结论 |
|---|---|---|---|
| 离线保守双信号门（method7 终态）| +0.00 | +0.00 | L1 成熟、L2/L3 缺位（"最优=不动"）|
| **在线 full（先验+反馈）** | −0.01 | **+2.83pp** | 反馈闭环创造收益（L2/L3）|
| 消融：无反馈（≈离线）| −0.45 | −0.34 | **反馈是因果杠杆**（去掉就回负）|
| 消融：无 capability 先验 | −8.02 | +1.49 | 先验保冷启动 + 加速 |
| oracle-per-domain / per-cell | — | +4.34 / +5.95 | 现实/理论上界 |

**学习曲线**（每域部署 20 轮）：`−0.1 → 1.6 → 2.0 → … → 2.9`，随反馈单调上升 = L3 持续增量。
**离线激进甜区** +0.15pp 对阈值敏感（feedbackm7 缺陷3 担忧成立）→ 不作部署收益源，仅研究发现。

### 5.1 抗漂移（解 feedbackm7 最大部署担忧"实验有效上线失效"）

`online_router` 加**遗忘因子 γ**（折扣后验，旧反馈指数衰减）。A→B 域切换（winner 改变）后 phase-2 恢复力（19 组配对）：

| 策略 | 漂移后 vs_base | 解读 |
|---|---|---|
| offline-static（调好即冻结）| **−1.13pp** | 离线调好就上线，漂移即失效（feedbackm7 担忧坐实）|
| online γ=1（无遗忘）| −0.08pp | domain 状态延续但被旧反馈拖住、翻转慢 |
| **online γ=0.6（折扣抗漂移）** | **+0.61pp** | 旧反馈衰减 → 漂移后快速翻到新 winner，恢复力最强 |

→ **折扣后验把"漂移即失效"变成"漂移可恢复"**；γ 是抗漂移旋钮（生产按漂移速率调）。这是工业可部署的关键补强。

### 5.2 在线收益的边界：取决于"域级结构稳定性"（诚实结论）

把在线 bandit 也接到**预测域**（未饱和，oracle 头寸 +23.9%relMAE，本该收益最大）：

| 任务 | 域级 winner 一致性 | 在线 full 收敛 | 无反馈消融 |
|---|---|---|---|
| 分类（38 域）| **0.78**（稳定）| **+2.83pp** | −0.34pp |
| 预测（6 域 few-shot）| **~0.46**（≈掷硬币，3/6 域级 gain 还为负）| +0.07%relMAE | −1.46%relMAE |

**关键洞察**：在线反馈的收益 **∝ 域级结构稳定性**——
- 分类：per-domain winner 稳定（之前"不可预测"确是 LODO 把 state 抹掉了，feedbackm7 缺陷2 成立）→ 反馈大幅变现。
- few-shot 预测：winner 连**域级**都不稳定（换 N/seed 就翻、半数域级 gain 为负）→ 反馈只把 −1.46 拉到 +0.07（仍 >无反馈，但绝对低）。

→ 诚实修正 feedbackm7 的乐观："反馈能救 gain"**成立但有条件**：需要**episode/域级存在稳定结构**。
分类满足、few-shot 预测不满足。工业落点：**先上有域级稳定专长的场景**（设备/电网/同源数据流），
对高方差 few-shot 预测，靠加长 episode、真异构专家（缺陷3）或更粗的 task-signature 聚合才有望变现。

### 5.3 异构专家池端到端（method8 落地，解缺陷3 + 真正变现 L2/L3）

把 method8 的**结构异构专家池**接进 4-agent 闭环、**真实执行模型**（`test/experts.py` + `pipelines.run_hetero_online_stream` + `run_hetero_test.py`）：
regime 多样部署流 → CuratorAgent 画像→task signature→regime tag → 在线 bandit（capability-affinity 先验）选专家 → 真跑专家→MAE→reward→更新（折扣后验，按 regime 分组）。

**12 异构专家**（trend/damped_trend/momentum/seasonal/harmonic/holt_winters/ar/mean_revert/robust/changepoint/spike/base）× 9 regime：

| 策略 | vs_base | 偏离率 | 说明 |
|---|---|---|---|
| **learned-cap（学出 capability profile，先验+反馈）** | **+13.74%relMAE** | 0.45 | 数据驱动先验，**可扩展到任意池** |
| learned-cap（只学出先验不更新）| **+14.08%relMAE** | 0.45 | 学出的能力画像本身就稳又强 |
| online+手工 affinity（先验+反馈）| +13.55%relMAE | 0.65 | **反馈救回坏先验**（手工先验差但反馈纠偏）|
| static 手工 affinity（不学）| **−24.94%relMAE** | 1.00 | **手工 signature 难随池扩展 → 失配净亏** |
| online-no-prior（只反馈）| +3.10%relMAE | 0.39 | 无先验、12 专家冷探索慢 |
| oracle 上界 | +46.78%relMAE | — | 每条选真最优 |

**扩库（6→12 专家）学到的关键经验**：
1. **手工 capability signature 不随池扩展**：6 专家时手工 affinity +19%，12 专家时 −24.94%（手调 12×7 必失配）。
2. **学出的 capability profile 才是正解**（feedbackm7 路线2 的正确实现）：在校准历史上记录每 (regime,expert) 平均 reward → 稳定 +14%、可扩展到任意池大小。
3. **反馈对坏先验有纠偏力**：手工先验 static −24.94% → 加反馈 +13.55%（在线更新把错先验拉回正）。
4. **杠杆随池结构转移**：同质池靠反馈（+2.83pp）；异构池靠 **capability 结构匹配**（学出 +14%），反馈是纠偏/兜底。
→ 印证 feedbackm7 缺陷3：换结构异构专家 + **学出的**能力画像，选择性行动从 +0.00 真正变现；L1 base 锚点仍兜底。

### 5.4 真实数据验证（关键边界）—— 异构必须"结构互补"非"架构不同"

把 learned-capability 接到**真实预测库**（`run_realcap_test.py`：14 真实 TSFM × 6 真实域 ETT/ECL/Weather/…，
每 cell 重建真实训练序列→regime，LODO by dataset 无泄漏）：

| 策略 | vs_base | 对比合成结构专家 |
|---|---|---|
| learned-cap static（只先验）| **−6.61%relMAE** | (合成 +14.08%) |
| online（学出先验+反馈）| −2.92%relMAE | (合成 +13.74%) |
| oracle-per-cell | +23.93%relMAE | — |

regime 分布（真实少样本）：shift 41 / noise 16 / ac1 14 / **seasonal 仅 1**。

**决定性 2×2（E15，`run_realstruct_test.py`）—— 重大诚实修正**：把**结构专家也跑到真实数据**，隔离"池 vs 数据"：

| | 结构专家 | 通用 TSFM |
|---|---|---|
| 合成 regime 数据 | **+14.08%** | — |
| **真实数据** | **−6.37%** | −6.6% |

→ **瓶颈不是池结构**：结构专家在真数据上也净亏（−6.37%，尽管 oracle 头寸 38.66%）。真实少样本序列被 regime 检测塌成 ac1/shift/noise（无 trend/seasonal），tag 内 winner 不稳定 → 路由失败。
**最硬的一课**：method8 的 +14% 是**合成 regime 数据的属性**（regime 按构造可观测+专家对齐），**不迁移到真数据**；真实 TS 的 winner 从 task-signature 不可预测（回到 F-R12.1，换结构专家也不例外）。
→ **真数据上唯一变现的是 per-domain 在线反馈**（分类 +2.83pp，domain=稳定单元）；capability-routing 在真预测域不成立。
**工业杠杆 = 存在稳定单元(域/episode) + 在线反馈，不是"换异构专家池"。**

---

## 6. 与 method7 的关系 & 诚实边界

- **继承**：四变量框架、conformal 避险（L1，F-R11.9）、双信号解耦（离线档仍在 `signal_router.py`）。
- **新增**：在线反馈（gain 后验）、episode 状态、capability 先验——把 L2/L3 从 0 做出来。
- **诚实边界**：
  - capability 先验只用其它域（无泄漏）；在线更新用本域**部署后真实反馈**（= 工业合法反馈）。
  - 当前 episode 用"同域 cells 多轮重放"模拟持续部署——演示反馈机制；真实部署 episode 更长、漂移更复杂。
  - 资产池仍偏同质（缺陷3 只部分解）；引入真异构专家（physics/transformer/expert-system）后 winner 更可预测、收益更高——留 method8。
  - 漂移自适应（distribution drift）：当前后验无遗忘；生产需加滑动窗/衰减——`online_router` 预留 `obs_noise/prior_strength` 调参位。

---

## 7. 运行

```bash
python -m test.run_full_test_v2   # 离线：双信号解耦 + 激进档 + 阈值扫描（L1 + 激进甜区）
python -m test.run_online_test    # 在线：Contextual Bandit 反馈闭环（L2/L3）+ 消融 + 学习曲线
```

## 8. 文件
- `signal_router.py` — 离线双信号解耦决策 + 激进 policy 拨盘 + oracle-capture 指标（L1 + 研究档）
- `online_router.py` — **Trust-Aware Online Router**：Thompson bandit + capability 先验 + episode 后验 + L1 护栏（L2/L3）
- `run_full_test_v2.py` / `run_online_test.py` — 离线 / 在线 全量 LODO 评测
- `agents.py` / `pipelines.py` — 4-agent 编排（Curator/SaturationPlanner/RouterAgent/Reporter）+ 三任务端到端
- 结果：`results_signal_router.jsonl`（离线）/ `results_online.jsonl`（在线，含学习曲线）

> **下一步（method8 方向，feedbackm7 路线1 最推荐）**：把在线 bandit 接真异构专家池 + 漂移自适应 +
> 真实部署反馈流，从"模拟重放"升级为"生产闭环"，把 +2.83pp 推向 per-domain 上界并验证抗漂移。
