# 工业时序自适应路由 Agent 系统（Multi-Agent，三大任务）

> 蒸馏自 `research/method1-7` + `finish1-7` 全部经验教训，覆盖**预测 / 分类 / 检测**三大任务。
>
> ## 🎯 系统目标（2026-06-07 升级：研究结论系统 → 工业收益系统）
> 在已有强 base 模型的生产环境里，做一个**风险可控、能持续创造增量收益**的模型路由 agent，
> 把三大时序任务（forecasting / classification / fault-detection）的模型选择**自动化 + 在线进化**。
> 价值分三层（对标 feedbackm7）：
> - **L1 避免犯错（已成熟）**：conformal trust = OOD/风险闸——不确定就守 base，永不显著变差（"Don't do stupid things"）。
> - **L2 自动选更优（在线达成）**：带反馈的 Contextual Bandit 在部署中学到"该域该用谁"，收益从 +0.00pp→收敛 **+2.83pp**。
> - **L3 持续创造增量（闭环达成）**：observe→act→reward→update 持续闭环，随部署累积反馈而升（学习曲线见 §0.2）。
>
> 设计哲学的演进：**method7「诚实判断何时该守 base」（离线、避险）→ 本系统「在线反馈中学会何时值得偏离」（闭环、获利）**。
> 关键认知（feedbackm7 纠正）：**"gain 离线特征不可学" ≠ "本质不可学"**——加上反馈闭环，收益就回来了。
> 完整设计思想与框架见 **`test/DESIGN.md`**。

---

## 0. v2 决策升级：双信号解耦（research Round 12 / F-R12.5）

> 最新结论：决策不能用单一 confidence/trust。**"避险"和"获利"是两个需不同信号的子问题**：
> - **避险**（偏离会不会变差）→ **conformal 可信度**（AUC 0.79）
> - **获利**（偏离能不能净赚）→ **saturation-headroom**（AUC 0.71）
> 决策律：**deviate iff conformal_safe AND saturation_headroom**（`test/signal_router.py`）。

**全量测试（扩展库/数据：38 数据集 × 10 分类器全量库，full LODO）**：
模型库与数据集均扩到全量——UCR 24（含新增 FordA/FordB）+ **UEA 14 多变量**（库 3→10，channel-flatten 适配 7 个单变量分类器）= **221 cells / 38 datasets**。

| 决策 | system | base | vs base | 偏离 |
|---|---|---|---|---|
| **双信号门 (trust≥.5 AND head≥.5)** | **80.00%** | **80.00%** | **+0.00pp** | 0/221 |
| ├ UCR(24, 单变量) | 84.35% | 84.35% | +0.00pp | 0 |
| └ UEA(14, 多变量) | 72.49% | 72.49% | +0.00pp | 0 |
| 自由偏离 (head≥0) | 79.71% | 80.00% | −0.30pp | 221 |
| oracle 上界 | 84.46% | — | +4.46pp | — |

**读法**：① 保守双信号门在全量 22 模型库上仍收敛到**精确 base（+0.00pp，0 偏离）**——oracle 头寸真实(5.95pp)、但逐-cell winner 不可预测，守 base 是诚实弃权；② oracle 上界 5.95pp 逐-cell 不可达。

### 0.1 激进策略（`policy=` 参数，不追求 pp）

> 按 method 结论：oracle 头寸真实(5.95pp)、winner 不可预测 → 保守门收敛守 base。
> 若**不追求 pp、要主动出手抓头寸**，`run_signal_router(cells, policy=...)` 提供激进档：

| 策略 | vs_base | 偏离率 | oracle命中 | 安全偏离率 |
|---|---|---|---|---|
| conservative（双信号门）| +0.00pp | 0% | — | — |
| **aggressive 头寸≥0.3 ★** | **+0.15pp** | 7% | 0.125 | 0.69 |
| aggressive 头寸≥0.15 | −0.00pp | 42% | 0.106 | 0.61 |
| aggressive-belief（始终偏离）| −2.46pp | 100% | 0.068 | 0.48 |
| oracle 上界 | +5.95pp | — | — | — |

**激进甜区**：丢掉 conformal 避险门、只在**高头寸(≥0.3)格主动出手** → 净 **+0.15pp**（保守门拿不到的真实正收益），偏离 7%、安全率 0.69。但更激进就崩（≥0.15 washes out，始终偏离 −2.46pp）——winner 不可预测使激进收益薄且对阈值敏感。
**"不追求 pp"的真义**：用激进换 oracle 头寸的**主动覆盖**（命中率/捕获率），pp 是副产物。
运行：`python -m test.run_full_test_v2`（含主决策 + 激进策略对比 + 阈值扫描）。

> 注：离线激进甜区(+0.15pp)对阈值敏感、是"研究发现"非"可部署收益源"（feedbackm7 缺陷3 担忧成立）。
> 真正的收益来源是下面的**在线反馈**——这才是工业可部署的 L2/L3。

### 0.2 在线反馈路由（L2/L3 工业升级，解 feedbackm7 三大缺陷）

> feedbackm7 三大工业缺陷：① 完全离线无反馈 ② 无状态/episode ③ 资产池同质。
> 对策 = `test/online_router.py` 的 **Trust-Aware Online Router（Contextual Bandit）**：
> `choose model → observe reward → update posterior` 持续闭环；后验按 **domain 持久化**（episode 状态）；
> 每模型后验用 **capability profile 先验**（该模型在其它域的平均 reward，LODO 无泄漏）暖启动；L1 conformal 护栏保留。

**关键证据（全量 22 模型库 / 38 域 / 每域部署 20 轮，5 seed 均值）**：

| 策略 | 冷启 vs_base | 收敛 vs_base | 说明 |
|---|---|---|---|
| offline 保守双信号门（method7 终态）| +0.00pp | +0.00pp | L1 避险，"最优=不动" |
| **online full（先验+反馈）** | −0.01pp | **+2.83pp** | 反馈闭环学到该域 winner（安全偏离 0.77）|
| 消融：无 capability 先验 | −8.02pp | +1.49pp | 先验保护冷启动 + 加速收敛 |
| 消融：无反馈（≈离线）| −0.45pp | −0.34pp | **没有反馈就回到负收益 → 反馈是因果杠杆** |
| oracle-per-domain（域级现实上界）| — | ≈+4.34pp | / oracle-per-cell 上界 +5.95pp |

**学习曲线 vs_base/epoch**：`−0.1 → 1.6 → 2.0 → … → 2.9`（20 轮，随反馈累积单调上升）。

**结论**：① 反馈把离线 +0.00pp（最优=不动）提到收敛 **+2.83pp**——直接验证 feedbackm7："gain 离线不可学 ≠ 本质不可学"；
② **winner 在 per-cell 静态特征上不可预测，但在 per-domain 反馈上可学**（缺陷2：之前是 state 被 LODO 抹掉了，非本质不可预测）；
③ 无反馈消融停在负收益 → 价值确由反馈闭环产生（L2/L3），非先验；④ L1 避险护栏在线仍在（安全偏离 0.77）。

**抗漂移（解 feedbackm7 最大担忧"实验有效上线失效"）**：`online_router` 加遗忘因子 γ。A→B 域切换后 phase-2 vs_base：
offline-static（调好即冻结）**−1.13pp**（漂移即失效）/ online γ=1 −0.08pp（翻转慢）/ **online γ=0.6 +0.61pp**（折扣后验快速恢复）。
→ 折扣后验把"漂移即失效"变成"漂移可恢复"，γ 是抗漂移旋钮。

**在线收益的边界（诚实）**：反馈价值 **∝ 域级结构稳定性**。分类 per-domain winner 一致性 0.78 → 在线 +2.83pp；
few-shot 预测一致性仅 ~0.46（换 N/seed 就翻、半数域级 gain 为负）→ 在线仅 +0.07%relMAE（仍 >无反馈 −1.46，但绝对低）。
→ 工业落点：**先上有域级稳定专长的场景**（设备/电网/同源流）；高方差 few-shot 预测需加长 episode/真异构专家才变现。
运行：`python -m test.run_online_test`（在线收益 + 消融 + 学习曲线 + 抗漂移 + 预测域）。

### 0.3 异构专家池端到端（method8 落地，真正变现 L2/L3）

> 解 feedbackm7 缺陷3（资产池同质）：把 **12 个结构异构专家**（trend/damped_trend/momentum/seasonal/harmonic/
> holt_winters/ar/mean_revert/robust/changepoint/spike+base，`test/experts.py`）接进 4-agent 闭环、**真实执行模型**
> ——CuratorAgent 画像→capability 路由（**学出的能力画像**）→真跑专家→反馈更新。9 regime 部署流。

| 策略 | vs_base | 说明 |
|---|---|---|
| **learned-cap（学出先验+反馈）** | **+13.74%relMAE** | 数据驱动，可扩展任意池 |
| learned-cap（只学出先验）| **+14.08%relMAE** | 学出的能力画像本身就稳 |
| online+手工 affinity | +13.55%relMAE | 反馈救回坏先验 |
| static 手工 affinity（不学）| **−24.94%relMAE** | 手工 signature 难随池扩展→失配 |
| online-no-prior（只反馈）| +3.10%relMAE | 12 专家冷探索慢 |
| oracle 上界 | +46.78%relMAE | — |

**扩库经验**：① 手工 capability signature **不随池扩展**（6 专家 +19%→12 专家 −25%）；② **学出的 capability profile 才是正解**
（feedbackm7 路线2 正确实现，稳定 +14%、任意池可扩）；③ 反馈对坏先验有纠偏力（−25%→+13.55%）；④ 杠杆随池结构转移：同质池靠反馈，异构池靠结构匹配。
→ method7 的 +0.00 是**同质池局部结论**；换异构专家 + 学出能力画像，选择性行动真正变现（L2/L3），L1 base 兜底。
运行：`python -m test.run_hetero_test`。

---

## 1. 架构：4 个 Agent 闭环

```
  序列/cell ─► CuratorAgent ──► SaturationPlanner ──► RouterAgent ──► 执行 ─► ReporterAgent
              (感知/画像)       (决策:该不该route)     (选模型:CV+gate)         (可解释trace)
```

| Agent | 职责 | 关键方法教训 |
|---|---|---|
| **CuratorAgent** | 诊断序列结构：趋势/季节/平稳/噪声/复杂度（多变量取通道均值）| method1/2 Curator |
| **SaturationPlanner** | 先判**饱和度**（base 是否已近 oracle）→ abstain/route；再判 **N-fallback**（极少样本 CV 不可信）| method5 F-R9.7/10.2 + method3 v10/B7v2 |
| **RouterAgent** | CV(训练集,无泄漏) + **saturation-自适应 margin** gate；越饱和偏离门槛越高 | method3 margin gate + F-R10.2 损害控制 |
| **ReporterAgent** | 每步决策汇成自然语言 + 结构化 trace | method3 M8 attribution 精神 |

**智能决策的核心**（区别于"无脑选 CV 最优"）：
- **Saturation-aware**：分类/检测（base 71-75% 已是 oracle）默认 **abstain**；预测（base 仅 25% 是 oracle）默认 **route**。
- **N-conditional fallback**：N/class < 7 时 CV 噪声大 → 强制 default（实证 +0.87pp，避免 BeetleFly N=3 式 −25pp 灾难）。
- **Honest**：决策只用部署可得信息（训练集 CV / walk-forward），**绝不读 test**（method3 M9 去泄漏教训）。
- **永不显著伤害 base**：偏离需超 saturation-自适应 margin；强 base 时只偏向强候选（finish §4 short-val overfitting 教训）。

---

## 2. 全量测试结果（本地 `tsci` env，UCR/UEA/ETT/合成故障）

| 任务 | cells | System | Base | System−Base | 偏离率 | 说明 |
|---|---|---|---|---|---|---|
| **分类** (8 UCR ds) | 48 | **90.02%** | 90.02% | **+0.00pp** | 0/48 | **trust-gate 升级后**：CV-饱和惩罚拦掉 2 次误偏离，达成精确 base 持平 |
| **检测** (合成 4-fault, ETTh1/ECL) | 12 | **50.62%** | 50.62% | +0.00pp | 0/12 | 全 fallback/abstain（4-class fault 本身难，base 已最优）|
| **预测** (6 ds × 3N × 2seed) | 36 | MAE 持平 | — | **−1.6% rel** | 4/36 (11%) | 3/6 数据集精确持平；route 模式只偏向强 TSFM |

> **trust-gate 升级（2026-06-02，method6 F-R11.5/F-R8.8）**：RouterAgent 增加可信度门——偏离不仅要过 CV margin，
> 还要过 trust=（CV 饱和惩罚 × fold 稳定性）。直击 few-shot CV↔test 背离：moment_1nn 在 BirdChicken N=10
> 上 CV=1.000（饱和）但 test 仅 0.80，trust 压到 0.00 → 拦截 → 守 Rocket 0.90。**分类从 −0.42pp 升到 +0.00pp，
> 偏离从 2 次→0 次，无任何 cell 受损**。检测/预测未变（本就 0/4 安全偏离）。

**可解释性示例**（ReporterAgent 自然语言 trace）：
```
[classification] 画像: N=20 2类×10/类 L=512 season=0.99 noise=0.05
  决策: abstain (sat=0.71) — base 大概率已近 oracle，仅 CV 强烈支持才偏离（F-R9.7/10.2）
  选模型: moment_1nn (偏离) — CV-winner 超 default +0.151≥margin0.15

[detection] 画像: N=12 4类×3/类 | extreme few-shot
  决策: fallback (sat=0.75) — N/class=3<7: CV 不可信，强制 default 'rocket'（B7v2 N-fallback）

[forecasting] 画像: N=50 trend=0.65 season=0.36
  决策: route (sat=0.25) — 有 routing 头寸，启用 CV 路由（未饱和，F-R10.1）
  选模型: chronos_bolt (偏离) — CV-winner 超 default +2.127≥margin0.43
```

---

## 3. 结果解读（诚实定位）

本系统**在每个任务上都做到"近-base 且永不显著伤害"**，这正是 Round 9-10 实证钉死的**强 base 时代最优行为**：
- 分类/检测**饱和**（Rocket≈oracle）→ 系统正确 abstain，−0.42pp/+0.00pp ≈ 持平，**避免了 always-route 的 −9.4pp 灾难**。
- 预测**未饱和**但 walk-forward CV 在小验证窗噪声大 → 系统用强 margin + 只偏向强 TSFM，把 naive 版的 −27% 收到 **−1.6%**。
- **价值不在 +Xpp 提分**（标准 benchmark 已被 base 饱和），而在**自适应地知道何时该动手、何时该退**——
  且全程**可解释**（每决策附 sat 分数 + CV + 偏离理由）+ **无数据泄漏**（LODO-grade 诚实）。

> 这与论文统一论点一致：**selective abstain > aggressive routing**（finish5 F-R10.2）。
> 系统是一个**诚实的自适应运行时**，不是一个"刷分 router"。

---

## 4. 运行

```bash
# 单任务
python -m test.run_full_test --task classification
python -m test.run_full_test --task detection
python -m test.run_full_test --task forecasting
# 全部
python -m test.run_full_test --task all
# 子集
python -m test.run_full_test --task classification --datasets Coffee,ECG200
```
输出：`test/results_<task>.jsonl`（每 cell 含 profile / plan_mode / sat_score / chosen_model / cv_scores / trace / nl_explanation）。

## 5. 文件
- `agents.py` — 4 个 agent（Curator/SaturationPlanner/RouterAgent/Reporter）+ 数据结构
- `pipelines.py` — 三任务端到端编排（调 research 模型库，CV 无泄漏）
- `run_full_test.py` — 全量测试 + 对照 base + 聚合摘要
- `results_*.jsonl` — 逐 cell 结果 + 可解释 trace

## 6. 复用的 research 模型库
- 预测：`research/agent/forecaster_reflect.STRATEGY_FN`（chronos2 / chronos_bolt / arima_ets / naive_* …）
- 分类/检测：`research/agent/clf_strategies`（rocket / moment_1nn/lr / dtw_1nn / euclid_1nn …）+ `clf_planner` CV
- 数据：`research/utils/{ucr_loader, uea_loader, data_loader, splitter, inject_fault}`
