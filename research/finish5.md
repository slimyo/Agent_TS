# Finish v5 — Round 10 实测（Saturation Detection）

> 版本：2026-06-01 起
> 方法写在 `method5.md`；外部 review 是 `feedback_m4.md`。本文件承接 Round 10（M12）实测 + Findings F-R10.x。
> 前置：`finish4.md`（Round 9 / learned belief 全证伪）。

---

## 0. 本轮目标

feedback_m4 三位 reviewer 一致：**停止造 router，转 Saturation Detection**（判断某域 base 是否已近 oracle →
该不该 route）。method5 把研究对象从"选哪个模型"改为"**该不该选**"。M12 在**三任务**（预测/分类/检测）
上做 LODO 实验，回答：**saturation 是否可检测？检测器能否跨任务正确决定 route vs abstain？**

环境：本地 `tsci`（纯 CPU 聚合 + 小 RF 回归，无 GPU/LLM）。数据 = 三任务既有 per-cell per-method 结果。

> ⚠️ 本轮**诚实结果**：saturation 是**部分可检测**的信号（gap 回归在预测上 corr 0.37），且 gate 提供
> **单调的"损害控制"**；但**没有任何 deployable 配置在任一任务上正向击败 base**——这反过来**强化** Round 9：
> 瓶颈是 routing 执行（per-cell winner 预测），不是检测。诚实 router 的最优仍≈abstain。

---

## 1. 三任务饱和度（M12 实测底座）

| 任务 | base | base=oracle 占比 | 平均 oracle gap | 饱和? |
|---|---|---|---|---|
| **Forecasting**（6 ds / 72 cell）| Chronos-2 | **25%** | 大（相对 MAE ~34%）| ❌ 未饱和 |
| **Classification**（UCR 22 ds / 128 cell）| Rocket | **71%** | 1.88pp | ✅ 饱和 |
| **Detection**（synth 4-class, 2 ds / 12 cell）| Rocket | **75%** | 1.35pp | ✅ 饱和 |

> 三任务饱和度差异显著 → 这是检验"Saturation Detection 跨任务"的理想 testbed。
> **注**：forecasting 的 gap/regret 原始数值很大（ILI/ECL 的 MAE 量纲大），下文统一用 **vs_base**
> （selected − base 的平均，越正越好）+ route_rate + safe_dev_rate 三个可解释量，不报原始 MAE regret。

---

## 2. M12 · Saturation gap 回归（LODO）

`ĝ_φ(z)` = RandomForest 回归 oracle gap，z=[N, n_methods, base_perf, log N]（全 deployment 可得，不读 test）。
留一数据集训练、留出预测，与真实 gap 比相关：

| 任务 | gap 预测相关 corr | 解读 |
|---|---|---|
| Forecasting | **0.371** | gap 有信号、可学（未饱和域更可学）|
| Classification | 0.158 | 弱信号（饱和域 gap 本就接近 0，难学）|
| Detection | −0.113 | 学不出（仅 2 数据集，LODO 退化）|

**F-R10.1**：oracle gap 在有头寸的域（forecasting corr 0.37）**确实是可学信号**；在饱和域（gap≈0）
信号本就微弱，检测器无从学起——这本身印证"饱和域没什么可检测/可路由"。

---

## 3. M12 · Selective routing（deployable，无 test-peek）

route 分支用 **deployable 备选**：held-out 数据集 → 训练集上均值最优的非 base 方法（LODO 转移，不看 test）。
对照 always-route / always-abstain(=base) / saturation-gated（τ 升序）。**vs_base 越正越好**。

### 3.1 Forecasting（未饱和）

| 策略 | vs_base | route_rate | safe_dev_rate |
|---|---|---|---|
| always-route | **−0.21** | 1.00 | 0.86 |
| gated τ low | −0.21 | 1.00 | 0.86 |
| gated τ mid | −0.08 | 0.82 | 0.93 |
| gated τ high | **+0.00** | 0.28–0.32 | **1.00** |
| always-abstain(=base) | 0.00 | 0.00 | — |

### 3.2 Classification（饱和）

| 策略 | vs_base | route_rate | safe_dev |
|---|---|---|---|
| always-route | **−0.094** | 1.00 | 0.18 |
| gated τ low→high | −0.062 → **−0.030** | 0.78 → 0.48 | 0.21 → 0.29 |
| always-abstain(=base) | 0.00 | 0.00 | — |

### 3.3 Detection（饱和，小样本）

| 策略 | vs_base | route_rate |
|---|---|---|
| always-route | **−0.045** | 1.00 |
| gated τ high | **−0.002** | 0.08 |
| always-abstain(=base) | 0.00 | 0.00 |

### 3.4 关键观察

- **F-R10.2（gate = 单调损害控制）**：三任务**一致**——always-route 在每个任务都**负**（fc −0.21 /
  clf −0.094 / det −0.045，因 deployable 单一备选无法 per-cell 选对），而 saturation gate 随 τ 升
  **单调把 vs_base 拉回 0**（=abstain to base）。gate 学到"预测 gap 小就别 route"，**在饱和域几乎全 abstain、
  在未饱和域保留高 route_rate 且 safe_dev 升到 1.0**。这正是 method5 要的行为：**永不显著伤害 base**。
- **F-R10.3（仍无正向 deployable 增益，强化 Round 9）**：没有任何配置在任一任务 vs_base **>0**。
  即便 forecasting **有** 34% oracle 头寸，**deployable 单一备选也吃不到**（哪个备选赢逐 cell 变化，
  与 F-R9.x"per-cell winner 不可学"同源）。→ **瓶颈被再次钉死在 routing 执行，不是 saturation 检测**：
  检测器能正确说"这里有/没有头寸"，但**把头寸变现仍需 per-cell winner 预测，而那个在小样本下学不出**。
- **safe_dev 的诚实信号**：forecasting gate 在高 τ 下 safe_dev=1.0（routing 的那部分确实安全），
  分类/检测 safe_dev 始终 ≈0.2–0.4（饱和域偏离基本是赌博）——**检测器正确区分了两类域的偏离质量**。

---

## 4. 对 feedback_m4 的回应与评价（实测后）

| reviewer 建议 | 实测结论 |
|---|---|
| 转 Saturation Detection（方向1）| ✅ **部分成立**：gap 在未饱和域可学（corr 0.37），gate 跨任务正确分流（饱和域 abstain / 未饱和域保留 route + safe_dev↑）|
| 别再 calibration（belief inversion）| ✅ 采纳，本轮未碰 belief 校准 |
| benchmark 是瓶颈 | ⚠️ **修正**：benchmark 饱和是分类/检测的瓶颈；但 forecasting 未饱和却**仍无 deployable 正增益** → 瓶颈更准确是 **routing 执行（per-cell winner 不可学）**，跨"饱和/未饱和"都成立 |
| 新 metric（regret / safe-dev / abstain）| ✅ 采纳，本轮主用 vs_base + route_rate + safe_dev_rate |
| 价值主张转"selective abstain"| ✅ **实测支持**：gate 的唯一稳健价值是"永不伤害 base"（损害控制），而非提分 |

**我的评价**：feedback_m4 方向正确（saturation 确实可检测、abstain 确实是饱和域最优），但 reviewer
把希望寄托于"离开饱和 benchmark 就能 route 成功"——**M12 forecasting 实测证伪了这个乐观**：未饱和域
**有头寸也变现不了**（deployable 备选吃不到 per-cell 异质性）。这把论文论点从"benchmark 饱和"收紧为
更强的 **"deployable routing 执行壁垒跨饱和/未饱和域普遍存在"**。

---

## 5. Findings F-R10.x

| ID | 内容 | 来源 |
|---|---|---|
| **F-R10.1** | **Saturation 是部分可检测信号**：oracle-gap LODO 回归在 forecasting corr=**0.37**（有头寸的域可学）、分类 0.16、检测 −0.11（饱和域 gap≈0 + 小样本，学不出）。检测器能学的程度 ∝ 域的真实头寸——饱和域"无可检测"本身即印证饱和 | §2 |
| **F-R10.2** | **Saturation gate = 单调损害控制（跨任务一致）**：always-route 在三任务全负（fc −0.21 / clf −0.094 / det −0.045，deployable 单一备选无法 per-cell 选对），saturation gate 随 τ 升**单调把 vs_base 拉回 0（abstain to base）**，饱和域几乎全 abstain、未饱和域保留 route 且 safe_dev→1.0。**gate 的稳健价值 = 永不显著伤害 base**，跨预测/分类/检测一致 | §3 |
| **F-R10.3** | **仍无正向 deployable 增益 → 瓶颈是 routing 执行而非 saturation 检测（强化 Round 9 跨任务）**：无任一配置在任一任务 vs_base>0。即便 forecasting 有 34% oracle 头寸，deployable 单一备选也变现不了（per-cell winner 逐 cell 变、不可学，同 F-R9.x）。**修正 feedback_m4**：瓶颈不止"benchmark 饱和"，而是 **routing 执行壁垒跨饱和/未饱和域普遍存在**——saturation 检测能告诉你"有没有头寸"，但把头寸变现仍卡在 per-cell winner 预测 | §3 |
| **F-R10.4** | **safe-deviation-rate 正确区分域类型**：forecasting gate 高 τ 下 safe_dev=1.0（其 routing 子集确实安全），分类/检测 safe_dev 恒 0.2–0.4（饱和域偏离≈赌博）。新 metric 比 vs-base±pp 更能反映"检测器是否认得该不该动手" | §3.4 |

---

## 6. 论文含义与下一步

**论文论点（Round 10 收紧版）**：
> 把 Round 9 的"分类饱和"升级为两条 domain-invariant 经验律：
> ① **Saturation 可检测**（gap 在有头寸域可学，corr 0.37）且**饱和域最优策略是 abstain**（gate 损害控制实证）；
> ② **更强的壁垒**：deployable routing 即使在**未饱和**域（forecasting 有 34% 头寸）也无正向增益——
> per-cell winner 预测是跨任务的根本壁垒。**论文价值 = "在强 base 时代，selective abstain 是诚实最优，
> 且 routing 执行壁垒比 benchmark 饱和更普遍"**——比单纯"benchmark 饱和"更深、更有冲击力。

**下一步（按 ROI）**：
1. **#94 Failure Diagnostics 章**（paper §4.12）：把 F-R9.2→9.5→9.6→9.7/9.8→**F-R10.1/2/3** 固化成
   一条完整证据链——learned belief 失败 → 表示无效 → gate 坍缩 → benchmark 饱和 → **even unsaturated 也变现不了**。
2. **#95 真 expert-switching 域**：M12 证明 forecasting 有头寸但 deployable 吃不到；要变现需 **per-cell
   online 反馈**（部署累积真 outcome）或**受控 synthetic regime benchmark**。这是唯一可能翻盘的路径。
3. 不再造离线 router（reviewer + F-R10.3 一致：离线 deployable 在任何域都不正向）。
