# Finish v7 — Round 12 实测（Trust-Aware Gain Routing）

> 版本：2026-06-03。方法见 `method7.md`；外部 review 是 `feedback_m6.md`；主线 `plan.md §零.2`。
> 本文件承接 Round 12（E4-E9，#101-106）实测 + Findings F-R12.x。前置 `finish6.md`（F-R11.x）。

---

## 0. 本轮目标与总判断

feedback_m6 三 reviewer 一致：Round 11 证明 **trust ≈ 风险估计器 ≠ 效用(gain)估计器**（F-R11.7），
缺 **gain model**。Round 12 完整做了 6 个方向（E4-E9）来回答："能不能预测‘值得偏离的收益’？"

> **总判断（诚实）**：**gain 在 cell 特征上不可靠预测**——E4(分类 corr 0.40 但 AUC 0.47)、
> E5(预测 corr 0.18，17% 头寸吃不到)、E6(组合输单 base) 三路否证。**但 E8 给出关键正向细化**：
> "避险"和"获利"需要**不同信号**——conformal 最善避险(AUC 0.79)、saturation 最善获利(AUC 0.71)，
> 朴素融合会稀释。E7 证 LLM 的 proposal 价值**可被小模型平替**（无需 API）。E9 证探索性偏离离线净负。
> → **Method7 的 gain 主目标未达成，但把"获利信号"从"不存在"细化到"是 saturation 而非 trust/belief"**。

---

## 1. E4 · Gain Model（#101，分类 10-clf 库 LODO）

`gain(z,m)=acc_m−acc_base` 逐候选回归；决策 `trust≥τ1 AND pred_gain≥τ2`。

| 指标 | 值 | 解读 |
|---|---|---|
| gain_pred_corr | 0.399 | gain **幅度**有弱相关 |
| gain_AUC（profit 分类）| **0.472** | 但**分不开**"该不该偏离"（<0.5）|
| trust-only acc | 0.875 | 偏离精度 0.08 |
| **trust+gain acc** | **0.882**（g>0.02）| 偏离精度 **0.118**（↑），calls 25→17 |
| always-commit / oracle | 0.884 / 0.888 | 仍 < 守默认 |

**F-R12.1**：**gain 幅度可弱回归（corr 0.40）但作"是否获利"分类器失败（AUC 0.47）**。
trust+gain 双门比 trust-only 精度翻倍(0.08→0.118)、更省偏离，**但被分类域 0.4pp 天花板锁死，仍 ≤ base**。
→ 印证 F-R11.7：gain 是独立信号，但在饱和分类域**头寸太薄无法变现**。

## 2. E5 · Forecasting 主战场（#102，6 数据集 72 cell LODO）

预测**未饱和**（chronos2 仅 25% 是 oracle，oracle 头寸 **17.2% rel-MAE**），按 reviewer #1 转主场。

| 指标 | 值 |
|---|---|
| gain_pred_corr | **0.178**（比分类更弱）|
| always-prop（偏离到 pred-best）| **−18.6% rel-MAE**（灾难）|
| gain-gate g>0 / g>0.05 / g>0.1 | −3.1% / −0.8% / **0.0%** |
| oracle 头寸 | +17.2% rel-MAE |
| safe-dev-rate | 0.286 |

**F-R12.2（关键否证）**：**即便预测域有 17% oracle 头寸，gain 仍预测不准（corr 0.18），吃不到**。
gain-gate 唯一安全工作点是 g>0.1 → 几乎不偏离 = base（0.0%）。→ **F-R11.7 的"获利难"跨饱和/非饱和域成立**：
头寸大不代表能学会抓——预测域 per-cell winner 比分类更难学。reviewer "转预测就能测出显著性"的乐观**被否证**。

## 3. E6 · Portfolio（#103，组合 vs 单 base）

不选一个、改加权组合（equal-topk / gain-weighted softmax），加权-acc 期望作软组合**线性上界**。

| 策略 | acc | vs base |
|---|---|---|
| base (rocket) | 86.08% | — |
| equal-topk | 85.94% | **−0.14pp** |
| gain-weighted | 82.48% | **−3.60pp** |
| oracle-single | 90.18% | +4.1pp |

**F-R12.3**：**组合（投资组合视角）输给单一 base**，连乐观线性上界都 −0.14pp（硬投票只会更低）。
saturated 库里没有"互补到值得分散"的专家结构——Rocket 太强，掺入弱模型只稀释。
→ portfolio 在当前 benchmark **不成立**；要成立需真正互补、无单一强 base 的资产池。

## 4. E7 · Proposal Network vs LLM（#104）

学一个轻量 proposal net（LogisticRegression(z)→非-base 候选）对比 belief-argmax 和 LLM defer。

| proposer | 提议命中 oracle | beats_base | profits |
|---|---|---|---|
| belief-argmax | 0.111 | 0.358 | 0.210 |
| **proposal-net** | 0.111 | **0.383** | **0.235** |
| LLM defer（E3 参照）| — | — | 0.267 |

**F-R12.4**：**轻量 proposal net ≈ LLM（profits 0.235 vs 0.267），且优于 belief-argmax**。
→ E3/F-R11.8 里 LLM 的"提偏离候选"价值**可被一个无需 API 的小模型平替**——LLM 的增益来自 proposal 角色
（可学），不是不可替代的推理。工程上：**用 proposal net 取代 LLM defer 省成本**，性能相当。

## 5. E8 · Meta-trust + risk-coverage（#105）—— 本轮最重要正向发现

四个 trust 源（conformal / ensemble-disagreement / feature-density / saturation）分别在
"避险(safe)" 和 "获利(profit)" 两维度上测 AUC：

| 信号 | AUC_safe（避险）| AUC_profit（获利）|
|---|---|---|
| **conformal** | **0.787** ✅ | 0.399 ✗ |
| disagreement | 0.590 | 0.602 |
| density | 0.465 | 0.567 |
| **saturation** | 0.352 ✗ | **0.714** ✅ |
| meta（logistic 融合）| 0.579 | 0.559 |

risk-coverage：按 meta-trust 降序放行偏离，safe-rate @cov25%/50%/75% = 0.571/0.517/0.395（base 0.448）。

**F-R12.5（关键机制细化）**：**"避险"和"获利"由不同信号主导**——conformal 最善避险(0.79)、
saturation 最善获利(0.71)，且二者在对方维度上都弱（conformal 获利 0.40 / saturation 避险 0.35）。
**朴素 logistic 融合反而稀释（safe 0.58 / profit 0.56），不如各自最优单源**。
→ 这把 F-R11.7（trust≠gain）**精确化为可操作结论**：决策应**双信号解耦**——
用 conformal 把"会变差"挡掉、用 **saturation（而非 belief/trust）**排"可能赚"的优先级，而非求一个万能 trust。

## 6. E9 · 探索性偏离（#106）

"unknown-unknown"区（belief 高置信但 base 与 top 都非 oracle）的探索价值（离线复盘）。

| 指标 | 值 |
|---|---|
| unknown-unknown cell 占比 | **28.8%**（38/132，真实存在）|
| uu 区 base / oracle | 74.4% / 86.1%（头寸大）|
| uu 区 explore（选 belief 最低候选）| 68.4%（**< base**）|
| explore-is-oracle / beats-base | 0.105 / 0.184 |

**F-R12.6**：**unknown-unknown 区大量存在（28.8%，头寸 12pp）但离线探索净负**——
belief 最不看好的候选 68.4% < base 74.4%，"低置信"通常是真差而非藏宝。探索作为离线动作不值得；
其唯一理论价值在**在线多轮信息增益**（本轮离线测不出）→ 归 future（需部署反馈循环）。

## 6b. E10 · Timer-S1 vs Chronos-2 base（换不换预测 base？远程 GPU 实测）

问题：预测 base 现为 Chronos-2，是否该换成更大的 **Timer-S1（ByteDance, 8.3B）**？
协议：复现既有 72 forecasting cell 的**确定性 few_shot_split**（split 只依赖 dataset,N,H,seed），
远程 2×RTX 5070 Ti（fp16, device_map=auto）跑 Timer-S1，按 (ds,N,seed) 配对本地记录的 Chronos-2 MAE，
**校验 start_idx 一致**（同测试窗口才可比）。ILI 12 cell 剔除（本地/远程序列长度不同 + Timer 输出 NaN）。

| 指标 | 值 | 解读 |
|---|---|---|
| 配对 cell | 60（5 数据集）| Weather/ETTh1/ETTh2/Exchange/ECL |
| Timer 胜率 | **45%** | < 一半 |
| 总体 mean rel-MAE | **−7.0%** | Timer 平均**更差** 7% |
| median rel-MAE | −1.6% | 中位也负 |
| 分数据集 | ETTh2 **+5.6%**(win58%)；ETTh1 −2.3%；ECL −7.5%；Exchange −9.2%；Weather **−21.8%** | 仅 1/5 数据集 Timer 占优 |
| **建议** | **保持 Chronos-2 ❌不换** | 仅 ETTh2 局部更好，整体显著负 |

**F-R12.7**：**更大的 TSFM（Timer-S1 8.3B）不构成更好的 few-shot 预测 base**——5 数据集 60 cell 上
相对 Chronos-2 平均 −7.0% rel-MAE、胜率 45%，仅 ETTh2 一域占优。→ 与饱和律一致：**base 之"强"不是参数量
决定，换更大模型不自动提分**；Chronos-2 维持为预测 base。可视化见 `results/timer_vs_chronos2.png`，
逐 cell 数据 `results/timer_vs_chronos2.jsonl`。
（工程：`research/baseline/timer.py` 加 `TIMER_FORCE_GPU=1` 多卡 fp16 加载 + 输入 dtype 对齐；
远程 env `tsci-remote` 无 chronos 包，故 Timer-only 远程跑、本地 merge。）

---

## 6c. E11 · trust 跨域可迁移性（#105 收口的第二块，`m23_trust_transfer.py`）

**动机**：F-R12.5 证 conformal 是最佳避险信号（同域 AUC≈0.79，用 m13 富 epistemic 特征）。#105 还问**跨域**是否成立。
本实验在 signal_router 全量 22-clf 库（UCR 140 + UEA 81 cell）上，用**部署级 4 维特征**，把 belief 头 + conformal 标定
只在源域拟合 → 目标域测 avoid-harm Trust-AUC。

| 口径 | Trust-AUC | n | 解读 |
|---|---|---|---|
| within_UCR (LODO) | **0.37** | 140 | 4 维特征下同域都弱（<0.5）|
| within_UEA (LODO) | 0.50 | 81 | ≈ 掷硬币 |
| cross_UCR→UEA | 0.42 | 81 | 跨域不成立 |
| cross_UEA→UCR | 0.575 | 140 | 唯一略 >0.5 |
| shuffle 对照 | 0.47 | 81 | 基准 |

**F-R12.8**：**trust 的避险判别力强依赖特征丰富度，且部署级轻特征下不可跨域迁移**。
F-R12.5 的 AUC 0.79 来自 m13 的**富 epistemic 特征**（30 维 + bagged belief 分歧/conformal）；
换成系统实际用的 **4 维部署特征**（N/候选数/base 水平/logN），同域 AUC 就掉到 0.37~0.50、跨域 0.42~0.575 ≈ 掷硬币。
→ **避险信号要可用必须带够 epistemic 信息（conformal nonconformity 需要分布表示），不能靠几个标量元特征**；
跨域迁移在轻特征下不成立。这给 method6 §2.5 理论界一个明确边界：**trust 的迁移性是"特征级"而非"任务级"属性**。
可视化/数据：`results/m23_trust_transfer.jsonl`。

---

## 7. Findings 汇总 F-R12.x

| ID | 一句话 | 出处 |
|---|---|---|
| **F-R12.1** | gain 幅度弱可回归(corr 0.40)但"是否获利"分类失败(AUC 0.47)；trust+gain 双门精度翻倍仍被 0.4pp 天花板锁死 | E4 |
| **F-R12.2** | 预测域 17% oracle 头寸也吃不到（gain corr 0.18，gate 安全点=不偏离）→ 获利难跨饱和/非饱和域成立，否证"转预测即显著" | E5 |
| **F-R12.3** | 投资组合输单 base（−0.14~−3.6pp，连线性上界都负）→ saturated 库无值得分散的互补结构 | E6 |
| **F-R12.4** | 轻量 proposal net ≈ LLM（profits 0.235 vs 0.267）→ LLM 的 proposal 价值可被无-API 小模型平替 | E7 |
| **F-R12.5（最重要）** | **避险↔conformal(AUC0.79)、获利↔saturation(AUC0.71)，不同信号；朴素融合稀释** → trust≠gain 精确化为"双信号解耦决策" | E8 |
| **F-R12.6** | unknown-unknown 区大(28.8%/头寸12pp)但离线探索净负；唯在线信息增益有价值 | E9 |
| **F-R12.7** | Timer-S1(8.3B) 不优于 Chronos-2 base（60cell mean −7.0% rel-MAE / win 45%，仅 ETTh2 占优）→ base 之强非参数量决定，维持 Chronos-2 | E10 |
| **F-R12.8** | trust 避险判别力强依赖特征丰富度：富 epistemic 特征 AUC 0.79，部署级 4 维特征同域掉到 0.37~0.50、跨域 0.42~0.575≈掷硬币 → trust 迁移性是"特征级"非"任务级" | E11 |

---

## 8. 最终结论与 method7 定位

**Round 12 没解决"获利"，但把它从"黑箱"拆成了可操作的结构**：

1. **gain 不可逐-cell 可靠预测**（E4/E5，跨饱和/非饱和、跨分类/预测一致）—— per-cell winner/gain 的不可学是
   **领域无关的根本壁垒**（再证 F-R10.3/F-R11.7）。
2. **但"该不该偏离"的两个子问题有各自最优信号**（E8，最重要）：
   - 避险（会不会变差）→ **conformal**（已闭环部署，F-R11.9）
   - 获利（能不能赚）→ **saturation 排序**（非 trust/belief）
   决策算子应为 **双信号解耦**：`deviate iff conformal_safe AND saturation_says_headroom`，而非单 trust 或单 gain。
3. **工程可平替**：proposal 角色用小模型即可（E7），不依赖 LLM API。
4. **组合/探索当前不成立**（E6/E9），留作"需互补资产池 / 在线反馈"的 future。

> **method7 定位（诚实）**：Trust-Aware **Gain** Routing 的"gain 回归"分支是**负结果**，
> 但产出一个更强的正向结论——**"避险"与"获利"是两个需不同信号、不可合并的子决策**（F-R12.5）。
> 这比"做一个 gain model"更接近问题本质，且给出可操作决策律。**主线收敛**：
> M4 inversion → M5 saturation → M6 trust 只避险 → **M7 避险↔conformal / 获利↔saturation 的双信号解耦**。

## 9. 下一步
1. 把 F-R12.5 的**双信号解耦决策**（conformal-safe × saturation-headroom）接进 `test/` 验证（类比 F-R11.9 trust 落地）。
2. 论文："Two Questions, Two Signals: why a single trust/confidence cannot drive selective action"——
   E4-E9 六实验构成完整证据链（gain 不可学 + 避险/获利信号分离 + proposal 可平替 + 组合/探索否证）。
3. 真互补资产池（E6 成立条件）/ 在线信息增益（E9）→ 长期 future。
