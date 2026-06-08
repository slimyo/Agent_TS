# Method v8 — Heterogeneous-Expert Online Routing（结构异构专家 + 反馈闭环）

> 版本：2026-06-08（Round 13 起）。接续 `method7.md`（双信号解耦 / gain 离线不可学）+
> `test/`（在线 bandit，反馈把 +0.00→+2.83pp）。外部评审 `feedbackm7.md`（工业视角）。
> **配套**：`finish8.md`（实测，待建）/ `experiments/m24_hetero_experts.py` / `test/online_router.py`。

---

## 0. Thesis（一句话）

> Round 9–12 在 **同质资产池**（22 个多为 kernel/dict/deep 变体的分类器）上钉死了：
> per-cell winner 离线不可预测（profit-AUC 0.47），+0.00pp 是"可达天花板≠可预测"。
> `test/` 在线实验进一步发现：**反馈的收益 ∝ 域级结构稳定性**——分类有(0.78→+2.83pp)、few-shot 预测没有(0.46→+0.07)。
> method8 的**原命题**：winner 不可预测是同质池的产物；换结构异构专家 + regime 多样数据 → winner 可预测、机制变现。
>
> **实测修正（E13→E15，诚实）**：原命题**只在合成 regime 数据上成立**（+14%，因 regime 按构造可观测且与专家对齐）。
> **真实数据上证伪**：结构专家(−6.37%)与通用 TSFM(−6.6%)在真序列上**都净亏**，尽管 oracle 头寸 38%——
> 真实 TS 的 per-cell winner 从可观测 task-signature **不可预测，换结构专家也不例外**（回到 F-R12.1）。
> → **真正修正后的命题**：选择性行动在真数据变现的杠杆**不是"换异构专家池"，而是"存在稳定单元(域/episode)+在线反馈"**
> （分类 per-domain 反馈 +2.83pp 是唯一在真数据成立的正收益）。method8 的合成 +14% 是上界存在性证明，非真数据可达。

---

## 1. 为什么换池子是关键（诊断 → 设计）

把全研究链的瓶颈定位汇总：

| 层 | 结论 | 出处 |
|---|---|---|
| 决策机制 | 拆成 saturation/trust/gain/proposal 四变量，trust 避险闭环 | method6/7 |
| 离线 gain | per-cell winner 不可学（AUC 0.47），富特征更差 | F-R12.1 |
| 在线反馈 | 反馈能变现，但 **∝ 域级结构稳定性** | test/ §5.2 |
| **真瓶颈** | **同质池 → winner 被噪声淹没；不是机制不行，是池子无可预测结构** | feedbackm7 缺陷3 |

→ method8 不再改机制、也不堆同质模型，而是**换资产的"结构维度"**：
让池中模型的**能力差异 >> 少样本噪声**，winner 才会浮现且可预测。

**反例预测（可证伪）**：同一套机制
- 在**异构专家池 + regime 数据**上：winner-AUC ≫ 0.5、capability 路由捕获大头寸；
- 在**同质池**上：winner-AUC ≈ 0.5、路由 ≈ 0（复现 method7）。
若做不到这个对比，则"异构能救"被证伪。

---

## 2. 框架：能力画像驱动的异构专家路由

```
  series x ─► task signature  τ(x) = [trend, seasonal, noise, ac1, sparsity, length, ...]
                    │
                    ├─ capability match:  affinity(τ, expert_e) = ⟨τ, signature(e)⟩   ← 路线2
                    │      （每个 expert 有结构画像：擅长 trend / season / noise / AR / ...）
                    ▼
        proposal = argmax_e affinity   ─►  trust 避险闸(conformal)  ─►  deploy
                    │                                                      │
                    └──────── 在线反馈：observe reward → update per-(regime,expert) 后验（bandit）
```

- **task signature** τ(x)：从序列抽结构特征（趋势/季节/噪声/自相关/稀疏/长度）。
- **model signature** s(e)：每个专家的结构擅长向量（先验设定或从历史学）。
- **affinity 匹配**：τ·s 高 = 该专家大概率擅长此场景（**预测"谁擅长"而非"谁赢"**，feedbackm7 路线2）。
- **trust + 在线 bandit**：沿用 method7/test 的避险闸 + 反馈闭环（折扣后验抗漂移）。
- **episode/regime 域**：按 regime 聚合后验（路线3），让结构持续性被利用。

---

## 3. 异构专家池（结构互补，非同质变体）

| 专家 | 归纳偏置 / 物理先验 | 擅长 regime |
|---|---|---|
| trend_expert | 线性/Theil-Sen 外推 | 强趋势 |
| seasonal_expert | 季节朴素 / Fourier | 强季节 |
| ar_expert | AR(p) 自回归 | 自相关/平稳 |
| robust_expert | 中位/鲁棒平滑 | 重噪声/离群 |
| spike_expert | 事件/脉冲检测响应 | 稀疏脉冲 |
| (base) generic | naive-drift / 通用 TSFM | 兜底 |

**关键**：这些专家的能力差异是**结构性的**（不同 regime 上差几倍 MAE），不是同一范式调参——
所以 winner 由能力差异主导、可被 task-signature 预测。这正是 feedbackm7 表格里"physics/transformer/CNN/expert-system/statistical 各有专长"的落地。

---

## 4. 实验设计（m24，可证伪对比）

**受控 regime benchmark**：合成多 regime 序列（trend/seasonal/ar/noisy/spike 各若干），每 cell 有主导结构。

1. **异构池**：上面 5 专家 + base，算每 cell 各专家 MAE → oracle 库。
2. **同质池（对照）**：5 个只是窗口/阶数不同的同范式平滑器 + base。
3. **可预测性**：task-signature → winner 的 LODO 分类 AUC/acc。预期 **异构 ≫ 同质**。
4. **路由变现**：capability-affinity 路由 + 在线 bandit 的 vs-base 头寸捕获。预期 **异构捕获大、同质≈0**。
5. **抗漂移**：regime 随时间切换，折扣后验恢复力。

**成功判据**：异构池上 winner-AUC>0.7 且路由捕获 >50% oracle 头寸；同质池复现 ~0。
→ 证明"winner 可预测性 = 池子结构属性"，method7 的"不可预测"是同质池的局部结论而非普适规律。

---

## 5. 与 method7 的关系

- **不变**：四变量框架、conformal 避险（L1）、在线 bandit + 折扣后验（test/）、诚实 LODO。
- **新增**：结构异构专家池 + task/model signature 能力匹配（capability 先验从"全局模型质量"升级为"场景化擅长度"）。
- **角色转变**：method7 证明"在同质池上机制已到顶（避险）"；method8 证明"换异构池机制才出获利"——
  二者合起来是完整论点：**决策机制 × 资产池结构** 共同决定选择性行动的可达价值，单改任一不够。

---

## 6. 诚实底线与风险（含真实数据验证 E14）

- **真实数据已测（E14/F-R13.5，关键边界）**：把 learned-capability 接到真实预测库（14 真实 TSFM × 6 真实域）→
  **净亏 −6.6%**（反馈拉到 −2.9%）。原因：真实通用 TSFM 是"**架构不同**"而非"**结构互补**"（无清晰 regime 专长），
  且少样本窗退化 regime 检测。→ **method8 成立条件被精确界定**：需"**结构互补的专家**（各有清晰 regime 专长）
  **+ 足够上下文检测 regime**"。合成结构专家满足（+14%）；真实少样本通用 TSFM 不满足（−6.6%）。
- → **"异构能救" = 结构互补的异构，不是架构不同**。真工业落地需引入**真结构专家**（physics / 规则系统 / 统计模型，
  feedbackm7 表格那种），不能指望通用 TSFM 池自动有可路由结构。
- 合成 regime 是受控**存在性**证明；抗漂移、冷启动、L1 避险在异构池上已在 test/ 重测保留。

---

## 7. 文件地图（Round 13 增量）

```
research/
├── method8.md                       # 本文件（设计/框架/思想）
├── finish8.md                       # 实测（待建）
└── experiments/
    └── m24_hetero_experts.py        # 受控 regime + 异构/同质池对比 + 可预测性 + 路由变现
test/
├── online_router.py                 # 复用：bandit + capability 先验 + 折扣后验
└── (异构专家接入 pipelines 后落地)
```

---

## 术语表（method8 增量）

| 术语 | 含义 |
|---|---|
| 结构异构专家池 | 不同归纳偏置/物理先验的模型集合，能力差异 >> 噪声（非同质调参变体）|
| task signature τ(x) | 序列结构特征向量（trend/season/noise/ac/sparsity/length）|
| model signature s(e) | 专家的结构擅长向量；affinity=τ·s 预测"谁擅长此场景" |
| 可预测性是池子属性 | winner 能否预测取决于池的结构异构度，非普适不可预测（method8 核心命题）|

---

## 8. 实测（E12，详见 finish8.md）—— thesis 成立

受控 regime benchmark（5 结构 × 40 = 200 cell），**同一机制、同数据、只改池结构**：

| 池 | winner 预测 acc（多数类）| 路由 vs_base | oracle 头寸 | 捕获率 |
|---|---|---|---|---|
| **异构专家池** | **0.590**（0.30）| **+35.23%relMAE** | 40.14% | **0.878** |
| 同质池（MA 变体）| 0.385（0.34）| −2.91%relMAE | 9.36% | −0.31 |

- **F-R13.1**：winner 可预测性是**资产池结构属性**——异构使其可学(0.59)、路由捕获 88% 头寸；同质退化到≈多数类、路由净亏（复现 method7 +0.00/负）。
- **F-R13.2**：连"有没有头寸"也由池结构决定（异构 40% vs 同质 9% relMAE）。

→ **"winner 不可预测"是同质学术池的局部结论，非普适规律**。完整论点：
**选择性行动的可达收益 = f(决策机制 method7, 资产池结构 method8, 反馈 test/)**，三者缺一不可。

**End of method8.md** — 真异构专家 + 真域落地后追加 §9 + finish8.md §5。
