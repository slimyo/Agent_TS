# Method v5 — Saturation-Aware Selective Routing

> 版本：2026-06-01（Round 10 起）
> 接续 `method4.md`（Round 9：learned belief 全证伪）。**配套**：`feedback_m4.md`（v4 外部 review · 3 reviewer 一致）/ `finish5.md`（Round 10 实测）/ `TODO.md`（#92-95）/ `paper_draft.md`。
>
> method4 把 belief 从 factor 加权升级为 learned distribution，并**用诚实负结果证伪了"learned belief 是银弹"**。
> feedback_m4 三位 reviewer 一致：**停止造 router，转 Saturation Detection**。method5 据此把研究对象从
> "如何选模型" 改为 "**该不该选模型**"——一个可跨任务证伪的新问题。

---

## 0. Thesis（一句话）

> Round 9 证明：在 base 已近 oracle 的域（Rocket@UCR/UEA），任何诚实 router 收敛到"不 route"。
> method5 的核心问题不再是 routing policy，而是 **Saturation Detection**：
> 给定一个 (域, base model)，**先预测 oracle gap ĝ(z)，再决定是否值得 route**——
> gap 大 → activate routing；gap 小 → abstain to base。
> **价值主张从"提分"彻底转为"诚实判断 base 何时已足够 + 只在有头寸时才动手"。**

---

## 1. feedback_m4 三位 reviewer 的一致诊断 + 我的评价

| reviewer 论点 | 我的评价 | method5 回应 |
|---|---|---|
| Round 9 已把"改进 router"做到边际收益≈0，别再迭代 belief/gate/shape | ✅ 完全同意（F-R9.1-9.8 已铁证）| 不再造 router |
| F-R9.2 是 **belief inversion**（方向反了）非 miscalibration → 别再做 calibration | ✅ 同意；calibration 假设"ranking 对、conf 不准"，但这里 ranking 都错 | 放弃 belief 校准路线 |
| **最大瓶颈是 benchmark 不是模型**（Rocket≈Oracle, gap<2pp）| ✅ 同意，且**实测强化**：forecasting 未饱和（见 §3）| Saturation Detection 跨任务验证 |
| 方向1：**Saturation Detection**（判断域是否饱和→该不该 route）| ✅ 最推荐，采纳为 M12 主线 | §2 / §4 |
| 方向2：**Failure Diagnostics**（为什么 router 失败）| ✅ 采纳为论文一章 | paper §4.12 + finish5 §归因 |
| 救 calibration：proper scoring / conformal / soft label | ⚠️ reviewer 自相矛盾（又说 ranking 全错别 calibration）；取后者，不投入 | — |
| 在线 belief 更新（reviewer4）| ⚠️ ROI 低于 Saturation Detection；留 future（#95）| — |

**我对 feedback 的一个实测强化（非 reviewer 原话）**：reviewer 说"UCR/UEA 都饱和"暗示 saturation 普遍；
但我聚合既有结果发现 **forecasting 显著未饱和**（chronos2 仅 25% cell 是 oracle，mean rel-MAE-gap 34%）。
这把 Saturation Detection 从"哲学转向"变成**可证伪的跨任务实验**：一个正确的检测器必须在 **forecasting 上
activate routing、在 classification 上 abstain**——若做不到，则 saturation 不可检测，论点失败。

---

## 2. 系统：Saturation-Aware Router

```
            ┌─────────────────────── per (域, base) 离线 ──────────────────────┐
  cell x ─► z=meta_features(x) ─► ĝ_φ(z) = 预测 oracle_gap(x)  (回归, LODO 训练)
            └───────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────┴───────────────┐
              ĝ(z) ≤ τ  (predicted saturated)   ĝ(z) > τ  (headroom)
                    │                              │
                    ▼                              ▼
            ABSTAIN → use base            ROUTE → learned/CV best-of-M
            (Rocket / Chronos-2)          (method4 belief or CV-winner)
```

- **`ĝ_φ(z)`**：回归头，输入 cell 元特征 z，输出**预测的 oracle gap**（= 最优模型 − base 的 acc/MAE 差）。
  训练标签 = 历史 cell 的真实 oracle gap（部署即"离线经验库"，合法）。**LODO 诚实协议**。
- **τ**：saturation 阈值。`ĝ≤τ` 判定该 cell 饱和 → abstain to base（已被 Round 9 证为饱和域最优）。
- **关键**：method5 不试图"选对模型"（Round 9 证明小样本下学不出），只试图**"判断该不该选"**——
  一个**更粗、更可学**的二元/标量信号（gap 回归比 winner 分类样本复杂度低得多）。

---

## 3. 三任务的饱和度（method5 的实验底座，实测聚合）

| 任务 | base | base=oracle 占比 | oracle gap | 饱和? | Saturation Detector 应 |
|---|---|---|---|---|---|
| **Forecasting**（6 ds × 72 cell）| Chronos-2 | **25%** | mean rel-MAE **34%** | ❌ 未饱和 | **activate routing** |
| **Classification**（UCR 22 ds × 128 cell）| Rocket | **71%** | 1.88pp | ✅ 饱和 | **abstain** |
| **Detection**（synth 4-class fault, 12 cell）| Rocket | **75%** | 1.4pp | ✅ 饱和 | **abstain** |

> 这是 method5 的核心 testbed：**同一检测器跨三任务**。forecasting 是"未饱和对照组"（routing 该有用），
> 分类/检测是"饱和组"（该 abstain）。检测器若能跨任务正确分流，则 Saturation Detection 成立。

---

## 4. M12 · 实验设计（finish5 §1-3）

### 4.1 元特征 z（轻量、跨任务通用）
cell 级统计：序列长度/数量/类数(分类)、per-channel 统计矩、谱熵、复杂度、信噪比代理。
**关键**：z 描述的是"任务难度/异质性"，不是"哪个模型赢"（后者 Round 9 证明学不出）。

### 4.2 saturation gap 回归（LODO）
- forecasting：gap = (base_MAE − oracle_MAE)/base_MAE（相对）
- 分类/检测：gap = oracle_acc − base_acc（绝对 pp）
- 留一数据集训 `ĝ_φ`，留出集预测 gap，与真实 gap 比 **(a) 相关性 (b) 排序 AUC（高 gap cell 能否排前）**。

### 4.3 端到端 selective routing 评估（新 metric，采纳 reviewer #93）
- **Regret-to-Oracle**：selected vs oracle 的差（越小越好）
- **Safe-Deviation-Rate**：在不显著伤害前提下成功偏离的比例
- **Abstain-Accuracy**：检测器判"饱和→abstain"时 base 确实≈oracle 的比例
- 对照：always-route（method4）/ always-abstain（=base）/ **saturation-gated（method5）**
  期望：saturation-gated 在 forecasting 接近 route 的收益、在分类接近 abstain 的安全 → **跨任务最优 envelope**。

---

## 5. 与既有方法的关系
- **复用**：method4 的 learned belief 作为"route 分支"的执行器；method3 factor engine 作 audit。
- **新增**：上层 `ĝ_φ(z)` saturation gate（这是 method5 的唯一新组件，符合"不堆模块"——它**替代**而非叠加决策层）。
- **诚实底线**：全 LODO；gap 标签是离线历史，不读查询 cell 的 test。

---

## 6. 文件地图（Round 10 增量）

```
research/
├── method5.md                       # 本文件
├── finish5.md                       # Round 10 实测 + F-R10.x
└── experiments/
    └── m12_saturation_router.py     # 跨三任务 saturation detection + selective routing LODO
```

---

## 术语表（method5 增量）

| 术语 | 含义 |
|---|---|
| Saturation Detection | 预测 (域,base) 的 oracle gap → 判断该不该 route |
| oracle gap | 最优可选模型 − base 的性能差（小=饱和）|
| ĝ_φ(z) | gap 回归头（LODO 训练）|
| abstain-to-base | 检测到饱和时退回 base，不 route（饱和域贝叶斯最优）|
| Regret-to-Oracle / Safe-Deviation-Rate | Round 10 新 success metric（取代 vs-Rocket±pp）|

---

**End of method5.md** — M12 实测后追加 §7+；论文 §4.12 "Saturation is Detectable" 据此成稿。
