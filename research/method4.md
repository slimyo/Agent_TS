# Method v4 — Learned Belief-State Runtime

> 版本：2026-05-31（Round 9 起）
> 接续 `method3.md`（Round 7-8：self-evolving factor router）。**配套**：`feedback_m3.md`（v3 外部 review · 3 位 reviewer）/ `finish4.md`（Round 9 实测）/ `TODO.md`（#84-90 路线图）/ `paper_draft.md`。
>
> method3 把 router 升级到 self-evolving factor system。method4 回应 feedback_m3 的**一致核心批评**：把系统最核心的 **Belief** 从"人工 factor 加权评分"升级为"**从数据学出来的概率信念**"，并以 belief 的**分布形状**驱动决策、以 belief 的**校准失败**作为归因抓手。

---

## 0. Thesis（一句话）

> method3 的 belief `b(M)=softmax(−Σ factor)` 本质是 **interpretable scoring function**；
> method4 让 `b(M|z)` **直接从历史 outcome 学**（softmax-CE 到 oracle-winner），
> 人工 factor 降级为 **M8 审计层**而非 belief 来源。
> 系统形态：**Representation → (learned) Belief → belief-shape-aware Decision → Outcome → Evolution**。

---

## 1. feedback_m3 三位 reviewer 的一致诊断

| 编号 | 缺陷（3 位高度一致）| method3 现状 | method4 回应 |
|---|---|---|---|
| **缺陷1（一致 #1）** | **Belief 不是学出来的**：E=Σ(人工 prior+likelihood factor)，M3/M8 只调权重 | factor 加和 | **M10 Learned Belief**（§3）|
| 缺陷2（最有趣理论）| 决策坍缩太廉价：decide() 只取点估计，不看 `b(M)` 形状 | argmax/thompson/risk-min | **belief-shape 决策**（§4）|
| 缺陷3（论文下限）| 缺"系统为何失败"的归因闭环 | M8 拆单次决策 | **负结果归因工具箱**（§5）|
| 问题2 | Regime 静态 k-means，cluster≠mechanism | RegimePrior 硬先验 | **regime-as-feature**（§6）|
| M1 存疑 | meta-bandit 可能学的是环境噪声非 decide_mode | 92% thompson | **regret ablation**（§7）|
| 数据 | UCR-5 已饱和，router≈Rocket；需 expert-switching | UCR-5 30 cell | **UCR-10 LODO**（§3.3）|
| 缺失（reviewer2）| 缺 Outcome Model：reactive 而非 predictive | observe-only | **future（§8）**|

> 三位都把 **Learned Belief Model 列为头号升级**（"这是 v4 与 v3 的真正分水岭"）。

---

## 2. 与 method3 的关系（什么变、什么留）

```
method3 (factor engine):                    method4 (learned belief + audit):
  z ─► Σ prior/likelihood factor ─► E         z ─► g_φ(z) ─► b(M|z)   [LEARNED]
       └─ M3 学 strength                            │
       └─ M8 拆 attribution                         └─ factor engine 仍在，但降级为
  b = softmax(−E)                                      ① cold-start 先验（数据不足时）
  decide(argmax/thompson/risk_min)                     ② M8 审计 b 的"为什么"
                                              decide(belief-shape-aware)   [§4]
```

**保留**：representation（§method3 Layer1）、factor engine（作 cold-start + 审计）、M9 去泄漏协议、M8 attribution。
**新增**：learned belief head（M10）、belief-shape 决策、负结果归因工具箱。
**诚实底线不变**：所有评测 **leave-one-dataset-out (LODO)**，比 method3 的 leave-one-cell-out 更严（held-out 整个数据集不进训练）。

---

## 3. M10 · Learned Belief Model（缺陷1，核心）

### 3.1 公式

$$b(M_k \mid z) = \mathrm{softmax}_k\big(g_\phi(z)\big),\qquad z = \text{featurize\_cell}(X_{\text{train}}) \in \mathbb{R}^{30}$$

训练目标 = reviewer 明确建议的 **cross-entropy 到 oracle-winner one-hot**：

$$\mathcal{L}(\phi) = -\sum_{\text{cells }i} \log b\big(M_{k^*_i} \mid z_i\big),\qquad k^*_i = \arg\max_k \text{acc}_k(\text{cell }i)$$

其中 $\text{acc}_k$ 是历史 cell 上 classifier $k$ 的真实 test acc（部署即"已积累的离线经验库"，合法）。

**M10 vs 既有 meta_router_v2 的本质区别**：v2 是 per-classifier **独立回归** acc（5 个 RFR head，各自预测一个标量），不构成分布；M10 学的是**单一 softmax 分布** `b(M|z)`，其形状（熵/gini/尾部）才是 §4/§5 决策与归因的抓手。这正是 reviewer 说的"learned belief" vs "heuristic score"的差别。

### 3.2 路由 + confidence gate

$$\hat M = \begin{cases}\arg\max_k b(M_k|z) & \text{if } b(\hat M)-b(\text{rocket}) > \tau\\ \text{rocket (default)} & \text{otherwise}\end{cases}$$

τ = belief-margin 门控：只在"学出来的信念明显偏离默认"时才偏离 Rocket（呼应 method3 的 N-fallback 哲学，但门控量从 CV margin 换成 **learned belief margin**）。

### 3.3 诚实评测协议（UCR-10 LODO）

- 数据：`taskb_ucr.jsonl`(UCR-5) + `taskb_extended_ucr.jsonl`(GunPoint/Strawberry/Wafer/ECG5000/Crop) = **10 数据集 / 56 cell**，每 cell 有 5 个 classifier 的真实 test acc。
- 协议：留一个数据集 → 其余拟合 `g_φ` → 留出集上路由。held-out 数据集**完全不进训练**（无 per-cell 也无 per-dataset 泄漏）。
- 为什么换 UCR-10：feedback_m3 明确"UCR-5 已饱和、不要再刷"；UCR-10 含 expert-switching（不同数据集 oracle winner 不同），是 router 能真正创造价值的场景。

实测见 `finish4.md §1`。

---

## 4. Belief-shape 敏感决策（缺陷2，待实现）

decide() 不再只取点估计，而是读 `b(M|z)` 的**分布形状特征**：

| 特征 | 公式 | 决策含义 |
|---|---|---|
| entropy | $-\sum_k b_k\log b_k$ | 整体不确定性 |
| gini (1−Simpson) | $1-\sum_k b_k^2$ | 优势模型统治力（低=有明确赢家）|
| top1−top2 gap | $b_{(1)}-b_{(2)}$ | 决策脆弱度 |
| tail mass | $\sum_{k\ge3} b_{(k)}$ | 黑天鹅风险 |

**升级 M1**：arm 从 `{argmax, thompson, risk_min}` 改为**带参变体**，参数由形状特征驱动——
例如 `explore(ε)`，`ε ∝ gini`（没有明显赢家时自动更激进探索）；门控 τ 随 top1−top2 gap 自适应
（信念脆弱时更保守、退回 default）。这就是 reviewer 说的"决策的元认知"。

> finish4 §1 已输出全部形状特征到 `m10_learned_belief.jsonl`，作为本节决策器的输入；决策器本体待实现。

---

## 5. 负结果归因工具箱（缺陷3，已实测）

把"干净的负结果"变成"为什么"的洞察：

1. **belief miscalibration**：router 偏离 default 且**错**时，`b(选中)` 的平均强度 vs 偏离且**对**时的强度。
   若两者接近 → 系统"**自信地犯错**"（calibration 失败），比单纯报告 −Xpp 更有价值。
2. **per-dataset oracle 分布**：每个数据集的 oracle-winner 分布，量化 expert-switching 程度——
   证明"no universal best classifier"并定位 router 的价值区间。
3. **LOFO post-mortem**（method3 M8 复用）：router 输给 Rocket 的 cell，画"Rocket 归因图 vs 选中失败模型归因图"，
   诊断**哪个 factor 系统性误导**。

实测见 `finish4.md §2`（F-R9.2 belief 校准失败是本轮最重要发现）。

---

## 6. Regime-as-feature（问题2，待实现，并入 #75）

取消 `RegimePrior` 的硬先验（`if regime==3: prior+=...`），改为把 **regime embedding** 与 `z` 拼接
一起喂 belief head：`g_φ([z; regime_emb])`。reviewer：cluster≠mechanism，硬先验在跨域时危险；
喂特征让模型自己决定 regime 信息的用法，"明显更稳"。

---

## 7. M1 regret ablation（M1 存疑，待实现）

reviewer 最怀疑 M1："92% thompson 可能只证明环境太简单，不证明 auto mode 有效"。
**实验**：固定 belief state，只比 `{argmax, thompson, risk_min, auto}` 的 **regret 曲线**是否真的下降。
否则 M1 归为"看起来高级但贡献有限"，论文不主推。

---

## 8. Outcome Model（reviewer2 补充，P3 future）

当前 `belief→decision→observe` 是 **reactive**；reviewer2 建议轻量 **Outcome Model**：
`belief→simulate future reward→decision`（predictive runtime）。比 feedback.md 的 world model 轻、更现实。
本期不实现，写 §5.4 future work。

---

## 9. 文件地图（Round 9 增量）

```
research/
├── method4.md                          # 本文件
├── finish4.md                          # Round 9 实测
├── experiments/
│   └── m10_learned_belief.py           # M10: learned belief LODO + 形状/校准分析
├── results/
│   └── m10_learned_belief.jsonl        # τ-sweep summary + 逐 cell belief 记录
└── agent/
    └── (belief head 待落地为 agent/belief_model.py + 接入 clf_planner)
```

---

## 术语表（method4 增量）

| 术语 | 含义 |
|---|---|
| learned belief | `b(M|z)=softmax(g_φ(z))`，从 oracle-winner CE 学出，**非** factor 加和 |
| belief-shape | b(M) 的 entropy/gini/top-gap/tail，驱动 §4 决策 |
| belief miscalibration | 偏离对/错时 belief 强度是否可分；不可分=自信地犯错 |
| expert-switching | 不同数据集 oracle-winner 不同的 benchmark（router 价值区间）|
| LODO | leave-one-dataset-out（比 method3 LOCO 更严的诚实协议）|

---

**End of method4.md** — belief-shape 决策器（§4）+ regime-feature（§6）+ M1 ablation（§7）落地后追加 §10+。
