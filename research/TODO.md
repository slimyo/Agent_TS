# TODO · 任务看板

> **重构：2026-05-30**（并入 M8/M9 结论 + 论文善后进度）
> 配套：`plan.md`（总纲）/ `method3.md`（方法）/ `finish3.md`（实测 F-R8.x）/ `feedback.md`（11 条硬伤）/ `paper_draft.md`（论文稿）。
> **本文件文首"路线图"是当前唯一权威优先级**；下方历史段仅作存档。

---

## 🟢 状态快照（一眼看清现在在哪）

- **Round 11（#96-100）+ Round 12（#101-106）全部闭环（2026-06-05）**。机制结论已成稿 `paper_mechanism.md`。
  - 决策四变量定稿：**belief 提候选 / saturation 有头寸 / trust 避险 / proposal**；获利(gain)经 E4-E11 **证不可学**。
  - 双信号解耦决策律：`deviate iff conformal_safe ∧ saturation_headroom`（F-R12.5）；regret bound 见 `paper_mechanism §4.5`。
  - 全量库审计（22 clf / 14 TSFM / 三任务）：扩库 un-saturate（rocket 即 oracle 71%→37.6%、头寸 4.85pp）但 winner 逐-cell 不可预测 → **+0.00pp 是"可达天花板≠可预测"的正确弃权，非窄库 artifact**。
- **已闭环**：trust 避险门部署进 `test/`，分类 −0.42pp→**+0.00pp 零代价避坑**（F-R11.9）。
- **Round 12 关键负结果**：gain 不可学(F-R12.1/2)、portfolio 输(12.3)、探索净负(12.6)、大 TSFM≠好 base(12.7)、trust 迁移是特征级(12.8)。
- **下一步（要正收益需换条件，非更多离线模型）**：未饱和域 / 真异构互补资产池 / 在线反馈（见 `paper_mechanism §8`）。
- 历史阶段（存档）：M8/M9 诚实化 → Round 9-10 饱和律 → Round 11 决策机制 → Round 12 gain 不可学 + 双信号解耦。

---

## 🧭 feedback 改进路线图（权威优先级，按 effort × value 排序）

### ✅ 已封堵的 feedback 硬伤

| feedback 条目 | 修复 | 产出 |
|---|---|---|
| **问题 1** fake Bayesian | M8 改 energy-based / belief-state framing | `bayesian_router.py` + `method3 §M8.1` |
| **问题 2** factor explosion / 黑盒 | M8 Factor Attribution（`factor_log_contributions` + LOFO `attribute_decision` + `FactorAttributionAccumulator`）| F-R8.5/8.6 |
| **问题 6** memory 数据泄漏（"致命"）| M9 CV-only 投票 + leave-one-cell-out 检索 | F-R8.7/8.8/8.9 + `method3 §M9` |

### 🔴 P0 · 论文善后（M9 直接引发，必须先做）

| Task | 内容 | 状态 | 依据 |
|---|---|---|---|
| **#70** | **全文撤回 "+0.89pp 击败 Rocket"** → 统一 **86.91% / −0.62pp**。已改：§4.5/§4.6 正文表 + §3.2 framing（前序）；**本轮补**：abstract、§1.2 表、§1.3 贡献#5、§3.0.1、§4.5 标题 | 🟡 **基本完成**（待全文 grep 复核 §5.2/§6/appendix 是否还有残留）| F-R8.7 |
| **#71** | **重写 TSC 主张 framing**：从"router 击败 Rocket"→"saturated benchmark 上 routing 无显著增益；**数据泄漏审计本身**作为方法学贡献" | 🟡 abstract/§4.6 已落，需扩到 §5.3 synthesis + §6 conclusion | feedback + F-R8.7 |
| **#72** | **memory 增益重定位**：isotonic 校准 CV→test **已实现+实测(2026-05-31)=诚实负结果** 86.91%→86.66%(−0.25pp 更差，见 finish3 §9/F-R8.10)；离线 CV 变换走不通，剩"在线真实反馈"子路径 | 🟡 校准分支已封(负结果)；在线反馈待做 | F-R8.8/8.10 |

### 🟠 P1 · feedback 理论/实证硬伤（本期主攻）

| Task | feedback | 内容 | effort | value |
|---|---|---|---|---|
| **#73** | 问题 2 + 建议 5 | **Factor 消融 + 减法实验**：用 M8 `redundancy_matrix` + `mean_abs_influence` 多数据集跑 6 prior×2 lik 逐项 ablation，剔除 inert/冗余 factor | 中 | ⭐⭐⭐ |
| **#74** | 问题 4 + 建议 2 | **分位数 / 分布鲁棒 bandit**：TS loss 重尾正偏，单高斯低估尾部；改 bootstrap 分位数 bandit 或 log-normal/t 共轭；Thompson 保留 + 拟合优度监控 | 中 | ⭐⭐⭐ |
| **#75** | 问题 3 + 建议 3 | **Regime 退化为 feature**：取消 `RegimePrior` 直接当先验（k-means purity 仅 82.4%）；改 one-hot 喂权值学习器，per-cluster loss 仅作历史参考 | 中 | ⭐⭐ |
| **#76** | 建议 1 | **学 factor 权重**（扩 M3 EB）：弃手调 `strength`，验证集上用 continuous bandit / BO 学对数线性权重 λ | 中 | ⭐⭐ |

### 🟡 P2 · 收敛 + 论文 scope（feedback 反复强调"不要再堆模块"）

| Task | 内容 |
|---|---|
| **#77** | 把系统重构成 **Representation `z=f(x)` → Belief `b(M)` → Decision `a~π(a|b)`** 三层；"posterior"统一改叫 **belief state**（呼应 #70；plan.md §三已落文档层，代码层待对齐）|
| **#78** | **论文 scope 收敛**：主论文只保留 "universal probabilistic routing runtime"；RCA / action layer / scheduler / telemetry / reflective loop 降为附录或单独短文 |
| **#79** | **路由开销/收益 Pareto**：每决策记 router 自身耗时，出"精度提升 vs 总耗时"帕累托（扩 `latency_analysis.py`）|
| **#80** | **冷启动 + 保护性回退**：bandit burn-in 用固定 baseline；校准器低置信时降级 robust ensemble |

### 🔵 P3 · ambitious future（写 §5.4 future work，本期不实现）

| Task | feedback | 内容 |
|---|---|---|
| #81 | 问题 3 | Dynamic regime geometry：soft regime field `p(r|z)` + regime transition → adaptive state-space router |
| #82 | 问题 4 | Neural Bayesian Bandit `p(ℓ|z,M)`：用 representation 直接预测 loss 分布，弥合 regime↔bandit 割裂 |
| #83 | 问题 5 | World model / latent env dynamics `s_t→s_{t+1}`：从 detect drift → **anticipate drift** |

---

## 📌 后续改进规划（执行序 + 验收标准）

> 把上面路线图落成"做什么 → 怎么验收 → 依赖"的可执行序。建议按下表自上而下推进。

| 序 | Task | 一句话动作 | 验收标准（怎么算 done）| 依赖 |
|---|---|---|---|---|
| 1 | **#70/#71 收尾** | 全文 grep `0.89`/`88.42`/`p=0.17`/`beat`/`Bayesian posterior`，逐条确认要么是"描述泄漏 artifact"要么改成诚实值；§5.3/§6 conclusion 同步 | grep 后无"作为正面战绩"的残留；abstract↔正文↔conclusion 数字一致 | 无（纯文档）|
| 2 | ~~**#72** 离线 CV 校准~~ | ✅ 已做完=负结果（86.66%/−0.87pp，更差）→ 离线变换判定走不通；改做 **在线真实反馈** 子路径（部署累积真 outcome）才有可能 | 在线反馈版 acc 报告不依赖任何 CV proxy | M9（done）/ F-R8.10 |
| 3 | **#73** factor 消融 | 多数据集跑逐 factor ablation，用 `mean_abs_influence`/`redundancy_matrix` 出表 | 给出"剔除哪些 factor 不掉点"的结论表，写进论文 §4.3 ablation | M8（已完成）|
| 4 | **#74** 鲁棒 bandit | `bandit.py` 加 bootstrap 分位数 / t-共轭分支 + 拟合优度监控 | 重尾 cell 上尾部覆盖率较单高斯改善；不掉平均点 | 无 |
| 5 | **#75** regime→feature | 取消 `RegimePrior` 硬先验，改 one-hot feature 喂权值学习器 | purity 噪声不再单点误导；ablation 显示 ≥ 持平 | #76 协同 |
| 6 | **#76** 学 factor 权重 | 验证集 BO/continuous bandit 学 λ，替换手调 `strength` | 每个 λ_k 非常数且优于手调；写进 §3 | M3 EB |
| 7 | **#77/#78** 收敛+scope | 代码层对齐三层命名；论文拆主文/附录 | 主文只剩 routing runtime；模块名与 plan.md §三一致 | #70-76 |
| 8 | **#79/#80** Pareto+冷启动 | 记录 per-decision 耗时出 Pareto；加 burn-in + 低置信降级 | 帕累托图入附录；冷启动不再 catastrophic | 无 |
| 9 | **#81-83** future work | 仅写 §5.4，不实现 | 论文 future work 段成稿 | — |

**里程碑**：
- **M-A（论文可投）**= 序 1 + 序 3 完成（诚实数字全文一致 + factor 消融实证）。
- **M-B（方法更扎实）**= 序 2 + 序 4 完成（去泄漏增益重定位 + 鲁棒 bandit）。
- **M-C（收敛成稿）**= 序 5-8 完成（三层抽象 + scope 收敛 + Pareto/冷启动）。

---

## 🧭 Round 12 路线图（2026-06-02 · feedback_m6）—— Method7：Trust-Aware **Gain** Routing

> feedback_m6 三 reviewer **一致**：Round 11 最大贡献是把单一 confidence 拆成
> **saturation / trust / gain / proposal 四个独立变量**，并证明 **trust ≈ 风险估计器 ≠ 效用估计器**（F-R11.7）。
> → method6 缺 **gain model**：决策应 `if trust>τ1 AND gain>τ2`，而非仅 `if trust>τ`。
> 设计=`method7.md`（待建），实测=`finish7.md`（待建）。详细方向变量分解见 `plan.md §零.1/零.2`。

| Task | 方向（reviewer 优先级）| 内容 | 状态 |
|---|---|---|---|
| **#101** | Gain Modeling（最高优先）| **E4**：回归 `gain(z)=oracle−base`（LODO），决策升级 `trust>τ1 AND gain>τ2`；测 gain-AUC + 是否把"获利"从 ~0 提到可测。直击 F-R11.7 缺口 | ✅ done（F-R12.1：gain corr 0.40 / profit-AUC 0.47 不可学，`m16`）|
| **#102** | 主战场转 Forecasting | **E5**：把 trust+gain 机制接预测域（chronos2/bolt/timesfm/moirai…，未饱和、正确偏离多），测 trust/gain 统计显著性 + horizon 增大时 trust 衰减 | ✅ E5 done（F-R12.2 否证）|
| **#102b** | 换不换预测 base | **E10**：远程 GPU 跑 **Timer-S1(8.3B) vs Chronos-2** 72cell（确定性 split + start_idx 校验，ILI 剔除）→ **保持 Chronos-2**（mean −7.0% rel-MAE / win 45%，仅 ETTh2 占优）| ✅ done（F-R12.7）|
| **#102c** | 远程 chronos2 跑通 | 新建 `tsci-c2` py3.10 env（torch cu128 + chronos-forecasting 2.2.2，含 Chronos2Pipeline；py3.9 装不了）→ chronos2 在远程 smoke OK，amazon/chronos-2 权重已缓存。后续预测全量 sweep 可端到端在远程（不必本地 merge）| ✅ done |
| **#103** | Counterfactual / Portfolio | belief 从 "winner 预测" → "各候选 outcome 分布 Y(a)"；再到组合配比（trust=风险 / gain=期望收益，投资组合视角，最有论文潜力）| ✅ E6 done（F-R12.3：组合输单 base −0.14~−3.6pp，`m18`）|
| **#104** | Proposal Network 替代 LLM | E3 证 LLM 价值在"提偏离候选"非推理 → 学一个 proposal net（AlphaGo 式 policy-propose + trust-verify）| ✅ E7 done（F-R12.4：proposal-net≈LLM 0.235 vs 0.267，`m19`）|
| **#105** | Meta-trust / 理论界 | trust=f(conformal,disagreement,density,saturation)；推 selective-action regret bound；trust 跨域可迁移性 | ✅ **done**：E8 meta-trust(F-R12.5) + E11 跨域可迁移性(F-R12.8：特征级非任务级，`m23`) + regret bound 理论(`paper_mechanism.md §4.5`) |
| **#106** | 探索性偏离（reviewer3）| "未知未知"区（高 trust 但集体错）主动选低置信候选最大化信息增益，把失败转为系统学习 | ✅ E9 done（F-R12.6：离线探索净负，`m21`）|

**Round 12 执行序**：#101 Gain Model（补缺口，最高 ROI）→ #102 forecasting 主场（测显著性）→ #103 portfolio（论文潜力）。
**诚实前提**：oracle 天花板低（分类 +0.4pp / 预测 ~19% rel）；gain model 目标是"避险已闭环上把获利从 ~0 提到可测（+1~2pp 即实质突破）"，非回到刷 SOTA。

---

## 🧭 主线路线图（Round 11 · 2026-06-02）—— 决策机制本身的设计科学（method6）

> **研究主线已转向**（见 `plan.md §零`）：不再追性能，以 **method4 的决策机制**为研究对象。
> 即使提不了分，"agent 何时该行动/退/信自己/为克制辩护"本身有发表价值。设计=`method6.md`，实测=`finish6.md`。
> **本期主攻方向 1+2 合并**（trust≠confidence + shape→action 相图），LLM 用 **DeepSeek**（已实测可用）。

| Task | 方向 | 内容 | 状态 |
|---|---|---|---|
| **#96** | 主线A（攻 F-R9.2）| **Trust≠Confidence**：epistemic 估计（deep ensemble 分歧 / MC-dropout / conformal）解耦 trust，trust-gate 后把 belief inversion 的 corr **从负翻正** | 🔬 method6 §1A / E1（`m13_trust_vs_confidence.py`）|
| **#97** | 主线B（攻缺陷2）| **决策相图**：`π(a｜saturation×trust×shape)→{commit/deviate/ensemble/explore/defer}`，画 (saturation×trust) 最优动作相图 + 测跨任务边界一致性 | 🔬 method6 §1B / E2（`m14_decision_phase_diagram.py`）|
| **#98** | LLM defer | **DeepSeek 第二意见**作为 defer 动作：低 trust+低 sat 时调 `deepseek-reasoner` 审议"该不该偏离"，测 defer vs ensemble 的 trust 校准改善 | 🔬 method6 §2 / E3（`m15_llm_defer.py`）|
| **#99** | 指标改造 | 主指标换为 **Inversion-coef / Regret-to-Oracle / Safe-Deviation-Rate / Abstain-Acc / Trust-AUC / 相图边界清晰度**（弃 vs-base±pp）| 🔬 随 #96-98 |
| **#100** | 论文 | 把机制研究写成 **"The Design of a Selective-Action Decision Mechanism under a Strong Default"**（独立于提分的机制论文）| ✅ **草稿成稿** `research/paper_mechanism.md`（2026-06-05；E1-E3 + F-R11.x/F-R12.x + 全量库 un-saturation/不可预测审计 + 三任务相图 m22）|

**⚙️ LLM 配置陷阱（务必记住）**：`demo/.env` 先加载且 `PROVIDER=zhipu`、DeepSeek key 在那里是注释掉的；
真 key 在 `research/.env`。用 DeepSeek 必须**显式注入** research/.env 的 `DEEPSEEK_API_KEY` 再设 `PROVIDER=deepseek`
（`MODEL=deepseek-reasoner` 思考 / `deepseek-chat` 快通道）。两者均已 live ping 通过（2026-06-02）。
`research/utils/llm.chat_cached` 带磁盘缓存，重复 prompt 零成本。

---

## 🆕 feedback_m4 路线图（Round 10 · 2026-06-01）—— 从 "造 Router" 到 "Saturation Detection"

> 三位 reviewer（`feedback_m4.md`）**一致**结论：Round 9 已把"如何改进 Router"做到边际收益极低；
> 继续在 belief/gate/shape/embedding 上迭代只会得 −0.5~0pp。**应停止造 router，转两个更高层问题**：
> ① **Saturation Detection**（判断某域 base 是否已近 oracle → 该不该 route）② **Failure Diagnostics**
> （为什么 router 系统性失败）。设计=`method5.md`，实测=`finish5.md`。

**我对 feedback_m4 的评价（采纳度）**：
- ✅ **完全采纳**："belief inversion"（非 miscalibration，方向反了→别再做 calibration）、"benchmark 是瓶颈非模型"、
  价值主张转 **Saturation Detection + Selective Abstain**。这三点 Round 9 数据已铁证（F-R9.2/9.7/9.8）。
- ✅ **采纳并强化**：reviewer 说"UCR/UEA 都饱和"——我**实测验证了关键反例**：**forecasting 没饱和**
  （chronos2 仅 25% cell 是 oracle，mean rel-gap 34%）vs 分类饱和（rocket 71-75%）。这把"Saturation Detection"
  从思辨变成**可证伪的跨任务实验**：检测器应在 forecasting 上 activate routing、在分类上 abstain。
- ⚠️ **部分保留**：reviewer 建议"换 proper scoring / soft label / conformal" 救 calibration——但他自己也说
  "ranking 都错了，别再 calibration"。两条矛盾；我取后者：**不再在饱和分类域救 belief**，把精力投 Saturation Detection。
- ⚠️ **保留**：reviewer4 段建议"在线信念更新"——Round 10 先做离线 Saturation Detection（更高 ROI），online 留 future。

| Task | feedback_m4 建议 | 内容 | 状态 |
|---|---|---|---|
| **#92** | 方向1（最推荐）| **M12 Saturation Detector**：`ĝ(z)→预测 oracle gap`，gap>τ 才 route 否则 abstain to base；跨三任务（forecast 不饱和 / clf 饱和 / detect 饱和）LODO 实测 | 🔬 进行中（method5/finish5）|
| **#93** | 新 success metric | **Regret-to-Oracle + Safe-Deviation-Rate + Abstain-Accuracy**：取代"vs Rocket ±pp"，度量"该 route 时 route、该 abstain 时 abstain" | 🔬 随 #92 |
| **#94** | 方向2 | **Failure Diagnostics 链**：把 F-R9.2(inversion)→9.5(repr 无效)→9.6(gate 坍缩)→9.7/9.8(饱和) 固化成论文一章 | 📋 待写（paper §4.12）|
| **#95** | 中期 | 真 expert-switching 域（工业/concept-drift/synthetic 受控 regime）| 📋 future |

**Round 10 论点**：把 Round 9 的"分类饱和"升级为 **"Saturation 是可检测的领域属性"**——
forecasting(未饱和,route 有用) vs classification(饱和,abstain 最优) 的对比，证明**同一 Saturation Detector
能跨任务正确决定该不该 route**。这比"再造一个 router"研究价值更高（reviewer 一致判断）。

---

## 🆕 feedback_m3 路线图（Round 9 · 2026-05-31）—— 从 "factor 加权" 到 "learned belief"

> 三位 reviewer（`feedback_m3.md`）**一致**核心结论：`b(M)=softmax(−E)` 的 E 全是人工 factor 之和，
> 是 **interpretable scoring function**，不是 **learned belief**；M3 学 strength / M8 学 attribution
> 只调权重，没改根本。次级共识：决策坍缩太廉价(不看 b 形状) / 缺"为什么失败"归因闭环 / regime-as-feature /
> 不要再刷 UCR-5(已饱和)→找 expert-switching / 缺 outcome model。设计=`method4.md`，实测=`finish4.md`。

| Task | feedback_m3 缺陷 | 内容 | 状态 |
|---|---|---|---|
| **#84** | 缺陷1（一致#1）| **M10 Learned Belief Model**：`b(M｜z)` 直接 softmax-CE 学 oracle-winner，取代 factor 加和 | ✅ **已实现+实测**=诚实负结果（finish4 §1，UCR-10 LODO 全 τ 输 Rocket −0.80~−1.46pp）|
| **#85** | 缺陷3（论文下限）| **负结果归因工具箱**：belief miscalibration + per-dataset oracle 分布 + LOFO post-mortem | ✅ **已实测**（finish4 §2，**F-R9.2 belief 强度是反指标**：错时 0.79 > 对时 0.51）|
| **#86** | 缺陷2（最有趣理论）| **belief-shape 决策**：读 b(M) entropy/gini/top-gap/tail → 上下文自适应 explore | 📋 形状特征已输出；待 belief 修可信后做 |
| **#86b** | F-R9.5 | **表示 z 升级**：z 30-d 手特征 → 512-d MOMENT embedding，重跑 M10 LODO | ✅ **已实测=更强负结果**（finish4 §6，−0.80→**−3.96pp**，证伪"z 是主瓶颈"；真瓶颈=元监督少+校准缺失）|
| **#86c** | F-R9.2 | **belief 校准 gate**：bagged belief + nested-LODO 训"P(偏离正确)"校准 gate | ✅ **已实测**（finish4 §7/F-R9.6，−3.96/−0.80→**−0.45pp**：消除灾难误偏离但坍缩回 Rocket；瓶颈=元监督样本复杂度，非决策器形式）|
| **#87** | 问题2 | **regime-as-feature**（并入 #75）：regime embedding 喂决策层，取消硬先验 | 📋 待做 |
| **#88** | M1 存疑 | **M1 regret ablation**：固定 belief 比 argmax/thompson/risk-min/auto regret，证 auto 真有效 | 📋 待做 |
| **#89** | F-R9.6→9.7 | **扩元数据**：12 新 UCR 数据集 ×3N×2seed×5clf（360 rows），oracle 库 10→22 数据集 | ✅ **已实测=证伪样本复杂度假说**（finish4 §8/F-R9.7：扩数据后 raw −2.90pp、gate +0.00pp 全坍缩；真相 = **Rocket 在 71% cell 就是 oracle，UCR 是 Rocket-饱和 benchmark**，routing ceiling≈1.88pp）|
| **#90** | reviewer2 | **Outcome Model**：reactive→predictive，决策前 simulate reward（比 world model 轻）| 📋 P3 future |

**本轮最终结论**（finish4 §1-8，Round 9 完整）：在 UCR 类 benchmark 上，**factor 加和 / 独立回归 / learned-belief（手特征+MOMENT）/ 校准 gate 四范式 + 扩数据(10→22ds) 全部 ≈/< Rocket**。
逐步钉死的诊断链：F-R9.1 learned belief 不是银弹 → F-R9.2 belief 强度是反指标 → F-R9.5 换强 embedding 更差（z 非瓶颈）→ F-R9.6 校准 gate 坍缩 Rocket →
**F-R9.7（最终）：根因不是样本复杂度，是 Rocket 在 UCR 上已近 oracle（71% cell / 17-22 ds / gap 1.88pp）——这是 forecasting TSH 在分类的镜像（Rocket 饱和 UCR）**。
**转向**：routing 提分需 expert 真分化域（多变量 UEA DTW>Rocket / 工业 blind-test），非 UCR；或把价值主张改为"**诚实诊断 base 何时已是 oracle**"（saturated 域最优 router = 不 route）。
→ 不再在 UCR 上换 router 形式；下一步 = 转 UEA/工业域 或 收敛论文（把 §8 写成 "Saturation in classification" 一节）。

**#91 转多变量 UEA（done=饱和复现）**：M11 在 14 UEA 数据集（81 cell）learned-belief LODO = **−4.83pp**（比 UCR 更差），
rocket 在 **75% cell 是 oracle**、11/14 数据集 top、gap 2.34pp（finish4 §9/F-R9.8）。"UEA DTW>Rocket"是 3 数据集 cherry-pick。
**Rocket-饱和 = domain-invariant（UCR↔UEA 一致）** → 与 forecasting TSH 共同构成统一论点"**TSFM/SOTA-base 饱和跨任务跨 domain**"。
**最终转向**：标准学术 benchmark（UCR/UEA）全被 Rocket 饱和，找 routing headroom 需**工业 blind-test 或在线反馈部署**；
否则**收敛论文**——Round 9 的最大产出是把 Saturation 从"forecasting 假说"升级为"跨任务跨 domain 经验定律"，论文 §5 可据此重写。

---

## 🗄️ 历史存档（仅供回查，非当前任务）

> ⚠️ 下方 thesis（B7 击败 Rocket）已被 F-R8.7 推翻，保留作演进记录。

### Phase 1-6 时间线（关键 finding）

| 阶段 | 内容 | 关键 finding |
|---|---|---|
| Phase 1-5 | Forecasting 边界：6 数据集 × 6 方法 144 cells | no-method-dominates；v11 0W/1L/23T parity Chronos-2；CRPS +0% |
| Phase 6.1 | RCA natural（v1 10-dim → v2 12-dim Curator）| R1 40% vs LLM-direct 0%（+40pp）|
| Phase 6.3 | TaskB UCR 210 cells | Rocket 87.5%；Agent direct −33pp；MOMENT 在 BeetleFly/BirdChicken 反超 |
| Phase 6.4a | Synthetic 4-class 84 cells | B6 33.7% vs Rocket 50.6%（−17pp）→ Agent direct 结构性弱 |
| Phase 6.4b | Agent-as-Router | B7v2 86.66%；B7v3 ~~88.42%~~ → **86.91% 去泄漏** |
| Round 7-8 | M1 Meta-bandit / M2 Model 淘汰 / M3 EB prior / M4 per-regime decay / M7-P1 anomaly | 见 finish3 §0 F-R8.1-8.4 |
| Round 8（2026-05-30）| **M8 Factor Attribution + framing / M9 泄漏修复** | F-R8.5-8.9 |

### Learned Routing 4-Level 路径（feedback 第四轮，归入 P3 future）

- L1 Learned Margin（task #50）：(Curator feat, optimal_margin) 回归头替换常量 margin — 已有 `learned_margin.py`，+0.49pp LODO
- L2 Meta-Router v2（task #51）：confidence-gated override + 扩训练数据 + regression-mode — `meta_router_v2.py`
- L3 Contextual Bandit RL（task #52）：每 cell 作 context，classifier 作 arm，Thompson 在线学
- L4 Meta-Learning via TSFM Transfer（task #53）：MOMENT/Chronos-2 embedding 作 universal repr

### Engineering backlog（feedback 之后、优先级低于 P0-P3）

> feedback 明确"不要再堆模块"，以下归 P2/P3 之后：
> M5 Memory importance sampling · M6 Curator 25→18 维剪枝 · M7-Phase2 per-fault Memory ·
> M7-Phase3 LLM RCA agent · R6-E2 forecaster_reflect ADAPTTS_ACTION 集成 ·
> R6-E3 Drift Engine 双向 pred_residual_z。
> 详细设计要点见 git 历史版本 TODO.md（grep "Round 8+ 自演化路线候选"）。

### 长尾候选

MLE 替代 PAV 校准 · Telemetry Reservoir sampling · GMM 替代 KMeans regime ·
regime stale 自动 resurrect · Per-task RouterConfig 模板。

---

## 📁 不会忘的文件 hook

- **权威优先级**：本文件文首"🧭 feedback 改进路线图" + "📌 后续改进规划"
- **总纲**：`plan.md`（三层抽象 + 三任务诚实结果）
- **方法**：`method3.md §M8/§M9`（最新）/ `method2.md §10-§15`（Round 5-8）
- **实测**：`finish3.md §0` Findings 索引 F-R8.1–8.9
- **原始 review**：`feedback.md`（11 条硬伤）
- **下次会话第一句问"下一步"**：先看本文件"📌 后续改进规划"表，从序 1（#70/#71 收尾）开始
