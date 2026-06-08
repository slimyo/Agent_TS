# Finish v6 — Round 11 实测（决策机制：Trust-Aware Selective Action）

> 版本：2026-06-02 起
> 方法写在 `method6.md`；主线见 `plan.md §零`。本文件承接 Round 11（E1 起）实测 + Findings F-R11.x。
> 前置：`finish4.md`（Round 9 / belief inversion F-R9.2）/ `finish5.md`（Round 10 / saturation F-R10）。

---

## 0. 本轮目标

主线（plan §零）：以**决策机制本身**为研究对象。E1 攻 F-R9.2 belief inversion —— 把"置信度
(confidence)"与"可信度(trust=1−epistemic)"解耦，验证 trust 能否排序"这次偏离对不对"，
进而把 inversion 校正。环境：本地 `tsci`，复用 m10 数据底座（UCR-22 LODO，128 cell）。

> ⚠️ **诚实结论**：本轮有**一个正向机制信号**（bagged ensemble 自身就抑制了单头 inversion）+
> **一个被数据卡住的验证**（饱和 UCR 上正确偏离太少，n=1/16，无法统计性证明 trust 有效）。
> → E1 的真正价值实验必须搬到**非饱和域（forecasting）**或 deviation-rich 设定。这是 method6
> 主线的第一个"环境约束"发现，本身指导后续 E2/E3 的数据选择。

---

## 1. E1 · Trust ≠ Confidence（UCR-22 LODO, K=20 bagged heads）

设置：K=20 bootstrap belief head（CE→oracle-winner），三种 epistemic（MI / JS / conformal），
trust = 1 − rank-normalized epistemic。决策 = argmax b̄；偏离 = 选非 rocket。

### 1.1 主结果

| 指标 | 值 | 解读 |
|---|---|---|
| 偏离数 | **16 / 128** | 饱和 UCR 上 ensemble 已很少偏离（符合 F-R9.7）|
| 偏离正确数 | **1 / 16** | **正确偏离极少**（饱和域几乎没有真头寸）|
| conf when correct / wrong | **0.70 / 0.625** | **未 inversion**（method4 单头是 0.51<0.79）|
| inversion_coef (raw) | +0.10 | 已非负（ensemble 修了单头的反指标）|
| Trust-AUC (ensemble MI) | **0.633** | trust 排序对错 > 随机，但靠单个正例 |
| Trust-AUC (conformal) | 0.40 | conformal trust 在此**失败** |
| dev_precision raw → trust-gated | 0.062 → 0.125 (n=8) | 门控后翻倍，但样本太小 |

### 1.2 两个发现（一正一卡）

**F-R11.1（正向机制信号）：bagging 本身抑制 belief inversion。**
method4 单头 learned belief 的 confidence 是**反指标**（对 0.51 < 错 0.79，F-R9.2）。换成 K=20
bootstrap ensemble 的均值 belief 后，**inversion 消失**：conf-when-correct 0.70 **>** wrong 0.625，
inversion_coef 从负转 +0.10。机制解释：单头的"高置信伪模式"是 high-variance 的，bagging 平均
把它压低 → ensemble 均值 belief 比单头更诚实。**这是 method6 主线 A 的第一个机制证据：
inversion 部分来自 epistemic variance，ensemble 是有效的一阶修正。**

**F-R11.2（被数据卡住）：饱和 UCR 无法验证 trust 的判别力。**
trust-AUC=0.633 看似 >0.5，但**正确偏离只有 1 个**（trust 当对/错 0.230 vs 0.172，几乎不分）——
AUC 完全由这单个正例驱动，**统计上不可信**。根因仍是 F-R9.7：UCR 饱和→真头寸≈0→偏离几乎全错→
没有足够"正确偏离"样本来证明 trust 能识别它们。conformal trust 直接失败（AUC 0.40）也是同因
（饱和域 nonconformity 信号弱）。**结论**：E1 的判别力验证**必须换到非饱和域**（forecasting，
chronos2 仅 25% 是 oracle，F-R10.1 已证有头寸），否则任何 trust 机制都缺正样本可验。

---

## 2. 对主线（method6）的影响

- **主线 A 部分成立**：ensemble 修 inversion 这一步**在饱和域已可见**（F-R11.1），不依赖头寸。
  → method6 §2.5.4 的"K 头 ensemble"应设为默认（不只是 epistemic 估计的副产品，本身是 inversion 修正器）。
- **环境约束**：trust 的**判别力**（能否排序对错）需要"正确偏离"正样本，饱和域给不了（F-R11.2）。
  → E2 决策相图、E3 LLM defer 都应**在 forecasting（非饱和）域为主场**跑，UCR/UEA 只作"饱和对照组"。
- **方法不变，数据重排**：method6 框架无需改；finish6 后续实验把主战场从分类移到预测。

---

## 3. Findings F-R11.x

| ID | 内容 | 来源 |
|---|---|---|
| **F-R11.1** | **Bagging 抑制 belief inversion**：method4 单头 learned belief confidence 是反指标（对0.51<错0.79，F-R9.2）；换 K=20 bootstrap ensemble 均值 belief 后 inversion 消失（对0.70>错0.625，inversion_coef −→+0.10）。机制：单头高置信伪模式是 high-variance 的，bagging 平均压低之。→ inversion 部分源于 epistemic variance，ensemble 是有效一阶修正（method6 主线 A 第一证据）| §1 |
| **F-R11.2** | **饱和域无法验证 trust 判别力**：UCR-22 上仅 16 偏离、1 正确，trust-AUC=0.633 全靠单正例驱动（trust 对/错 0.23 vs 0.17 几乎不分），conformal trust 失败(AUC 0.40)。根因 F-R9.7：饱和→无头寸→偏离全错→缺正样本。→ **trust 判别力实验必须搬到非饱和的 forecasting 域**（chronos2 仅 25% oracle，有头寸）| §1.2 |
| **F-R11.3** | **方法论**：method6 框架经 E1 验证可跑且诚实，但暴露"验证环境"约束——决策机制的**修正**部分（ensemble 修 inversion）饱和域可验，**判别**部分（trust 排序对错）需非饱和域。E2/E3 据此把主场设在 forecasting，分类作对照 | §2 |

---

## 5. E1-libplus · 扩候选库到 10 分类器（un-saturate UCR → trust 判别力被激活）

**动机**：F-R11.2 指出 trust 判别力需非饱和域。除了换 forecasting，另一条路是**扩候选库**——
把分类器从 5 个扩到 **10 个**（+catch22 / mantis_1nn / mantis_lr / minirocket / weasel，
`expand_clf_library_sweep.py`，22 数据集 × 660 行全跑通，0 NaN）。

### 5.1 扩库直接 un-saturate 了 UCR（关键前提）

| 库规模 | rocket 即 oracle 占比 | 平均 oracle gap | 性质 |
|---|---|---|---|
| 5 分类器（旧）| **71%** | 1.88pp | 饱和 |
| **10 分类器（新）** | **38%** | **5.87pp** | **未饱和（3× 头寸）** |

oracle winner 分布（132 cell）：rocket 50 / **minirocket 27** / **weasel 19** / moment_1nn 7 /
catch22 7 / euclid 7 / mantis_lr 6 / mantis_1nn 5 / moment_logreg 3 / dtw 1。
→ **minirocket、weasel 真的在大量 cell 上强过 rocket**，UCR 不再是 rocket 一家独大。

### 5.2 E1 在 10-clf 库上的结果（trust 机制被激活，正向）

| 指标 | 5-clf（F-R11.1/2）| **10-clf（本节）** |
|---|---|---|
| 偏离数 | 16/128 | **58/132** |
| 偏离正确数 | 1（无法统计）| **26（vs 错 32，平衡）** |
| 偏离精度 (raw) | 0.062 | **0.448** |
| inversion_coef (raw) | ≈0 | **+0.475**（明确正相关）|
| conf 对/错 | 0.70/0.625 | **0.607/0.483**（无 inversion）|
| **Trust-AUC (conformal)** | 0.40（失败）| **0.796** ✅ |
| Trust-AUC (ensemble MI) | 0.633（单例驱动）| 0.55（弱）|
| conformal trust 对/错均值 | — | **0.744 / 0.624**（真分离，n=26/32）|

### 5.3 Findings

| ID | 内容 | 来源 |
|---|---|---|
| **F-R11.4** | **扩候选库直接 un-saturate 基准**：5→10 分类器使 rocket-即-oracle 从 71%→**38%**、oracle gap 1.88→**5.87pp**（minirocket 赢 27 / weasel 19 cell）。证明 F-R9.7 的"UCR 饱和"**是候选库太小的产物**，不是 UCR 本质——**扩库是 un-saturate 的有效手段**，比换数据集更直接 | §5.1 |
| **F-R11.5** | **trust 判别力在非饱和库上被激活（主线 A 关键正果）**：10-clf 库上 **conformal Trust-AUC=0.796**（5-clf 仅 0.40），conformal trust 当偏离对(0.744) 显著 > 错(0.624)，26/32 平衡样本非单例驱动。同时 belief inversion 彻底消失（inversion_coef +0.475，conf 对 0.607>错 0.483）。**验证 method6 主线 A：trust≠confidence 在有真头寸时确实能排序"这次偏离对不对"**——F-R11.2 的"环境约束"被扩库解除 | §5.2 |
| **F-R11.6** | **conformal trust ≫ ensemble-分歧 trust**：10-clf 上 conformal AUC 0.796 vs ensemble-MI 0.55。conformal 的 nonconformity（"这点在不在已见分布内"）比集成内部分歧更能识别"该不该信这次偏离"。→ method6 §2.5.4 应**以 conformal 为默认 trust 估计**，ensemble 分歧降为辅助 | §5.2 |

---

## 6. E2 · 决策相图（(saturation × trust) → 最优动作，10-clf 库 LODO）

**设置**（`m14_decision_phase_diagram.py`）：每 cell 算 saturation（gap 回归归一）+ trust（conformal）+
belief-shape，标"事后最优动作"a*∈{commit-base, deviate}，在"belief 想偏离"的 58 个决策 cell 上画相图、
比固定策略 vs 学习 π（决策树，LODO）。

### 6.1 相图（行=saturation 低→高，列=trust 低→高；值=P(deviate 最优)）

```
 sat[0..1]   全空（10-clf 库下几乎无低饱和预测 cell）
 sat[2]                       trust[2]=0.33(6)
 sat[3]  0.19(21) | 0.42(12) | 0.00(16) | 0.00(3)
```
**所有决策 cell 落在高-saturation 带**——即便候选库已 un-saturate（全局 gap 5.87pp），
**per-cell 的 saturation 回归仍把绝大多数 cell 判为"高饱和/低头寸"**（头寸集中在少数 cell）。

### 6.2 策略对比（58 个决策 cell 的实际 acc）

| 策略 | acc | 说明 |
|---|---|---|
| always-commit (=base) | 0.884 | 守默认 |
| always-deviate | 0.831 | 全偏离（亏 5pp）|
| **rule-phase**（trust≥0.5 & sat≤0.5 才偏离）| **0.884** | = always-commit（规则学到"几乎别偏离"）|
| learned-π（决策树）| 0.879 | 略**差**于守默认（小过拟合）|
| oracle-action（上界）| 0.888 | 完美决策天花板 |

决策 cell 上 deviate-最优占比仅 **19%**；oracle 天花板比守默认只高 **0.4pp**。

### 6.3 关键负结果（F-R11.7）—— 与 E1 的张力及其澄清

E1（F-R11.5）说 conformal trust-AUC=0.796 能排序"偏离对/错"；但 E2 在决策 cell 上发现：
**deviate-最优的 trust 均值 0.335 反而 < commit-最优的 0.426**——**trust 在这里不分离**。两者**不矛盾**，
区别在"对"的定义：
- E1 "correct deviation" = 选中 ≥ base（**含平局**，且统计全部 58 偏离）→ trust 能排"会不会变差"。
- E2 "deviate optimal" = 选中 **严格 >** base（真有净增益）→ 这是更苛刻的"值不值得偏离"，trust 排不动。

**机制结论**：trust 善于回答**"偏离会不会伤"（避险，E1 ✓）**，但不善于回答**"偏离能不能赚"（获利，E2 ✗）**。
前者是损害控制（F-R10.2 的精确化），后者需要预测净增益——而净增益逐 cell 不可学（F-R10.3 的再次印证）。

### 6.4 Findings

| ID | 内容 | 来源 |
|---|---|---|
| **F-R11.7** | **决策相图退化为"几乎恒 commit"，trust 只避险不获利**：10-clf un-saturated 库上，per-cell saturation 回归仍把决策 cell 全压到高饱和带，deviate-最优仅占 19%，oracle 天花板比守默认仅高 0.4pp；规则相图/学习 π 都 ≈ 或 < always-commit。与 E1 不矛盾——trust 能排"偏离会不会变差"(E1 含平局, AUC 0.80)，但排不动"偏离能不能净赚"(E2 严格>base, trust 0.335<0.426)。**决策机制的可达价值是避险(损害控制)，非获利**；获利需逐-cell 净增益预测，仍不可学（F-R10.3 再证）| §6.1-6.3 |

---

## 7. E3 · LLM defer（DeepSeek 第二意见 → 补 trust 排不动的"获利"判断）

**动机**（接 F-R11.7 开口）：trust 只能避险、排不动"偏离能不能净赚"。本实验在 58 个决策 cell
（belief 想偏离）上，让 **DeepSeek** 用**纯训练侧信息**（Curator 画像 + 各候选训练集平均 CV +
belief 倾向 + "历史多少比例 base 即最优"）做结构化第二意见，输出 JSON `{should_deviate, toward, ...}`。
诚实：无 test 泄漏；`deepseek-chat`，磁盘缓存。`m15_llm_defer.py`。

### 7.1 策略对比（58 决策 cell 实际 acc）

| 策略 | acc | 偏离判断精度 | 召回（11 个真获利偏离）|
|---|---|---|---|
| always-commit (=base) | 0.884 | — | — |
| always-deviate | 0.831（亏 5.3pp）| 0.19 | 11/11 |
| trust-gate(≥0.5) | ~0.86 | **0.08** | **2/11** |
| **LLM-defer (DeepSeek)** | **0.8839** | **0.267** | **4/11** |
| oracle-action（上界）| 0.888 | 1.0 | 11/11 |

### 7.2 关键结果（F-R11.8）—— LLM 部分补上了"获利"判断

- **DeepSeek 在"该不该偏离"上 2–3× 优于 trust**：精度 0.267 vs trust 0.08，召回 4/11 vs trust 2/11。
  即 **LLM 第二意见确实能识别一部分 trust 排不动的"净获利偏离"**——正是 F-R11.7 的开口。
- **且不伤 base**：llm-defer 0.8839 ≈ always-commit 0.884（只发 15 次偏离请求，远少于 always-deviate 58 次），
  **既避开了 always-deviate 的 −5.3pp 灾难，又抓到了 4 个真获利**。
- **但远未解决**：召回仅 36%（11 个里抓 4 个），且总头寸本就只有 0.4pp → 净增益微小。
  LLM 是"更好的偏离提议器"，不是"获利问题的解"。

### 7.3 三动作的能力分工（method6 主线收敛）

E1+E2+E3 合起来给出决策机制的清晰分工：

| 信号/动作 | 擅长 | 实证 |
|---|---|---|
| **saturation `ĝ(z)`** | 判"该不该进场"（全局头寸）| F-R10.1 corr 0.37 |
| **trust（conformal）** | **避险**："偏离会不会变差" | F-R11.5 AUC 0.80 |
| **LLM defer** | **获利提议**："哪个偏离可能净赚"（部分）| F-R11.8 精度 2-3× trust |
| commit-base | 兜底：饱和/无把握时守默认 | F-R11.7 几乎恒 commit |

### 7.4 Findings

| ID | 内容 | 来源 |
|---|---|---|
| **F-R11.8** | **LLM 第二意见部分补上 trust 排不动的"获利"判断**：DeepSeek 用纯训练侧信息（CV+画像+案例先验）在决策 cell 上判"该不该偏离"，精度 0.267、召回 4/11，**2–3× 优于 trust-gate(精度 0.08、召回 2/11)**；llm-defer 实际 acc 0.884 ≈ 守默认且只发 15 次偏离（避开 always-deviate 的 −5.3pp 灾难）。→ **LLM 是更好的"偏离提议器"，能识别一部分 trust 识别不了的净获利偏离**，但召回仅 36%、总头寸 0.4pp，**部分缓解而非解决"获利"问题**。三动作分工成形：saturation 进场 / trust 避险 / LLM 获利提议 / commit 兜底 | §7.1-7.3 |

---

## 8. 机制落地 · trust-gate 接入 test/ 多-agent 系统（research → 部署验证）

把 method6 主线 A 的 trust 避险门接进 `test/` 系统的 RouterAgent（`test/agents.py` + `test/pipelines.py`），
在三任务全量上验证"避险"机制是否真能减少误偏离：

- **trust 实现（部署版，无 test）**：`trust = CV饱和惩罚 × fold稳定性`。**CV 饱和惩罚**是关键——
  few-shot 下 best 候选 CV≥0.95 视为饱和（已撞天花板、失去判别力，F-R8.8），trust 线性压向 0。
- **偏离条件升级**：CV margin **AND** trust≥0.6（F-R11.5 "confident AND trustworthy"）。

| 任务 | trust-gate 前 | trust-gate 后 | 偏离 |
|---|---|---|---|
| 分类（48 cell）| 89.61%（−0.42pp）| **90.02%（+0.00pp）** | 2→**0** |
| 检测（12）| +0.00pp | +0.00pp（不变）| 0→0 |
| 预测（36）| −1.6% rel | −1.6%（不变）| 4→4 |

**F-R11.9**：**trust 避险门在部署系统上消除了误偏离**。分类的 2 次误偏离全是 BirdChicken N=10→moment_1nn，
其 CV=1.000（饱和）但 test 仅 0.80——典型 CV↔test 背离（F-R8.8/F-R9.2）。CV-饱和惩罚把 trust 压到 0.00 → 拦截 →
守 Rocket 0.90，**分类从 −0.42pp 升到精确 base 持平（+0.00pp），无任何 cell 受损**。检测/预测的安全偏离未被误伤
（仍 0/4）。→ **验证 F-R11.5 的工程价值：trust 的"避险"在真实多-agent 系统里把"近-base"变成"精确-base"，
代价为零**。这印证主线结论——决策机制可达的稳健价值是损害控制（避险），且**可直接落地**。

---

## 4. 下一步

1. **E1-续（forecasting 版）**：把 m13 的 belief/trust 机制接到预测域（候选 chronos2/bolt/arima/...，
   "正确偏离"= 偏离后 MAE < base）。预期：非饱和域有足够正确偏离 → 能真正测出 trust-AUC 是否 >0.5。
2. **E2 决策相图**（`m14`）：(saturation × trust) → action，主场 forecasting + 分类对照。
3. **E3 LLM defer**（`m15`）：低 trust + 低 saturation 时 DeepSeek 第二意见。
4. method6 §2.5.4 把 "K 头 ensemble" 标为默认 belief（F-R11.1 依据）。

> **诚实定位**：E1 没"一击证明 trust 万能"，但给出**一个干净机制结论**（bagging 修 inversion）+
> **一个环境约束**（判别力需非饱和域）。这正是机制研究该有的样子——把"trust 有没有用"拆成
> "修正/判别"两问，分别定位它们成立的条件，而非笼统宣称成功。
