# Agent + 时序 · 工业少样本多任务自适应系统

> PPT 文字稿（**工程导向**）。听众：工程/研究所人员。弱化理论，突出工程可行性。
> 指标均为**绝对值 / 上下界 / 估计类型**（基于已知数据集先验）。实验为**测试阶段**结论，后续会调整。
> 更新：2026-06-02。配套：`method6.md` / `finish3-6.md`。

---

## Slide 1 · 总览

### 工业背景（为什么要做）
- **即插即用刚需**：新产线 / 新工艺 / 新传感器接入即用 —— 要求 **zero-shot（零样本）+ 长视野预测**，没有时间为每个新场景重训模型。
- **单一 TSFM 不够用**：同一个时序基础模型在不同领域**饱和度差异极大** —— Weather 上 Chronos-2 是 SOTA，但 Exchange/ECL 上落后**领域专家模型 35–44%**。没有"一招鲜"。
- **工业更看重置信区间**：部署决策对 **CRPS（概率质量）比 MAE（平均绝对误差）更敏感** —— 工程师要的是"预测 ± 多少"，而不只是一个点。

### 解决思路
工业时序三大任务 —— **预测 / 分类 / 机械诊断** —— 统一于**一套"选择性自适应路由"系统**：

> **同一框架：诊断序列 → 估计该不该换模型 → 在模型库里选 → 永不显著伤害默认模型。**

| 任务 | 工业场景 | 默认强模型 | 系统做的事 |
|---|---|---|---|
| 预测 | 负荷/产量/库存 | Chronos-2 (TSFM) | 有把握时换更优 TSFM，否则守默认 |
| 分类 | 故障类型/质量分级 | Rocket | 多分类器路由 + 少样本兜底 |
| 机械诊断 | 振动/电流异常+根因 | 规则基线 | LLM 结构化根因 + 规则兜底 |

**统一点**：三任务共用 *诊断 → 决策(该不该换) → 选模型 → 解释* 的同一条流水线，换任务只换模型库。

---

## Slide 2 · 任务一 · 预测 (Forecasting)

- **背景**：新产线/新传感器要 zero-shot 长视野预测；单一 TSFM 跨域饱和度差异大（Weather 上 Chronos-2 SOTA，Exchange/ECL 落后专家模型 35–44%）；工业看 CRPS（要置信区间）甚于 MAE。
- **输入**：少样本（N=10–100）单/多变量序列；输出未来 H 步点预测 + 置信区间。
- **模型库**：Chronos-2 / Chronos-Bolt / TimesFM-2 / Moirai(2) / TiRex / Toto / Time-MoE / Sundial / Timer-S1 + ARIMA/naive。
- **决策**：walk-forward 验证 + 强阈值门控，**只在明显更优时**才偏离 Chronos-2（避免小验证窗"假赢"）。
- **工程结论（绝对/上界）**：
  - 默认 Chronos-2 在 **25%** 的 cell 上已是最优（基于 6 数据集先验）。
  - **理论路由增益上界 ≈ 19% rel-MAE**（oracle 对 Chronos-2，即"完美选模型"的天花板）。
  - 系统实测：与 Chronos-2 **持平**（不伤默认），偏离仅在高把握时发生。

---

## Slide 3 · 任务二 · 分类 (TSC)

- **输入**：few-shot（每类 3–10 样本）单/多变量；输出类别。
- **模型库（10 个）**：Rocket / MiniRocket / WEASEL / Catch22 / MOMENT(1nn/lr) / Mantis(1nn/lr) / DTW / Euclid。
- **决策**：训练集内 CV + 少样本兜底（每类 <7 时强制默认，CV 不可信）。
- **工程结论（绝对/上界，基于 22 数据集先验）**：
  - 默认 Rocket 绝对准确率 **86.1%**；**oracle 上界 90.2%** → **理论路由增益上界 +4.1pp**。
  - 模型库大小决定头寸：5 模型时 Rocket 在 71% cell 最优；**10 模型时降到 38%**（MiniRocket/WEASEL 接管）。
  - → **工程含义：扩模型库是提升头寸最直接的手段。**

---

## Slide 4 · 任务三 · 机械诊断 (Anomaly + RCA)

- **输入**：振动/电流/温度时序（+ 可选维修文本）；输出 异常 + 故障类型 + 严重度。
- **方法**：规则基线兜底 + LLM Agent 结构化根因推理（LLM 不做预测，只做诊断+解释）。
- **工程结论（绝对/双面）**：
  - 合成 4-class 故障：默认 Rocket 绝对 **50.6%**，oracle 上界 52.0%（**理论增益上界仅 +1.4pp** → 该域路由头寸小，宜守默认）。
  - 根因(RCA)：结构化 Agent 比无结构 LLM **+40pp**；但对工程规则基线 **−37pp** → **保留规则基线，LLM 提供可解释根因而非取代规则**。
  - OOT（未知故障）检测：abstain + 强 LLM + 数据集先验 三路叠加可达 **100% recall**。

---

## Slide 5 · 总体模型框架

> 〔此页放 AI 生成的模型总览图 —— 占位，后续替换〕
>
> ```
> ┌─────────────────────────────────────────────────────────┐
> │                                                           │
> │            [ 模型框架总览图 · 占位 ]                       │
> │      (使用本页下方提示词由 AI 生成后插入)                  │
> │                                                           │
> └─────────────────────────────────────────────────────────┘
> ```
>
> 五阶段流水线（文字版，供对照）：
> ① 诊断/编码 → ② 信念+可信度估计（该不该换） → ③ 决策（用默认/换模型/集成/求助LLM）
> → ④ 执行（永不伤默认） → ⑤ 解释（决策依据可审计）

---

## Slide 6 · 难点

| 难点 | 工程表现 | 应对 |
|---|---|---|
| **默认模型已很强（饱和）** | 强 base 在 71–75% cell 上即最优，路由头寸 <5pp | 检测饱和→守默认，不硬路由 |
| **逐 cell 最优模型难预测** | 换模型常"换错"，离线选不准 | 强阈值门控 + 只在高把握偏离 |
| **置信度不可信** | 模型"越自信越容易错" | 用 conformal 可信度替代裸置信度 |
| **极少样本** | 每类 <7 时 CV 噪声大、误判 | N-条件兜底，退回默认 |
| **数据评估易作弊** | 记忆若用测试信息→虚高 +1.5pp | 严格留一/只用训练侧（已审计修正）|

> 核心工程原则：**系统永不显著伤害默认模型**——最差也就是"等于直接用 Chronos-2/Rocket"。

---

## Slide 7 · 可行性（硬件 / 部署）

| 维度 | 需求 |
|---|---|
| **CPU-first** | Chronos-2 / Rocket / MiniRocket / WEASEL / Mantis：单 cell **亚秒~十几秒**，纯 CPU 可跑 |
| **GPU（可选）** | 仅 8B 级大 TSFM（Timer-S1 等）需消费级 GPU；实测 2× RTX 5070 Ti(16GB) 足够 |
| **内存/显存** | 边缘可部署轻量库（Rocket+MiniRocket+Euclid 全 CPU）；大 TSFM 走远程 |
| **LLM** | DeepSeek（开源/免费 API，带磁盘缓存，重复零成本）；可换本地 Ollama |
| **依赖** | conda 单环境；模块正交可插拔（库可增删，不重训）|
| **零样本** | 所有模型**不重新训练**，新产线接入即用 |
| **可解释** | 每次决策输出：该不该换的依据 + 选中模型 + 反事实理由（可审计）|

---

## Slide 8 · 关键数值锚点（绝对 / 上下界 / 估计）

| 数值 | 类型 | 含义 |
|---|---|---|
| **86.1%** | 绝对 | 分类默认 Rocket 准确率（22 数据集先验）|
| **90.2%** | 上界 | 分类 oracle 准确率（完美选模型天花板）|
| **+4.1pp** | 增益上界 | 分类路由理论最大增益 |
| **19% rel-MAE** | 增益上界 | 预测路由理论最大增益（oracle vs Chronos-2）|
| **25% / 38% / 75%** | 估计（先验占比）| 默认模型即最优的 cell 占比：预测 / 分类(10库) / 检测 |
| **+1.4pp** | 增益上界 | 检测路由理论最大增益（头寸小→宜守默认）|
| **+40pp / −37pp** | 绝对（双面）| RCA Agent vs 无结构 LLM / vs 规则基线 |
| **亚秒~十几秒** | 估计 | 主力模型单 cell CPU 延迟 |

> 说明：以上为**测试阶段**基于现有公开数据集先验的估计，工程上界用于评估"路由最多能带来多少"，
> 后续随模型库/数据扩展会更新。

---

## 附录 · 模型框架图 · AI 生成提示词（顶会论文风格）

> 用于 Slide 5 占位图。建议生成英文标注（避免中文渲染问题），16:9，矢量论文图风格。

**英文提示词（推荐直接用）：**
```
A clean top-conference-style system architecture diagram for a machine-learning
paper, horizontal left-to-right data flow, 16:9, white background, flat vector
style, thin labeled arrows, muted academic palette (slate blue, teal, warm gray),
crisp sans-serif English labels only, no photorealism.

LEFT — input: a small multivariate time-series panel (3 stacked signal curves)
labeled "Few-shot Series x  (N=3–100)".

STAGE 1 box "Curator / Encoder": gear+magnifier icon, output a short feature
vector chip labeled "z = f(x)  (trend / season / noise / complexity)".

STAGE 2 box "Belief + Trust Estimator": two parallel sub-bars —
top "Confidence  b(M|z)  (which model)",
bottom "Trust = 1 − epistemic  (conformal)";
plus a small dial "Saturation  ĝ(z)  (is base already enough?)".

STAGE 3 central decision diamond "Policy  π(saturation, trust, shape)"
with five labeled output arrows fanning right:
"commit-base", "deviate", "ensemble", "explore", "defer-to-LLM".

RIGHT — a "Model Library" shelf of stacked cards:
forecasting cards (Chronos-2, TimesFM-2, Moirai, TiRex, Timer-S1) and
classification cards (Rocket, MiniRocket, WEASEL, MOMENT, Mantis, DTW);
the selected card glows with a check mark.

BOTTOM — a thin feedback arrow from "Outcome" back to the estimator,
labeled "observe (offline / online)", and an "Explainer" note card
"decision card: why-not counterfactual + retrieved cases".

THREE downstream task icons at far right bottom:
"Forecasting (curve + confidence band)", "Classification (colored buckets)",
"Mechanical Diagnosis (gear + warning triangle)".

Guiding principle banner under the policy block (small text):
"never significantly hurt the default model".
Minimal text, every arrow labeled, high-contrast, publication-quality.
```

**中文备注（给你看，不进图）**：图要传达三件事——(1) 一条流水线接三任务；
(2) 决策核心是"该不该换模型"(saturation+trust)；(3) 模型库可插拔 + 永不伤默认 + 可解释。

**备选极简版提示词**（若上图太密）：
```
A minimalist 3-tier ML system diagram, white background, flat vector, 16:9,
English labels. Tier 1 "Series x → Encoder z". Tier 2 "Belief + Trust + Saturation
→ Policy π" (one rounded box). Tier 3 "Model Library shelf → selected model →
{Forecasting, Classification, Diagnosis}". A dashed feedback arrow loops outcome
back to Tier 2. Three colors only (blue boxes, gray shelf, orange feedback arrow),
clean publication figure quality, no clutter.
```

---

## 附录 · 每页讲稿（口语化）+ 术语解释

> 给演讲者照着念的逐页口语稿；每页末尾解释当页出现的术语。

### Slide 1 总览 · 讲稿
"我们先说为什么做这件事。工业上经常遇到三种'新'——新产线、新工艺、新换的传感器，它们一接上来就要能用，
没时间为每个场景重新训练模型，这叫**零样本**。而且工业关心的是**长视野预测**，要预测很远。
问题是：现在最火的时序基础模型（TSFM），换个领域表现差很多——比如在天气数据上 Chronos-2 是最强的，
但到了汇率、电力数据上，它比那些专门做这个领域的模型差 35% 到 44%。所以**没有一个模型通吃**。
还有一点，工厂要的不是一个干巴巴的预测数字，而是'预测值正负多少'这个区间，所以我们更看 CRPS 这个指标。
我们的思路就是：不造新模型，而是搭一套**会自己挑模型**的系统——先诊断这条数据，判断该不该换模型，
在一个模型库里挑，而且保证一条底线：**永远不会比直接用默认模型更差**。三个任务都用这一套流程。"
- **术语**：*TSFM*=时序基础模型（像 GPT 那样预训练好、能直接用的时序大模型）；
  *zero-shot 零样本*=不用本场景数据训练，直接拿来用；*SOTA*=当前最好水平；
  *CRPS*=连续排序概率分数，衡量"预测的概率分布准不准"，比只看点误差(MAE)更全面；
  *MAE*=平均绝对误差，预测值和真值差多少的平均。

### Slide 2 预测 · 讲稿
"第一个任务是预测。输入是很少的几段历史数据，输出未来一段 + 置信区间。我们手里有十几个预测模型
（Chronos-2、TimesFM、Moirai、TiRex 这些大模型，加上传统的 ARIMA）。系统的做法是：
拿训练数据的尾巴做个'小考'(walk-forward 验证)，只有当别的模型**明显**考得更好，才换掉默认的 Chronos-2，
否则就守着它——因为小考样本少，容易'蒙对'，乱换反而坏事。
结论用大白话说：默认模型在 1/4 的情况下本来就是最优的；就算我们有上帝视角每次都选对，
最多也只能把误差再降 19%——这是**天花板**；我们实测做到的是'和默认持平、但绝不更差'。"
- **术语**：*walk-forward 验证*=用历史数据的后半段模拟'未来'来试各模型；
  *oracle/上界*=假设每次都选到最优模型的理想成绩，是理论天花板，实际达不到；
  *rel-MAE*=相对误差降幅（百分比）。

### Slide 3 分类 · 讲稿
"第二个任务是分类，比如判断是哪种故障。难点是**样本极少**，每类可能就 3 到 10 个。
我们有 10 个分类器（Rocket、MiniRocket、WEASEL、MOMENT 这些）。系统先在训练集内部交叉验证估每个的准度，
但如果每类少于 7 个样本，交叉验证不可信，就直接用默认的 Rocket 兜底。
关键发现：默认 Rocket 准确率 86%，如果每次都选对能到 90%，也就是路由最多帮我们提 4 个百分点。
还有个很实用的点——**模型库越大，默认模型越不够用**：只有 5 个模型时 Rocket 在 71% 情况下最好，
扩到 10 个模型后掉到 38%，因为 MiniRocket、WEASEL 在很多情况下接管了。所以**多放几个模型，提升空间就大**。"
- **术语**：*few-shot 少样本*=每个类别只有很少训练样本；*交叉验证(CV)*=把训练数据轮流当测试来估准度；
  *Rocket/MiniRocket/WEASEL/MOMENT*=不同的时序分类算法；*pp*=百分点。

### Slide 4 机械诊断 · 讲稿
"第三个任务是从振动、电流这些信号判断机器出了什么问题，不只是'有没有异常'，还要给**根因**——
是轴承坏了还是转子失衡。我们的做法是：**规则基线兜底 + 大模型(LLM)做结构化根因分析**，
注意 LLM 不负责预测，只负责'讲清楚为什么'。
结论是两面的：在我们造的合成故障上，路由空间很小（最多提 1.4 个点），所以这个场景建议老老实实守默认。
根因分析上，带结构的 Agent 比让大模型瞎猜强 40 个点，但比工程师写的规则还差 37 个点——
所以**规则不能扔，LLM 的价值在于给出能看懂、能扩展的根因解释**，而不是取代规则。"
- **术语**：*RCA*=根因分析（找出故障的根本原因）；*LLM*=大语言模型；
  *规则基线*=工程师按经验写的判断规则（如阈值）；*OOT*=超出已知故障类型（遇到没见过的故障）。

### Slide 5 总体框架 · 讲稿
"这页是系统全貌（图后面补）。一句话：数据进来，先**诊断/编码**，再估两件事——
'哪个模型可能最好'(信念)和'这次判断该不该信'(可信度)，外加'当前默认是不是已经够好'(饱和度)；
然后**决策**：用默认、换模型、做集成、还是求助大模型；执行时守住底线**绝不比默认更差**；
最后给一张**决策卡**说明为什么这么选，可以审计。整套换任务只换右边的模型库。"
- **术语**：*信念(belief)*=系统估计的'各模型谁会赢'的概率；*可信度(trust)*=这次估计本身可不可靠；
  *饱和度(saturation)*=默认模型是不是已经接近最优（越饱和越该守默认）；*集成*=多个模型结果合在一起。

### Slide 6 难点 · 讲稿
"五个真实难点，都是我们踩过的坑：①默认模型本来就强，多数情况换不动；②就算想换，'这条数据该用哪个模型'
其实很难离线预测准；③模型给的置信度不可信，经常'越自信越错'，所以我们改用 conformal 这种更靠谱的可信度；
④样本太少时交叉验证会骗人，必须兜底；⑤评估时一不小心就会'作弊'（用到测试信息），我们专门审计修正过。
贯穿所有的工程铁律是这句：**系统永远不会明显比默认模型差**，最坏也就是'等于直接用 Chronos-2 或 Rocket'。"
- **术语**：*饱和*=默认模型已接近最优、没多少提升空间；*conformal*=一种给预测配可靠'可信度分'的统计方法；
  *数据泄漏*=不小心把测试答案用进了训练/决策，导致虚高。

### Slide 7 可行性 · 讲稿
"落地很现实。主力模型（Chronos-2、Rocket、MiniRocket 这些）**纯 CPU 就能跑**，单次几秒到十几秒；
只有 80 亿参数的超大模型才需要显卡，一张消费级 5070 Ti（16G）就够，还能放远程。
大模型用 DeepSeek 的免费 API，带缓存，重复调用不花钱，也能换成本地 Ollama。
所有模型**不用重训**，新场景接上就用；每个决策都有可解释的依据，能审计。边缘设备可以只装轻量的几个模型。"
- **术语**：*CPU-first*=优先用普通处理器、不强依赖显卡；*VRAM*=显卡内存；
  *Ollama*=本地跑大模型的工具；*正交可插拔*=模块互不依赖，可单独增删。

### Slide 8 数值锚点 · 讲稿
"最后给几个可以记住的数字，分三类：**绝对值**（默认模型实际多准）、**上界**（完美选模型的天花板）、
**估计**（基于现有数据集先验的占比）。比如分类：默认 86%、天花板 90%、所以路由最多帮 4 个点；
预测最多帮 19%；检测只有 1.4 个点（所以那个场景别折腾）。要强调：这些是**测试阶段**的数，
基于现在的公开数据集，随着模型库和数据扩展还会变——它们的用途是帮工程上判断'路由值不值得做'。"
- **术语**：*绝对值/上界/估计*=三类指标口径；*先验*=基于已有数据观察到的规律；
  *增益上界*=理论上最多能提升多少（达不到，仅作参考）。

---

**End of ppt.md**（2026-06-02 · 工程导向重写；含背景 + 逐页讲稿）
