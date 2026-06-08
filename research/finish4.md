# Finish v4 — Round 9 实测（Learned Belief）

> 版本：2026-05-31 起
> 方法写在 `method4.md`；外部 review 是 `feedback_m3.md`。本文件承接 Round 9（M10 起）实测 + Findings F-R9.x。
> 前置：`finish3.md`（Round 7-8 / F-R7.x-F-R8.x）。

---

## 0. 本轮目标

回应 feedback_m3 三位 reviewer **一致核心批评**："belief 不是学出来的"。实现 **M10 Learned Belief Model**
并诚实实测（UCR-10 LODO），同时产出 reviewer 要的两件分析料：**belief 校准失败**（缺陷3）和
**expert-switching per-dataset 分布**（数据缺陷）。环境：本地 `tsci`（py3.10，纯 CPU 手特征，无 LLM/MOMENT/GPU）。

> ⚠️ **本轮是诚实负结果**：learned belief（浅层 logistic + 30-d 手特征）**未击败 Rocket**（最好 −0.8pp），
> 且 belief 校准严重失败（自信地犯错）。这反而是比"刷一个 +Xpp"更有价值的论文素材——它精确定位了
> learned-belief 范式的瓶颈。**不夸大、不掩盖。**

---

## 1. M10 · Learned Belief Model · UCR-10 LODO（56 cell）

**设置**：`b(M|z)=softmax(LogisticRegression(z))`，z=30-dim `featurize_cell`，训练目标 CE→oracle-winner；
leave-one-dataset-out（10 数据集轮流留出，held-out 整集不进训练）。候选 = {rocket, moment_1nn,
moment_logreg, dtw_1nn, euclid_1nn}。对照 Rocket-alone + Oracle（per-cell 最优）。

### 1.1 τ-sweep（confidence gate = belief margin）

| τ | learned-belief acc | vs Rocket | Oracle | regret | 偏离数 | 偏离精度 | belief(偏离对) | belief(偏离错) |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 82.09% | **−1.20pp** | 86.37% | 4.28pp | 10 | 0.200 | 0.507 | **0.791** |
| 0.1 | 81.82% | **−1.46pp** | 86.37% | 4.54pp | 9 | 0.111 | 0.545 | **0.791** |
| 0.2 | 82.27% | **−1.02pp** | 86.37% | 4.10pp | 8 | 0.125 | 0.545 | **0.831** |
| 0.3 | 82.49% | **−0.80pp** | 86.37% | 3.88pp | 7 | 0.143 | 0.545 | **0.878** |

**Rocket-alone = 83.29%**（UCR-10 上，比 UCR-5 的 87.53% 低 ~4pp——less-saturated，oracle 86.37%，有 routing headroom）。

### 1.2 关键观察（全部诚实）

- **learned belief 在所有 τ 下都输 Rocket（−0.80 ~ −1.46pp）**。即便换到有 headroom 的 UCR-10，
  浅层 learned belief 也没创造净价值。这与 method3 factor-router（UCR-5 honest −0.62pp）量级相当——
  **换用"学出来的 belief"本身并未翻越 Rocket 这道线**（修正了早前误记的 +0.70pp；以本表为准）。
- **confidence gate 的悖论**：τ↑ → 准确率略升（82.09→82.49%）但**偏离精度反而下降**（0.20→0.143）。
  原因见 §2.1：belief 在**错的时候反而更高**，所以"belief margin>τ"门控**留下的恰是高信念的错误偏离**、
  挡掉的是低信念的（部分正确）偏离。准确率小升只因偏离总数减少（10→7）、净负偏离被砍掉。
- **regret 3.88~4.54pp**：离 oracle 还很远，routing 远未被"解决"。

---

## 2. 归因分析（缺陷3 + 数据缺陷）

### 2.1 belief miscalibration —— 本轮最重要发现（F-R9.2）

| 偏离 default 且… | 平均 belief(选中)（τ=0）|
|---|---|
| **对**（chosen acc ≥ rocket）| 0.507 |
| **错**（chosen acc < rocket）| **0.791** |

**belief 在偏离错误时反而显著更高（0.791 vs 0.507）**，且随 τ 收紧，错误偏离的 belief 升到 0.878。
即模型不仅"自信地犯错"，而且**越自信越容易错**——belief 强度与正确性**负相关**。配合偏离精度仅 0.20
（≈5 次偏离错 4 次），结论刺眼：

> **learned belief 的绝对强度是反指标。** softmax 头在 oracle-winner CE 下学到的"高置信"，
> 对应的往往是**训练集里被过度代表的伪信号**（如某些 z 区域历史上 moment 偶然赢过），
> 在 held-out 数据集上系统性失效。这正是 feedback_m3 缺陷3 预言的 calibration 失败，且比预期更严重。

→ 直接推论：method4 §4 的 belief-shape 决策**不能信 belief 强度**，必须先叠校准层（且 F-R8.10 已证
isotonic 离线校准对 CV 无效——belief 校准可能同样困难）。这是 learned-belief 范式的真正瓶颈，
**比 +Xpp 更有论文价值**。

### 2.2 per-dataset —— expert-switching 实证（F-R9.3）

| dataset | cells | Rocket | learned(τ0) | Oracle | oracle-winner 分布 |
|---|---|---|---|---|---|
| GunPoint | 6 | 82.4 | **85.6 (+3.2)** | 91.6 | dtw 3 / rocket 1 / moment_lr 1 / euclid 1 |
| TwoLeadECG | 6 | 96.7 | 96.8 (+0.1) | 98.9 | rocket 4 / moment_1nn 2 |
| ECG5000 | 2 | 91.5 | 91.5 (0) | 93.4 | moment_1nn 2 |
| Strawberry | 2 | 95.1 | 95.1 (0) | 95.7 | rocket 2 |
| Wafer | 6 | 96.5 | 96.5 (0) | 98.3 | moment_1nn 3 / euclid 2 / dtw 1 |
| Crop | 2 | 47.9 | 47.9 (0) | 50.6 | rocket 1 / euclid 1 |
| Coffee | 6 | 96.4 | 95.8 (−0.6) | 98.8 | rocket 3 / moment_lr 1 / dtw 1 / euclid 1 |
| ECG200 | 6 | 79.3 | 74.2 (−5.1) | 84.5 | rocket 2 / euclid 2 / moment_1nn 1 / dtw 1 |
| BeetleFly | 6 | 83.3 | 78.3 (−5.0) | 93.3 | moment_1nn 4 / rocket 2 |
| BirdChicken | 6 | 77.5 | 65.8 (−11.7) | 87.5 | rocket 4 / moment_1nn 2 |

**5 个 classifier 全部在 ≥1 数据集上当 oracle-winner**（rocket / moment_1nn / moment_logreg / dtw_1nn /
euclid_1nn）——UCR-10 是真正的 expert-switching benchmark（对比 UCR-5 上 rocket 主导）。这正面回答
reviewer "不要再刷 UCR-5、去找 expert-switching 数据"。

- **唯一稳健正向**：GunPoint +3.2（dtw 主场，learned belief 正确识别了 dtw 偏好）。
- **重灾区**：BirdChicken −11.7、ECG200 −5.1、BeetleFly −5.0——共同点是 oracle 其实多为 rocket 或
  moment，但 learned belief 以高信念误偏离到错误 expert（§2.1 校准失败的犯罪现场）。
- **结论**：learned belief 能感知 expert-switching（GunPoint 证明了上限存在），但**校准太差导致
  净负**——它在"该偏离"时偶尔对、在"不该偏离"时自信地错。这是清晰、可改进的失败模式。

---

## 3. 与 method3 / method 既有路线的衔接

| 路线 | 结果 | M10 的位置 |
|---|---|---|
| method3 factor-router (UCR-5 honest) | −0.62pp | factor 加和 |
| method3 #72 CV 校准 (UCR-5) | −0.87pp（更差，F-R8.10）| factor + 离线校准 |
| meta_router_v2 (per-clf 回归, LODO) | ~−0.01pp tied（finish §4.9）| 独立回归 score |
| **M10 learned belief (UCR-10 LODO)** | **−0.80pp(τ0.3)** | **学出来的分布** |

**诚实总结**：从 factor 加和 → 独立回归 → learned belief，三种范式在诚实协议下**都没稳定击败 Rocket**。
这强烈印证 **TSFM/SOTA Saturation**：在单模型已强的 TSC 上，routing 的增益空间被 expert-switching 程度
死死限制，而**当前表示 z（30-d 手特征）+ 浅层 belief 不足以可靠识别该偏离的少数 cell**。
reviewer 的"learned belief 是分水岭"判断方向正确，但**本轮证明：换成 learned belief 还不够——
真正的瓶颈是 (a) 表示 z 太弱（手特征），(b) belief 校准失败**。

---

## 6. #86b · 表示升级实验 —— MOMENT embedding z（检验 F-R9.4 假说）

**动机**：F-R9.4 推断 learned belief 输 Rocket 的瓶颈之一是 **z 太弱（30-d 手特征）**。本实验把 z 换成
**MOMENT-1-small frozen encoder 的 cell embedding**（512-d，mean-pool over training samples），其余
完全照 M10 的 UCR-10 LODO 协议（`experiments/m10b_moment_belief.py`，56 cell，GPU）。

### 6.1 结果（决定性反驳"表示是瓶颈"）

| z 表示 | belief acc | vs Rocket | regret | 偏离精度 | belief(偏离对/错) |
|---|---|---|---|---|---|
| 30-d 手特征（M10）| 82.49%(τ0.3) | **−0.80pp** | 3.88pp | 0.143 | 0.545 / 0.878 |
| **512-d MOMENT（M10b）** | **79.33%(all τ)** | **−3.96pp** | 7.04pp | 0.250 | **0.846 / 0.963** |

**MOMENT embedding 不仅没修好，反而把 routing 做得更差（−0.80 → −3.96pp）**，且 τ-gate 完全失效
（所有 τ 同分——belief 太尖锐，margin 恒 >0.3，gate 形同虚设）。belief 校准更糟：错误偏离 belief 高达
**0.963**（vs 正确 0.846）—— embedding 让模型"更自信地犯错"。

### 6.2 per-dataset（tau=0，实测值）

| dataset | Rocket | MOMENT-belief | Δ |
|---|---|---|---|
| ECG200 | 79.2 | 79.7 | +0.5 ✓ |
| BirdChicken / ECG5000 / GunPoint / TwoLeadECG / Wafer | — | =Rocket | **+0.0**（未偏离）|
| Crop | 50.3 | 47.6 | −2.6 ✗ |
| Strawberry | 72.7 | 67.6 | −5.1 ✗ |
| **Coffee** | 98.8 | **85.7** | **−13.1** ✗ |
| **BeetleFly** | 83.3 | **65.8** | **−17.5** ✗ |

**注意**：与手特征 M10 不同，MOMENT-emb belief 在 GunPoint/BirdChicken 等 **完全不偏离**（=Rocket），
唯一净正只有 ECG200 +0.5；而在 Coffee（−13.1）、BeetleFly（−17.5）灾难性误偏离——这两个 Rocket/Moment
主场被 embedding 拽向错误 expert。28 次偏离仅 7 次对（精度 0.25），偏得多、错得狠。

### 6.3 解读（F-R9.5）

表示升级**证伪了"z 是主瓶颈"的乐观假设**。MOMENT embedding 对**分类判别**很强（B4 baseline 在
BeetleFly/BirdChicken 上反超 Rocket），但对 **"哪个 classifier 会赢"这个元任务**反而引入更多
spurious 方向：512-d 高维 + 56 训练 cell → LODO 下严重过拟合，belief 又尖锐又错（连手特征版在
GunPoint dtw 主场拿到的 +3.2 都丢了，退化为不偏离或错偏离）。**真瓶颈不是 z 的判别力，而是
(a) meta 监督信号太少（56 cell）(b) belief 校准缺失**。这把下一步从"换更强 embedding"明确转向
**"校准 + 扩元数据"**（#86c + 更多数据集），是比继续刷表示更有价值的方向定位。

---

## 7. #86c · Calibrated selective-deviation gate（直击 F-R9.2）

**动机**：F-R9.2 证 belief 强度是反指标（错时更高），F-R9.5 证换强 embedding 更糟。两者共同指向
"不是模型不够强，是 belief 不可信"。本轮**不换模型**，改两件事（`experiments/m10c_selective_gate.py`）：

1. **Bagged belief**：25 个 bootstrap LogisticRegression 头的 softmax 平均，压掉 F-R9.5 的尖锐过拟合方差。
2. **Calibrated gate**：训二级分类器 `g(belief-shape) → P(本次偏离正确)`，**只在 g≥阈值时才偏离 rocket**，
   而不是信 raw belief 强度。gate 输入是 belief 的形状量（entropy/gini/top-gap/tail/强度），不含原始 z。

**诚实性 = nested LODO**（关键）：外层留出 D_test；gate 的训练标签由训练集上的**内层 LODO**
生成（out-of-fold 偏离对/错对），gate 从不见 D_test，belief head 从不预测自己训过的 cell。无泄漏。

### 7.1 结果

| 系统 | acc | vs Rocket | regret | 偏离数 | 偏离精度 |
|---|---|---|---|---|---|
| Rocket-alone | 83.29% | — | 3.08pp | — | — |
| M10 raw-belief（手特征 τ0.3）| 82.49% | −0.80pp | 3.88pp | 7 | 0.143 |
| M10b raw-belief（MOMENT-emb）| 79.33% | −3.96pp | 7.04pp | 28 | 0.250 |
| **M10c gated（bagged + 校准 gate）** | **82.84%** | **−0.45pp** | 3.53pp | **1** | 0.000 |

阈值 0.5/0.6/0.7 结果**完全相同**（gated 82.84%，dev=1）——gate 学到的结论很坚决：**UCR-10 上几乎
没有可信的偏离机会**，于是把 28→1 次偏离几乎全部抑制，坍缩回近-Rocket。

### 7.2 per-dataset（threshold=0.6）

| dataset | Rocket | gated | Δ | 偏离 |
|---|---|---|---|---|
| **BeetleFly** | 83.3 | 79.2 | **−4.2** | 1（唯一，且错）|
| 其余 9 个数据集 | — | =Rocket | **+0.0** | 0 |

唯一存活的偏离是 **BeetleFly N=3 s=1 → moment_1nn**，gate 给它 **p=0.99** 却仍判错——这恰是 F-R9.2 的
余响：连校准 gate 也在这个极端 few-shot（N=3）cell 上被高置信地骗过。其余偏离全被 gate 正确否决。

### 7.3 解读（F-R9.6）

**校准 gate 把灾难性误偏离基本消除**（M10b −3.96 / M10 −0.80 → −0.45pp），代价是**几乎放弃所有偏离**
（含 GunPoint +3.2 的真机会）。这是一个**诚实但悲观**的结论：在 56-cell 的元监督预算下，
"哪次偏离可信"本身**学不出来**——gate 唯一稳健的策略是"几乎永远别偏离"，等价于"用 Rocket"。
这与 F-R8.10（CV 离线校准也救不了 TSC router）同构：**TSC routing 的瓶颈是元监督的样本复杂度，
不是决策器/表示/校准器的形式**。要真正解锁，唯一出路是 **大幅扩元数据**（几百个数据集的 oracle 标签，
即 reviewer 说的"找更多 expert-switching 数据"）或 **在线真实反馈**（F-R8.8）——离线小样本下，
任何 router 形式都收敛到 ≈Rocket。**§8 直接做了这个扩数据实验来检验该假说。**

---

## 8. #89 · 扩元数据实验 —— 检验"样本复杂度是瓶颈"假说（F-R9.6）

**动机**：F-R9.6 推断 56-cell 元监督太少是 router 学不出来的根因，预言"扩元数据应解锁 routing"。
本实验跑 **12 个新 UCR 数据集 × 3N × 2seed × 5 分类器**（`experiments/expand_oracle_sweep.py`，360 rows），
把 oracle 标签库从 **10 数据集/56 cell → 22 数据集/128 cell（2.3×）**，然后重跑 M10/M10c（`M10_EXPANDED=1`）。

### 8.1 结果（假说被部分证伪）

| 库规模 | Rocket | M10 raw-belief(τ0.3) | M10c 校准 gate | oracle | rocket=oracle 占比 |
|---|---|---|---|---|---|
| 10 ds / 56 cell | 83.29% | 82.49%(−0.80pp) | 82.84%(−0.45pp, dev1) | 86.37% | — |
| **22 ds / 128 cell** | **86.08%** | 83.18%(**−2.90pp**) | **86.08%（+0.00pp, dev0）** | 87.96% | **71%** |

**扩数据没有解锁 routing**：raw belief 反而更差（−0.80→−2.90pp，更多数据 = 更多 Rocket 主场误偏离），
**校准 gate 在 22 数据集上偏离 0 次、完全坍缩到 Rocket（+0.00pp）**。阈值 0.5/0.6/0.7 同结果。

### 8.2 为什么（F-R9.7 —— 比"样本复杂度"更深的诊断）

扩数据揭示了真相：**这些 benchmark 是 Rocket-主导的，不是 expert-switching 的**。

- **rocket 在 128 cell 里 71% 就是 oracle**（91/128），22 数据集里 **17 个 Rocket 是 top oracle**。
- oracle gap 仅 **1.88pp**（86.08→87.96）——可路由的空间本就极小。
- 只有 5/22 数据集 Rocket 非最优（BeetleFly/ECG200/Wafer/ItalyPowerDemand/DistalPhalanx），且都是窄优势。

→ **校准 gate 的"永远 defer Rocket"不是失败，是对的**：在 Rocket-主导的分布上，最优 router 就该恒选 Rocket。
F-R9.6 说"样本复杂度"只对了一半——更准确是：**UCR 这类 benchmark 上 Rocket 已近 oracle，routing 的
ceiling（1.88pp）本身就接近零**，再多同分布数据只会强化"别偏离"。这与 TSFM Saturation Hypothesis 完全一致，
只是把"Chronos-2 饱和 forecasting"换成"**Rocket 饱和 UCR 分类**"。

### 8.3 含义（钉死 + 转向）

本实验**证伪了"扩同分布数据能救 TSC router"**，把结论收紧为：
> **在 Rocket-主导的 UCR 类 benchmark 上，任何诚实 router 的上限 ≈ Rocket+1.88pp，且校准后最优策略是恒选 Rocket。**
> routing 真正有价值需要**分布上 expert 真正分化**的场景（reviewer 的"expert-switching 数据"），
> 而 UCR/UEA 主流集不满足。下一步不是再扩 UCR，而是 **(a)** 转向 expert 真分化的领域（多变量 UEA 已显示 DTW>Rocket，
> 或工业 blind-test），或 **(b)** 把 routing 价值主张从"提分"彻底改为"**诚实诊断 base model 何时已是 oracle**"
> （即论文 §5.3.2 "no method dominates" 的反面：**在 saturated 域，最诚实的 router 就是不 route**）。

---

## 4. Findings F-R9.x

| ID | 内容 | 来源 |
|---|---|---|
| **F-R9.1** | **learned belief（softmax-CE 到 oracle-winner）在诚实 UCR-10 LODO 上未击败 Rocket**（τ-sweep 全负：−0.80 ~ −1.46pp，最好 τ=0.3 −0.80pp，regret 3.88pp）。即便换到有 routing headroom 的 less-saturated benchmark、即便 belief 是"学出来的"（回应 feedback_m3 缺陷1），也没翻越 Rocket。与 method3 factor-router(−0.62pp)、meta_router_v2(tied) 量级一致 → **换 learned belief 本身不是银弹** | §1 |
| **F-R9.2** | **本轮最重要：learned belief 的绝对强度是反指标**。偏离 default 时，belief 在**错的时候(0.791)显著高于对的时候(0.507)**，且 τ 越严错误偏离 belief 越高(0.878)；偏离精度仅 0.20。即模型"**越自信越容易错**"——CE 头学到的高置信对应训练集过度代表的伪信号，held-out 系统性失效。这正是 feedback_m3 缺陷3 的 calibration 失败，且比预期严重。**推论**：belief-shape 决策不能信 belief 强度；learned-belief 范式的真正瓶颈在**校准**，不在排序 | §2.1 |
| **F-R9.3** | **UCR-10 是真正 expert-switching benchmark**：5 个 classifier 全部在 ≥1 数据集当 oracle（dtw→GunPoint, moment→BeetleFly/Wafer/ECG5000, euclid→ECG200, rocket→Coffee/TwoLeadECG/BirdChicken）。learned belief 在 dtw 主场 GunPoint **+3.2pp** 证明感知 expert-switching 的**上限存在**，但在 rocket/moment 主场误偏离致命（BirdChicken −11.7）。正面回答 reviewer "去找 expert-switching 数据"——价值区间存在，但需更强表示+校准才能稳定开采 | §2.2 |
| **F-R9.4** | **三范式收敛于同一边界**：factor 加和 / 独立回归 / learned belief 在诚实协议下都 ≈ 或 < Rocket。强化 Saturation：TSC routing 增益 ∝ benchmark expert-switching 程度，且受限于 (a) z 表示力 (b) belief 校准。reviewer "learned belief 是分水岭"方向对，但本轮证明它是**必要非充分** | §3 |
| **F-R9.5** | **表示升级（#86b）证伪"z 是主瓶颈"**：把 z 从 30-d 手特征换成 512-d MOMENT embedding，belief routing **从 −0.80pp 恶化到 −3.96pp**（regret 3.88→7.04pp），且 belief 更尖锐更错（错误偏离 belief 0.878→0.963，τ-gate 完全失效）。MOMENT 对**分类判别**强（B4 baseline 证），但对**"谁会赢"的元任务**在 56-cell LODO 下严重过拟合、引入更多 spurious 方向。**真瓶颈不是 z 判别力，而是 (a) 元监督太少(56 cell) (b) belief 校准缺失** → 下一步从"换更强 embedding"转向"校准+扩元数据"（#86c）。这是本轮把研究方向钉死的关键负结果 | §6 |
| **F-R9.6** | **校准 gate 消除灾难误偏离但坍缩回 Rocket**：bagged belief + nested-LODO 训练的"P(偏离正确)"校准 gate 把 M10b −3.96pp / M10 −0.80pp 收回到 **−0.45pp**（28→1 次偏离），灾难性误偏离基本消除；但代价是连真机会（GunPoint +3.2）也一并放弃——gate 学到"UCR-10 上几乎没有可信偏离"，阈值 0.5/0.6/0.7 同结果。唯一存活偏离 BeetleFly N=3 gate 给 p=0.99 仍错（F-R9.2 余响）。**结论与 F-R8.10 同构**：离线小样本下任何 router 形式都收敛 ≈Rocket。当时归因为"样本复杂度"，§8 扩数据后修正见 F-R9.7 | §7 |
| **F-R9.7** | **扩数据证伪"样本复杂度是瓶颈"——真相是 benchmark Rocket-饱和**：把 oracle 库 10→22 数据集（56→128 cell, 2.3×），raw belief **反更差（−0.80→−2.90pp）**，校准 gate **偏离 0 次、+0.00pp 完全坍缩 Rocket**。根因诊断：22 数据集里 **rocket 在 71% cell 就是 oracle**、17/22 数据集 Rocket 是 top，oracle gap 仅 **1.88pp**——**UCR 是 Rocket-主导而非 expert-switching benchmark**。校准 gate "永远 defer Rocket" 不是失败而是**最优**（Rocket-饱和分布上恒选 Rocket 正确）。把结论从 F-R9.6 收紧为：**Rocket 在 UCR 上已近 oracle，routing ceiling≈1.88pp 本就接近零，扩同分布数据只强化"别偏离"**。这是 forecasting TSH（Chronos-2 饱和）在分类的镜像（**Rocket 饱和 UCR**）。真出路 = 转 expert 真分化域（多变量 UEA / 工业）或把 routing 价值改为"诚实诊断 base 何时已是 oracle" | §8 |
| **F-R9.8** | **Rocket 饱和是 domain-invariant——多变量 UEA 也一样**：在 14 个 UEA 多变量数据集（81 cell）上重做 learned-belief LODO，结果 **−4.83pp（比 UCR 更差）**，regret 7.17pp，偏离精度仅 0.33。saturation 描述统计与 UCR 几乎同构：**rocket 在 75% cell 就是 oracle**（61/81）、11/14 数据集 Rocket 是 top、gap 仅 2.34pp。"多变量 UEA DTW>Rocket"是 3 数据集（AtrialFibrillation/FingerMovements/NATOPS）的小样本 cherry-pick，全集上 routing 头寸极小。15 次偏离全集中在 3 个数据集的灾难误偏离（Cricket −49.8 / Heartbeat −24.2 / HandMovementDirection −3.4），其余 11 数据集 belief 正确 defer Rocket。**结论**：**Rocket-饱和跨 univariate(UCR)↔multivariate(UEA) 一致**，TSC routing 的 Saturation 不是单一 benchmark 偶然，是 Rocket 作为 strong base 的系统性属性 → 与 forecasting TSH 共同构成"**TSFM/SOTA-base 饱和**"的统一论文论点 | §9 |

---

## 9. 转向多变量 UEA · saturation domain-invariance（F-R9.8）

**动机**：F-R9.7 把 UCR 钉为 Rocket-饱和，并提"转多变量 UEA（曾报 DTW>Rocket）"。本实验在
**14 个 UEA 多变量数据集 / 81 cell**（`experiments/m11_uea_belief.py`，候选 {rocket,dtw,euclid}，
z=每通道 hand feature 跨通道 mean-pool，LODO）上重做 learned-belief 路由。

### 9.1 结果（saturation 跨 domain 复现）

| benchmark | Rocket | learned belief | oracle | regret | rocket=oracle | dev 精度 |
|---|---|---|---|---|---|---|
| UCR-22（univariate）| 86.08% | 83.18%(−2.90pp) | 87.96% | 4.78pp | 71% | 0.15 |
| **UEA-14（multivariate）** | **72.49%** | **67.66%(−4.83pp)** | 74.82% | 7.17pp | **75%** | 0.33 |

learned belief 在 UEA 上 **−4.83pp，比 UCR 更差**，τ-sweep 全同（belief 太尖锐，gate 不动）。

### 9.2 saturation 描述统计（与 UCR 几乎同构）

- **rocket 在 75% UEA cell 就是 oracle**（61/81），**11/14 数据集** Rocket 是 top oracle，gap 仅 **2.34pp**。
- 只有 3 数据集 Rocket 非最优：**AtrialFibrillation / FingerMovements / NATOPS**——正是此前"DTW>Rocket"
  报告的那批，证明那是**小样本 cherry-pick**，全集上 routing 头寸极小。

### 9.3 per-dataset（tau=0，偏离全是灾难）

15 次偏离**全集中在 3 个数据集**，且全是灾难误偏离：

| dataset | rocket | belief | Δ | dev |
|---|---|---|---|---|
| **Cricket** | 99.5 | 49.8 | **−49.8** | 6 |
| **Heartbeat** | 61.0 | 36.8 | **−24.2** | 3 |
| HandMovementDirection | 33.1 | 29.7 | −3.4 | 6 |
| 其余 11 数据集 | — | =Rocket | **+0.0** | 0 |

Cricket（rocket 99.5% 主场）被 belief 误判到非 rocket，崩到 49.8——又一个 F-R9.2"自信地犯错"实例。
其余 11 数据集 belief **正确 defer Rocket**（dev=0），与 §8 校准 gate 在 UCR 上的行为一致。

### 9.4 含义（统一论点成形）

UEA 实测把 Round 9 的核心发现从"UCR 特有"提升为 **domain-invariant**：
> **Rocket 作为 strong TSC base，在 univariate(UCR) 和 multivariate(UEA) 上都已近 oracle
> （71%/75% cell 是 oracle，gap 1.88/2.34pp），诚实 router 在两个 domain 上都收敛到 ≈Rocket 且最优策略是不 route。**

这与 forecasting 的 TSFM Saturation Hypothesis（Chronos-2 饱和）**完全平行**，二者共同支撑论文统一论点：
**"在 strong-base 时代（TSFM forecasting / Rocket TSC），LLM-Agent/learned-router 的诚实角色是
selective abstain 而非提分——且这一边界跨任务、跨 univariate/multivariate domain 不变。"**
（论文 §5.3.2 "no method dominates" 的精确反面：**在 saturated 域，最诚实的 router 就是不 route**。）

> **诚实定位**：本轮没找到 expert-switching 翻盘场景（UEA 全集也饱和），但这本身是**更强的论文素材**——
> 把 Saturation 从"forecasting 一个任务的假说"升级为"跨任务跨 domain 的经验定律"。下一步真要找 routing
> headroom，需离开标准学术 benchmark（UCR/UEA 都被 Rocket 饱和），转**工业 blind-test** 或 **在线反馈**部署场景。

---

## 5. 下一步（method4 §4-§8 + TODO #86-90）

按本轮实测**修正后**的优先级（数据说话，非 reviewer 原序）：

1. **表示 z 升级（最高优先，F-R9.4）**：把 z 从 30-d 手特征换成 **MOMENT/Chronos2 embedding**，
   重跑 M10 LODO。验证 reviewer 缺陷1 的"表示层断层"是否是真瓶颈（预期：embedding z 让 GunPoint 式
   正向扩散到更多数据集）。
2. **belief 校准（F-R9.2）**：在 learned belief 上叠校准层 / 用 ensemble 降方差；目标把"偏离对/错"的
   belief 强度分开（当前 0.507 vs 0.791 是**反的**，先做到正相关）。
3. **#86 belief-shape 决策**：形状特征已输出到 `m10_learned_belief.jsonl`；待 1+2 把 belief 修可信后再做。
4. **#88 M1 regret ablation** / **#87 regime-as-feature**：见 method4 §6/§7。

> **诚实定位**：本轮没做出性能突破（learned belief 输 Rocket），但做出了**方法论诊断**：
> 定位了 learned-belief 范式的两个真瓶颈（表示力 + 校准），并证伪了"换 learned belief 即可反超"的乐观假设。
> 论文价值 = 《Why learned belief routing still loses: a representation-and-calibration post-mortem》。
