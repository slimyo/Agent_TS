# Method v6 — The Decision Mechanism: Trust-Aware Selective Action

> 版本：2026-06-02（Round 11 起）
> 接续 `method4.md`（Round 9：learned belief 全证伪）/ `method5.md`（Round 10：saturation detection）。
> **配套**：`feedback_m4.md`（v4 review）/ `finish6.md`（Round 11 实测，待建）/ `plan.md §零`（研究主线）。
>
> **研究主线转向（plan §零）**：不再以"提升性能"为目标，而以 **method4 暴露的决策机制本身** 为研究对象。
> 即使性能持平，回答 **"一个自适应 agent 何时该行动、何时该退、凭什么相信自己的判断、如何为克制辩护"**
> 本身有发表价值。method6 把这个机制形式化、可证伪化。

---

## 0. Thesis（一句话）

> Round 9-10 证明：强 base 饱和域里，决策算子的最优解几乎恒为 abstain（F-R9.6）。
> 但 method4 同时暴露两个**机制级真现象**——belief 强度**反指标**（F-R9.2）、决策只用点估计（缺陷2）。
> method6 把研究对象锁定为**决策算子本身** `π(a | trust, shape, saturation)`：
> **不是"选哪个模型"，而是"凭什么相信、该采取哪类动作"**——
> 把 belief 的 *置信度(confidence)* 与 *可信度(trust/epistemic)* 解耦，
> 并把"belief 分布形状 → 动作"的映射刻画成一张可证伪的**决策相图（phase diagram）**。

---

## 1. 两条合并主线（方向 1 + 方向 2）

### 主线 A · Trust ≠ Confidence（攻击 F-R9.2 belief inversion）

**现象**：softmax-CE 学出的 `b(M|z)` 强度与正确性**负相关**——偏离错时 belief 0.79 > 对时 0.51（F-R9.2）。
**诊断**：raw belief 是 *aleatoric-blind* 的，它表达"我觉得是 X"，不表达"我该不该信我这次的判断"。
高 belief 往往对应 **训练分布里被过度代表的伪模式**（spurious），在 OOD cell 上系统性失效（F-R9.5 强化）。

**机制改造**：决策输入从单一 `b(M)` → **解耦双信号**：

$$\text{confidence} = \max_k b(M_k\mid z)\quad\text{（选谁）}\qquad
\text{trust} = 1 - \widehat{\text{epistemic}}(z)\quad\text{（信不信这次）}$$

epistemic 不确定性的三种估计（择优/对比，全部 LODO-honest）：
1. **Deep ensemble 分歧**：K 个 bootstrap belief head 的预测分布方差 / JS 散度。
2. **MC-dropout / 子采样分歧**：同一头多次扰动的 argmax 翻转率。
3. **Conformal nonconformity**：用训练集 split 算 nonconformity score，给每个决策一个"是否在已见分布内"的 p 值。

**决策规则**：`deviate` 仅当 `confidence 高 AND trust 高`；其余 commit-base。
**可证伪预期（核心图）**：按 trust gate 后，"偏离对/错"的 confidence 应从**负相关翻成正相关**——
即 inversion 被消除。**无论提不提分，这张"inversion → corrected"图本身是 method6 的主结论。**

### 主线 B · Shape → Action 决策相图（攻击缺陷2 + F-R9.6）

**现象**：method4 的 decide() 只取点估计（argmax + margin），丢掉了 belief 分布的全部形状信息。
**机制改造**：决策算子升级为显式策略，动作集扩展：

$$\pi: \big(\underbrace{\text{saturation}}_{\hat g(z)},\ \underbrace{\text{trust}}_{1-\text{epistemic}},\ \underbrace{\text{shape}(b)}_{\text{entropy/gini/top-gap/tail}}\big)\ \longrightarrow\ a\in\mathcal{A}$$

$$\mathcal{A} = \{\textsf{commit-base},\ \textsf{deviate},\ \textsf{ensemble},\ \textsf{explore},\ \textsf{defer-to-LLM/human}\}$$

**关键洞察（来自 F-R9.6）**：饱和域最优 `π` 几乎恒选 `commit-base`——所以本主线的价值**不是让它多动**，
而是**刻画动作边界**：在 **(saturation × trust)** 二维平面上画出最优动作的**相图**：

```
        trust 高 ┌─────────────┬─────────────┐
                 │  DEVIATE    │  COMMIT-BASE│   ← 饱和高：即使 trust 高也守 base
                 │ (有头寸+可信)│  (base≈oracle)│
        trust 低 ├─────────────┼─────────────┤
                 │ ENSEMBLE/   │  COMMIT-BASE│
                 │ DEFER-LLM   │  (别赌)      │
                 └─────────────┴─────────────┘
                  saturation 低    saturation 高
```

**可证伪预期**：(a) 存在一条清晰的最优-动作边界；(b) 该边界**跨任务（预测/分类/检测）形状一致**
→ 又一个 domain-invariant 机制结论，呼应 plan §零。

---

## 2. LLM 作为 "defer" 动作的执行器（DeepSeek reasoning）

动作集里的 `defer-to-LLM` 用 **DeepSeek** 落地（用户偏好 + 已实测可用）：
- 当 `trust 低 AND saturation 低`（有头寸但模型自己没把握）→ 把 cell 的 **Curator 画像 + 候选 CV + 历史相似案例**
  喂给 `deepseek-reasoner`，让它做**结构化第二意见**（不是直接预测，是审议"该不该偏离 + 偏向谁"）。
- **诚实边界**：LLM 只读训练侧信息（画像/CV/案例），绝不读 test（沿用 M9 协议）。
- **配置**（已实测，注意 env 陷阱）：
  ```python
  import os
  # demo/.env 先加载且 PROVIDER=zhipu、DeepSeek key 注释掉 → 必须显式注入 research/.env 的 key
  for line in open("research/.env"):
      if line.startswith("DEEPSEEK_API_KEY="):
          os.environ["DEEPSEEK_API_KEY"] = line.strip().split("=", 1)[1]
  os.environ["PROVIDER"] = "deepseek"
  os.environ["MODEL"] = "deepseek-reasoner"   # 思考模型；deepseek-chat 为快通道
  from research.utils.llm import chat_cached    # 带磁盘缓存，重复 prompt 零成本
  ```
- **研究问题**（非提分）：LLM 的"第二意见"能否**改善 trust 校准**（让低 trust 决策少犯错）？
  即 defer 是不是一个比 ensemble 更好的"低 trust 动作"？这是相图右下角的实验。

---

## 2.5 完整方法框架与模型结构（实现蓝图）

> 本节是 method6 的工程契约：给出端到端数据流、每个组件的精确输入/输出、数据结构、
> trust 估计的数学、决策算子 `π` 的具体形式、以及训练/推理协议。按本节即可直接编码。

### 2.5.1 端到端架构（5 阶段 pipeline）

```
 cell x                         ┌──────────────── 离线 LODO 训练（每个 held-out 域一套）────────────────┐
 (序列/few-shot)                │                                                                       │
   │                            │  历史 cells（其它域）→ 真实 per-model 表现 → 标签                       │
   ▼                            │     ├─ oracle_gap(x)         → 训 SaturationHead  ĝ_φ                  │
 ┌─────────────┐                │     ├─ winner one-hot(x)     → 训 K×BeliefHead   {b_θ^(j)}             │
 │ ① Encoder   │ z = f(x)       │     └─ split-conformal       → 存 nonconformity 分位数               │
 │  z ∈ R^d    │                └───────────────────────────────────────────────────────────────────┘
 └──────┬──────┘
        ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │ ② Belief + Trust 层（主线 A）                                                   │
 │   confidence: b̄(M|z) = mean_j softmax(b_θ^(j)(z))        ← K 头 ensemble 均值   │
 │   epistemic : u(z) = disagreement({b_θ^(j)(z)})           ← 三估计之一（§2.5.4） │
 │   trust     : t(z) = 1 − ǔ(z)                             ← u 经 held-out 归一   │
 │   saturation: s(z) = ĝ_φ(z)（method5 复用）              ← 预测 oracle gap       │
 └──────┬───────────────────────────────────────────────────────────────────────┘
        ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │ ③ 决策算子 π（主线 B）                                                          │
 │   state σ = (s, t, shape(b̄))                                                   │
 │   a = π(σ) ∈ {commit-base, deviate, ensemble, explore, defer-LLM}              │
 └──────┬───────────────────────────────────────────────────────────────────────┘
        ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │ ④ Executor（动作落地）                                                          │
 │   commit-base→base; deviate→argmax b̄ 的非 base; ensemble→CV 加权池;            │
 │   explore→ε 采样次优; defer-LLM→DeepSeek 第二意见（§2）                          │
 └──────┬───────────────────────────────────────────────────────────────────────┘
        ▼
 ┌─────────────┐
 │ ⑤ Reporter  │  决策卡：σ 三量 + 选中动作 + 反事实辩护 + 检索案例（可解释）
 └─────────────┘
```

**唯一的"运行时回路"**（区别于 method4 的一次性预测）：④ 执行后观测到的 outcome（在线设定，
方向5/future）可回灌更新 trust 的归一化分位 + π 的 bandit 统计。本期先做离线版（④ 后即停）。

### 2.5.2 核心数据结构（dataclass 契约）

```python
@dataclass
class CellInput:
    x: np.ndarray            # 序列 [L] 或 few-shot [N,L]/[N,C,L]
    task: str                # "forecasting" | "classification" | "detection"
    base_model: str          # 该任务的 strong default（chronos2 / rocket）
    candidates: list[str]    # 模型库（含 base）
    meta: dict               # {dataset, N, seed, season_m, ...}（仅元信息，无 test）

@dataclass
class BeliefState:
    z: np.ndarray                    # encoder 输出
    belief: dict[str, float]         # b̄(M|z)，K 头均值，∑=1
    confidence: float                # max_k belief
    epistemic: float                 # u(z) ∈ [0,1]
    trust: float                     # t = 1 − ǔ(z)
    saturation: float                # s = ĝ_φ(z) ∈ [0,1]（gap 归一）
    shape: dict[str, float]          # entropy / gini / top1_top2_gap / tail_mass
    per_head: list[dict[str,float]]  # K 个头各自的 belief（审计用）

@dataclass
class ActionDecision:
    action: str                      # commit-base|deviate|ensemble|explore|defer-LLM
    chosen_model: str                # 最终执行的模型（或 "ensemble:{...}"）
    state: dict                      # (s, t, shape) 快照
    justification: str               # 反事实辩护（Reporter）
    retrieved_cases: list[dict]      # LOCO 检索的相似历史 cell（解释 grounding）
    llm_opinion: Optional[dict]      # defer 时 DeepSeek 的结构化意见
```

### 2.5.3 ① Encoder `f(x) → z`

- **统一接口**：`encode(cell: CellInput) -> np.ndarray`。
- **三种 z**（消融对比，全部已在 repo 有实现）：
  - `hand`：`research/utils/series_features.featurize_cell`（30-d，默认，F-R9.5 证它不比 embedding 差）。
  - `moment`：`research/agent/representation.MomentEmbedding`（512-d，多变量 mean-pool）。
  - `chronos2`：`Chronos2Embedding`（768-d，长序列）。
- few-shot 分类：z = cell 内逐样本特征的均值 + 元信息（N/类、类数、长度、通道）。
- 多变量 [N,C,L]：逐通道特征跨通道 mean（沿用 m11 做法）。
- **诚实**：只用训练侧；z 不含任何 test 信息。

### 2.5.4 ② Belief + Trust 层（主线 A 的模型结构）

**BeliefHead（K 头 ensemble）**：每个头是一个轻量分类器 `b_θ^(j): R^d → Δ^{|M|}`
（softmax over 候选模型），训练目标 = oracle-winner 的 cross-entropy（method4 M10 已实现单头）。
> **⚠️ E1 实证（F-R11.1）**：K 头 ensemble 的均值 belief **本身就修正了单头的 belief inversion**
> （单头 conf 对0.51<错0.79 → ensemble 对0.70>错0.625）。故 **K 头 ensemble 设为默认 belief**，
> 不只是 epistemic 估计的副产品，而是 inversion 的一阶修正器。
- 实现：`sklearn` LogisticRegression / 小 MLP，K=10~25 个 bootstrap 头（method4 M10c 已有 bagging）。
- confidence = `max_k b̄(M_k|z)`，`b̄ = (1/K)∑_j softmax(b_θ^(j)(z))`。

**Epistemic 估计 `u(z)`（三选一，E1 对比）**：

| 估计器 | 公式 | 实现要点 |
|---|---|---|
| **(a) Ensemble 分歧** | `u = mean_pairwise_JS({softmax(b_θ^(j)(z))})` 或 `u = mean_j H(b^(j)) − H(b̄)`（mutual info）| K 头已有；JS/MI 直接算 |
| **(b) MC-dropout** | 单 MLP 头开 dropout，T 次前向，`u = argmax 翻转率` 或 预测方差 | 需把头换成带 dropout 的 torch MLP |
| **(c) Conformal** | split-conformal：训练分一半算 nonconformity `α_i=1−b(y_i\|z_i)`，`u(z)=` 新点 α 的分位排名 | 纯 numpy；给"是否在已见分布内"的 p 值 |

**归一化 ǔ**：u 在 held-out 训练 cells 上做 min-max 或 rank 归一 → `t = 1 − ǔ ∈ [0,1]`（部署可得，无 test）。

**Shape 特征**（决策算子输入）：`entropy = −∑ b̄ log b̄`、`gini = 1−∑ b̄²`、
`top1_top2_gap = b̄_(1) − b̄_(2)`、`tail_mass = ∑_{k≥3} b̄_(k)`。

**Saturation `s(z)`**：直接复用 method5 的 `ĝ_φ`（RandomForest 回归 oracle gap，LODO），gap 归一到 [0,1]。

### 2.5.5 ③ 决策算子 `π(σ) → a`（主线 B 的核心模型）

state `σ = (s, t, entropy, gini, top_gap, tail)`。**两种实现，并行对比**：

1. **规则相图版（可解释基线）**：手设阈值网格，把 (s, t) 平面切成动作分区（method6 §1B 那张图），
   用于**画干净相图 + 给 reviewer 看边界**。形式：
   ```
   if s ≥ s_hi:                      a = commit-base        # 饱和：F-R9.6 最优
   elif t ≥ t_hi and confidence_margin > m:  a = deviate    # 有头寸+可信
   elif t < t_lo and s < s_hi:       a = defer-LLM          # 没头寸把握→第二意见
   elif gini high (无明显赢家):       a = ensemble           # 分散→池化
   else:                             a = commit-base         # 损害控制兜底
   ```
2. **学习版（oracle-action 监督）**：对每个历史 cell 事后标"最优动作"
   `a* = argmax_a U(a, cell)`（U=该动作落地后的真实表现），训一个小分类器 `π_ψ(σ)→a`（LODO）。
   - **关键**：a* 用历史真实 outcome 标（离线经验，合法）；σ 全部 deployment-可得。
   - 学习版与规则版的 agreement + 各自 regret 对比 = E2 的核心结果。

**动作集语义**（Executor 落地）：
| action | 落地 | 何时最优（先验，待 E2 实证）|
|---|---|---|
| commit-base | 用 base_model | s 高（饱和）或兜底 |
| deviate | argmax b̄ 的非 base 模型 | t 高 + confidence margin 够 |
| ensemble | CV/belief 加权池（forecasting quantile pool / clf 软投票）| gini 高、无明显赢家 |
| explore | ε 采样次优模型（仅在线/有反馈设定有意义）| future（方向5）|
| defer-LLM | DeepSeek-reasoner 第二意见（§2）| t 低 + s 低（有头寸但自己没把握）|

### 2.5.6 ④ Executor + ⑤ Reporter

- **Executor**：`execute(action, cell) -> (prediction, model_used)`，调 `research` 模型库
  （forecasting `STRATEGY_FN` / clf `predict_with`）。任何异常→兜底 base（永不伤 base，F-R10.2）。
- **Reporter**（可解释一等公民）：输出 `ActionDecision.justification`，模板：
  > "选 {action}：饱和度 s={s:.2f}（{饱和/有头寸}）、可信度 t={t:.2f}（{可信/存疑}）；
  >  belief 形状 {分散/集中}；检索到 k 个相似历史 cell 中 base {n}/{k} 次≈oracle；
  >  若偏离到 {alt}，历史预期 regret={r:+.3f}。"
  - 检索 = LOCO（剔除自身，防 M9 泄漏）top-k 相似 z 的历史 cell。
  - defer 时附 DeepSeek 的结构化意见（JSON：`{should_deviate, toward, reason}`）。

### 2.5.7 训练 / 推理协议（诚实底线）

```
for held_out_domain in domains:                 # LODO 外层
    train_cells = all cells NOT in held_out_domain
    # 1) 训 K 个 BeliefHead（bootstrap, CE→winner）
    # 2) 训 SaturationHead ĝ_φ（RF 回归 oracle gap）
    # 3) split-conformal: train_cells 再分一半算 nonconformity 分位
    # 4) (学习版 π) 标 a* + 训 π_ψ
    for cell in held_out_domain:                # 留出域逐 cell 推理
        z = f(cell); 算 belief/confidence/epistemic/trust/saturation/shape
        a = π(σ); pred = execute(a, cell)
        记录 BeliefState + ActionDecision + 真实 outcome（仅评估，不进决策）
```
- **无泄漏铁律**：belief/trust/saturation/π 的训练标签都来自**其它域**的历史 outcome；
  conformal/epistemic 只用训练 split；LLM 只读训练侧画像/CV/案例；test 仅用于最后算指标。

### 2.5.8 模块 ↔ 已有代码映射（最大化复用，最小化新写）

| method6 组件 | 复用 | 新写 |
|---|---|---|
| ① Encoder | `utils/series_features` / `agent/representation` | — |
| ② BeliefHead (K 头) | `experiments/m10c_selective_gate`(bagging) | epistemic 三估计 + trust 归一 |
| ② SaturationHead | `experiments/m12_saturation_router`(ĝ_φ) | — |
| ③ 决策算子 π | method5 planner（规则版雏形）| 5-动作 π + 学习版 π_ψ |
| ④ Executor | `forecaster_reflect.STRATEGY_FN` / `clf_strategies.predict_with` | 统一 execute() |
| ④ defer-LLM | `utils/llm.chat_cached`(DeepSeek) | prompt 模板 + JSON 解析 |
| ⑤ Reporter | method3 M8 attribution 精神 | 反事实辩护 + LOCO 检索解释 |

> **总计新代码 ≈ 3 个实验文件**（m13/m14/m15，§5），核心组件 80% 是把 Round 9-10 已验证的零件
> 重新接成"trust-aware + 相图"的新决策回路——符合 plan §零"不堆模块，深耕机制"。

---

## 2.6 实现级细节（伪代码 / 超参 / prompt / 指标公式 / 算例）

> 本节把 §2.5 的契约降到"照抄即可跑"的粒度：可直接对照编码，无需再做设计决策。

### 2.6.1 标签构造（离线经验库 → 三类训练标签，全 LODO-safe）

输入：每个历史 cell 的 per-model 真实表现 `perf[cell][model]`（分类=acc 越大越好；
预测=−MAE 越大越好；统一记为"效用 U，越大越好"）。base 一定在 perf 里。

```python
def build_labels(cells, base):
    rows = []
    for c in cells:
        u = cells[c]                      # {model: utility}
        base_u  = u[base]
        oracle_u = max(u.values())
        winner   = max(u, key=u.get)      # oracle winner（BeliefHead 标签）
        gap      = oracle_u - base_u      # SaturationHead 标签（≥0）
        rows.append(dict(key=c, base_u=base_u, oracle_u=oracle_u,
                         winner=winner, gap=gap, u=u))
    return rows
```
归一：`gap` 在**训练域** min-max → `s∈[0,1]`（held-out cell 用训练域的 min/max 变换，不偷看自身）。

### 2.6.2 ② Belief + Trust（E1 主体）伪代码

```python
def fit_belief_heads(Z_tr, winner_idx_tr, K=20, seed0=0):
    heads = []
    rng = np.random.default_rng(seed0)
    n = len(Z_tr)
    for j in range(K):
        idx = rng.integers(0, n, n)                  # bootstrap
        if len(set(winner_idx_tr[idx])) < 2: continue
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Z_tr[idx], winner_idx_tr[idx])
        heads.append(clf)
    return heads

def belief_and_trust(heads, z, classes, u_norm):
    P = np.zeros((len(heads), len(classes)))
    for j,h in enumerate(heads):
        p = h.predict_proba([z])[0]
        for ci,c in enumerate(h.classes_): P[j, classes.index(c)] = p[ci]
    bbar = P.mean(0)                                  # 均值 belief
    confidence = float(bbar.max())
    # epistemic (a) ensemble 分歧 = mutual information = H(b̄) − mean_j H(b^j)
    H = lambda q: float(-(q*np.log(q+1e-12)).sum())
    epi = H(bbar) - np.mean([H(P[j]) for j in range(len(heads))])
    trust = 1.0 - u_norm(epi)                         # u_norm: 训练域 epi 的 rank/minmax 归一
    return bbar, confidence, float(epi), float(trust)
```
**三种 epistemic 都实现、E1 对比**：(a) 上面的 MI；(b) `mean_pairwise_JS(P)`；
(c) conformal——`α_i = 1 − b̄(winner_i|z_i)` 在训练 split 上排序，`u(z)=` 新点 α 的分位。

**超参默认**：K=20（与 m10c 一致）；LogisticRegression C=1.0 / max_iter=2000；
trust 归一用 **rank-normalize**（对异常 epi 鲁棒，优于 min-max）。

### 2.6.3 ③ 决策算子 π —— 规则版默认阈值（E2 起点）

```python
THRESH = dict(s_hi=0.50, t_hi=0.55, t_lo=0.30, conf_margin=0.10, gini_hi=0.60)

def policy_rule(s, t, bbar, base_idx, th=THRESH):
    gini = 1.0 - float((bbar**2).sum())
    margin = float(bbar.max() - bbar[base_idx])      # confidence margin vs base
    if s >= th['s_hi']:                  return "commit-base"   # 饱和：F-R9.6
    if t >= th['t_hi'] and margin > th['conf_margin']:
                                          return "deviate"       # 有头寸+可信
    if t < th['t_lo']:                    return "defer-LLM"     # 没把握→第二意见
    if gini >= th['gini_hi']:             return "ensemble"      # 无明显赢家
    return "commit-base"                                         # 损害控制兜底
```
阈值由 **E2 在训练域上网格搜**（grid: s_hi/t_hi ∈ {0.4,0.5,0.6,0.7}，conf_margin∈{0.05,0.1,0.15}），
按训练域 **Regret-to-Oracle** 选，再在 held-out 域评估——**阈值也 LODO，不偷看留出域**。
学习版 `π_ψ`：特征 σ=(s,t,entropy,gini,top_gap,tail)，标签 a*（§2.5.5），RF 分类器。

### 2.6.4 ④ defer-LLM（DeepSeek）prompt 模板 + JSON 契约

```python
SYS = ("你是时序模型选择的审议助手。只能用给定的训练侧信息，禁止假设测试标签。"
       "判断是否应当偏离默认模型，输出严格 JSON。")
def build_prompt(profile, cv_scores, base, retrieved):
    return [{"role":"system","content":SYS},
      {"role":"user","content":
        f"任务画像: {profile}\n默认模型(base): {base}\n"
        f"各候选训练集CV效用: {cv_scores}\n"
        f"相似历史案例(LOCO检索, base是否=oracle): {retrieved}\n"
        '请输出 JSON: {"should_deviate": bool, "toward": "<model|null>", '
        '"confidence": 0~1, "reason": "<=40字"}'}]
# 调用：os 注入 research/.env 的 DEEPSEEK_API_KEY → PROVIDER=deepseek, MODEL=deepseek-reasoner
# out = chat_cached(build_prompt(...), max_tokens=2000); 解析 JSON（容错：找第一个 {..}）
```
**defer 落地规则**：LLM `should_deviate=True` 且 `toward` 在候选里 → 执行 `toward`；否则 commit-base。
**诚实**：retrieved 只含训练域案例 + "base 是否=oracle" 的历史布尔，不含留出 cell 任何 test。

### 2.6.5 指标公式（finish6 主指标，精确定义）

| 指标 | 公式 | 方向 |
|---|---|---|
| **Inversion-coef** | `corr(confidence_i, 1[selected_i 正确])` over 偏离决策 | 负→正 为目标（E1）|
| **Trust-AUC** | ROC-AUC(`trust_i`, `1[decision_i 不伤 base]`) | ↑ 越能排序对错 |
| **Regret-to-Oracle** | `mean_i (oracle_u_i − selected_u_i)` | ↓ |
| **Safe-Deviation-Rate** | `#{偏离且 selected_u ≥ base_u} / #{偏离}` | ↑ |
| **Abstain-Accuracy** | `#{判 commit-base 且 base_u ≈ oracle_u} / #{commit-base}` | ↑ |
| **Phase-boundary clarity** | 相图上动作分区的 silhouette / 决策树纯度 | ↑ 越可分 |
| **vs-base**（仅作诚实参照，非主指标）| `mean(selected_u − base_u)` | ≈0 即"不伤" |

### 2.6.6 一个完整算例（手验 trace，便于对照实现）

```
cell = (BirdChicken, N=10, classification)；base=rocket
① z = featurize_cell(...)  → 30-d
② K=20 头：b̄ = {rocket:0.46, moment_1nn:0.42, euclid:0.06, dtw:0.04, moment_lr:0.02}
   confidence = 0.46（moment 很接近 → 决策脆弱）
   epistemic(MI) 高（头之间在 rocket/moment 上分歧大）→ trust = 0.28（低）
   saturation s = ĝ_φ(z) = 0.68（分类域饱和先验 + 该 cell gap 预测小 → 高）
③ π_rule: s=0.68 ≥ s_hi=0.50 → "commit-base"        # 饱和优先，直接守 rocket
④ execute → rocket
⑤ Reporter: "选 commit-base：饱和度 0.68(高)，base≈oracle；虽 belief 显示 moment 接近
   (gap 0.04)，但 trust 仅 0.28(存疑) → 不赌。历史 8/10 相似 cell rocket=oracle。"
对照真值：该 cell rocket=0.90, moment=0.80 → commit-base 正确（避免 F-R9.x 的 −10pp 误偏离）✓
```
> 这个 trace 就是 method4 BirdChicken 误偏离案例（finish4）在 method6 下被 **trust+saturation 双门**
> 正确拦截的对照——E1/E2 要在全数据上系统复现这种"该拦的拦住、该放的放行"。

---

## 3. 实验设计（finish6 §1-3）

复用 Round 9-10 的诚实底座（UCR-22 / UEA-14 / forecasting-72 / synth-detect，全 LODO），但**指标全换**：

| 旧指标（弃）| 新指标（method6 主指标，采纳 feedback_m4 #93）|
|---|---|
| vs-base ±pp | **Inversion 系数**：corr(confidence, correctness)，目标从负翻正 |
| acc/MAE | **Regret-to-Oracle**：selected vs oracle 差 |
| — | **Safe-Deviation-Rate**：偏离且不伤 base 的比例 |
| — | **Abstain-Accuracy**：判"该守 base"时 base 确实≈oracle 的比例 |
| — | **决策相图的边界清晰度**（动作可分性 / silhouette）|
| — | **Trust-AUC**：trust 能否排序"这次决策对不对" |

### 三个核心实验
- **E1（主线 A）**：3 种 epistemic 估计 × trust-gate，画 inversion-correction 图，比谁把 corr 翻正最干净。
- **E2（主线 B）**：在 (saturation × trust) 平面上，对每个 cell 标最优动作（事后 oracle），训 `π`，画相图 + 测边界跨任务一致性。
- **E3（LLM defer）**：相图右下角（低 sat 低 trust）启用 DeepSeek 第二意见，测 defer vs ensemble 的 trust 校准改善。

---

## 4. 与既有方法的关系（不堆模块）

- **复用**：method4 的 learned belief（confidence 来源）；method5 的 saturation detector（`ĝ(z)`）；method3 M8 attribution（解释）。
- **新增**：**唯一新组件 = trust estimator + 显式决策算子 `π`**——它**替代** method4 的 `argmax+margin`，不是叠加。
- **诚实底线**：全 LODO；epistemic/conformal 只用训练 split；LLM 只读训练侧；无 test 泄漏。

---

## 5. 文件地图（Round 11 增量）

```
research/
├── method6.md                          # 本文件
├── finish6.md                          # Round 11 实测（待建）
└── experiments/
    ├── m13_trust_vs_confidence.py      # E1: epistemic 估计 + inversion 校正
    ├── m14_decision_phase_diagram.py   # E2: (saturation×trust)→action 相图
    └── m15_llm_defer.py                # E3: DeepSeek 第二意见作为 defer 动作
```

---

## 术语表（method6 增量）

| 术语 | 含义 |
|---|---|
| confidence | `max_k b(M_k|z)`——"选谁"的强度（method4 已有，被证为反指标）|
| trust | `1 − epistemic(z)`——"该不该信这次 belief"（method6 新，主线 A）|
| belief inversion | confidence 与正确性负相关（F-R9.2）；method6 要把它校正 |
| decision phase diagram | (saturation × trust) → 最优动作 的相图（主线 B 核心产物）|
| defer-to-LLM | trust 低时调 DeepSeek-reasoner 做结构化第二意见的动作 |

---

**End of method6.md** — E1/E2/E3 实测后追加 §6；论文可据此写 "The Design of a Selective-Action
Decision Mechanism under a Strong Default"（机制论文，独立于提分）。
