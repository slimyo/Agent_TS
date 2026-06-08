# plan.md — 项目总纲（Routing, Not Competing）

> 本文档是项目的**总纲 / single source of truth（高层）**。
> 配套：`method*.md`（方法细节）/ `finish*.md`（实测 + Findings）/ `TODO.md`（任务看板·权威优先级）/ `feedback.md`（外部 review 11 条硬伤）/ `classifier.md`（分类框架）/ `paper_draft.md`（论文稿）。
> **最近重构：2026-05-30**（并入 M8 energy-based framing + M9 泄漏审计的结论）
> **🧭 主线确立：2026-06-02** —— 见 §零（研究主线已从"提升性能"转为"决策机制本身的设计科学"）

---

## 零、研究主线（2026-06-02 确立 · 权威方向）

> **主线：以 method4 的"决策机制"为研究对象，而非以"性能提升"为目标。**
>
> Round 9-10 已用诚实负结果钉死：在强 base（Chronos-2 / Rocket）饱和的 benchmark 上，
> 任何 router 形式都收敛到 ≈base（F-R9.7/9.8/10.3）。**继续追性能是低 ROI 的死路。**
> 但 method4 暴露出的**决策机制**本身是一个未被充分研究、且独立于"能否提分"的科学对象：
>
> > **一个自适应 agent 何时该行动、何时该退、凭什么相信自己的判断、如何为自己的克制辩护？**
>
> 这条主线把项目从 "another routing paper" 重定位为 **"the design science of a selective-action
> decision mechanism under a strong default"**——即使性能持平，机制研究本身有发表价值
> （feedback_m4 三 reviewer 一致：Saturation Detection + Failure Diagnostics 比再造 router 更有价值）。
>
> **核心研究对象 = 决策算子 `π(a | b(M|z))`**：从 belief 到"route/abstain/探索/求助"动作的映射。
> 已知的机制级现象（待系统刻画，全是 method4 的真发现）：
> - **F-R9.2 belief inversion**：belief 强度与正确性**负相关**（越自信越错）——决策不能信 raw 强度。
> - **F-R9.6 gate collapse**：诚实校准的 gate 在饱和域**最优解=永不行动**（abstain 是贝叶斯最优）。
> - **F-R10.2 damage control**：决策机制的稳健价值是"**永不显著伤害 base**"，而非提分。
> - **F-R10.1 saturation 可检测**：oracle-gap 在未饱和域可学（corr 0.37）→ "该不该行动"是可学信号。
>
> 衡量标准也随之改变（采纳 feedback_m4 #93）：主指标 = **Regret-to-Oracle / Safe-Deviation-Rate /
> Abstain-Accuracy / 决策可解释性**，而非 vs-base ±pp。
>
> 具体优化方向见 `TODO.md` 文首"feedback_m4 路线图"+ 本次新增的"决策机制研究方向"。
> 设计沉淀到 `method6.md`，实测到 `finish6.md`。

### 零.1 决策变量四分解（Round 11 收敛 · feedback_m6 三 reviewer 一致认定为最大贡献）

> Round 11 把过去 method2-5 混成单一 "confidence" 的东西，**拆成四个独立随机变量**——
> 这是当前研究线最有价值的结构性发现（独立于提分）：

| 变量 | 回答 | 现状 | 实证 |
|---|---|---|---|
| **saturation** `ĝ(z)` | 有没有头寸（该不该进场）| ✅ 可学（未饱和域 corr 0.37）| F-R10.1 |
| **trust** `≈P(action safe)` | 偏离会不会**死**（避险）| ✅ conformal 可学（AUC 0.80）+ 已部署（零代价避坑）| F-R11.5/11.9 |
| **gain** `≈E[Δreward]` | 偏离能不能**赚**（获利）| ❌ **尚缺，下一步核心** | F-R11.7（trust 排不动 gain）|
| **proposal** | 谁来提偏离候选 | 🟡 belief / LLM 部分（LLM 2-3× trust）| F-R11.8 |

**关键定论（F-R11.7，三 reviewer 公认比 F-R11.5 更重要）**：`trust ≈ 风险估计器 ≠ 效用估计器`。
→ method6 缺的正是 **gain model**：决策应从 `if trust>τ` 升级为 `if trust>τ1 AND gain>τ2`。

### 零.2 下一阶段路线 · Method7 = Trust-Aware **Gain** Routing

> 演进逻辑：M4 发现 inversion → M5 发现 saturation → M6 证明 trust 只避险 →
> **M7 自然问题：如何预测"值得偏离的收益"，而非仅"偏离是否安全"。**

主线管线（method7 待建）：`saturation → proposal → trust(避险) → gain(获利) → action`。
五个候选方向（按 reviewer 优先级 + 我方 ROI 排序，详见 `TODO.md` Round 12 路线图）：
1. **Gain Modeling**（最高优先，补 F-R11.7 缺口）：直接回归 `gain(x)=oracle−base` / 反事实 `Y(a)`。
2. **主战场转 Forecasting**（reviewer #1）：未饱和、正确偏离样本多，能测 trust/gain 的统计显著性。
3. **Counterfactual / Portfolio**：从"选一个专家"→"预测各候选 outcome 分布"→"组合配比"（投资组合视角）。
4. **Proposal Network 替代 LLM**（AlphaGo 式 policy-propose + trust-verify）。
5. **Meta-trust / 理论界**：trust=f(conformal,disagreement,density,saturation)；selective-action regret bound。

⚠️ 诚实前提（不夸大）：oracle 天花板本就低（分类 +0.4pp / 预测 ~19% rel），gain model 的目标是
**在"避险已闭环"基础上把获利从 ~0 提到可测**（哪怕 +1~2pp 也是实质突破），而非回到"刷 SOTA"。

---

## 一、项目目标（一句话）

复现 TSci（Time-Series Scientist）多 Agent 框架，并在 **few-shot（N=10–100）** 场景下系统回答一个问题：
**当 TSFM（时间序列基模型）已经很强时，LLM-Agent 在系统里到底该扮演什么角色？**

**论文答案（已被实验证据收敛）**：Agent 不该做"竞争性预测器"，而应做围绕基模型的
**选择性路由 / abstain 网关**（selective router）。即把 Agent 的角色从 *prediction* 重定位为
*meta-decision*：$m^*(x) = \arg\max_{m\in\mathcal{M}} \mathbb{E}[U(m,x)]$。

---

## 二、核心论点链（3 条，全部已有实证支撑）

1. **TSFM Saturation Hypothesis**：当基模型预训练分布覆盖测试分布时，任何 wrapper 的
   期望增益 → 0，但方差 > 0（部分 cell 改善、部分变差，期望相消）。
2. **Meta-Decision reduction**：Agent 不参数化预测器本身，只参数化"选哪个预测器 / 是否弃权"
   的选择策略，可统一约化为 selective prediction $(f, g)$。
3. **单一架构横跨三任务**：forecasting / RCA / classification 用同一套
   Curator + Model-Cards + Memory 骨架，失败点相同（tiny-N CV 噪声误路由）、
   修复手段相同（N-conditional abstain 退回 base）、上限相同（平均持平、niche 正增益）。

---

## 三、三层系统抽象（method3 M8 后的权威表述）

> 旧"Bayesian posterior"措辞已弃用 → 改 **energy-based / belief state**（M8.1，更诚实）。

```
  x ──► Representation z=f(x) ──► Belief b(M) ──► Decision a~π(a|b)
        Curator 特征              energy score      risk/cost gate / abstain
        representation.py         bayesian_router.py clf_planner.py
        series_features.py        (softmax 归一化,    action_layer.py
                                   非 Bayesian 后验)  drift_engine.py
```

---

## 四、当前进度速览（2026-05-30）

| Phase | 内容 | 状态 |
|---|---|---|
| 1 | TSci 复现 + 环境 | ✅ |
| 2 | Baseline 矩阵（Chronos-2 / TimesFM / MOMENT / Rocket / DTW …）| ✅ |
| 3 | 边界评估 144 cells（no-method-dominates）| ✅ |
| 4 | E1–E4 增强模块 + A1–A9 ablation | ✅ |
| 5 | 概率评估 CRPS | ✅ |
| 6.1 | RCA natural（+40pp vs LLM-direct）| ✅ |
| 6.3 | TaskB UCR router | ✅ |
| 6.4 | Agent-as-Router 重定位 | ✅ |
| Round 8 | M1–M4 / M7-P1 自演化模块 | ✅ |
| **M8** | Factor Attribution + energy framing（问题 1+2）| ✅ |
| **M9** | **Memory 泄漏审计修复（问题 6）** | ✅ |
| **善后** | 论文撤回"击败 Rocket"+ feedback 路线图 | 🔬 进行中（TODO P0–P2）|

---

## 五、三任务最终诚实结果（post-M9）

| 任务 | Agent 角色 | 最强 baseline | 我们最终 | 诚实差距 |
|---|---|---|---|---|
| Forecasting（24 cells）| Chronos-2 wrapper | Chronos-2 | v11 parity wrapper | 0%（CRPS，Wilcoxon p=0.32）|
| RCA（30 cells / 5-fault）| 结构化根因分析 | LLM-direct | Curator+Cards | **+40pp R1** vs LLM-direct（但 −37pp vs 规则 baseline，诚实负结果）|
| Few-shot TSC（30 cells UCR）| router | Rocket-alone | B7v3（router+memory, 去泄漏）| **−0.62pp（持平）**；去泄漏前虚高 +1.51pp |

> ⚠️ **关键修正（F-R8.7）**：去泄漏后 TSC router **不再击败 Rocket**。论文价值从
> "击败 SOTA"转为 **"saturated benchmark 上 routing 的诚实评估 + 泄漏审计方法学"**。

---

## 六、方法版本谱系（简表，详见 method*.md）

**Forecasting**：v5c 基线 → v8 TSFM 扩充（ECL/Exchange 反转）→ v9 margin gating →
v10 N<15 fallback → v11 memory safety-net（parity）→ v12 entropy gate → v13 联合。

**Classification**：B3 Rocket(87.5%) → B6 Agent direct(54.3%, −33pp) → B7v1 catastrophic →
B7v2 N<7 fallback(86.66%) → **B7v3 router+memory(86.91% 去泄漏, −0.62pp)**。

两条谱系**同构**：同样的失败（直接竞争 SOTA）→ 同样的中间症状（few-shot CV 噪声误路由）→
同样的修复（N-conditional fallback + cross-series memory）。

---

## 七、关键文件索引

| 文件 | 作用 |
|---|---|
| `agent/adapt_ts.py` / `forecaster_reflect.py` | Forecasting 主 wrapper（v5c–v13）|
| `agent/clf_planner.py` | 分类 router（B7）|
| `agent/bayesian_router.py` | Energy-based 路由 + Factor Attribution（M8）|
| `agent/clf_memory.py` / `memory.py` / `memory_decay.py` | Memory 层（M9 去泄漏）|
| `agent/representation.py` / `utils/series_features.py` | 特征 embedding |
| `agent/drift_engine.py` / `action_layer.py` | Decision 层 |
| `utils/llm.py` | LLM 接口（SiliconFlow / DashScope / DeepSeek，开源免费优先）|
| `baseline/` | 所有基线模型封装 |
| `experiments/` | ~60 sweep/eval 脚本（如 `taskb_router_v3_honest_sweep.py`）|
| `results/` | 150+ 实验结果 jsonl |

---

## 八、下一步（权威优先级见 TODO.md）

当前唯一权威优先级 = `TODO.md` 文首 **"feedback 改进路线图 #70-83"**：
**P0 #70-72（论文善后撤回）→ P1 #73-76（理论/实证硬伤）→ P2 #77-80（收敛+scope）→ P3 #81-83（future）**。

---

## 九、环境与运行

- conda/mamba env `tsci`（本地）/ `tsci-remote`、`tsci-remote-tx440`（远程 GPU）
- LLM：SiliconFlow / DashScope / DeepSeek（开源免费优先，不用付费官方服务）
- 远程 GPU：c220@10.192.43.66，2× RTX 5070 Ti
- 运行：`python -m experiments.<name>` 或 `experiments/*.sh`
