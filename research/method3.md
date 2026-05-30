# Method v3 — Self-Evolving Router

> 版本：2026-05-29（Round 7 起）
> 前文：`method.md` (Round 4-A) / `method2.md` (Round 5 + 6) / `feedback.md`（外部 review）
> 配套：`finish3.md`（Round 7 实测） / `paper_draft.md`（论文）

---

## 0. Thesis

> method2 把 router 升级到 **self-adaptive runtime + decision engine**（B2/B3/E1/R6-E 自适应 + Action 层）。method3 进一步把 router 升级到 **self-evolving system**：除了 runtime 自适应，还要让模型库本身能**自我淘汰**、Prior 权重能**自我学习**，去掉 method2 中残留的手调常数。

---

## 1. 与 method2 的关系

method2 §11.5 的 Round 6 后半已经做到：
- 单次决策的 calibration + cost-min decision（B2 + E1）
- 周期性 drift 自适应（B3 4+1 信号 + 3 动作 + refit）
- 预算感知的 inference 调度（R6-E）

但残留两个**手调常数**：
1. **PriorFactor 的 `strength` 常数**：`NPrior(strength=2.0)`、`IndustrialPrior(strength=2.0)`、`OperationalReliabilityPrior(strength=1.5)` 全是经验值，**无法 auto-adapt**。这与 feedback 前§2.B "Empirical Bayes" 期望直接冲突。
2. **Model library 永久污染**：即使某个 model 在某 regime 下系统性地差，bandit 也只会让它 belief 变高（loss 大）→ 在 Thompson 模式下仍可能被采样；libraries 越长越垃圾。

method3 (Round 7) 通过 **M2** + **M3** 两件套封堵：

| ID | 名称 | 解决问题 | 对应 feedback |
|---|---|---|---|
| **M1** | Meta-bandit on decide_mode | `decide_mode="argmax"` 是手调常数 | 前§2.C "Meta-bandit" |
| **M2** | Model 自动淘汰 | model library 永久污染 | 前§4 表 "Model 淘汰机制" |
| **M3** | Empirical Bayes Prior strength | 手调常数 + Prior 权重不自适应 | 前§2.B "Empirical Bayes" |
| **M4** | Per-regime bandit decay | global decay 把 hot/cold regime 一刀切 | 前§2.C "Per-regime decay" |

---

## 2. M1 · Meta-bandit on decide_mode

### 2.1 设计 (M1)

> 把 `{argmax, thompson, risk_min}` 当作 3 个 arm 的 meta-level bandit。每次决策 `select_mode(meta_state)` 选 mode；观察到 outcome 后 `meta_state.observe(mode_used, outcome)` 更新 Gaussian conjugate (n, s, sq)。冷启动期 round-robin 填 `cold_start_K` 个 obs/mode，之后 Thompson sample 持续 explore。

```
RouterConfig.decide_mode = "auto"        ──► adaptive_decide
                                              │
                                              ▼
                                       select_mode(state.meta_bandit, cfg)
                                              │
                                              ▼
                          mode ∈ {argmax, thompson, risk_min}
                                              │
                                              ▼
                              router.decide(ctx, ev, mode=mode)
                                              │
adaptive_observe(state, plan, outcome) ─────► meta_bandit.observe(mode, outcome)
```

### 2.2 状态

`state.meta_bandit_dict` 持久化（save/load round-trip）。包含 `decay` + `prior_mu/var/n` + `counts: dict[mode → (n, s, sq)]`。

### 2.3 触发开关

新增 `RouterConfig` 旋钮：

| 名称 | 默认 | 作用 |
|---|---|---|
| `meta_bandit_enable` | False | 显式开关 |
| `decide_mode="auto"` | — | 等价于 `meta_bandit_enable=True` |
| `meta_bandit_cold_K` | 10 | 冷启动每 arm 至少观测 N 次 |
| `meta_bandit_selection` | `"thompson"` | 或 `"greedy"` |
| `meta_bandit_decay` | 0.995 | 老 obs 指数衰减 |

### 2.4 与 §1.2 三个 decide mode 的关系

method2 §1.2 把这 3 个 mode 设计成"用户选一个 + 文档说明各自适用场景"。M1 把"用户选"自动化掉 —— 系统按真实 outcome 自学应该用哪个 mode；扰动来时（B3 drift 触发）可清空 meta_bandit 让其重新 explore（后续可加）。

---

## 3. M2 · Model 自动淘汰

### 3.1 设计 (M2)

> 每 `cull_every` 次决策后，按 per-regime belief μ 排序模型，淘汰底部 `cull_fraction` 的"差模型"。被淘汰的 `(regime, model)` 对在后续 router 决策中通过 `EliminationPrior` 硬屏蔽（log_prior = -∞）。

```
state.bandit._state[(regime, model)] = (n, s, sq)
        │
        ▼
cull_models(state, config):
    for each regime r:
        rank models by μ = s/n  (lower = better)
        eligible = {m : n_r,m ≥ min_observations  AND  m ∉ protect}
        cull bottom ⌈fraction × |eligible|⌉
        guard: keep_at_least min_keep models per regime
        state.culled.setdefault(r, set()).update(culled)
```

**约束**（互锁防止单点全挂）：
- `min_keep_per_regime ≥ 2`：保证至少有 2 个候选
- `protect: tuple[str]`：永远不淘汰（如 `naive_drift` 兜底 + `chronos2` 默认）
- `min_observations`：未观察够的模型不淘汰（避免冷启动误杀）
- 可选 **resurrection**：被淘汰 model 在 K 次决策后或 `drift_engine` 触发 `boost_exploration` 时自动复活

### 3.2 EliminationPrior

```python
@dataclass
class EliminationPrior(PriorFactor):
    state_ref: object = None
    log_factor: float = -50.0   # 实际等价于 -∞，但保持数值稳定

    def log_prior(self, candidates, ctx):
        regime = ctx.features.get("regime") if ctx.features else None
        culled = getattr(self.state_ref, "culled", {}).get(regime, set())
        return {m: (self.log_factor if m in culled else 0.0) for m in candidates}
```

### 3.3 集成 (M2)

- `adaptive_observe` 在 `n_observations % cull_every == 0` 时调一次 `cull_models`
- `adaptive_decide` 将 regime 写入 `ctx.features["regime"]` 供 EliminationPrior 消费
- `state.culled: dict[int, set[str]]` 加入 `RouterState` 持久化

---

## 4. M3 · Empirical Bayes Prior Strength

### 4.1 设计 (M3)

> 不假设 prior strength 是固定常数；从 `state.telemetry` 的 (log_prior_F(chosen), outcome) 关系里**学**每个 prior 的有效权重。Pearson 正相关 = factor 有用 → 加强；负相关 = factor 反向 → 削弱。

```
for each PriorFactor F with .strength attribute:
    collect (xs, ys):
        xs = log_prior_F(chosen)   from prior_contribs[F.name][chosen]
        ys = -outcome               higher = better
    r = Pearson(xs, ys)
    F.strength ← clip(F.strength · (1 + lr · r), 0, max_strength)
```

直觉：
- F 把高 log_prior 分配给最终 outcome 好的 model → `r > 0` → strength 升
- F 持续推荐 outcome 差的 model → `r < 0` → strength 降到 0（事实上 prune 该 factor）
- 训练样本不足时（< `min_samples`）不更新

### 4.2 边界 + 安全

- 只学有 `strength` 属性的 PriorFactor（NPrior / IndustrialPrior / OperationalReliabilityPrior 等）
- `r` 用 winsorize 防止极端 outcome 主导
- `lr` 默认 0.05，max_strength 默认 5.0
- 持久化：把学到的 strengths 写入 `state.learned_prior_strengths: dict[name, float]`，下次 `adaptive_decide` 构 prior 时读取

### 4.3 集成 (M3)

- `adaptive_observe` 在 `n_observations % eb_learn_every == 0` 时调 `learn_prior_strengths(state, router)`
- `state.learned_prior_strengths` 加入 `RouterState` 持久化
- `adaptive_decide` 构 prior 时读取 `state.learned_prior_strengths.get(name, default)` 覆写常数

---

## 4.5 M4 · Per-regime bandit decay

> `BanditState.decay` 原本是全局标量，所有 (regime, model) 共享。问题：热门 regime 大量更新会快速冲淡冷门 regime 的有效样本数；扰动来时也只能整体调节。M4 让每个 regime 拥有独立 decay。

### 4.5.1 设计

```python
class BanditState:
    decay: float = 1.0                 # 全局 fallback
    regime_decay: dict = {}            # {regime → decay}, empty = 用 scalar
    def _effective_decay(self, r):
        return self.regime_decay.get(r, self.decay)
    def observe(self, regime, model, loss):
        d = self._effective_decay(regime)
        if d < 1.0: n*=d; s*=d; sq*=d
        ...
    def set_regime_decay(self, regime, decay):  # clip to (0,1]
        ...
```

### 4.5.2 与 B3 drift 联动

`drift_engine.apply_actions` 在 `boost_exploration` 触发时自动收紧最近 30 条 telemetry 中出现的 regime 的 decay：

```python
tighten_to = max(0.85, 1.0 - 0.10 * a.magnitude)
for rg in recent_regimes:
    bandit.set_regime_decay(rg, min(current, tighten_to))
```

效果：扰动只让 "正在被使用" 的 regime 加速遗忘，**不影响安静的 regime** —— 解决了 feedback 前§2.C 的核心痛点。

### 4.5.3 持久化

`BanditState.save/load` 序列化 `regime_decay` 字段（meta 行新增 `"regime_decay": {str → float}`）。

---

## 5. RouterConfig 新增旋钮

| 名称 | 默认 | 作用 |
|---|---|---|
| `cull_every` | 200 | 每 N obs 调一次 `cull_models` |
| `cull_fraction` | 0.15 | 每次淘汰底部多少比例 |
| `cull_min_keep` | 2 | 每 regime 最少保留模型数 |
| `cull_protect` | `("naive_drift", "chronos2")` | 永不淘汰的模型 |
| `cull_min_observations` | 5 | 模型在该 regime 至少观察过几次才能被淘汰 |
| `eb_learn_every` | 100 | 每 N obs 学一次 prior strengths |
| `eb_lr` | 0.05 | Empirical Bayes 学习率 |
| `eb_max_strength` | 5.0 | strength 上界（防止数值爆炸）|
| `eb_min_samples` | 30 | 至少 N 条带 outcome 的 telemetry 才学 |
| `meta_bandit_enable` | False | M1: 启用 meta-bandit（`decide_mode="auto"` 等价）|
| `meta_bandit_cold_K` | 10 | M1: 冷启动每 arm 最少观测次数 |
| `meta_bandit_selection` | `"thompson"` | M1: `"thompson"` 或 `"greedy"` |
| `meta_bandit_decay` | 0.995 | M1: 老 obs 指数衰减 |

---

## 6. 与 §11.5 (method2) 的关系

| method2 §11.5 节 | method3 接续 |
|---|---|
| 11.5.3 Drift Engine（5 信号 / 3 动作）| M2 culling 在 drift_engine 触发 `boost_exploration` 时清空 culled（resurrection），实现"扰动来时重新探索"|
| 11.5.4 Action Layer（cost-min）| 不变 |
| 11.5.5 Inference Scheduler（utility）| 被 culled 的模型从 candidates 移除，自动减少 scheduler 待选 |
| §3.4 BayesianRouter 6 priors | M3 让其中带 strength 的几个 prior 权重 self-tune |

---

## 7. 实测与 Findings → `finish3.md`

Round 7 全部实测 + Findings 写在 `finish3.md`。本文件只承担方法描述。

---

## 8. M7 · Anomaly Detection Phase 1（feedback 前§5 务实版）

> Phase 1 显式不做：LLM RCA / 新 Memory / Anomaly-Transformer 论文模型本体。本节只装最小可跑闭环。Phase 2/3 留给后续 round。

### 8.1 设计 (M7)

```
window  ──►  AnomalyTypePrior.compute(window) ──► {fault_type → prior}
        │                                              │
        ├──►  RuleBaselineDetector.detect(window) ──┐  │
        │                                            │  │
        ├──►  ResidualScoreDetector.detect(window) ─┤  │
        │       (站位 "Anomaly-Transformer"，无需   │  │
        │        深度模型，Phase 2 可替换)           │  │
        │                                            ▼  ▼
        └────────────────────►   detect_anomaly  ──► AnomalyResult
                                  (softmax over             ├─ is_anomaly
                                   detector scores)         ├─ score
                                                            ├─ suspected_type
                                                            ├─ detector
                                                            ├─ type_prior
                                                            └─ detector_posterior
```

### 8.2 AnomalyTypePrior 规则

| Fault type | 触发条件 | 输入特征 |
|---|---|---|
| `trend_break`      | `level_shift_z > 2`    | `(mean_second_half − mean_first_half) / σ_first` |
| `variance_explode` | `variance_ratio > 2`   | `std(tail) / std(head)` |
| `outlier_burst`    | `max_outlier_z > 4`    | `max|x − μ| / σ` |
| `normal`           | baseline = 1.0 | (兜底) |

logits softmax → 归一化概率。`strength` 默认 1.5，由 **M3 Empirical Bayes** 自动学习（与 NPrior 同管道）。

### 8.3 与 method2 子系统的复用

| Round 6 子系统 | M7 复用方式 |
|---|---|
| B2 Calibration       | `AnomalyResult.score` 可直接作 raw_conf 喂进 `ConfidenceCalibrator.calibrate()` |
| B3 Drift Engine      | 检测到的 fault_type 序列可注入 `state.telemetry.ctx_summary` 让 drift 跟踪 fault distribution 漂移 |
| E1 Action Layer      | `is_anomaly=True` + calibrated_conf → 已有 5-tier 介入决策无缝接 |
| R6-E Scheduler       | 多个 detector = 多个 candidate；scheduler.utility 同样可计算（accuracy_gain 改成 detection_gain）|
| M2 Culling           | 长期表现差的 detector 自动淘汰 |
| M3 EB                | `AnomalyTypePrior.strength` 与 `detector_strength[*]` 由 EB 自适应 |

Phase 1 不需要新机制 —— 全部沿用 Round 5/6/7 已有基础设施。这印证 feedback §5 "核心还是尽量少引入新组件"。

### 8.4 Phase 2 / 3 预留接口

- Phase 2 · per-fault Memory：复用 `failure_memory.py` 的 `FailureCase`，按 `fault_type` 而非 `model` 分桶
- Phase 3 · LLM RCA agent：可关闭模块，输入 `(window, AnomalyResult)`，输出 natural-language 根因 + 推荐 intervention

---

## 10. M8 · Factor Attribution + Bayesian framing 修正（feedback 问题 1+2）

> feedback 两条理论硬伤：**问题 1** "你的 posterior 还不是真 posterior"（factor 非生成式、非条件独立，不该 claim exact Bayesian）；**问题 2** "factor explosion 已经开始失控"（everything becomes a factor → unidentifiable / 黑盒），并 **强烈建议新增 Factor Attribution Analysis**。M8 同时封堵这两条，且**不新增任何运行时模块**（符合 feedback "收敛 abstraction，不要继续堆模块"）。

### 10.1 Framing 修正（问题 1）

`bayesian_router.py` 模块 docstring 显式声明：本系统是 **factorized posterior-inspired energy model**（`p_k = softmax(−E_k)`, `E_k = −Σ_i w_i f_i(x)`），不是 exact Bayesian inference。论文/method 文本用 "Bayesian-style compositional decision model" / "energy-based routing"，**不**写 "exact Bayesian posterior"。`BayesianRouter` 类名保留仅为兼容。

### 10.2 Factor Attribution Analysis（问题 2）

新增三个**纯分析**工具（无状态、不改路由行为）：

| 接口 | 回答的问题 | 方法 |
|---|---|---|
| `BayesianRouter.factor_log_contributions(ctx, ev)` | 每个 factor 给每个候选贡献了多少 log-term | 拆开 `log_posterior` 的加和项；label 去重 (`name` / `name#1`)；Σ 重构 log_posterior 误差 = 0 |
| `attribute_decision(router, ctx, ev)` | **本次**决策里谁是决定性 factor | Leave-One-Factor-Out (LOFO)：去掉 factor 后 argmax 是否翻转 + KL(full‖without) + Δmargin(chosen−runner) |
| `FactorAttributionAccumulator` | **跨**决策里哪两个 factor 在重复表达 | 收集每次 centred log-term 向量（softmax 平移不变 → 只看 centred），跨决策 concat 求 Pearson；`mean_abs_influence` 标出 inert factor |

**硬 mask 处理**：`AvailabilityPrior` 用 ±1e6 是约束非偏好，会数值上淹没软 factor。`FactorAttributionAccumulator.clip=50.0` 把 centred 贡献裁到 log-space 饱和界，使软/硬 factor 在同一可读量纲对比。

### 10.3 实测 → `finish3.md` §7（F-R8.5）

---

## 11. M9 · 分类 Memory 数据泄漏修复（feedback 问题 6 · "致命 gap"）

> feedback 把 `clf_memory` 列为唯一"致命"工程硬伤：存储 / 检索用的是 **测试集 acc**（`all_clf_accs` 来自 sweep 的 `r["acc"]`），部署时测试标签不可得 → router 用了未来信息；且 memory bank 与评测集是同 30 个 cell，存在 **self-membership** 泄漏（查询 cell 自身 case 以 sim≈1 排第一回灌自己的 outcome）。任何 memory 增益都不可复现。

### 11.1 两类泄漏 + 修复

| 泄漏 | 旧行为 | 修复 |
|---|---|---|
| **(A) value 泄漏** | 投票权重 `1/(1-test_acc)`；`best_classifier`=test-winner | 改为训练集内 **CV** 估计：`cv_accs` 是唯一可投票字段，`best_classifier`=CV-winner；`test_acc`/`all_clf_accs` 降级为 **AUDIT ONLY**，决策代码不可读 |
| **(B) self-membership** | query 不排除自身 | `query/query_diverse/memory_consensus` 加 `exclude_meta`，按 `{dataset,N_per_class,seed}` **leave-one-cell-out** 剔除查询 cell |

### 11.2 实现（无新模块，改既有边界）

- `clf_memory.py`：`ClfCase` 加 `votable_accs()`（只返 `cv_accs`）+ `is_leaky()`；`_eligible()` 实现 LOCO；`consensus_winner_inv_loss` 改用 `votable_accs()`，legacy 无 cv 的 case **跳过**而非回退 test acc（回退=重新泄漏）。
- `bayesian_router.py`：`MemoryLikelihood` 读 `cv_accs`；`Evidence.memory_neighbors` 注明禁止传 test acc。
- `clf_planner.py`：按 `dataset/seed/n_classes` 构 `exclude_meta` 串进两条 memory 路径，neighbor dict 传 `cv_accs`。
- `experiments/build_clf_memory_v2.py`：每 cell 用 `loo_cv_acc` 现算 CV → `cv_accs` + CV-winner；test 仅作 audit。

### 11.3 实测 → `finish3.md` §8（诚实 vs 泄漏数字对比）

---

## 9. 文件地图（Round 7-8 增量）

```
research/
├── method3.md                  # 本文件（Round 7 方法）
├── finish3.md                  # Round 7 实测
├── agent/
│   ├── meta_bandit.py          # M1: Meta-bandit on decide_mode (Round 8)
│   ├── model_culling.py        # M2: cull_models + EliminationPrior
│   ├── prior_learning.py       # M3: learn_prior_strengths (Empirical Bayes)
│   └── anomaly.py              # M7 Phase 1: AnomalyTypePrior + 2 detectors
└── experiments/
    └── (TBD Round 7 demos)
```
