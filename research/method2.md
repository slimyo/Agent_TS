# Method v2 — Universal Bayesian Adaptive Time-Series Routing

> 版本：2026-05-28（Round 5 + Phase 2/4 全部合并后）
> 取代 `method.md`（Round 4-A 版本）。**配套**：`finish-1.md`（实测）/ `feedback.md`（外部 review）/ `paper_draft.md`（论文）/ `ppt.md`（讲述）。
>
> 自顶向下完整描述系统现状。每节脚注真实文件路径作代码地图。

---

## 0. Thesis（一句话总结）

> **任何时序任务都是 `p(model | series_regime, history)` 的概率决策。**
> 我们把这一抽象实现为单一 `BayesianRouter`，统一 forecasting / classification / anomaly+RCA 三类 task；所有 hand-coded 规则（N-fallback / entropy gate / industrial override / margin gate / diverse retrieval / ε-greedy）重 frame 为 prior / likelihood factor 或 decision mode。
> 系统的**核心创新不在新 TSFM，而在把现有 TSFM library 重构为 principled adaptive inference system**。

---

## 1. 数学公式

### 1.1 决策规则

$$\hat{M}(x, h, t) = \mathrm{decide}\Big( p(M_k \mid x, h, t) \Big), \quad p(M_k \mid x, h, t) \propto \exp\!\Big(\sum_i \log \pi_k^{(i)}(z) + \sum_j \log L_k^{(j)}(z, h, t)\Big)$$

其中：

| 符号 | 含义 | 来源 |
|---|---|---|
| $x$ | 输入序列 | raw data |
| $z = f_\phi(x)$ | learned embedding | Phase 4 / `representation.py` |
| $r(z)$ | k-means cluster (regime label) | Phase 4 / `RegimeAssigner` |
| $\pi_k^{(i)}(z)$ | 第 i 个 prior factor 的 log-likelihood for model k | `bayesian_router.py` PriorFactor |
| $L_k^{(j)}(z, h, t)$ | 第 j 个 likelihood factor | `bayesian_router.py` LikelihoodFactor |
| $h$ | 历史观测（CV losses / memory neighbors）| `Evidence` |
| $t$ | 时间步（bandit context）| `BanditState` |
| $\mathrm{decide}(\cdot)$ | argmax / Thompson / risk-min | `BayesianRouter.decide()` |

### 1.2 三个 decide mode

| Mode | 公式 | 替换了什么 |
|---|---|---|
| **argmax** | $\arg\max_k p(M_k \mid x, h)$ | hand if/else + L0 trust threshold |
| **Thompson** | $r̃_k \sim p(M_k \mid x, h)$；$\arg\max(r̃_k)$ | ε-greedy patch |
| **risk_min** | $\arg\min_k \big( \mathbb{E}[\ell_k] + \lambda \cdot \mathrm{Var}[\ell_k] \big)$ | margin gate |

### 1.3 在线更新（Phase 2 Contextual Bandit）

per-(regime, model) Gaussian conjugate update：

$$
\begin{aligned}
n_t &= \mathrm{decay} \cdot n_{t-1} + 1 \\
\mu_t &= \frac{\mathrm{decay} \cdot n_{t-1} \cdot \mu_{t-1} + \ell_t}{n_t} \\
\sigma_{\mu, t}^2 &= \frac{\hat{\sigma}^2}{n_t}
\end{aligned}
$$

decay < 1 用于非稳态环境。

---

## 2. 系统架构

```
┌─────────────┐
│   Series x  │
└──────┬──────┘
       ↓
┌──────────────────────┐
│  Curator Agent       │  curator_uq.py + series_features.py
│  → 25-d 诊断 + 置信度 │  (Diagnosis: trend×3, season×3, stat×3)
└──────┬───────────────┘
       ↓
┌──────────────────────┐
│  Embedding f_φ        │  representation.py
│  HandFeature / MOMENT / Chronos2 → z ∈ R^{25|512|768}
└──────┬───────────────┘
       ↓
┌──────────────────────┐
│  RegimeAssigner       │  representation.py:RegimeAssigner
│  k-means(K=6/8) → r   │  (替代 dataset 名)
└──────┬───────────────┘
       ↓
┌──────────────────────────────────────────────────┐
│  BayesianRouter (bayesian_router.py)             │
│  ─────────────────────────────────────────────    │
│  Prior factors (composable):                      │
│    AvailabilityPrior(local ∪ remote?)             │
│    CRPSPrior(dataset)        ← Round 4-A 兼容     │
│    RegimePrior(assigner)     ← Phase 4 替代       │
│    TypePrior(POINT × 0.3)                         │
│    NPrior(N_threshold, strength)                  │
│    EntropyPrior(beta)                             │
│    IndustrialPrior(strength)                      │
│                                                   │
│  Likelihoods:                                     │
│    CVLikelihood(sigma_sq)                         │
│    MemoryLikelihood (consensus_winner_inv_loss)   │
│    RepresentationLikelihood (z-space kNN)         │
│                                                   │
│  Online bandit state (optional):                  │
│    BanditState[(regime, model)] → (μ, σ)          │
│                                                   │
│  decide(ctx, ev, mode) ∈ {argmax, thompson,       │
│                            risk_min, ucb}         │
└──────┬───────────────────────────────────────────┘
       ↓                          ↑
┌──────────────────────┐          │
│ Model Library Shelf   │   observe(z, chose, loss)
│ (12 forecast + 11 TSC)│          │
└──────┬───────────────┘          │
       ↓                          │
┌──────────────────────┐          │
│  Prediction          │──────────┘    (online loop closure)
└──────────────────────┘
```

---

## 3. 关键模块 (Round 5 + Phase 2/4 合并后)

### 3.1 Curator + Features

`research/agent/curator_uq.py` + `research/utils/series_features.py`

- **Diagnosis**: `(trend, season, stat) × (stat_conf, llm_conf, xc_conf)` = 9 个置信度信号
- **25-d feature vector** (`featurize_cell`):
  - basic (5): mean / std / median / iqr / range
  - trend (4): slope / r² / curvature / dominance
  - freq (4): spectral entropy / 3 dominant freqs
  - complexity (4): PermutationEntropy / DFA / SampEn / approximate
  - outlier (3): tail ratio / mad outlier / max-z
  - industrial (5): acf_decay / quant_bits / level_count / step_count / signature
- **Use**: 既作 router input (Context features) 也作 memory key + industrial 信号源

### 3.2 Embedding (Phase 4 NEW)

`research/agent/representation.py`

```python
class Embedding(Protocol):
    dim: int; name: str
    def embed(self, series) -> z       # [L] → [dim]
```

| Embedding | dim | 来源 |
|---|---|---|
| `HandFeatureEmbedding` | 25 | series_features (baseline) |
| `MomentEmbedding` | 512 | AutonLab/MOMENT-1-small (frozen TSFM encoder) |
| `Chronos2Embedding` | 768 | amazon/chronos-2 T5 encoder mean-pool |

### 3.3 RegimeAssigner (Phase 4 NEW)

```python
assigner = RegimeAssigner(K=8, embedding=MomentEmbedding())
assigner.fit(stored_Z, stored_losses)
assigner.regime_prior(z) -> {model: π_k}
```

- k-means on z (cosine, L2-normalized)
- per cluster 聚合 per-model loss → π_k = 1/loss 归一
- 实测 K=8 regime purity = **82.4%**，18% 在 cross-dataset cluster

### 3.4 BayesianRouter (Round 5 CORE)

`research/agent/bayesian_router.py`

三个核心类：

| Class | 接口 | 责任 |
|---|---|---|
| `Context` | dataset, N, H, entropy, industrial, features={"z":...}, allow_remote | routing 输入捆绑 |
| `Evidence` | cv_losses, cv_std, memory_neighbors | per-call 观测 |
| `BayesianRouter` | `decide(ctx, ev, mode, lam)` | 单一决策入口 |

6 PriorFactor + 2 LikelihoodFactor，对应 feedback Round 4 全部改进（详见 §6 mapping 表）。

### 3.5 ContextualBanditRouter (Phase 2 NEW)

`research/agent/bandit.py`

```python
state = BanditState(prior_mu, prior_var, prior_n, decay)
router = ContextualBanditRouter(candidates=[...], bandit=state, regime_fn=regime_fn)

# decide:
chosen, scores = router.decide(z, mode="thompson")   # | "greedy" | "ucb"

# online update:
router.observe(z, chosen, actual_loss)
state.save("research/results/bandit_state.jsonl")   # persistent
```

实现 feedback Round 4 §二 Thompson Routing。三个 mode：
- `thompson`: r̃_k ~ N(μ, σ); argmin
- `greedy`: argmin μ (pure exploitation)
- `ucb`: argmin (μ − β·σ) (optimistic exploration, β=2)

### 3.6 Memory Layer

| | Forecasting | TSC |
|---|---|---|
| 文件 | `agent/memory.py` | `agent/clf_memory.py` |
| Case 字段 | feature, best_strategy, mae, all_strategy_maes | diag_feature, best_classifier, test_acc, **all_clf_accs** |
| 检索 | top-K cosine | top-K cosine + `query_diverse` |
| 投票 | safety-net override | `consensus_winner_inv_loss` (1/CRPS-style) |

**反事实存储**：`all_clf_accs = {clf: test_acc}` 完整字典（不只 winner）。
**Diversity retrieval**：top-K 全 default → 替换最低 sim default 为最高 sim non-default。
**1/CRPS vote**：每邻居为所有 candidate 按 `sim · 1/(1-acc+ε)` 投票。

### 3.7 Model Library

#### Forecasting (12 baselines)

| 类别 | 实例 |
|---|---|
| trivial point | naive_drift / naive_seasonal / arima_ets / llmtime |
| Chronos family | chronos (60M) / chronos_bolt (200M) / chronos2 (120M, **default**) |
| TSFM 主流 | timesfm2 (500M) / moirai (311M) / moirai2 (11M) |
| Niche specialists | tirex (128M, xLSTM) / **toto (151M, observability ⭐)** / toto2 (4M) |
| Remote large | time_moe (50M) / sundial (128M) / timer (8.3B MoE) |

每模型 5 字段 Model Card (`research/agent/model_cards.py`)：class / assumes / strengths / weaknesses / typical_failure。

#### TSC (11 classifiers)

distance (2) + kernel (2) + TSFM-embed (4) + dictionary (1, **WEASEL** NEW SOTA +2.7pp) + feature (1) + LLM (1).

---

## 4. Track 流程

### 4.1 Forecasting (`agent/forecaster_reflect.py`)

```
input: (train, val, H, diag, dataset)
  │
  ├─ planner = env.ADAPTTS_PLANNER:
  │     bandit       → ContextualBanditRouter (Phase 2)
  │     bayesian     → BayesianRouter         (Round 5)
  │     prior_aware  → planner_prior_aware    (Round 4-A baseline)
  │     <unset>      → make_plan (legacy 4-model)
  │
  ├─ z = embedding.embed(train)         ← Phase 4 (bandit/bayesian only)
  ├─ regime = regime_fn(z)              ← Phase 4
  ├─ chosen, post = router.decide(...)
  │
  ├─ if level == L1: single-model run
  │  else:          ensemble_predict(plan)    (quantile linear pool, §3.5 quantile_ensemble.py)
  │
  ├─ pred = STRATEGY_FN[chosen](train, val, H, season_m)
  │
  └─ observe_outcome(trace, y_true=test, y_pred=pred)
        → bandit.observe(z, chosen, actual_mae)   (online loop closed)
```

### 4.2 Classification (`agent/clf_planner.py`)

```
input: (X_train, y_train, X_test, season_m, dataset)
  │
  ├─ if use_bayesian or ADAPTTS_CLF_PLANNER=bayesian:
  │     ┌─ CV losses = {clf: 1 - cv_acc}
  │     ├─ industrial_p = f(acf_decay, quant_bits)        # continuous, not threshold
  │     ├─ memory_neighbors via query_diverse if enabled
  │     ├─ BayesianRouter([NPrior, IndustrialPrior, ...]) + [CVLikelihood, MemoryLikelihood]
  │     ├─ chosen, post = router.decide(ctx, ev, mode)
  │     └─ predict_with(chosen, ...)
  │  else:
  │     ┌─ N < n_min → force default
  │     ├─ CV → margin gate → memory consensus → industrial override (legacy hand path)
  │     └─ predict_with(chosen, ...)
```

---

## 5. 实测发现 (Findings F1-F12)

详见 `finish-1.md §0` 完整索引。Method 节直接引用：

| ID | 一句话 | 对方法的影响 |
|---|---|---|
| F1 | Wrapper 改进上界 2.43%（TSH） | 放弃 wrapper 路线 → 走 library expansion |
| F2 | 5-way oracle == 3-way (-5.24% ceiling) | dynamic MoE 写入 future work |
| F3 | niche 互补 >> 模型数 | library curate by domain KL，不按 param 数 |
| F4 | TSFM normalize 隐性要求 | wrapper 必须 scale-stress test |
| F5 | naive top-K 自我强化 default | 必须 query_diverse + 反事实存储 |
| F6 | Soft router 双轨 NEGATIVE | hard top-1 优于 soft mixture in saturation |
| F7 | N≤5 CV catastrophic | N-fallback 不是 hack，是 sample complexity bound |
| F8 | Specialist TSFM niche-clean | pair-up：generic + specialist |
| F9 | 1/MAE prior 自动复现专家划分 | feedback "all heuristics → prior" 可行 |
| F10 | Cost asymmetry across env | routing 必须含 deployment cost 项 |
| F11 | heuristic stack ≠ principled framework (meta) | 触发 Round 5 Bayesian 重 frame |
| F12 | (mean, std) Pareto 仅 {C2, TiRex}；λ=5 切换 | prior 必须含 variance 项 |

---

## 6. feedback Round 4 → 实现 mapping

| feedback 项 | 落点 | 文件 |
|---|---|---|
| Phase 1 Bayesian unification | §3.4 + §1.1 | bayesian_router.py |
| **Phase 2 Contextual bandit (Thompson)** | §3.5 + §4.1 observe loop | bandit.py + forecaster_reflect.py |
| Phase 3 Dynamic MoE | ❌ F2 ceiling 压低 ROI → future work | — |
| **Phase 4 Learned representation** | §3.2 + §3.3 | representation.py |
| §四 Risk-sensitive | risk_min decide + risk_metrics | utils/risk_metrics.py |
| §六 Cost-aware | cost_metrics + env_penalty | utils/cost_metrics.py |
| **§七 Regime manifold** | RegimePrior + k-means | representation.py |
| §八 Universal framework | 同一 BayesianRouter 接 forecasting + TSC | bayesian_router + clf_planner |
| §九 NOT add heuristics | Round 5 严格遵守 | — |

**完成 8.5/9 (94%)**。仅 Phase 3 因实测上界压低 ROI 放弃，作为 NEGATIVE finding 收入论文 §6 future work。

---

## 7. 评估指标

`research/utils/{prob_metrics, risk_metrics, cost_metrics}.py`

### 7.1 准确性

| 指标 | 用途 |
|---|---|
| MAE | 点预测主指标 |
| CRPS | 概率预测主指标 (`∫(F̂(z) - 1[z ≥ y])² dz`) |
| Pinball @ α | 单分位精度 |
| 80% coverage / width | 区间校准 |
| Accuracy / Macro-F1 | TSC |

### 7.2 风险敏感 (Phase 4-B P1)

$$\mathrm{risk}_k = \mathbb{E}[\ell_k] + \lambda \cdot \mathrm{std}[\ell_k]$$

实测 Pareto 仅 {chronos2, tirex} 非支配；λ=5 时 tirex 反超（F12 tipping point）。

### 7.3 Cost-aware (Phase 4-B P2)

$$c_k = \alpha \log(\mathrm{lat}) + \beta \log(\mathrm{params}) + \gamma \cdot \mathrm{env\_penalty} + \delta \log(\mathrm{VRAM})$$

`env_penalty`: 0 = main / 1 = alt-local / 2 = remote.

---

## 8. 部署矩阵

5 个 conda env：

| Env | python | torch | transformers | 用途 |
|---|---|---|---|---|
| `tsci` (local main) | 3.10 | 2.x cu118 | 4.45+ | 8/12 forecasting + 11/11 TSC |
| `tsci-py312` (local) | 3.12 | 2.5+ | <4.46 | moirai2 / toto2 |
| `tsci-remote` (192.168.1.102) | 3.9 | 2.8.0+cu128 | **4.57.1** | Timer-S1 (Blackwell GPU) |
| `tsci-remote-tx440` (远程) | 3.9 | 2.8.0+cu128 | **4.40.1** | time_moe / sundial |

远程 SSH: `c220@192.168.1.102` (2× RTX 5070 Ti 16GB)；HF cache `/data2/c220/hz/hf_cache/` + `HF_ENDPOINT=https://hf-mirror.com`。

---

## 9. Configuration Knobs

| Var | 取值 | 作用 |
|---|---|---|
| `ADAPTTS_PLANNER` | `bandit` / `bayesian` / `prior_aware` | 选 router |
| `ADAPTTS_DECIDE` | `argmax` / `thompson` / `risk_min` / `greedy` / `ucb` | 决策模式 |
| `ADAPTTS_RISK_LAM` | float (1.0) | risk_min λ |
| `ADAPTTS_ALLOW_REMOTE` | 0 / 1 | 远程模型纳入候选 |
| `ADAPTTS_BANDIT_PATH` | path | bandit state 持久化文件 |
| `ADAPTTS_BANDIT_DECAY` | float (1.0) | 非稳态遗忘率 |
| `ADAPTTS_CLF_PLANNER` | `bayesian` / unset | TSC 启用 Bayesian 路径 |
| `CLF_MEM_K` / `CLF_MEM_K_MIN` | int | TSC memory 检索 K |

---

## 10. 论文级 framing

**核心公式**：
$$\boxed{\hat{M}(x, h, t) = \arg\max_k \log \pi_k(z) + \log L_k(z, h, t), \quad z = f_\phi(x)}$$

**消融矩阵**（6 priors × 2 likelihoods × 3 modes × 3 embeddings = ~108 配置可独立开关）：

- prior on / off：each of {Availability, CRPS, Regime, Type, N, Entropy, Industrial}
- likelihood on / off：each of {CV, Memory, Representation}
- decide mode：argmax / thompson / risk_min / ucb
- embedding：hand25 / MOMENT / Chronos2
- bandit decay：1.0 (stationary) / 0.99 (drift)

每一维都对应一个**可独立证伪的命题**（F1-F12 中至少一条）。

---

## 11. 后续 Roadmap (Round 6+)

| 优先级 | 内容 | 入口 | 状态 |
|---|---|---|---|
| **P0** | 当前 162-cell sweep + full library sweep 跑完 → method.md §5 实测数据补全 | 已在跑 | 进行中 |
| P1 | Anomaly + RCA (问题三) 实现 | 见 **附录 A** | 未起 |
| **P2** | **online drift simulator → 验证 Thompson > greedy** | streaming benchmark | **由 §11.5 B3 部分覆盖** |
| P3 | MomentEmbedding 完整 sweep (vs hand25 baseline) | 已就位待跑 | 未跑 |
| P4 | Cross-env routing dispatcher (subprocess for remote models) | future work | 未起 |
| ~~P5~~ | ~~Dynamic MoE~~ | ❌ F2 ceiling，留 future paper | 已弃 |
| **R6-A** | 架构收敛：AdaptiveRouter + 统一 RouterState | `bayesian_router.py` / `router_state.py` / `adaptive_planner.py` | ✅ 完成 |
| **R6-B** | Self-Adaptive Closed Loop（Calibration + Drift + Action） | 见 §11.5 | ✅ 完成 |
| **R6-C** | Memory 演化：Failure Memory + Decay | `failure_memory.py` / `memory_decay.py` | ✅ 完成 |
| **R6-D** | Telemetry + Health Report | `telemetry.py` | ✅ 完成 |
| **R6-E** | Inference Scheduler（latency / VRAM-aware 升级链） | `inference_scheduler.py` | ✅ 完成（§11.5.5） |
| **R6-G/H/J** | 端到端 demo + 多数据集 stress + 长走查收敛 | `experiments/{g_real_demo, h_stress_demo, j_drift_convergence}.py` | ✅ 完成（实测见 `finish-1.md` §12） |

---

## 11.5 Round 6 后半 · Self-Adaptive Closed Loop (B2 + B3 + E1)

> 动机来自 `feedback.md` 后段：当前 Router 是 *single-shot* — 决策完就结束，缺 reflection / drift handling / calibration / action layer。本节是 Round 6 后半为弥补这些 gap 落的三个互锁子系统。

### 11.5.1 设计动机（Why）

`method2.md` §1-10 描述的 BayesianRouter 把每一步决策建模成「given Context+Evidence, choose model M_k」。这在**单次**决策上已经接近最优，但在**长时间运行**时缺：

| 缺口 | 工业后果 |
|---|---|
| 没人监控 Router 自身在退化 | 模型 ranking 早就过时却不知道 |
| posterior gap 当 confidence 用 | 高 raw conf 也可能历史成功率 < 30% |
| 决策止步 `chosen_model` | 系统不会说 "这个预测说要 SHUTDOWN" |

Round 6 后半三件套精准对应：
- **B2 Calibration** — *什么时候不能信自己*
- **B3 Drift Engine** — *系统正在偷偷退化时自我修正*
- **E1 Action Layer** — *从预测跨到决策*

三者通过 `RouterState` 单一容器共享状态，闭环到下一步 `adaptive_decide`。

### 11.5.2 B2 · Confidence Calibration

文件：`agent/calibration.py`（已并入 Round 6 B2）。

```
raw confidence ─►  ConfidenceCalibrator (isotonic, PAV monotone bins)
                         │
                         └─► calibrated P(correct | conf=c)  ─►  4-tier behavior
```

- 训练数据：`state.telemetry` 里每条 `(posterior_max, outcome ≤ threshold_quantile)`
- 4 档对应 Round 6 reflective_loop 的升级链：
  - `≥0.9` → fast_single (L0)
  - `≥0.5` → ensemble (L1)
  - `≥0.2` → specialist_escalate (L2)
  - `<0.2` → human_in_loop (L3)
- 增量重训：每 50 obs 由 `_get_or_fit_calibrator` 自动 refit，缓存在 `state._calibrator`

### 11.5.3 B3 · Drift Engine

文件：`agent/drift_engine.py`。完整闭环：

```
adaptive_observe
  └── 每 drift_check_every 次 obs (默认 50) → run_drift_step
         │
         ├── compute_drift  (5 signals — incl. F-R6.1 fix)
         │     ├─ feature_kl       ← memory_cases.z 投影直方图 KL
         │     ├─ residual_ks      ← outcome 两样本 KS 统计量
         │     ├─ routing_kl       ← chosen 分布 KL (recent vs history)
         │     ├─ memory_mismatch  ← per-regime outcome ±2σ shock rate
         │     └─ pred_residual_z  ← Welch mean shift | E[rec]−E[hist] | / σ_hist
         │                          (路由无关；F-R6.1 修复)
         │
         ├── recommend_actions
         │     ├─ boost_exploration   ← feature ∪ routing
         │     ├─ lower_memory_trust  ← residual ∪ memory
         │     └─ mark_regime_stale   ← memory
         │
         ├── apply_actions (mutate state via setattr — 不破坏 save/load)
         │     ├─ state.memory_trust ∈ (0,1]      → MemoryLikelihood 退权
         │     ├─ state.bandit_explore_scale ≥ 1  → BanditLikelihoodFactor 温度退火
         │     └─ state.regime_stale = True
         │
         ├── auto refit_regimes (when regime_stale)
         │     ├─ KMeans 重训 centroids + per-regime π
         │     ├─ bandit (n,s,sq) × 0.3 软重置
         │     └─ regime_stale → False
         │
         └── append state.drift_history (持久化到 router_state.jsonl)
```

**关键约束**：所有 drift 状态都是 `state` 上的「软」属性 (`setattr`)，BayesianRouter 的核心因子通过 `getattr(state, ..., default)` 读取。这意味着旧代码 / 没传 `state_ref` 的因子完全无感知，行为与 Round 5 一致。

**消费侧的真正落地**：
- `MemoryLikelihood.log_lik = log(votes) * memory_trust`（trust=0.3 ⇒ 30% 影响）
- `BanditLikelihoodFactor`：`log_lik /= explore_scale` 退火 + Thompson σ × `explore_scale`

### 11.5.4 E1 · Action Layer

文件：`agent/action_layer.py`。把 forecast 升级成 decision：

```
ForecastDist (mean, std)            ─┐
ActionContext (upper/lower threshold)─┼─► assess_risk → expected_costs → choose_intervention
calibrated confidence  (来自 B2)     ─┘
                                        │
                                        ▼
                                 ActionDecision
                                 ├─ intervention ∈ {MONITOR, INSPECT, THROTTLE, SHUTDOWN, ESCALATE}
                                 └─ reason (calibration provenance + tier + cost-min trace)
```

**策略**：3 条硬覆盖（解释性优先）+ 默认 argmin 期望成本
- 高风险 (p≥0.7) + 高 calibrated conf (≥0.7) → **SHUTDOWN**
- 高风险 + 低 conf (<0.3) → **ESCALATE**（不盲目动作）
- 低 conf + 中风险 (p≥0.4) → **INSPECT**（先查证再决定）
- 其他 → argmin Σ_action [ p_breach · C(action, breach) + (1-p) · C(action, safe) ]

成本矩阵默认值仅为示例，生产部署需按 asset / 业务覆写。

### 11.5.5 R6-E · Inference Scheduler

文件：`agent/inference_scheduler.py`。对应 feedback 后§4 "Runtime Orchestrator"。把"router 选出 top-1 直接跑"升级成预算感知的升级链：

```
posterior  ─►  rank by mass
                 ▼
        for each candidate (top-K):
            U(M) = accuracy_gain(M) × (1 − current_confidence)
                 − w_latency · latency(M)
                 − w_vram    · vram(M)
                 − (remote_penalty if remote)
            ├─ U > 0 AND budget OK  → run, bump confidence (+agreement_bonus)
            └─ else                  → skip, early_stop=True
```

**4 个独立 stop 条件**：`U ≤ 0` / `latency_budget_s` / `vram_budget_gb` / `min_confidence_stop ≥ 0.95`。

**集成**：`schedule(posterior, profiles, config, init_conf)` 任意可调；`schedule_from_state(plan, state)` 自动消费 `AdaptivePlan.posterior + state._calibrator` (B2) calibrated conf。调度器**只决定 run-list**，实际 ensemble 由上层；与 B1 reflective_loop 互补（B1 是 run-后 retry，R6-E 是 run-前规划）。`default_profiles()` 给 15 个现有模型一组 (latency, vram, accuracy_gain) 缺省值，部署时按线下测量覆写。

### 11.5.6 与 §1-§10 的关系

| 旧节 | Round 6 后半补充 |
|---|---|
| §1.1 决策规则 (single-shot) | 由 E1 升级成 forecast → action decision |
| §1.3 在线更新 (bandit decay) | 由 B3 升级成 4 信号自适应（decay 全局 → per-state） |
| §3.4 BayesianRouter 6 priors + 2 likelihoods | `MemoryLikelihood` / `BanditLikelihoodFactor` 接入 `state_ref` 消费 drift state |
| §3.6 Memory Layer | C1 Failure Memory + C2 Decay 已在 Round 6 中半；B3 用 `memory_trust` 给 Memory 加 runtime 信任度 |
| §4.1 forecasting 入口 ("跑 top-1") | R6-E Inference Scheduler 改成预算感知升级链；调度+ensemble 分离 |
| §8 部署矩阵 | R6-E 直接消费 latency / VRAM 预算，edge / cloud 部署差异由 SchedulerConfig 表达 |
| §10 论文级 framing | Round 6 后半把 "router" 升级到 "self-adaptive runtime + decision engine" |
| §5 实测发现 | 新增 F-R6.1 ~ F-R6.4 四条；详见 `finish-1.md` §12 |

### 11.5.7 实测与 Findings → `finish-1.md` §12

R6 后半的全部实测（合成 e1_action_demo / 真实 g_real_demo / 多数据集 stress h_stress_demo / 长走查 j_drift_convergence）和 4 条 Findings（F-R6.1 路由无关 drift 信号 pred_residual_z / F-R6.2 单模型 std 残差回退 / F-R6.3 扰动方向 + 量级敏感 / F-R6.4 drift 修正幂等 + 终态收敛）已剥离到 `finish-1.md` §12，本节不再重复。

---

## 12. 文件地图

```
research/
├── README.md                  # 项目目标 + 路线图
├── method2.md                 # 本文件（自顶向下方法）
├── method.md                  # 旧版（Round 4-A，已被 method2 取代）
├── finish.md                  # Phase 1-6 实测
├── finish-1.md                # Round 2-5 实测 + Findings F1-F12
├── feedback.md                # 当前 round review（每轮覆盖）
├── ppt.md                     # 讲述用文字稿
├── paper_draft.md             # 论文稿
│
├── agent/
│   ├── curator_uq.py          # Curator 诊断
│   ├── model_cards.py         # 16 forecast cards
│   ├── clf_model_cards.py     # 11 TSC cards
│   ├── representation.py      # Phase 4: Embedding + RegimeAssigner + RegimePrior + RepresentationLikelihood
│   ├── prior_crps.py          # Round 4-A: dataset-keyed CRPS prior + BMA
│   ├── bayesian_router.py     # Round 5/6: Context / Evidence / BayesianRouter + 6 priors + 2 likelihoods
│   │                          #            Round 6 B3: Memory/Bandit factor state_ref → memory_trust / explore_scale
│   ├── bandit.py              # Phase 2: BanditState + ContextualBanditRouter
│   ├── planner_prior_aware.py # Round 4-A baseline (for ablation)
│   ├── planner_adaptive.py    # legacy (Round 0-3 baseline)
│   │
│   │ ── Round 6 后半 · self-adaptive closed loop (§11.5) ──
│   ├── router_state.py        # R6-A2: unified RouterState (bandit+memory+regime+telemetry+drift_history)
│   ├── adaptive_planner.py    # R6-A3: adaptive_decide/observe — single entry; B3 auto-trigger inside observe
│   ├── calibration.py         # R6-B2: ConfidenceCalibrator (isotonic PAV) + behavior_tier
│   ├── drift_engine.py        # R6-B3: 4 signals + 3 actions + refit_regimes + drift_history
│   ├── action_layer.py        # R6-E1: ForecastDist → assess_risk → choose_intervention → ActionDecision
│   ├── telemetry.py           # R6-D1: HealthReport (auto-pulls drift_engine signals)
│   ├── failure_memory.py      # R6-C1: FailureCase store + signal pattern aggregation
│   ├── memory_decay.py        # R6-C2: exp time decay + regime drift KL
│   ├── reflective_loop.py     # R6-B1: progressive inference L0→L1→L2→L3
│   ├── inference_scheduler.py # R6-E:  latency/VRAM/remote-aware cascade — chooses which models to RUN
│   ├── forecaster_reflect.py  # forecasting 主入口 + observe_outcome
│   ├── clf_planner.py         # TSC 主入口
│   ├── clf_memory.py          # TSC Memory + diverse + inv_loss vote
│   ├── memory.py              # Forecasting memory
│   ├── quantile_ensemble.py   # L2 soft pool
│   └── gated_residual.py      # NEGATIVE finding F1 实证
│
├── baseline/                  # 12 forecast + 11 TSC predict()
├── utils/                     # series_features / prob_metrics / risk_metrics / cost_metrics / data_loader
├── experiments/               # bandit_full_sweep / full_library_sweep / a3_prob_metrics / ...
│   ├── e1_action_demo.py      # R6-E1 端到端 demo: forecast → action → observe → drift loop（合成数据）
│   ├── g_real_demo.py         # R6-G 实数据端到端: ETTh1 + adaptive_decide + scheduler + ensemble + action + drift
│   ├── h_stress_demo.py       # R6-H 多数据集 × 4 故障注入 sanity (ETTh1/ETTh2/Exchange × clean/trend/var/outlier)
│   └── j_drift_convergence.py # R6-J 140-step 长走查：验证 drift 修正幂等 + pred_residual_z 在 injected 路径独立 fired
├── scripts/                   # remote_sweep / remote_smoke
└── results/                   # *.jsonl 实测数据
```

---

# 附录 A · 问题三 Anomaly + RCA 解决方案设计（待实现）

> 本附录是 **future implementation spec**，不是已实现内容。给出系统设计、模块接口、评估指标，作为下一阶段工程入口。

## A.0 任务定义

输入：
- 多通道传感器时序：振动 (vibration_xyz) / 电流 (current_3ph) / 温度 (temp) / 转速 (rpm)
- 设备元数据：设备类型 / 维护周期 / 历史工单
- （可选）现场图片：油液 / 表面磨损

输出：
1. **二元 anomaly score** (是否异常，∈ [0, 1]) + 置信度区间
2. **根因 (RCA) 分类**：unbalance / misalignment / bearing_fault / gear_wear / electrical_fault / belt_loose / lubrication / 其它
3. **严重度等级**：observe → warning → action → shutdown
4. **预估剩余寿命 (RUL)** + uncertainty

## A.1 系统架构

```
多通道时序
  ↓
┌─────────────────────────────────────┐
│ Pre-Anomaly Layer                    │  utils/anomaly_features.py (新)
│  ├─ 滑窗 baseline: rolling MAE/STD   │
│  ├─ FFT band energy (8 freq bands)   │
│  ├─ MCSA (electrical signature)      │
│  └─ rule baseline: ADF / IQR / 工程经验
└─────────────────────────────────────┘
  ↓ (continuous anomaly score)
┌─────────────────────────────────────┐
│ BayesianRouter (复用现有)             │  bayesian_router.py
│  Candidates = {                       │
│    anomaly_transformer,               │  baseline/anomaly_transformer.py (新)
│    gpt4ts,                            │  baseline/gpt4ts.py (新)
│    rule_baseline,                     │  baseline/rule_anomaly.py (新)
│    rocket_classifier (already),       │
│    moment_zero_shot (already)         │
│  }                                    │
│  + AnomalyTypePrior (新 PriorFactor) │
└─────────────────────────────────────┘
  ↓ (anomaly type label + score)
┌─────────────────────────────────────┐
│ RCA-LLM Agent                        │  agent/rca_llm.py (新)
│  Input:                               │
│    - anomaly windows                  │
│    - Curator diagnostic               │
│    - Selected classifier output       │
│    - Model Cards (rendered)           │
│    - Memory retrieval (similar cases) │
│  Output:                              │
│    - 根因分类 + 推理链                 │
│    - 严重度                           │
│    - RUL estimate + CRPS              │
│    - 建议行动 (检查 / 停机)            │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ Counterfactual Memory                │  agent/rca_memory.py (新)
│  存储:                                │
│    (anomaly_window, ground_truth,    │
│     all_classifier_outputs,           │
│     LLM reasoning chain)              │
│  retrieval: query_diverse + inv_loss  │
└─────────────────────────────────────┘
```

## A.2 新模块接口设计

### A.2.1 `utils/anomaly_features.py`

```python
def anomaly_features(series: np.ndarray, window: int = 100,
                    sample_rate: float = 1.0) -> dict:
    """Returns 12-d anomaly-specific feature vector.

    {rolling_mae_z, rolling_std_z,
     fft_band_energy[0..7],   # 8 frequency bands
     adf_pvalue,
     iqr_outlier_ratio}
    """
```

### A.2.2 `baseline/rule_anomaly.py`

```python
def predict_anomaly_score(series, window=100, threshold_z=3.0) -> tuple[np.ndarray, dict]:
    """Pure rule baseline.
    Returns (score_per_step ∈ [0,1], metadata={'method': 'rule', 'threshold': z}).
    F4 saturation hypothesis: TSFM 异常检测未必胜规则 → 强制保留 baseline.
    """
```

### A.2.3 `baseline/anomaly_transformer.py`

```python
def predict_anomaly_score(series, model_path="thuml/Anomaly-Transformer") -> tuple[np.ndarray, dict]:
    """Anomaly Transformer (ICLR 2022) wrapper."""
```

### A.2.4 `agent/rca_llm.py`

```python
class RCAAgent:
    def __init__(self, llm_model: str = "qwen-2.5-72b",
                 memory_path: str = "research/results/rca_memory.jsonl",
                 max_retrieval: int = 5):
        ...

    def diagnose(self, series, anomaly_score, candidate_outputs: dict,
                 device_meta: dict) -> RCAResult:
        """Multi-step reasoning:
        1. Retrieve top-K similar cases from memory (diversity-enforced)
        2. Render Model Cards + retrieved cases into prompt
        3. LLM proposes root cause + severity + RUL
        4. Self-check: re-rank by counterfactual likelihood
        5. Return structured RCAResult
        """
        ...

@dataclass
class RCAResult:
    root_cause: str           # one of {unbalance, misalignment, ...}
    severity: str             # observe / warning / action / shutdown
    rul_days: float | None    # estimated remaining useful life
    rul_crps: float | None
    confidence: float         # ∈ [0, 1]
    reasoning_chain: list[str]
    retrieved_cases: list[dict]
    counterfactuals: dict     # "if root_cause were X, expected signal would be Y"
```

### A.2.5 `agent/rca_memory.py`

```python
class RCAMemory:
    """Counterfactual memory for RCA.

    Each case stores:
        - anomaly_window: np.ndarray
        - feature_vec: 12-d anomaly features
        - ground_truth: {root_cause, severity, rul_actual}
        - candidate_outputs: dict[classifier_name -> output]
        - llm_reasoning: list[str] (chain)
        - outcome: was the recommended action correct?
    """
    def add(self, case: RCACase): ...
    def query_diverse(self, feat: np.ndarray, k: int = 5,
                       default_cause: str = "unbalance") -> list[RCACase]: ...
```

### A.2.6 BayesianRouter 扩展

新增 `AnomalyTypePrior(PriorFactor)`：

```python
@dataclass
class AnomalyTypePrior(PriorFactor):
    """Prior over candidate classifiers conditioned on anomaly_features.

    e.g. high FFT band 3 (gear mesh freq) → boost gear_wear classifier
         high MCSA sideband → boost electrical_fault
    """
    fft_to_classifier_map: dict[str, dict[str, float]]  # learned offline
    def log_prior(self, candidates, ctx):
        if ctx.features is None or "anomaly_feats" not in ctx.features:
            return {m: 0.0 for m in candidates}
        af = ctx.features["anomaly_feats"]
        # weighted sum based on dominant band
        ...
```

## A.3 评估指标

| 指标 | 公式 | 数据集 |
|---|---|---|
| **Precision @ alert** | TP / (TP + FP) | 工业告警频次约束 |
| **Recall @ severity≥action** | TP / (TP + FN) on critical | 漏报代价高 |
| **F1 per root cause** | per-class | 不平衡评估 |
| **RUL CRPS** | continuous ranked probability score on remaining life | 寿命预测置信 |
| **Time-to-alert latency** | t_alert − t_anomaly_onset | online 延迟 |
| **Reasoning faithfulness** | LLM 推理链是否一致 | 人工评分 |

## A.4 标准数据集

| 数据集 | 用途 |
|---|---|
| CWRU Bearing | 轴承故障 4 类 × 多速度 |
| IMS Bearing | 长时间运行至失效 (RUL ground truth) |
| MFPT Bearing | 振动 + 多 RPM |
| Paderborn | 多故障 + 多负载 |
| C-MAPSS | 涡扇 RUL (NASA) — 4 子集 FD001-004 |
| PHM 2008 / 2010 | 机械健康竞赛集 |

UCR archive 的 InsectWingbeat / EthanolConcentration 也可作 anomaly proxy。

## A.5 实施计划（粗）

| Phase | 时间 | 目标 | 阻塞 |
|---|---|---|---|
| RCA-P1 | 1 周 | rule baseline + anomaly_features + CWRU smoke | none |
| RCA-P2 | 1 周 | wire Anomaly-Transformer + GPT4TS into BayesianRouter | env compat |
| RCA-P3 | 2 周 | RCAAgent (LLM) + retrieval-grounded prompt | LLM API 配额 |
| RCA-P4 | 1 周 | RCAMemory + counterfactual storage + diverse retrieval | — |
| RCA-P5 | 1 周 | end-to-end on CWRU + IMS + C-MAPSS | dataset 下载 |
| RCA-P6 | 1 周 | evaluation: precision/recall/CRPS/latency vs SKF rule-only baseline | — |

## A.6 重用现有 Round 5 框架的程度

| 现有组件 | 在 Anomaly+RCA 中的复用 |
|---|---|
| `BayesianRouter` | ✅ 直接复用，只换 candidates + 加 AnomalyTypePrior |
| `RegimeAssigner` | ✅ 在 anomaly features 上聚类 → fault regime |
| `MemoryLikelihood` | ✅ 复用，case 字段扩展 |
| `BanditState` | ✅ per-(fault_regime, classifier) belief, online update |
| `decide(thompson)` | ✅ exploration 用于发现新故障模式 |
| `decide(risk_min)` | ✅ 严重度高时 (action/shutdown) λ↑ |
| `cost_metrics` | ✅ LLM API 调用是显性 cost |
| Model Cards | ✅ 每个 anomaly classifier 写 5 字段卡 |

**估计 70% 代码量是新的（baselines + LLM agent + RCA-specific features）；30% 是配置 + 接入现有 BayesianRouter。**

## A.7 论文级展开

新 anomaly + RCA paper 可独立写：
- §1-3 引用本框架（不重复 method）
- §4 Anomaly-specific contributions:
  - Counterfactual reasoning chain (LLM grounded by retrieved cases)
  - RUL with CRPS confidence interval
  - Reasoning faithfulness metric
- §5 实测 vs SKF / GE Bently Nevada rule-only baseline
- §6 与 ICLR 2026 TSFM-anomaly 质疑论文对话：哪些场景 TSFM 真有帮助，哪些不

---

**End of method2.md**
