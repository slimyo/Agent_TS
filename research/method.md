# Method · TSci-Demo Multi-Agent Time-Series System

> 版本：2026-05-27 (Round 3 闭环后)
> 配套文档：`finish.md`（实测日志）/ `finish-1.md`（Round 2/3 优化）/ `feedback.md`（外部 review）/ `paper_draft.md`（论文稿）

本文件**自顶向下**说明完整方法：从问题、库构成、特征工程，到 prior-aware hierarchical planner、quantile-pool 软路由、反事实记忆、ε-greedy 探索的端到端 pipeline。读完即可在不打开代码的前提下复述系统全貌；同时也是**代码地图**——每节脚注真实文件路径，方便对照。

---

## 0. 问题与范围

我们构建一个 **零样本/小样本时序基础模型路由系统**，覆盖两类下游任务：

| Track | Task | Eval | 默认 baseline |
|---|---|---|---|
| **A. Forecasting** | 给 N 个历史样本 + horizon H，输出点预测 + 9 quantile 预测分布 | MAE / CRPS / pinball / 80%-coverage / 80%-width | Chronos-2 (C2) |
| **B. TSC** | 给 N-per-class few-shot 训练集，对 X_test 输出类别 | accuracy / macro-F1 | Rocket |

**核心 thesis**（feedback 第 1 轮 confirmed）：单个 TSFM 在所有数据集上**不存在 dominance**；正确的架构是 **library + prior-aware router**，把单一模型的 saturation 转化为可发现的 routing space。

---

## 1. 顶层架构

```
┌─────────────┐
│   Series    │                                                                     
│ (train,val) │
└──────┬──────┘
       ↓
┌──────────────────────┐    诊断+置信度    ┌───────────────────────────┐
│  Curator             │  ───────────────▶│  Prior-Aware Planner       │
│  curator_uq.py       │                  │  planner_prior_aware.py    │
│  series_features.py  │                  │  ─────────────────────────  │
└──────────────────────┘                  │  1. compose_prior:          │
                                          │     static π_k (1/MAE)      │
                                          │     × type-prior            │
┌──────────────────────┐                  │     × N-prior (N<15 → C2)   │
│  prior_crps.py       │ ────── π_k ─────▶│     × availability mask     │
│  (Item 2, BMA)       │                  │  2. L0 triage:              │
└──────────────────────┘                  │     if π(C2)≥0.45 → L1      │
                                          │     else                → L2 │
┌──────────────────────┐                  │  3. BMA posterior (if CV):  │
│  Model Cards         │ ──── meta ──────▶│     ∝ exp(-loss/σ²)·π_k     │
│  model_cards.py      │                  │  4. (optional) ε-greedy     │
│  clf_model_cards.py  │                  │     epsilon_greedy_perturb  │
└──────────────────────┘                  └────────────┬───────────────┘
                                                       │ PriorPlan
                                                       ↓
            ┌──────────────────────────────────────────────────────────┐
            │ L1: single Chronos-2          │ L2: top-K ensemble        │
            │  baseline/chronos2.py         │  quantile_ensemble.py     │
            │  (zero overhead)              │  weighted linear pool     │
            └────────────────┬─────────────────────┬────────────────────┘
                             │                     │
                             ↓                     ↓
                  ┌──────────────────────────────────────┐
                  │  Memory (counterfactual + diverse)   │
                  │  agent/memory.py  /  clf_memory.py   │
                  │  all_clf_accs + query_diverse +      │
                  │  consensus_winner_inv_loss           │
                  └────────────────────┬─────────────────┘
                                       ↓
                  ┌──────────────────────────────────────┐
                  │  Output                              │
                  │   - point pred  (median)             │
                  │   - quantile  [9, H]                 │
                  │   - prob metrics (CRPS, pinball...)  │
                  └──────────────────────────────────────┘
```

---

## 2. 输入：序列特征 (Curator + featurize)

### 2.1 Curator (诊断 + 置信度)

`research/agent/curator_uq.py`

对单条序列输出 `Diagnosis` 对象（三维度 × 三置信源）：

| 维度 | 三值 | 置信度来源 |
|---|---|---|
| `trend` | rising / flat / falling | `_stat_confidence` (统计) + `_llm_confidence` (LLM) + `_cross` (交叉) |
| `season` | strong / weak / unclear | 同上 |
| `stat` (稳态性) | stationary / mean_shift / variance_explode | 同上 |

每维度 confidence ∈ {low, mid, high}；下游 planner 用最低置信值决定保守度（feedback §三.3 L0 信号之一）。

### 2.2 完整特征向量 (25-dim, z-scored)

`research/utils/series_features.py`

| 子集 | dim | 项 |
|---|---|---|
| basic | 5 | mean/std/median/iqr/range |
| trend | 4 | slope/r2/curvature/dominance |
| freq (FFT) | 4 | spectral entropy / 3 dominant freqs |
| complexity | 4 | PermutationEntropy / DFA / SampEn / approximate entropy |
| outlier | 3 | tail ratio / mad outlier frac / max-z |
| industrial | 5 | acf_decay / quant_bits / level_count / step_count / signature |

用于：
- **Memory key** (TSC `featurize_cell` 25-d → z-score → L2-norm 给 ClfMemory 检索)
- **Industrial-regime override** (Wafer-like detection → prefer Euclid 1-NN)

---

## 3. Model Library (Round 3 整理)

### 3.1 Forecasting (12 models, 16 cards)

`research/baseline/`（predict API）+ `research/agent/model_cards.py`（结构化能力卡）

| 类别 | 模型 | params | env | card |
|---|---|---|---|---|
| **trivial point** | naive_drift, naive_seasonal, arima_ets, llmtime | <1M / LLM | tsci | ✅ |
| **Chronos family** | chronos (60M), chronos_bolt (200M alias), **chronos2 (120M, default)** | 60M-200M | tsci | ✅ |
| **TSFM 主流** | timesfm2 (500M), moirai (311M), moirai2 (11M) | 11M-500M | tsci/tsci-py312 | ✅ |
| **niche specialists** | tirex (128M, xLSTM), toto (151M observability), toto2 (4M) | 4M-151M | tsci/tsci-py312 | ✅ |
| **remote-only large** | time_moe (50M), sundial (128M), timer (8.3B MoE) | 50M-8.3B | tsci-remote / tsci-remote-tx440 | ✅ |

**Model card 五字段**：`class / assumes / strengths / weaknesses / typical_failure` — 反思层 prompt 直接 render，让 LLM 基于显式先验做选型。

### 3.2 TSC (11 classifiers, B7v3/v4 router)

`research/baseline/tsc_classical.py` + `research/agent/clf_strategies.py` + `research/agent/clf_model_cards.py`

| 类别 | 实例 |
|---|---|
| distance | dtw_1nn, euclid_1nn |
| kernel | rocket (**default**), minirocket |
| TSFM-embed | moment_1nn, moment_logreg, mantis_1nn, mantis_lr |
| dictionary | weasel (**aggregate +2.7pp NEW SOTA**) |
| feature | catch22 |
| LLM | llm_direct |

---

## 4. Prior-Aware Hierarchical Planner

> 核心方法贡献。`research/agent/planner_prior_aware.py`

### 4.1 4-层先验栈 (`compose_prior`)

```python
prior = static_π_k(dataset)              # Item 2: 1/MAE 归一
prior = apply_type_prior(prior, 0.3)     # feedback §二.2: 点预测器降权
prior = apply_n_prior(prior, N)          # feedback §二.3: N<15 → C2=0.9
prior = apply_availability(prior, allow_remote)  # local vs remote mask
```

各层公式：

| 层 | 公式 |
|---|---|
| 1. Static π_k | π_k = (1/loss_k) / Σ_j (1/loss_j)，loss 从 `gated_residual_cells.jsonl` + `*_vs_c2.jsonl` 自动聚合 |
| 2. Type prior | w_k ← w_k · 0.3，**if** k ∈ POINT_PREDICTORS (naive_*, arima_ets, llmtime) |
| 3. N prior | **if** N<15: w_C2=0.9，其余瓜分 0.1 (按比例)。否则保留原值 |
| 4. Availability | drop k ∉ LOCAL ∪ (REMOTE if allow_remote) |

### 4.2 L0/L1/L2 分层 (`make_prior_plan`)

```python
if π(C2 | dataset) ≥ 0.45  AND  N ≤ 500  AND  cv_losses is None:
    → L1: PriorPlan(strategies=["chronos2"], weights=[1.0])    # 快通道，0 ensemble cost
else:
    → L2: top-K by (BMA posterior if cv_losses else π_k)
         PriorPlan(strategies=[m1,m2,m3], weights=[w1,w2,w3])
```

### 4.3 BMA 后验 (Bayesian Model Averaging)

feedback §二.4 数学闭环：

$$p(M_k \mid \text{data}) \propto \exp(-\text{loss}_k / \sigma^2) \cdot \pi_k$$

实现在 `prior_crps.bma_posterior`：log-sum-exp 数值稳定。σ² 控温度：
- σ² 小 → posterior 集中到 min-loss 模型（hard 决策）
- σ² 大 → 接近先验

### 4.4 ε-Greedy 探索 (feedback §三.4)

```python
plan_perturbed, was_explored = epsilon_greedy_perturb(plan, eps=0.2, rng)
```

以 ε 概率从 non-top-1 posterior 抽样替换 top-1。**目的**：闭环反事实数据收集 — memory 否则会陷入 default 单峰。L1 单模型 plan 自动跳过（无替代）。

### 4.5 与旧 `planner_adaptive.py` 的接续

`planner_adaptive.py` 是手调启发式（4 模型库 + 硬编码权重），保留作为 baseline。运行时切换：
```bash
export ADAPTTS_PLANNER=prior_aware    # 启用新 planner
export ADAPTTS_ALLOW_REMOTE=1         # 含远程模型
```

---

## 5. 软路由 / Quantile Ensemble

> `research/agent/quantile_ensemble.py` — feedback §三.1 软路由的概率版

### 5.1 公共栅格

`TARGET_LEVELS = [0.1, 0.2, ..., 0.9]`（9-level，TSFM 最小公分母）。
- Chronos-2 native 21-grid → `_align_to_target` 线性插值
- TiRex/Timer/Time-MoE/Sundial/Toto/TimesFM/Moirai 直接 9-level
- 点预测器 (naive/arima/llmtime) → degenerate quantile（Dirac at median）

### 5.2 线性池公式

$$q^{\text{ens}}_{\ell, t} = \sum_k w_k \cdot q^{(k)}_{\ell, t}$$

精确实现 feedback §三.1 伪代码。一致性保证（CRPS 是 proper scoring rule，线性池可直接最小化）。

### 5.3 调用

```python
plan = make_prior_plan(dataset="ECL", N=100, H=96, allow_remote=False)
# → PriorPlan(level="L2", strategies=["toto","tirex","chronos2"], weights=[0.45,0.30,0.25])

result = ensemble_predict(plan, train, val, H=96, seed=42)
# → EnsembleResult(median=[H], quantiles=[9,H], per_model_quantiles={...}, per_model_weights={...})
```

---

## 6. Memory Layer

两套并行实现，结构对称：

| | Forecasting | TSC |
|---|---|---|
| 文件 | `agent/memory.py` | `agent/clf_memory.py` |
| Case 字段 | feature, best_strategy, mae, all_strategy_maes | diag_feature, best_classifier, test_acc, **all_clf_accs** (反事实) |
| 检索 | top-K cosine | top-K cosine + `query_diverse` |
| 投票 | safety-net override | `consensus_winner_weighted` (top-1) / `consensus_winner_inv_loss` (1/CRPS) |

### 6.1 反事实存储 (feedback §三.5 Item 4)

`ClfCase.all_clf_accs = {clf_name: test_acc}` — 完整候选集结果，不只 best winner。
- 旧 vote 仅看 best_classifier，浪费 K-1 个候选信息
- 新 `consensus_winner_inv_loss` 按 `sim · 1/(1-acc+ε)` 加权所有候选 → 利用反事实完整信号

### 6.2 多样性强制检索 (`query_diverse`)

```python
top_k = retrieve_by_cosine(query, k=5)
if all(c.best_classifier == default for _, c in top_k):
    # 全 default winner → 丢最低 sim default，插入最高 sim non-default
    replace_lowest_with_best_alt(top_k)
```

破解 "memory 陷入 default 单峰" 的 hindsight bias（feedback §三.5）。

### 6.3 1/CRPS 加权投票（Item 3）

$$\text{vote}(\text{clf}) = \sum_{n \in \text{neighbors}} \text{sim}(n) \cdot \frac{1}{1 - \text{acc}_n[\text{clf}] + \epsilon}$$

每邻居为**全部**候选投票（不仅 top-1），权重 = 检索相似度 × 反损失。

---

## 7. Track A 流程: Forecasting

`research/agent/forecaster_reflect.py · forecast_with_reflection()`

```
input: (train, val, H, diag, dataset?)
  │
  ├─ N < 12 → cold_start_plan (chronos+naive_drift) [legacy adaptive]
  │
  ├─ if ADAPTTS_PLANNER=prior_aware:
  │     plan = make_prior_plan(dataset, N, H, allow_remote=$)
  │   else:
  │     plan = make_plan(diag, N, H, conf_source="xc")   [legacy 4-model]
  │
  ├─ if walk_forward:
  │     cv_losses = walk_forward_mae(plan.strategies, train, H_val)
  │     plan.weights = softmax(-cv_losses / τ)
  │
  ├─ A1/v12 entropy gating: if C2 quantile spread > τ_entropy:
  │     plan.margin ← higher (more skeptical of CV deviation)
  │
  ├─ if reflect:
  │     model_cards → LLM 反思 → maybe revise diagnosis or swap strategy
  │
  ├─ if plan.level == "L1":
  │     pred = chronos2.predict(...)              # 0 cost ensemble
  │   else:
  │     result = ensemble_predict(plan, ...)      # §5 quantile pool
  │     pred = result.median ;  quantiles = result.quantiles
  │
  └─ A3 prob metrics: CRPS / pinball / coverage / width on quantiles
```

### 7.1 v11 wrapper (safety-net memory)

C2 一定有 baseline pred；wrapper 只在 memory 强支持时才偏离。最终战绩 0W/1L/23T MAE = guaranteed parity（finish §3.1.27）。

### 7.2 v13 entropy + memory 联合

A1 entropy gating + v11 memory → 同样 0W/0L/16T（finish §3.1.23-25）。CRPS 无退步（§3.1.26）。

---

## 8. Track B 流程: Classification

`research/agent/clf_planner.py · classification_planner()`

```
input: (X_train, y_train, X_test, season_m, dataset?)
  │
  ├─ N_per_class < 7 → force default (rocket)    [B7v2 N-fallback, +0.87pp]
  │
  ├─ run LOO/K-fold CV across candidates
  │   → cv_accs = {clf: acc_estimate}
  │
  ├─ Margin gating:
  │     if best_other - default ≥ margin (0.10) → chosen = best_other
  │     else → chosen = default
  │
  ├─ if use_memory:
  │     featurize_cell(X_train) → z-score → L2 norm
  │     if use_diverse_retrieval:                              [Item 4]
  │         neighbors = mem.query_diverse(feat, k=5)
  │     else:
  │         neighbors = mem.query(feat, k=5)
  │     vote_fn = inv_loss if vote_method="inv_loss"           [Item 3]
  │              else consensus_winner_weighted
  │     mem_winner = vote_fn(neighbors, min_vote_ratio=0.6)
  │     if mem_winner: override CV chosen
  │
  ├─ industrial_signature override:
  │     if acf_decay high AND quant_bits low → prefer euclid_1nn
  │
  └─ predict_with(chosen, X_train, y_train, X_test)
```

### 8.1 B7 series 进化

| Version | Δ vs Rocket | 关键改动 |
|---|---|---|
| B7 (task #35) | -2.7pp | initial LOO + margin |
| B7v2 (#37) | -0.87pp | + N<7 fallback |
| B7v3 (#41) | **+0.89pp** ⭐ | + 25-dim z-scored features + weighted vote (任务 #38+#39) |
| B7v4 (#66) | +0.95pp | + industrial-regime override |

### 8.2 B7v5 (本轮新)

启用 Item 3+4：
```python
classification_planner(..., use_memory=True, use_enhanced_features=True,
                       vote_method="inv_loss", use_diverse_retrieval=True)
```

预计：反事实利用 + 多样性 → 进一步降低 catastrophic mis-route。**实测待 sweep**。

---

## 9. 评估指标

`research/utils/prob_metrics.py` + `research/utils/metrics.py`

### 9.1 Forecasting

| 指标 | 公式 | 用途 |
|---|---|---|
| MAE | $\frac{1}{H} \sum |y_t - \hat{y}_t|$ | 点预测主指标 |
| CRPS | $\int (\hat{F}(z) - \mathbb{1}[z \geq y])^2 \, dz$ ≈ $\sum_\ell \frac{2}{|Q|}(\alpha_\ell - \mathbb{1}[y < q_\ell])(q_\ell - y)$ | 概率主指标 |
| Pinball @ α | $(\alpha - \mathbb{1}[y < q_\alpha])(y - q_\alpha)$ | 单分位精度 |
| 80% coverage | $\Pr[q_{10} \leq y \leq q_{90}]$ (empirical) | 区间校准 |
| 80% width | $q_{90} - q_{10}$ | 区间紧致度 |

### 9.2 TSC

- Accuracy / Macro-F1（class imbalance 时 macro 优先）

---

## 10. 部署矩阵 (Local vs Remote, finish-1 §4)

### 10.1 5 个 conda env

| Env | python | torch | transformers | 用途 |
|---|---|---|---|---|
| `tsci` (local) | 3.10 | 2.x cu118 | 4.45+ | 主 env (8/12 forecasting + 11/11 TSC) |
| `tsci-py312` (local) | 3.12 | 2.5+ | <4.46 | moirai2 / toto2 (uni2ts 2.0 兼容) |
| `tsci-tx440` (local, deprecated) | 3.10 | 2.x | 4.40.1 | (旧) time_moe / sundial 回归 |
| `tsci-remote` (`c220@192.168.1.102`) | 3.9 | 2.8.0+cu128 | **4.57.1** | Timer-S1 (Blackwell GPU) |
| `tsci-remote-tx440` (远程) | 3.9 | 2.8.0+cu128 | **4.40.1** | time_moe / sundial (旧 API) |

### 10.2 远程 SSH

```bash
ssh c220@192.168.1.102        # 2× RTX 5070 Ti, 16GB each, sm_120
密码: cinter
workdir: /data2/c220/hz/agent_ts/
HF cache: /data2/c220/hz/hf_cache/ + HF_ENDPOINT=https://hf-mirror.com
```

### 10.3 Cross-env routing 调度（future work）

当前一个 plan 内不能跨 env 调用模型。`allow_remote=True` 仅在远程 sweep 离线收集 `*_vs_c2.jsonl` 后，本地 prior_crps 自动纳入这些模型的 π_k。在线运行需 subprocess dispatcher (paper §5 future work)。

---

## 11. Module 依赖图

```
                    ┌─────────────────────────────────────┐
                    │  utils/                             │
                    │   ├─ series_features.py  (25-dim)   │
                    │   ├─ prob_metrics.py     (CRPS)     │
                    │   ├─ data_loader.py / splitter.py   │
                    │   └─ uea_loader.py / ucr_loader.py  │
                    └─────────────┬───────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────────┐
        ↓                         ↓                             ↓
┌──────────────────┐  ┌─────────────────────┐    ┌────────────────────────┐
│  baseline/       │  │  agent/              │    │  experiments/          │
│  - chronos*.py   │  │  - curator_uq.py     │    │  - taska_*.py          │
│  - timesfm2.py   │  │  - model_cards.py    │    │  - taskb_*.py          │
│  - tirex.py      │  │  - clf_model_cards   │    │  - gated_residual_run  │
│  - timer.py      │  │  - planner_adaptive  │    │  - soft_router_v2      │
│  - time_moe.py   │  │  - planner_prior_aw  │◀──┤  - three_way_forecast  │
│  - sundial.py    │  │  - prior_crps.py     │    │  - tirex_vs_c2.py      │
│  - toto*.py      │  │  - quantile_ensemble │    │  - online_routing_sim  │
│  - moirai*.py    │  │  - memory.py         │    │                        │
│  - tsc_classical │  │  - clf_memory.py     │    │  scripts/              │
│  - mantis_*.py   │  │  - clf_planner.py    │    │  - remote_smoke.py     │
│  - moment_*.py   │  │  - clf_strategies    │    │  - remote_sweep.py     │
│  - llmtime.py    │  │  - dataset_priors    │    │                        │
│  - naive.py      │  │  - forecaster_refl   │    │                        │
│  - arima_ets.py  │  │  - rca.py            │    │                        │
└──────────────────┘  └──────────────────────┘    └────────────────────────┘
```

---

## 12. Configuration & Env Vars

| Var | 取值 | 作用 |
|---|---|---|
| `ADAPTTS_PLANNER` | `prior_aware` / 未设 | 启用 §4 新 planner（默认仍用旧 adaptive） |
| `ADAPTTS_ALLOW_REMOTE` | `1` / `0` | prior 池是否纳入远程模型 |
| `ADAPTTS_DEFAULT` | model_name | N<15 强制 fallback (v10 兼容) |
| `ADAPTTS_GATE` | `entropy` / 未设 | A1 entropy gating |
| `CLF_MEM_K` | int (默认 5) | TSC memory 检索 K |
| `CLF_MEM_K_MIN` | int (默认 5) | 旧 vote 最小 support |
| `HF_HOME` | path | HuggingFace cache 位置 |
| `HF_ENDPOINT` | `https://hf-mirror.com` | 国内镜像 |

---

## 13. 与 feedback 23 项改进的对应表

| feedback # | 改进 | 落点 |
|---|---|---|
| 一.1 | 7 forecast TSFM 扩 | §3.1（11/12 wired） |
| 一.2 | TSC 模型扩 (MantisV2 + KairosHope) | §3.2（Mantis ✅ / Kairos ⏳ repo 未公开） |
| 一.3 | 不盲加 TSFM 做异常检测 | §0 scope（B0-rule baseline 保留） |
| 二.1 | 静态 π_k = 1/CRPS | §4.1 / `prior_crps.py` |
| 二.2 | 类型先验 (point preds 降权) | §4.1 / `_apply_type_prior` |
| 二.3 | N 值先验 (N<15 → C2=0.9) | §4.1 / `_apply_n_prior` |
| 二.4 | BMA 后验 | §4.3 / `bma_posterior` |
| 三.1 | 软路由 (quantile linear pool) | §5 / `quantile_ensemble.py` |
| 三.2 | 残差门控 | finish-1 §2 (NEGATIVE finding) |
| 三.3 | L0/L1/L2 分层 | §4.2 / `make_prior_plan` |
| 三.4 | 路由信号增强 (CRPS + ε-greedy) | §4.4 + a3_prob_metrics.py |
| 三.5 | 记忆改进 (反事实 + 多样性 + 1/CRPS vote) | §6 / `clf_memory.py` |

完成度 22/23 (95.7%)。未做 1 项：forecasting soft router 的 NEGATIVE-finding 实证（依赖远程 sweep 数据回传 → §5 pipeline 已就绪，等数据）。

---

## 14. 复现指引

### 14.1 一次完整 forecasting 调用

```python
import os
os.environ["ADAPTTS_PLANNER"] = "prior_aware"

from research.agent.forecaster_reflect import forecast_with_reflection
from research.agent.curator_uq import diagnose
from research.utils.data_loader import load_series
from research.utils.splitter import few_shot_split

series, meta = load_series("ECL")
sp = few_shot_split(series, N=100, H=96, seed=42)
diag = diagnose(sp.train, season_m=meta.season_m)
pred, trace = forecast_with_reflection(
    train=sp.train, val=sp.val, H=96, diag=diag,
    season_m=meta.season_m, dataset="ECL",
)
# trace.final_plan is a PriorPlan with level / strategies / posterior
```

### 14.2 一次完整 TSC 调用

```python
from research.agent.clf_planner import classification_planner

chosen, y_pred, trace = classification_planner(
    X_train, y_train, X_test,
    season_m=meta.season_m,
    use_cv=True, use_memory=True,
    memory_path="research/results/clf_memory_v5.jsonl",
    use_enhanced_features=True,
    vote_method="inv_loss",            # Item 3
    use_diverse_retrieval=True,        # Item 4
    use_industrial_signature=True,
    default_classifier="rocket",
    margin=0.10,
)
```

### 14.3 远程模型 sweep

```bash
# 本地：rsync 代码 + cells
sshpass -p cinter rsync -az research/scripts/remote_sweep.py \
    research/results/gated_residual_cells.jsonl \
    c220@192.168.1.102:/data2/c220/hz/agent_ts/research/...

# 远程：跑 sweep
ssh c220@192.168.1.102 'source ~/anaconda3/etc/profile.d/conda.sh \
    && conda activate tsci-remote-tx440 \
    && HF_HOME=/data2/c220/hz/hf_cache HF_ENDPOINT=https://hf-mirror.com \
    && cd /data2/c220/hz/agent_ts && python research/scripts/remote_sweep.py time_moe'

# 拉回结果
sshpass -p cinter rsync -az c220@192.168.1.102:/data2/c220/hz/agent_ts/research/results/time_moe_vs_c2.jsonl \
    research/results/
```

---

## 15. 论文映射

| paper section | method.md 节 | 论文 contribution |
|---|---|---|
| §3 setup | §0, §9 | 问题 + 评估 |
| §4 method | §1-7 | 主体框架 |
| §4.5 model library | §3 | 12 forecast + 11 TSC |
| §4.6 hierarchical router | §4 | L0/L1/L2 + prior 栈 |
| §4.7 BMA | §4.3 | 数学闭环 |
| §4.8 quantile soft routing | §5 | feedback §三.1 实现 |
| §4.9 counterfactual memory | §6 | feedback §三.5 实现 |
| §5 ablations | §4.2 / §6 / §7 | L1-only / no-memory / no-inv-loss / no-diverse |
| §6 future work | §10.3 | cross-env dispatcher / ε-greedy live |

---

## 16. 术语表

| 缩写 | 全称 / 含义 |
|---|---|
| TSFM | Time-Series Foundation Model |
| C2 | Chronos-2 (Amazon, 2025) — 当前 forecasting default |
| TSH | TSFM Saturation Hypothesis (finish-1 §2 提出): Cov(C2,D)→1 时 E[Δ]→0 |
| CRPS | Continuous Ranked Probability Score |
| BMA | Bayesian Model Averaging |
| LODO | Leave-One-Dataset-Out (cross-dataset CV) |
| LOO | Leave-One-Out (within-cell CV) |
| π_k | Static prior for model k (1/loss normalized) |
| σ² | BMA temperature; sharper σ²↓ → posterior 集中 |
| ε | ε-greedy exploration probability |
| `all_clf_accs` | TSC ClfCase 的反事实字段：{clf: test_acc} 完整字典 |

---

**End of method.md** — 后续 sweep 落实/新模块加入后追加 §17+。
