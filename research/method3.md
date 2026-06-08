# Method v3 — Self-Evolving Belief-State Routing Runtime

> 版本：2026-05-30 重构（Round 7-8 全部合并 + M8/M9 后诚实化）
> 接续 `method.md`（Round 4-A）/ `method2.md`（Round 5-6）。**配套**：`finish3.md`（Round 7-8 实测 + Findings F-R7.x/F-R8.x）/ `feedback.md`（外部 review 11 条硬伤）/ `paper_draft.md`（论文稿）/ `plan.md`（总纲）/ `TODO.md`（权威优先级）。
>
> 自顶向下完整描述系统现状。每节脚注真实文件路径作代码地图。method2 把 router 升级到 *self-adaptive runtime + decision engine*；method3 进一步升级到 **self-evolving system**：模型库自我淘汰、Prior 权重自我学习、decide-mode 自我选择，并把 feedback 两条理论硬伤（fake-Bayes / factor-explosion）和唯一"致命"工程硬伤（memory 泄漏）逐条封堵。

---

## 0. Thesis（一句话总结）

> **任何时序任务都是 belief state `b(M | z, h, t)` 上的决策**；系统不只在运行时自适应，还要让自己的**结构参数**（模型库成员、prior 权重、decide-mode）随真实 outcome **自我演化**，去掉 method2 残留的全部手调常数。

三点收敛(呼应 feedback "三层统一结构"，见 §10)：

```
  x ──► Representation z=f(x) ──► Belief b(M)=softmax(−E) ──► Decision a~π(a|b)
        Curator 25-d + embedding   energy-based factor 组合      argmax/Thompson/risk-min
        series_features.py          bayesian_router.py            + abstain / action layer
        representation.py           (M8: 非 exact Bayes)          drift_engine / action_layer
```

**关键诚实化(M8/M9)**：
- 旧称 "Bayesian posterior" → 改 **energy-based / belief state**（M8.1，弱化 fake-Bayes claim）。
- 旧 TSC "+0.89pp 击败 Rocket" → 去数据泄漏后 **−0.62pp（持平）**（M9，F-R8.7）；论文价值转向"泄漏审计方法学"。

---

## 1. 数学公式

### 1.1 决策规则（energy-based, 非 exact Bayesian）

$$\hat{M}(x,h,t)=\mathrm{decide}\big(b(M_k\mid z,h,t)\big),\qquad
b(M_k)=\mathrm{softmax}(-E_k)=\frac{\exp(-E_k)}{\sum_j\exp(-E_j)}$$

$$-E_k=\sum_i \underbrace{\log\pi_k^{(i)}(z)}_{\text{prior factors}}+\sum_j \underbrace{\log L_k^{(j)}(z,h,t)}_{\text{likelihood factors}}$$

| 符号 | 含义 | 来源 |
|---|---|---|
| $z=f_\phi(x)$ | learned embedding | `representation.py` |
| $r(z)$ | regime label (k-means) | `RegimeAssigner` |
| $\pi_k^{(i)}$ | 第 i 个 prior factor | `bayesian_router.py:PriorFactor` |
| $L_k^{(j)}$ | 第 j 个 likelihood factor | `bayesian_router.py:LikelihoodFactor` |
| $E_k$ | 候选 k 的 energy（log 量纲）| factor 加和取负 |
| $\mathrm{decide}$ | argmax / Thompson / risk-min / auto | `BayesianRouter.decide()` + M1 meta-bandit |

> **M8.1 framing 修正**：factor 非生成式、非条件独立，$b$ 不是 exact posterior。论文/method 统一写 **"factorized posterior-inspired energy model"** / **"belief state"**，不写 "exact Bayesian posterior"。`BayesianRouter` 类名仅为兼容保留。

### 1.2 三个 decide mode（+ M1 自动选择）

| Mode | 公式 |
|---|---|
| argmax | $\arg\max_k b(M_k)$ |
| Thompson | $\tilde r_k\sim b(M_k)$；$\arg\max\tilde r_k$ |
| risk_min | $\arg\min_k(\mathbb{E}[\ell_k]+\lambda\,\mathrm{Var}[\ell_k])$ |

**M1 Meta-bandit**：把 `{argmax, thompson, risk_min}` 当 3 arm 的 meta-level bandit，`decide_mode="auto"` 时由真实 outcome 自学该用哪个 mode（取代手调常数）。

### 1.3 在线更新（per-(regime,model) 高斯共轭 + M4 per-regime decay）

$$n_t=d_r\,n_{t-1}+1,\quad
\mu_t=\frac{d_r\,n_{t-1}\,\mu_{t-1}+\ell_t}{n_t},\quad
\sigma_{\mu,t}^2=\frac{\hat\sigma^2}{n_t}$$

**M4**：decay $d_r$ 从全局标量升级为 **per-regime** $d_r=\text{regime\_decay}.get(r,\text{decay})$；B3 drift 触发 `boost_exploration` 时只收紧"最近活跃 regime"的 decay（实测 fast 0.80 vs slow 0.99 适应速度差 **3.40×**，F-R8.3）。

### 1.4 M3 · Empirical Bayes Prior strength（自学权重）

$$r=\mathrm{Pearson}\big(\log\pi_F(\text{chosen}),\,-\text{outcome}\big),\qquad
\text{strength}_F\leftarrow\mathrm{clip}\big(\text{strength}_F\cdot(1+\eta\,r),\,0,\,\text{max}\big)$$

正相关(factor 把高 prior 给了 outcome 好的模型)→ 加强；持续负相关 → 削到 0(等价 prune)。实测 NPrior `2.000→2.246`(+12.3%, r=+0.616)。

### 1.5 M8 · Factor Attribution（可解释性，问题 2）

- **拆解**：$-E_k=\sum_i c_{i,k}$，重构误差 $\max_k|\sum_i c_{i,k}-(-E_k)|=0$（严格可加，F-R8.5）。
- **LOFO**（单次决策决定性 factor）：去掉 factor $i$ 后 $\arg\max$ 是否翻转 + $\mathrm{KL}(b\,\|\,b_{-i})$ + $\Delta\text{margin}$。
- **跨决策冗余**：centred log-term 向量跨决策求 Pearson，$|\text{corr}|\ge0.8$ 即冗余对（当前 6 forecasting factor **无**冗余对，F-R8.6）。

---

## 2. 系统架构（三层 belief-state runtime）

```
┌─────────────┐
│   Series x  │
└──────┬──────┘
       ↓  ───────────────────────  Layer 1 · Representation ───────────────────────
┌──────────────────────┐   curator_uq.py + series_features.py (25-d 诊断+置信度)
│ Curator + Embedding  │   representation.py: HandFeature(25)/MOMENT(512)/Chronos2(768)
│   z = f_φ(x)         │   RegimeAssigner: k-means(K=8) → r(z)   (purity 82.4%)
└──────┬───────────────┘
       ↓  ───────────────────────  Layer 2 · Belief ────────────────────────────────
┌────────────────────────────────────────────────────────────┐  bayesian_router.py
│ BayesianRouter — energy-based belief b(M)=softmax(−E)        │
│   Prior factors:  Availability / CRPS / Regime / Type /      │
│                   N / Entropy / Industrial / AnomalyType(M7) │
│   Likelihoods:    CV / Memory(CV-acc, M9) / Representation   │
│   Online:         BanditState[(r,M)]→(μ,σ)  + per-regime decay(M4)│
│   Self-evolving:  M2 culling  · M3 EB-strength · M8 attribution │
│   decide(mode) ∈ {argmax, thompson, risk_min}  ←auto by M1   │
└──────┬───────────────────────────────────────────┬─────────┘
       ↓                                            ↑ observe(z,chosen,loss)
       │  ───────────  Layer 3 · Decision ──────────┤
┌──────────────────────┐  drift_engine.py (B3: 5 signals→3 actions→refit)
│ Decision / Action    │  action_layer.py (E1: forecast→intervention)
│  abstain / route /   │  inference_scheduler.py (R6-E: 预算感知升级链)
│  intervention        │  calibration.py (B2: isotonic 4-tier)
└──────┬───────────────┘  reflective_loop.py (B1: L0→L3)
       ↓
┌──────────────────────┐  telemetry.py (health) → drift_history → 下一步 adaptive_decide
│  Prediction / Action │  (闭环)
└──────────────────────┘
```

**自演化闭环**：`adaptive_observe` 按周期触发 M2 culling（`cull_every`）/ M3 EB（`eb_learn_every`）/ B3 drift（`drift_check_every`）；drift 的 `boost_exploration` 通过 `resurrect()` 复活被淘汰模型(M2↔B3 互锁)。

---

## 3. 三大任务

> 同一 belief-state runtime 接三类 task，区别只在候选集 $\mathcal{M}$、likelihood、decision 头。三任务**诚实结果**(post-M9)：

| 任务 | Agent 角色 | 候选集 $\mathcal{M}$ | 最强 baseline | 我们最终 | 诚实差距 |
|---|---|---|---|---|---|
| **A Forecasting** | Chronos-2 wrapper | {chronos2, bolt, arima_ets, llmtime, tirex, toto…} | Chronos-2 | v11 parity wrapper | **0%**（CRPS, Wilcoxon p=0.32）|
| **B TSC** | router | {rocket, moment_1nn/lr, dtw_1nn, euclid_1nn} | Rocket-alone | B7v3(router+memory, 去泄漏) | **−0.62pp**（持平；泄漏前虚高 +1.51pp）|
| **C Anomaly+RCA** | 结构化根因 | {rule, residual detector} + fault-type prior | LLM-direct / B0-rule | Curator+Cards | **+40pp** vs LLM-direct（但 −37pp vs 规则，诚实负结果）|

### 3.1 Track A · Forecasting（`agent/forecaster_reflect.py`）

```
(train,val,H,diag,dataset) → z=embed → regime → BayesianRouter.decide
  ├─ N<15 → NPrior 把 chronos2 拉到 0.9（v10 fallback）
  ├─ walk-forward CV → CVLikelihood softmax(−cv/τ)
  ├─ entropy gate (v12) → 高 spread 时提高偏离 margin
  ├─ L1 单模型(0 cost) / L2 quantile linear pool
  └─ memory safety-net(v11) 仅强支持时偏离 → observe(actual_mae)
```
最终 v11 = **0W/1L/23T MAE, CRPS 0%** = guaranteed parity（三机制 memory/entropy/abstain 都收敛到 C2 均值，TSH 直接证据）。

### 3.2 Track B · TSC（`agent/clf_planner.py`）

```
(X_tr,y_tr,X_te,dataset,seed) 
  ├─ N_per_class<7 → force rocket (B7v2 fallback, +0.87pp 回血)
  ├─ LOO/k-fold CV → cv_accs   (M9: 只有 cv_accs 可投票)
  ├─ margin gate: best_other−rocket≥0.10 → 偏离
  ├─ memory: featurize_cell→z-score→L2; query_diverse(exclude_meta=LOCO)
  │          consensus_winner_inv_loss 用 votable_accs()(=cv_accs)
  └─ predict_with(chosen)
```
**M9 去泄漏后**：86.91% / −0.62pp，路由 `rocket15/moment10/euclid4/dtw1`（本轮重跑复现，见 finish3 §8）。

### 3.3 Track C · Anomaly + RCA（`agent/anomaly.py` + `agent/rca.py`）

M7 Phase 1：`AnomalyTypePrior`(3 统计特征 level_shift_z / variance_ratio / max_outlier_z) + 2 轻量 detector（rule + residual），softmax 出 `{fault_type, score}`，4/4 故障类型正确识别(F-R7.3)，**不引入深度模型 / LLM**。Phase 2(per-fault memory)/Phase 3(LLM RCA)为预留接口。

---

## 4. 模型库

### 4.1 Forecasting（12 models / 16 cards · `baseline/` + `agent/model_cards.py`）

| 类别 | 模型 | params | env |
|---|---|---|---|
| trivial point | naive_drift, naive_seasonal, arima_ets, llmtime | <1M / LLM | tsci |
| Chronos family | chronos(60M), chronos_bolt(200M), **chronos2(120M, default)** | 60-200M | tsci |
| TSFM 主流 | timesfm2(500M), moirai(311M), moirai2(11M) | 11-500M | tsci/py312 |
| niche specialist | tirex(128M xLSTM), toto(151M observability), toto2(4M) | 4-151M | tsci/py312 |
| remote large | time_moe(50M), sundial(128M), timer(8.3B MoE) | 50M-8.3B | tsci-remote(-tx440) |

每模型 5 字段 card：class / assumes / strengths / weaknesses / typical_failure。

### 4.2 TSC（`baseline/tsc_classical.py` + `agent/clf_strategies.py`）

distance(dtw_1nn, euclid_1nn) · kernel(**rocket default**, minirocket) · TSFM-embed(moment_1nn, moment_logreg, mantis_*) · dictionary(weasel) · feature(catch22) · LLM(llm_direct)。

### 4.3 Anomaly detectors（`agent/anomaly.py`）

rule_baseline · residual_score（站位 Anomaly-Transformer，Phase 2 可替换）+ `AnomalyTypePrior`(strength 由 M3 EB 学)。

---

## 5. 数据集

### 5.1 Forecasting（6 数据集 × N∈{10,20,50,100} × 3 seeds）

| Dataset | Sampling | Len | H | m | 用途 |
|---|---|---|---|---|---|
| ETTh1/ETTh2 | 1h | 17,420 | 96 | 24 | 主表 |
| ECL(MT_001) | 1h | 26,304 | 96 | 24 | TSFM coverage |
| Exchange(rate_0) | 1d | 7,588 | 96 | 7 | low-coverage |
| Weather(OT) | 10min | 52,696 | 96 | 144 | OOD memory 测试 |
| ILI(OT) | 1w | 966 | 24 | 52 | 量纲极端 |

### 5.2 TSC（UCR univariate + UEA multivariate · `datasets/ucr`、`datasets/uea`）

- **UCR-5 核心**：Coffee / ECG200 / TwoLeadECG / BeetleFly / BirdChicken（N_per_class∈{3,5,10}×2 seeds=30 cells）
- **扩展(less-saturated)**：GunPoint / Strawberry / Wafer / ECG5000 / Crop / FordA/B
- **UEA 多变量**：BasicMotions / ERing / AtrialFibrillation（DTW>Rocket 反转，更大 routing space）

### 5.3 Synthetic 4-class fault + RCA cells

ETTh1/ECL 注入 {normal, trend_break, seasonal_break, outlier_burst}；RCA 自然失败 30-50 cells（从 forecasting catastrophic 选）。

---

## 6. 自演化模块 M1-M9（Round 7-8）

| ID | 名称 | 解决的手调常数 / 硬伤 | 文件 | 实测 |
|---|---|---|---|---|
| **M1** | Meta-bandit on decide_mode | `decide_mode` 手选 | `meta_bandit.py` | thompson 60步→83.3% / 200步→92% (F-R8.1/8.2) |
| **M2** | Model 自动淘汰 + EliminationPrior | model library 永久污染 | `model_culling.py` | per-regime μ 排序淘汰底部，protect+min_keep 互锁 |
| **M3** | Empirical Bayes prior strength | `strength` 经验常数 | `prior_learning.py` | NPrior 2.0→2.246 (§1.4) |
| **M4** | Per-regime bandit decay | global decay 一刀切 | `bandit.py`+`drift_engine.py` | fast/slow 适应差 3.40× (F-R8.3) |
| **M7** | Anomaly Phase 1 | (新任务，最小闭环) | `anomaly.py` | 4/4 故障类型识别 (F-R7.3) |
| **M8** | Factor Attribution + framing | 问题 1 fake-Bayes + 问题 2 黑盒 | `bayesian_router.py` | 重构误差 0；无冗余对 (F-R8.5/8.6) |
| **M9** | Memory 去泄漏 | 问题 6 致命泄漏 | `clf_memory.py`+`clf_planner.py`+`build_clf_memory_v2.py` | +1.51pp 泄漏挤出；88.42%→86.91% (F-R8.7/8.8/8.9) |

**M9 两类泄漏修复**：(A) value 泄漏——投票权重从 test-acc 改 **训练集内 CV**（`votable_accs()` 只返 `cv_accs`，test 降级 AUDIT ONLY）；(B) self-membership——`exclude_meta` 按 `{dataset,N,seed}` **leave-one-cell-out** 剔除查询 cell 自身。

### 6.1 RouterConfig 新增旋钮

| 旋钮 | 默认 | 作用 |
|---|---|---|
| `cull_every` / `cull_fraction` / `cull_min_keep` / `cull_protect` | 200 / 0.15 / 2 / (naive_drift,chronos2) | M2 |
| `eb_learn_every` / `eb_lr` / `eb_max_strength` / `eb_min_samples` | 100 / 0.05 / 5.0 / 30 | M3 |
| `meta_bandit_enable` / `decide_mode="auto"` / `meta_bandit_cold_K` / `meta_bandit_decay` | False / — / 10 / 0.995 | M1 |

---

## 7. 远程模型 / 部署矩阵

### 7.1 5 个 conda env

| Env | python | torch | transformers | 用途 |
|---|---|---|---|---|
| `tsci`(local main) | 3.10 | 2.x cu118 | 4.45+ | 8/12 forecasting + 全部 TSC（本地主力，CUDA 可用）|
| `tsci-py312`(local) | 3.12 | 2.5+ | <4.46 | moirai2 / toto2（uni2ts 2.0）|
| `tsci-tx440`(local, deprecated) | 3.10 | 2.x | 4.40.1 | 旧 time_moe/sundial 回归 |
| `tsci-remote`(`c220@10.192.43.66`) | 3.9 | 2.8.0+cu128 | **4.57.1** | Timer-S1（Blackwell sm_120）|
| `tsci-remote-tx440`(远程) | 3.9 | 2.8.0+cu128 | **4.40.1** | time_moe / sundial（旧 API）|

### 7.2 远程 SSH + sweep

```bash
ssh c220@10.192.43.66          # 2× RTX 5070 Ti 16GB, sm_120 ; 密码 cinter
# workdir /data2/c220/hz/agent_ts/ ; HF cache /data2/c220/hz/hf_cache/ + HF_ENDPOINT=https://hf-mirror.com

# 本地 rsync 代码+cells → 远程跑 → 拉回 *_vs_c2.jsonl
sshpass -p cinter rsync -az research/scripts/remote_sweep.py c220@10.192.43.66:/data2/c220/hz/agent_ts/research/...
ssh c220@10.192.43.66 'conda activate tsci-remote-tx440 && cd /data2/c220/hz/agent_ts && \
   HF_HOME=/data2/c220/hz/hf_cache HF_ENDPOINT=https://hf-mirror.com python research/scripts/remote_sweep.py time_moe'
sshpass -p cinter rsync -az c220@10.192.43.66:.../results/time_moe_vs_c2.jsonl research/results/
```

### 7.3 Cross-env routing（future）

当前单 plan 不能跨 env 调模型；`allow_remote=True` 仅离线把远程模型的 $\pi_k$ 纳入本地 prior。在线需 subprocess dispatcher（paper §5.4 future work）。

---

## 8. 评估指标

| 任务 | 指标 |
|---|---|
| Forecasting | MAE / **CRPS** $\approx\sum_\ell\frac{2}{|Q|}(\alpha_\ell-\mathbb{1}[y<q_\ell])(q_\ell-y)$ / pinball / 80%-coverage / width |
| TSC | Accuracy / Macro-F1 / routing trace / Oracle gap |
| RCA | R1 Top-1 / R2 Top-3 / R4 keyword-F1 / OOT recall |
| Risk / Cost | $\text{risk}_k=\mathbb{E}[\ell_k]+\lambda\,\text{std}[\ell_k]$ ; $c_k=\alpha\log\text{lat}+\beta\log\text{params}+\gamma\,\text{env\_penalty}+\delta\log\text{VRAM}$ |

---

## 9. 配置旋钮（核心）

| Var | 取值 | 作用 |
|---|---|---|
| `ADAPTTS_PLANNER` | `bandit`/`bayesian`/`prior_aware` | 选 router |
| `ADAPTTS_DECIDE` | `argmax`/`thompson`/`risk_min`/`auto` | 决策模式（auto=M1）|
| `ADAPTTS_ALLOW_REMOTE` | 0/1 | 远程模型纳入候选 |
| `ADAPTTS_CLF_PLANNER` | `bayesian`/unset | TSC Bayesian 路径 |
| `CLF_MEM_K`/`CLF_MEM_K_MIN` | int | TSC memory 检索 K |

---

## 10. feedback 11 条硬伤 · 解决状态回顾（vs 之前版本）

> feedback 列 5 条理论 + 6 条工程硬伤。下表是 **method3(Round 7-8) 相对 method2(Round 5-6) 的封堵进度**。

| # | feedback 硬伤 | method2 状态 | method3 改动 | 现状 |
|---|---|---|---|---|
| 理-1 | "贝叶斯"名不符实（factor 非生成式）| claim "Bayesian posterior" | **M8.1** 改 energy-based / belief-state framing | ✅ **已封堵**（framing 诚实化）|
| 理-2 | factor explosion / 黑盒不可识别 | 6 factor 无审计工具 | **M8.2** Factor Attribution（LOFO+KL+冗余矩阵），重构误差 0 | ✅ **已封堵**（工具就位；当前无冗余对）|
| 理-3 | Regime 根基不牢（静态 manifold，purity 82.4%）| k-means 硬聚类 + RegimePrior 直接当先验 | B3 `refit_regimes` 漂移时重训；但仍硬聚类 | ⚠️ **部分**（drift 重训有了，regime→feature 仍 open=P1 #75）|
| 理-4 | Thompson 太浅（单高斯假设脆弱，重尾低估）| 全局高斯共轭 | **M1** meta-bandit + **M4** per-regime decay 提升自适应 | ⚠️ **部分**（自适应↑，但仍单高斯=P1 #74 robust bandit open）|
| 理-5 | 缺 world model（不理解环境动力学）| 无 | 未做 | ❌ **未动**（P3 #83 future work）|
| 工-4 | 复杂度爆炸缺性能证据 | 堆模块无诚实 ablation | M9 诚实评估 + M8 attribution 标 inert factor | ⚠️ **部分**（诚实数已出；逐 factor 减法=P1 #73 open）|
| 工-5 | scheduler/router 开销未分析 | 无 Pareto | `latency_analysis.py` + Appendix C2(28-118×) | ⚠️ **部分**（有数；精度/耗时 Pareto=P2 #79 open）|
| 工-6 | **记忆层泄漏（"致命"）** | 投票用 test-acc + self-membership | **M9** CV-only 投票 + leave-one-cell-out | ✅ **完全封堵**（−0.62pp 本轮复现，F-R8.7）|
| 工-7 | 校正/漂移自洽性陷阱 | calibration↔drift 可能互相喂噪声 | drift 软属性 setattr 不破坏 save/load；冷启动回退=P2 #80 | ⚠️ **部分** |

**总结**：method3 **完全封堵 3 条**（理-1 framing / 理-2 attribution / 工-6 memory 泄漏，即 feedback 最强调的"撤回 fake-Bayes + 修致命泄漏"），**部分缓解 5 条**，**1 条未动**（world model）。下一步主攻 P1 #73-76（robust bandit / regime-as-feature / factor 消融与权重学习），见 `TODO.md`。

---

## 11. 实测 → finish3.md

Round 7-8 全部实测 + Findings **F-R7.1~7.4 / F-R8.1~8.9** 在 `finish3.md`：
- §0 M1 meta-bandit / M4 per-regime decay · §1 M2 culling · §2 M3 EB · §3 M2+M3 walk-forward
- §5 M7 anomaly · §7 **M8 Factor Attribution**（F-R8.5/8.6）· §8 **M9 泄漏诚实对比**（F-R8.7/8.8/8.9，逐-cell 表）

> **本轮新增**：重跑 `taskb_router_v3_honest_sweep.py` 复现 honest **86.91% / −0.62pp**（routing rocket15/moment10/euclid4/dtw1），与 finish3 §8 完全一致 → 论文撤回"击败 Rocket"有实测背书。

---

## 12. 文件地图（Round 7-8 增量，全图见 method2 §12）

```
research/agent/
├── bayesian_router.py     # 核心：energy belief + 6-8 priors + 3 likelihoods + M8 attribution
├── meta_bandit.py         # M1 decide-mode meta-bandit
├── model_culling.py       # M2 cull_models + EliminationPrior
├── prior_learning.py      # M3 learn_prior_strengths (Empirical Bayes)
├── bandit.py              # M4 per-regime decay (扩 BanditState)
├── anomaly.py             # M7 Phase 1: AnomalyTypePrior + 2 detectors
├── clf_memory.py          # M9 去泄漏: votable_accs()/cv_accs + LOCO exclude_meta
├── clf_planner.py         # M9: 传 dataset/seed 构 exclude_meta
├── drift_engine.py        # B3: 5 signals→3 actions→refit (M4 联动)
├── adaptive_planner.py    # 自演化闭环: observe 周期触发 M2/M3/B3
└── representation.py      # Layer1: embedding + RegimeAssigner
research/experiments/
├── build_clf_memory_v2.py        # M9: CV-based memory bank (test 仅 audit)
└── taskb_router_v3_honest_sweep.py  # M9 诚实 sweep → results/taskb_router_v3_honest_ucr.jsonl
```

---

## 术语表（method3 增量）

| 缩写 | 含义 |
|---|---|
| belief state | $b(M)=\mathrm{softmax}(-E)$，弱化版"posterior"（M8 后正名）|
| LOFO | Leave-One-Factor-Out（M8 单次决策归因）|
| LOCO | Leave-One-Cell-Out（M9 检索剔除自身）|
| EB | Empirical Bayes（M3 自学 strength）|
| F-R8.7 | 去泄漏后 TSC router −0.62pp（不再击败 Rocket）|
| TSH | TSFM Saturation Hypothesis（Cov→1 ⇒ E[Δ]→0）|

---

**End of method3.md** — 后续 P1（robust bandit / regime-as-feature / factor 消融）落地后追加 §13+。
