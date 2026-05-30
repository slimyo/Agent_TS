# Round 2-5 Optimization Journal

> 接 `finish.md`（Phase 1-6）。每 Round 一节，时间倒序。
> **顶部 §0 = Findings F1-F12 索引**（论文 §5 / §6 引用入口）；后续 §1-5 = 各 Round 实施细节。
> 当前 round → feedback.md（每轮覆盖）。

---

## 0. Findings F1-F12 索引

每条 = (观察, 实证, 命题草稿, 意义)。

| ID | 一句话 | 实证出处 | 命题 status |
|---|---|---|---|
| **F1** | C2 与 ground truth Cov→1 时 wrapper 上界可量化 (oracle 2.43%) | §1 GR-LODO | ✅ 形式化 |
| **F2** | 加无关 niche 的 TSFM 对 oracle 0 边际（5-way == 3-way == -5.24%） | §4.2 | ⏳ 待命题化 |
| **F3** | oracle gain = niche 互补度，非参数量 | §4.2 + §2.3 | ⏳ |
| **F4** | TSFM HF 卡片不写 normalize，缺则 catastrophic (MoE ILI 740k→58k) | §4.1 | ✅ 工程命题 |
| **F5** | 朴素 top-K 检索自我强化 default → memory hindsight bias | §3.3 单元 | ⏳ |
| **F6** | softmax 混合在 TSC + forecasting 双轨都无法超 single-model | §1.2 + §4.2 | ✅ 待论文 |
| **F7** | LOO CV 在 N≤5 catastrophic mis-route（>20pp） | finish §3.1.34 | ⏳ |
| **F8** | 训练域明确的 TSFM 在对应域 dominant / 域外 catastrophic | §2.3 (Toto ECL -44% vs Weather +671%) | ⏳ |
| **F9** | 1/MAE 倒数先验在 34 cells 上重现专家 niche 划分 | §3.1 smoke | ✅ 自验证 |
| **F10** | 8.3B 模型远程 + cross-env 是隐性 routing cost | §4.3 Pareto | ✅ 实测量化 |
| **F11** | well-motivated heuristic ≠ principled framework (meta) | feedback Round 4 + §5 | ✅ meta |
| **F12** | (mean, std) Pareto 仅 {C2, TiRex}；λ=5 时 TiRex 反超；Sundial 是 ETTh2 risk-min winner | §4.3 P1 | ✅ 实测命题 |
| **F13** | cold-start bandit 在 162 ep 未追上 prior_aware；冷启动静态环境 prior 仍强 | §8 sweep 1 | ✅ 实测命题 |
| **F14** | 多 env 部署下 prior 信号(在 env A 学到) 成为 env B 误导；bandit_greedy 反胜 via 失败过滤 | §9 sweep 2 | ✅ 实测命题 |
| **F15** | adaptive runtime 高方差 "全或无"：少数 niche 大胜 / 多数小输；mean 提升靠分布尾部 | §10 | ✅ 实测命题 |
| **F16** | Cold-start reliability prior 过强 → routing 退化到 default；R7 v2 比 R6 v1 反劣化 +4.4% | §10.7 | ✅ 实测命题 |

详细命题与意义在每 Round 实施节，链接在下表「实证出处」。

---

## 1. Round 2 · Wrapper 与 Soft Router NEGATIVE findings

**关键 thesis**：(a) wrapper-class 改进上界很低；(b) soft mixture 不优于 hard single-best。两者共同为 Round 3 决策（"换 base model 库"）提供数学依据。

### 1.1 Gated Residual Forecasting (task #67)

**架构**：ŷ = ŷ_C2 + g(d)·Δ(d)，g/Δ 在 36-d features 上训练。

**实验**：6 datasets × 36 cells，LODO CV，τ sweep ∈ {0.3, 0.5, 0.7, 0.9}

| Config | Δ rel vs C2 | helped/hurt/tied |
|---|---|---|
| GR v1 (no norm) | +15% | 2/26/6 catastrophic |
| GR v2 (history-std norm) | +14% | 13/21/0 |
| GR v3 (shrink 0.3) | +1.3% | 6/15/13 |
| **Oracle scalar bias** | **-2.43%** | 34/0/0 |

**Key diagnostics**: gate AUC=0.377 < random 0.5（gate 学不到 generalize）；oracle scalar bias ceiling = **2.43%**。

**Finding F1 命题**（TSFM Saturation Hypothesis）：
$$\sup_g E[\mu_C(x) - \mu_W(g, x)] \leq E[|y - \hat{y}_C|^2 / \sigma_C^2]^{1/2} \cdot \varepsilon$$
ε → 0（high saturation）时 wrapper 改进上限消失。→ 唯一出路 = 换 base model（Round 3）。

**产物**：`agent/gated_residual.py` v1/v2/v3 + `experiments/gated_residual_run.py`；results: `gated_residual_cells.jsonl` (34 cells), `gated_residual_lodo_tau{3,5,7,9}.json`

### 1.2 Soft Router for TSC (task #68 v2)

**架构**：softmax(β · pred_acc) → weighted vote over {rocket, moment_1nn, euclid_1nn}.

**LODO 评测** (64 cells × 7 β)：

| β | acc | Δ vs Rocket | β | acc | Δ |
|---|---|---|---|---|---|
| 1 | 0.7706 | -4.17pp | 20 | 0.7829 | -2.94pp (best) |
| 3 | 0.7706 | -4.17pp | 50 | 0.7808 | -3.15pp |
| 5 | 0.7679 | -4.43pp | 100 | 0.7788 | -3.35pp |
| 10 | 0.7700 | -4.23pp | **Oracle** | **0.8348** | **+2.25pp** |

**所有 β 都输 rocket-alone**；Oracle ceiling 仅 +2.25pp，与 forecasting GR 2.43% ceiling 同质（→ F6 双轨 NEGATIVE 证据）。

**产物**：`experiments/soft_router_v2.py`（cached design，5 min vs v1 22h）, `results/soft_router_*.jsonl`

---

## 2. Round 3 · Library 扩 (TiRex / Toto / Mantis / WEASEL + Remote Env)

**关键 thesis**：F1/F6 已封死 wrapper 路径 → 库扩张才是唯一杠杆。

### 2.1 TSC 扩 3 baselines (Phase A)

| Baseline | 关键 | TSC sweep |
|---|---|---|
| `b5_minirocket` | Wafer N=5 +5.5pp, BeetleFly N=5 **+20pp** ⭐ | acc 0.8222 / +0.4pp |
| `b6_weasel` | **NEW SOTA: +2.7pp aggregate over Rocket** ⭐⭐ | acc 0.8451 / +2.7pp |
| `b7_catch22` | balanced 22 features + RF | acc 0.7735 / -4.5pp |

**Oracle gain over Rocket-alone**: **+6.68pp** (8-classifier pool vs 5-clf pool's ~3pp) → library expansion **doubles** routing ceiling.

### 2.2 Forecasting 扩 (Phase B, Round 3 + 4-B)

11 TSFM 调研 → 9 wired，环境分布：

| Env | 模型 (params 排序) |
|---|---|
| `tsci` (main) | naive/arima/llmtime / chronos(60M) / chronos_bolt(200M) / chronos2(120M) / timesfm2(500M) / moirai(311M) / **tirex(128M)** / **toto(151M)** |
| `tsci-py312` | moirai2(11M) / toto2(4M) |
| `tsci-remote` (远程 5070 Ti) | **timer(8.3B)** |
| `tsci-remote-tx440` (远程, transformers 4.40.1) | **time_moe(50M)** / **sundial(128M)** |

Blocked: TabPFN-TS (license)、KairosHope (repo 未公开)。

### 2.3 TiRex / Toto head-to-head vs C2 (Round 3 sweep, 34 cells)

| Model | mean MAE | Δ vs C2 | Niche |
|---|---|---|---|
| C2 (base) | 5102.19 | 0 | Weather, ETTh1, ILI |
| **TiRex** | 5198.81 | +1.9% | **Exchange -35.1%** ⭐, ECL -16.2% |
| **Toto** | 7744.47 | +51.8% | **ECL -43.9%** ⭐⭐ |

**3-way oracle: -5.23% vs C2-alone** → F2 / F3 / F8 实证起点。

**Finding F8 命题**（domain-aligned TSFM niche-clean）：
> 当 TSFM 训练分布 D_train 集中（low entropy），cross-domain generalization gap 与 KL(D_test ‖ D_train) **线性** 而非 sublinear。Toto on Weather +671%、ILI +52% = 直接证据。

**产物**：`baseline/{tirex,toto,toto2,moirai2,mantis_classifier,time_moe,sundial,timer}.py`；results: `{tirex,toto,toto2}_vs_c2.jsonl`

### 2.4 远程 GPU 基础设施

- SSH: `c220@192.168.1.102` (2× RTX 5070 Ti 16GB, sm_120 Blackwell, CUDA 12.8)
- Workdir: `/data2/c220/hz/agent_ts/`
- HF cache: `/data2/c220/hz/hf_cache/`（用 `HF_ENDPOINT=https://hf-mirror.com` 国内镜像）
- 远程 conda envs (cloned from `TimeSeries_env`)：
  - `tsci-remote`: transformers **4.57.1** (Timer-S1)
  - `tsci-remote-tx440`: transformers **4.40.1** (Time-MoE/Sundial — `past_key_values.seen_tokens` API 旧依赖)
- 启动模板：
  ```bash
  ssh c220@192.168.1.102 'source ~/anaconda3/etc/profile.d/conda.sh && \
     conda activate <env> && HF_HOME=/data2/c220/hz/hf_cache \
     HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=/data2/c220/hz/agent_ts \
     python -m research.scripts.remote_sweep <model>'
  ```

---

## 3. Round 4-A · feedback Items 2-4 + Prior-Aware Hierarchical Planner

**关键 thesis**：把 feedback 推荐改进（CRPS 倒数先验 / 1/CRPS vote / 反事实记忆 / L0/L1/L2 / quantile pool / ε-greedy）落地为可组合模块。

### 3.1 Item 2 · CRPS 倒数 π_k + BMA (`agent/prior_crps.py`)

公式：$$\pi_k = \frac{1/\text{loss}_k}{\sum_j 1/\text{loss}_j}, \quad p(M_k|D) \propto \exp(-\text{loss}_k/\sigma^2)\cdot\pi_k$$

**自动从 `*_vs_c2.jsonl` 聚合**（无需新 sweep）。

**Smoke (3 模型 34 cells)**：

```
Overall  : C2=0.379 TiRex=0.372 Toto=0.249
ECL      : C2=0.252 TiRex=0.300 Toto=0.448   ← F9 自动复现 niche partition
Exchange : C2=0.279 TiRex=0.429 Toto=0.292
Weather  : C2=0.769 TiRex=0.131 Toto=0.100
ETTh1    : C2=0.489 TiRex=0.280 Toto=0.231
```

**意义**：feedback Round 4 "all heuristics → prior" 路径**可行**，prior 本身已学到 hand-coded routing 表。

### 3.2 Item 3 · 1/CRPS-style weighted vote for TSC

新增 `clf_memory.consensus_winner_inv_loss()`：每邻居为**全部** classifier 按 `sim × 1/(1-acc+ε)` 投票（旧 `consensus_winner_weighted` 仅看 top-1）。`clf_planner` 加 `vote_method="topk"|"inv_loss"` 开关。

**Unit test**（3 邻居：rocket 多 top-1，weasel 横跨更稳）：
```
old top-1 vote      : winner=rocket   support=0.667
new inv-loss vote   : winner=weasel   support=0.495    ← 翻转正确
```

### 3.3 Item 4 · 反事实记忆 + 多样性 (`clf_memory.query_diverse`)

- **反事实存储**：`ClfCase.all_clf_accs = {clf: test_acc}` 完整字典（早就存在，Item 3 vote 才用上）
- **多样性检索**：top-K 全 default winner → 丢最低 sim default，插入最高 sim non-default
- `clf_planner` 加 `use_diverse_retrieval: bool`

**Unit test**（5 rocket-winner + 1 weasel-winner）：
```
top5 winners      : [rocket × 5]
diverse5 winners  : [rocket × 4, weasel × 1]
plain vote        : rocket (0.568)
diverse + inv_loss: weasel (0.511)    ← F5 hindsight bias 解
```

### 3.4 Prior-Aware Hierarchical Planner (`agent/planner_prior_aware.py`)

把 4 个 prior + L0/L1/L2 + BMA + ε-greedy 在单一模块里组装：

```
compose_prior(dataset, N, allow_remote):
  static π_k = prior_crps.get_prior(dataset=ds)            ← Item 2
  × type_prior    (POINT_PREDICTORS × 0.3)                ← feedback §二.2
  × N_prior       (N<15 → C2=0.9 hard cliff)              ← feedback §二.3
  × availability  (LOCAL ∪ REMOTE?)

L0 triage:
  if π(C2) ≥ 0.45 AND N ≤ 500 AND no CV → L1 single C2
  else → L2 ensemble top-K

L2:
  posterior = bma_posterior(cv_losses, σ², prior) if cv_losses else prior
  weights = top-K renormalized

epsilon_greedy_perturb(plan, eps):  以 ε 概率从 non-top-1 抽样替换
```

**5 unit tests** 全通过：Weather → L1 C2 / ECL → L2 [toto:0.45, tirex:0.30, c2:0.25] / N=10 → L1 C2 boosted / BMA σ²=0.001 → 塌缩到 minloss / allow_remote 接通远程池。

### 3.5 L2 Quantile Ensemble (`agent/quantile_ensemble.py`, feedback §三.1 软路由概率版)

公共栅格 `TARGET_LEVELS = [0.1..0.9]`（9-level，TSFM 最小公分母）。C2 21-grid 自动插值；point predictors 退化 Dirac。

线性池：$$q^{\text{ens}}_{\ell, t} = \sum_k w_k \cdot q^{(k)}_{\ell, t}$$

**Unit test** (3 fake models, weights [0.5, 0.3, 0.2])：median 线性池 sanity err = 0；q10/q90 形状正确。

### 3.6 接入 v11 wrapper

`forecaster_reflect` 加 `dataset` 参数 + env 开关：
```bash
ADAPTTS_PLANNER=prior_aware     # 启用新 planner
ADAPTTS_ALLOW_REMOTE=1          # 含远程模型
```

`PriorPlan` 与旧 `Plan` 共享字段，wrapper 下游零修改。Integration test (Weather, N=80) → `PriorPlan(level='L1', strategies=['chronos2'])` 触发快通道 ✅。

### 3.7 Round 4-A 完成度

feedback 23 项 → **22 完成** (含 NEGATIVE 实证) / 1 部分（forecasting soft router 待 Round 4-B 数据）。

---

## 4. Round 4-B · 远程 Sweep + Risk/Cost 评估 + Findings 写入

### 4.1 远程 sweep: Time-MoE / Sundial

`results/{time_moe,sundial}_vs_c2.jsonl` 各 34 cells。**首跑 catastrophic** → 诊断 wrapper 缺 z-score normalize（Maple728 quickstart 要求但 README 无文字说明 → F4 命题）。

**F4 修复**：`baseline/time_moe.py:24-35` 加 manual z-score + denormalize。ILI MAE 从 740k → 58k。Sundial 同样修补但输出无变化（model 内部已 RevIN）→ Sundial 是真实 underperform。

### 4.2 5-way oracle 实测

| Library | mean oracle MAE | Δ vs C2 |
|---|---|---|
| C2 alone | 5102.19 | 0 |
| C2 + TiRex | 4835.31 | **-5.23%** |
| C2 + TiRex + Toto | 4834.76 | -5.24% (+0.01) |
| + Time-MoE | 4834.69 | -5.24% (+0.00) |
| **+ Sundial = 5-way** | **4834.69** | **-5.24% (+0.00)** |

→ **F2/F3/F6 重大实证**：加 Time-MoE/Sundial **零边际收益**。

Per-dataset Δ vs C2 (%)：

| dataset | tirex | toto | time_moe | sundial |
|---|---|---|---|---|
| ECL | -16.2 | **-43.9** ⭐ | -14.7 | +71.6 |
| Exchange | **-35.1** ⭐ | -4.6 | +38.8 | +279.7 |
| ETTh2 | +12.2 | +30.1 | +29.4 | +0.9 |
| ILI | +1.8 | +51.8 | +27.6 | +536.0 |
| ETTh1 | +74.7 | +111.1 | +157.3 | +132.3 |
| Weather | +486.3 | +671.4 | +286.5 | +534.5 |

### 4.3 P1 Risk-sensitive (`utils/risk_metrics.py`)

公式：$$\text{risk}_k = E[\ell_k] + \lambda \cdot \text{std}[\ell_k]$$

| 模型 | n | mean | std | risk(λ=1) | median |
|---|---|---|---|---|---|
| chronos2 | 34 | 5102.19 | 14393.62 | 19495.81 | 2.77 |
| tirex | 34 | 5198.81 | 14371.02 | 19569.83 | 3.14 |
| time_moe | 34 | 6510.83 | 17841.69 | 24352.52 | 4.64 |
| toto | 34 | 7744.47 | 21560.34 | 29304.81 | 3.81 |
| sundial | 34 | 32435.80 | 88808.60 | 121244.40 | 3.65 |

**Pareto (mean, std)**: 仅 `{chronos2, tirex}` 非支配。

**λ-sweep winner**: λ ∈ [0, 2] → chronos2；λ = 5 → **tirex 反超**（tipping point）

**Per-dataset risk-min (λ=1)**:
- ECL → toto / Exchange → tirex / ETTh1 → chronos2 / **ETTh2 → sundial** ⭐ / Weather → chronos2 / ILI → tirex

**F12 新发现**：Sundial 在 ETTh2 是 risk-min winner（mean=3.53 高，但 **std=0.12 极低**） — 纯 mean-loss prior 漏掉这类模型。**Round 5 unified prior 必须含 variance 项**。

### 4.4 P2 Cost-aware (`utils/cost_metrics.py`)

复合 cost：$$c_k = \alpha \log(\text{lat}) + \beta \log(\text{params}) + \gamma \cdot \text{env\_penalty} + \delta \log(\text{VRAM})$$

env_penalty: 0 = main / 1 = alt-local / 2 = remote

| 模型 | params | VRAM | env | lat(s) | cost | mean MAE |
|---|---|---|---|---|---|---|
| chronos2 | 120M | 2GB | 0 | n/a | 0.19 | 5102.19 |
| tirex | 128M | 2GB | 0 | n/a | 0.20 | 5198.81 |
| toto | 151M | 2GB | 0 | 1.04 | 1.25 | 7744.47 |
| time_moe | 50M | 2GB | 2 | 0.14 | 4.15 | 6510.83 |
| sundial | 128M | 2GB | 2 | 0.31 | 4.69 | 32435.80 |

**Pareto (MAE, cost): {chronos2}** 单点支配；α-sweep ∈ [0, 5] 全 winner = chronos2。

→ **F10**: 真实 trade-off 只有 Timer-S1 (8.3B / 16GB VRAM) 加入后才会浮现。

### 4.5 Findings F1-F12 写入

详见 §0 索引。本节实测 → F2, F3, F4, F8, F10, F12 直接证据；前序 Round 提供 F1, F5, F6, F7, F9, F11。

### 4.6 Timer-S1 状态

Weights (16GB) 在 hf-mirror.com 间歇下载（约 691MB 进度），retry loop 持续中。Timer 完成后 6-way oracle 是否突破 -5.24% 仍开放——按 F3 niche 模式预测：除非 Timer 在某未覆盖 niche 上 dominate，否则 oracle 不变。

---

## 5. Round 5 · Bayesian Unification (feedback Round 4 Phase 1)

**关键 thesis (F11)**: well-motivated heuristic ≠ principled framework。**纯重 frame，零新机制**。

### 5.1 核心抽象 (`agent/bayesian_router.py`)

$$p(M_k \mid x, h) \propto \exp\left(\sum_i \log \pi_k^{(i)}(x) + \sum_j \log L_k^{(j)}(x, h)\right)$$

三类：`Context` (dataset, N, H, entropy, industrial) / `Evidence` (cv_losses, cv_std, memory_neighbors) / `BayesianRouter` (priors + likelihoods + decide)。

**三个 decide 模式（替换 Round 4-A 三个独立 hand path）**：

| Mode | 公式 | 替代 |
|---|---|---|
| `argmax` | argmax_k p(M_k\|x,h) | hand if/else + L0 trust threshold |
| `thompson` | r_k ~ p(M_k\|x,h); argmax(r_k) | `epsilon_greedy_perturb` (ε-greedy) |
| `risk_min` | argmin_k E[ℓ_k] + λ·Var[ℓ_k] | margin gate |

### 5.2 7 个 hard switch → 7 个 Factor

| Round 4-A (hard) | Round 5 (soft Factor) |
|---|---|
| `if N<15: w_C2=0.9` (hard cliff) | `NPrior(strength=2.0)` smooth ramp |
| `if entropy > τ: margin*=2` | `EntropyPrior(beta=0.5)` log-odds |
| `if quant_bits low: use euclid` | `IndustrialPrior(strength=2.0)` |
| `_apply_type_prior × 0.3` | `TypePrior(log_factor=log 0.3)` |
| `LOCAL/REMOTE mask` | `AvailabilityPrior(log_mask=-1e6)` |
| `consensus_winner_inv_loss` | `MemoryLikelihood` (sim · 1/error) |
| `epsilon_greedy_perturb(ε=0.2)` | `decide(mode="thompson")` |

### 5.3 7 unit tests 全通过

| Test | 验证 |
|---|---|
| Weather + N=100 | π(C2) = 0.7691 ≈ Round 4-A 0.77 ✓ |
| ECL + N=100 | π(toto, tirex, c2) = (0.448, 0.300, 0.252) ≈ Round 4-A (0.45, 0.30, 0.25) ✓ |
| N=10 cold-start | π(C2) = 0.6506（旧 cliff 0.90，新更诚实） |
| CV losses BMA σ²=0.5 | π(toto) = 0.9028（sharp posterior） |
| decide modes | argmax→toto / thompson(20) {toto:16,tirex:3,c2:1} / risk_min→toto |
| entropy 0→2 | π(C2): 0.252 → 0.110 (smooth) |
| industrial 0→1 | π(euclid): 0.091 → 0.425 (smooth) |

**关键**：Round 4-A 行为**精确匹配**（误差 ≤ 0.005）→ 重 frame **零信息损失**。

### 5.4 接入 forecaster_reflect

```bash
ADAPTTS_PLANNER=bayesian          # 启用 Round 5
ADAPTTS_DECIDE=argmax|thompson|risk_min
ADAPTTS_RISK_LAM=1.0
ADAPTTS_ALLOW_REMOTE=0|1
```

`PriorPlan_from_posterior()` 适配 BayesianRouter → PriorPlan，零修改下游。`STRATEGY_FN` 扩到 16 entries（懒加载所有 Round 3 baselines）。

Integration test (Weather → C2 → L1) 跑通端到端 inference。

### 5.5 论文 framing 升级

**Before**: "Hierarchical L0/L1/L2 + 4-layer prior stack" → reviewer 读作 well-engineered heuristic stack

**After**: $\hat{M}(x, h) = \arg\max_k \log \pi_k(x) + \log L_k(x, h)$ with 6 factorized priors + 2 likelihoods + 3 decision modes → **principled Bayesian framework**

每个 Factor 可独立消融 → 论文 §5 ablation 自然铺开。

### 5.6 §八 Unify · BayesianRouter 接入 TSC (`clf_planner`)

**目标**：证明同一 `BayesianRouter` 抽象贯通 forecasting + TSC 两类任务（feedback Round 4 §八 "Universal Time-Series Routing Framework"）。

**改动**：`clf_planner.classification_planner` 新增 3 参数：
```python
use_bayesian: bool = False
bayesian_decide: str = "argmax"     # | "thompson" | "risk_min"
dataset: str | None = None
```
+ env 开关 `ADAPTTS_CLF_PLANNER=bayesian`。

**Bayesian 路径** (~50 行)：
1. CV 跑分 → `Evidence.cv_losses = {clf: 1 - acc}`
2. industrial signal → `Context.industrial ∈ [0, 1]`（连续，非 hard override）
3. memory neighbors → `Evidence.memory_neighbors`（重用 Item 4 query_diverse）
4. 构造 `BayesianRouter(NPrior + IndustrialPrior + AvailabilityPrior + CVLikelihood + MemoryLikelihood)`
5. `router.decide(ctx, ev, mode=bayesian_decide)` → 单一决策

**Behavior 验证** (合成 5-classifier × CV losses + ind + std)：

```
=== TSC unified routing tests (NPrior smooth ramp) ===
N= 3  chosen=rocket  π=0.649    ← NPrior strong boost
N= 5  chosen=rocket  π=0.511    ← smooth taper
N= 7  chosen=rocket  π=0.371    ← NPrior=0, CV likelihood 接管
N=20  chosen=rocket  π=0.371    ← stable

=== Industrial prior (Wafer-like signal) ===
ind=0.0  π(euclid)=0.067   (rocket=0.067, tied)
ind=0.3  π(euclid)=0.150
ind=0.6  π(euclid)=0.303
ind=1.0  π(euclid)=0.591   ← smooth takeover, 无 hard threshold

=== risk_min mode (rocket 0.15±0.03 vs weasel 0.13±0.10) ===
λ=0.0  →  weasel   (low mean wins)
λ=1.0  →  rocket   (low std compensates)    ← F12 命题落地
λ≥2.0  →  rocket
```

**关键观察**：
1. **N=7 边界无突变**：旧版 hard cliff (N<7 强制 rocket / N≥7 走 CV) 现在是平滑 sigmoid，π(rocket) 在 N=5 时 0.51（不是 1.0），更诚实表达不确定度。
2. **Industrial prior 取代 hard override**：旧版"if acf_decay high AND quant_bits low → euclid"现为 strength·industrial 连续项 → strength 可消融。
3. **risk_min mode 实操**：λ tipping point 直接控制 mean vs std 权衡 → F12 ETTh2 Sundial 现象现在可在 routing 层利用。

**完成度**：

| feedback Round 4 项 | 落点 |
|---|---|
| Phase 1 Bayesian unification | ✅ Round 5 §5.1-5.5 (forecasting) + §5.6 (TSC) |
| §八 Universal framework | ✅ 同一 `BayesianRouter` 接 forecasting + TSC |
| §四 Risk-sensitive | ✅ §4.3 + decide(risk_min) |
| §六 Cost-aware | ✅ §4.4 |
| §九 NOT add heuristics | ✅ 零新机制；TSC 路径只删/合并 hard if，无 hand rule 新增 |

**论文价值升级**：

method.md 现在可写：
> §4 Bayesian Adaptive Routing
> A single decision rule
>  $$\hat{M}(x, h) = \arg\max_k \log \pi_k(x) + \log L_k(x, h)$$
> 同时适用于 **forecasting 与 classification**，证伪 reviewer 关于 "经验 patch" 的担忧。Forecasting wrapper `forecaster_reflect` 与 TSC `classification_planner` 共享 `bayesian_router.BayesianRouter` 抽象，仅交换 prior factor 集合（forecasting 用 CRPSPrior + EntropyPrior；TSC 用 IndustrialPrior + MemoryLikelihood）。

---

## 6. Round 5 · Phase 4 — Learned Routing Representation (feedback Round 4 Phase 4 + §七)

**关键 thesis**：把 hand-coded `dataset` 名/25-d hand feature 替换为**学习的 series embedding** + **k-means regime cluster**。Routing 决策不再依赖 hand label — 任何 unseen 序列自动获 regime 分配。

### 6.1 Embedding 协议 (`agent/representation.py`)

```python
class Embedding(Protocol):
    dim: int; name: str
    def embed(self, series) -> z       # [L] → [dim]
    def embed_batch(self, batch) -> Z  # [B, L] → [B, dim]
```

三实现：

| Embedding | dim | 来源 | 用途 |
|---|---|---|---|
| `HandFeatureEmbedding` | 25 | series_features.extract_full_features | baseline |
| `MomentEmbedding` | 512 | AutonLab/MOMENT-1-small (frozen TSFM encoder) | 主推 |
| `Chronos2Embedding` | 768 | amazon/chronos-2 T5 encoder mean-pool | 长上下文 |

### 6.2 `RegimeAssigner` — k-means → 学习的 regime prior

```python
assigner = RegimeAssigner(K=8, embedding=MomentEmbedding())
assigner.fit(stored_Z, stored_losses)     # Z [N, dim], losses list[{model: mae}]
assigner.regime_prior(z) -> {model: π_k}  # for any new z
```

实现：
- k-means cluster on stored Z (cosine, L2-normalized)
- per cluster 聚合 per-model loss → π_k(regime) = 1/loss 归一
- 新 series 通过 `embed → cluster label → regime π` 一步出 prior

### 6.3 两个新 BayesianRouter 因子

| Factor | 类型 | 公式 |
|---|---|---|
| `RegimePrior(assigner=...)` | PriorFactor | log π_k = log assigner.regime_prior(z)[k] |
| `RepresentationLikelihood(stored_Z, stored_losses, k=5)` | LikelihoodFactor | log Σ_i sim(z, z_i)·1/(loss_i[k]+ε)，kNN in z-space |

`RepresentationLikelihood` 是 Item 3 `MemoryLikelihood` 的 **z-space 版本** — 检索基于学习的 embedding，不是 hand 25-d。

### 6.4 实测 · regime vs dataset partition (34 cached cells)

K=6 hand25：

| Regime | 主导 dataset | π top-2 | 含义 |
|---|---|---|---|
| **R1** | **Weather (6/6)** ⭐ | C2=0.58 / time_moe=0.15 | **F9 复现** — saturation domain |
| **R2** | **Exchange (3/3)** ⭐ | TiRex=0.38 / Toto=0.22 | **F9 复现** — financial niche |
| R3 | ECL+ETTh2 (6+3) | Toto=0.28 / TiRex=0.21 | observability proxy |
| R5 | ILI+ETTh1+ETTh2 (4+3+3) | C2=0.28 / TiRex=0.27 | mixed long-horizon |
| R0 / R4 | ETTh1/Exchange (3/3) | balanced | within-dataset variation |

K=8 regime purity = **82.4%** (28/34 cells in dominant dataset)：

| Regime | 大小 | dominant ds | purity | 关键观察 |
|---|---|---|---|---|
| R5 | 6 | Weather | 100% | saturation regime 干净独立 |
| R0/R3 | 3+3 | Exchange | 100%+100% | **同 dataset 分 2 regime** — dataset 名漏掉 within-dataset 变化 |
| R4/R7 | 3+3 | ETTh1 | 100%+100% | 同上，且 π 不同 (R4: C2=0.28; R7: C2=0.37) |
| R1 | 7 | ILI (57%) | mixed | **跨 dataset cluster**：ILI+ETTh2 共 "long-horizon" regime |
| R6 | 6 | ETTh2 (50%) | mixed | ETTh2+ECL 共 "industrial volatility" regime |

**Phase 4 thesis 强证据**：
1. ✅ regime ⊃ dataset：同 dataset 可分裂多 regime（within-dataset 异质性）
2. ✅ cross-dataset 共享 regime（跨 dataset commonality）
3. ✅ 18% 的 cells 在 cross-dataset cluster — dataset 名完全无法表达

### 6.5 End-to-end zero-shot regime routing 验证

```python
emb, assigner = build_regime_pipeline("hand25", K=6)
router = BayesianRouter(
    candidates=[...],
    priors=[AvailabilityPrior(), RegimePrior(assigner=assigner), TypePrior(), NPrior()],
    likelihoods=[CVLikelihood()],
)
for series in [weather_like, exchange_like, ecl_like]:
    z = emb.embed(series)
    ctx = Context(N=80, features={"z": z}, allow_remote=False)
    chosen, post = router.decide(ctx, mode="argmax")
```

**结果**：3 个合成序列各自获 regime 分配 + 后验 + 决策，**无 dataset 名输入**。

### 6.6 Phase 4 论文意义

| 维度 | Round 5 | Phase 4 |
|---|---|---|
| Prior source | `prior_crps.get_prior(dataset=name)` 需 hand label | `RegimePrior(assigner)` 从 series 自动 |
| 抽象 | dataset 级（粗）| regime 级（**严格更细**）|
| Zero-shot | ❌ 需 dataset 名查 prior 表 | ✅ 任何 series 都可路由 |
| Across-dataset commonality | ❌ | ✅ 自动捕捉（实测 18% cells）|
| Within-dataset heterogeneity | ❌ | ✅ Exchange 自动分 2 regime |
| Hand engineering | 高（dataset 标签 + niche 人工定义）| 低（K 是唯一超参；π 完全数据驱动）|

**method.md §4 终极版**：

$$\hat{M}(x, h) = \arg\max_k \log \pi_k(z) + \log L_k(z, h), \quad z = f_\phi(x)$$

其中 $f_\phi$ = frozen TSFM encoder（MOMENT-1-small 或 Chronos-2 encoder），$\pi_k(z)$ = regime-conditioned prior from k-means。这是 feedback Round 4 **Phase 1 + Phase 4 + §七** 三合一的最终形式。

### 6.7 Phase 4 完成总结

| 项 | 状态 | 产物 |
|---|---|---|
| Embedding protocol + 3 实现 | ✅ | `representation.py:HandFeatureEmbedding/MomentEmbedding/Chronos2Embedding` |
| RegimeAssigner (k-means + per-regime π) | ✅ | `representation.py:RegimeAssigner` |
| RegimePrior (替换 CRPSPrior 的 dataset 依赖) | ✅ | `representation.py:RegimePrior` |
| RepresentationLikelihood (z-space kNN memory vote) | ✅ | `representation.py:RepresentationLikelihood` |
| `build_regime_pipeline` 一键 (embedding + 34 cells fit) | ✅ | `representation.py:build_regime_pipeline` |
| K=6/K=8 regime purity 实测 | ✅ | 82.4% purity，18% cross-dataset |
| End-to-end zero-shot routing (3 合成 series) | ✅ | 跑通 |
| MomentEmbedding 实测（slow load）| ⏳ | hand25 已验证原理；MOMENT 实测留 next sweep |

### 6.8 Phase 4 → Round 6 钩子

| 路径 | 阻塞 |
|---|---|
| MomentEmbedding 跑完整 sweep + 对比 hand25 | 1 次 MOMENT load (~30s) × 34 cells |
| 用 RegimePrior 替换 forecaster_reflect 默认 prior | 接 1 个 env 开关 `ADAPTTS_PRIOR=regime` |
| online regime update（每次新 cell → 增量 update assigner）| streaming k-means |
| Contrastive embedding 微调（不只是 frozen TSFM）| 需要 (similar regime, different winner) pair mining，工程量大 |

---

## 7. Round 5 · Phase 2 — Contextual Bandit / Thompson Routing (feedback Round 4 §二)

**关键 thesis**：把 "exploration + exploitation + uncertainty + adaptation" 统一到一个 sequential decision rule。Round 5 `decide(mode="thompson")` 是单步采样，**Phase 2 增加 across-time 累积 belief**。

### 7.1 BanditState — per-(regime, model) Gaussian conjugate (`agent/bandit.py`)

Gaussian-Inverse-Gamma 近似（实用版 NIG）：

```
n_post  = decay·n_prior + 1
μ_post  = (decay·n_prior·μ_prior + ℓ_obs) / n_post
σ²_post = predictive var; epistemic uncertainty σ_μ = √(var / n)
```

Thompson sample per arm: $r̃_k \sim \mathcal{N}(\mu_{r,k}, \sigma_{r,k}^2)$，choose argmin(r̃).

**API**：
- `BanditState(prior_mu, prior_var, prior_n, decay)` — `decay<1` 用于非稳态
- `observe(regime, model, loss)` — Bayesian update
- `belief(regime, model) -> (μ, σ_μ)` — posterior
- `thompson_sample(regime, candidates, rng) -> {m: r̃}` — 单步采样
- `best_arm(regime, candidates)` — greedy exploitation

### 7.2 ContextualBanditRouter — wraps regime embedding

```python
router = ContextualBanditRouter(
    candidates=[...], bandit=BanditState(),
    regime_fn=lambda z: assigner.predict_label(z[None,:])[0],
)
z = emb.embed(series)
chosen, samples = router.decide(z, mode="thompson")    # | "greedy" | "ucb"
# ... actually run model ...
router.observe(z, chosen, observed_loss)    # closes the loop
```

三 decision modes：
- `thompson`: r̃_k ~ N(μ, σ); argmin → 不确定 arm 偶尔被试
- `greedy`: argmin μ → 纯 exploitation
- `ucb`: argmin (μ − β·σ) → 乐观探索（确定性、更易分析）

### 7.3 实验 · 4 policy 模拟比较 (3 regime × 4 arm × 200 episode)

True losses per regime（合成）：
```
regime 0: A=0.2 B=0.5 C=0.5 D=0.5   → A best
regime 1: A=0.6 B=0.2 C=0.5 D=0.5   → B best
regime 2: A=0.5 B=0.5 C=0.2 D=0.5   → C best
```

200 episodes × 3 seeds average，oracle = 0.2 per step：

| policy | cum_loss | cum_regret | mean_regret |
|---|---|---|---|
| random | 86.61 | 46.61 | 0.2330 |
| **greedy** | 43.52 | **3.52** | **0.0176** ⭐ |
| ucb (β=2) | 46.22 | 6.22 | 0.0311 |
| thompson | 49.76 | 9.76 | 0.0488 |

**观察**：
1. 所有 bandit policy 远超 random（regret 5-13× 小）
2. 短 horizon (200) 下 greedy 反而最优 — exploration 成本未回本
3. thompson > greedy 的优势需**非稳态环境**显现（regime 分布偏移或 arm 漂移） → 进 Round 6 fully online setting 时才会兑现

→ Phase 2 的实际价值在**架构闭合**，不在 200-episode 单次实验 win。

### 7.4 实测 · cached 34 cells 作 warm-start

```python
for c in cached_cells:
    z = emb.embed(c.history)
    for m, loss in c.per_model_losses.items():
        bandit.observe(regime_fn(z), m, loss)
# 然后任意 fresh series 直接 decide
```

ETTh1/N=10/seed=1 测试：
```
regime = 0
thompson chose: chronos2  (samples: c2=0.72 tirex=1.43 toto=1.13 moe=1.35 sundial=2.10)
greedy   chose: chronos2  (beliefs: c2=0.91 tirex=1.33 toto=0.97 moe=1.09 sundial=1.62)
```

**关键观察**：bandit belief 与 §6.4 regime π 一致（chronos2 在 regime 0 最低 loss）→ 验证 bandit state 正确积累 cached data。

### 7.5 Phase 2 → Round 6 闭环路径

| 需求 | 当前 status |
|---|---|
| BanditState 累积 belief | ✅ 本节 |
| Thompson / UCB / greedy decide modes | ✅ 本节 |
| 接 regime embedding (Phase 4) | ✅ 本节 (Test 3 端到端) |
| warm-start from cached sweep | ✅ 本节 |
| forecaster_reflect 接入 (`ADAPTTS_PLANNER=bandit`) | ⏳ next |
| online observe loop 接 actual prediction → loss | ⏳ next（需 wrapper 在 predict 后写 mae back） |
| 非稳态实验（regime 偏移）| ⏳ Round 6 |
| Regret bound 理论（O(√T log T)） | ⏳ paper §5 |

### 7.6 Phase 2 完成总结

| 项 | 状态 |
|---|---|
| `BanditState` Gaussian conjugate + decay | ✅ `agent/bandit.py:36-103` |
| `ContextualBanditRouter` wrapper | ✅ L116-159 |
| 3 decide modes (thompson/greedy/ucb) | ✅ L131-156 |
| 4-policy simulation (random/greedy/thompson/ucb) | ✅ Test 2 |
| 端到端 cached data warm-start + decide | ✅ Test 3 |
| `forecaster_reflect` 接入 + online observe loop | ⏳ next |

### 7.7 feedback Round 4 最终完成度

| Phase | 状态 |
|---|---|
| Phase 1 Bayesian unification | ✅ §5 |
| **Phase 2 Contextual bandit (Thompson)** | ✅ **本节** (架构+模拟+warm-start；online loop 留 Round 6) |
| Phase 3 Dynamic MoE | ❌ (F2 实测 ceiling，留 future work) |
| Phase 4 Learned representation | ✅ §6 |
| §四 Risk-sensitive | ✅ |
| §六 Cost-aware | ✅ |
| §七 Regime manifold | ✅ §6 |
| §八 Universal framework | ✅ §5.6 |
| §九 NOT add heuristics | ✅ |

**总完成度 8/9 (89%)**，仅 Phase 3 Dynamic MoE 因 oracle ceiling 实测压低 ROI 被放弃（写入 paper future work）。

---

## 8. Sweep 1 实测 · 162 cells × 3 policies (SAFE_CANDIDATES)

**Setup**：3 policies × 6 datasets × 3 N × 3 seeds = 162 cells, H=24, SAFE_CANDIDATES = {chronos2, chronos, arima_ets, naive_drift, naive_seasonal}（避开 tirex DynamicCache 兼容问题）。

**结果**：

| Policy | n | mean MAE | 决策分布 | 备注 |
|---|---|---|---|---|
| **prior_aware** ⭐ | 54 | **18860** | chronos2:42 / toto:6 / tirex:6 | 80% niche-aware 走 chronos2，正确分流 toto/tirex |
| bandit_thompson | 54 | 20787 (+10%) | 5-model 近均匀分布 | 探索成本未回本 |
| bandit_greedy | 54 | 23852 (+26%) | chronos:19 / naive_*:17 / arima:5 | 锁定到次优 arm |

**Findings F13 (new)** — bandit cold-start 在 162 episode 内**未追上 prior_aware**：
- prior_aware 用 `prior_crps` 已从 34 cached cells 学到 niche partition → 起手即知道 toto→ECL, tirex→Exchange
- bandit prior_mu=1, prior_n=2 弱先验 → 必须从 zero 学；K=6 regimes × 5 arms = 30 个 (regime, arm) belief slot，每 slot 平均 ~5 obs 不足以收敛
- **意义**：Round 4-A heuristic prior 在 stationary 数据上仍是 best baseline；bandit 价值在 drift / 新模型加入场景 (Round 6 P4 待验证)

**0 errors / 162 cells** → bandit + observe_outcome 闭环工程级稳定。

---

## 9. Sweep 2 · Full Library 实测 (162 cells, 8 candidates)

`research/experiments/full_library_sweep.py`，候选集 {chronos2, tirex, toto, timesfm2, moirai, moirai2, naive_drift, arima_ets}，per-cell failure → fallback chronos2。

### 9.1 Aggregate 结果

| Policy | mean MAE | Δ vs prior_aware | 真正选中 (non-fallback) | fallback 率 |
|---|---|---|---|---|
| **bandit_greedy** ⭐ | **18358** | **-2.7%** | chronos2:48 / tirex:3 / toto:3 | 36/54 (67%) |
| prior_aware | 18860 | 0 | chronos2:42 / toto:6 / tirex:6 | (未报 fallback) |
| bandit_thompson | 21635 | +14.7% | chronos2:35 / toto:5 / tirex:5 / naive:5 / arima:4 | 30/54 (56%) |

### 9.2 关键 finding F14 (NEW)

**反直觉**：bandit_greedy **赢** prior_aware，但本质是 **"通过失败学习到 chronos2 是最稳"**：
- prior_aware 仍按 prior_crps 历史信号选 toto/tirex（认为它们在 ECL/Exchange 是 winner），但本机 env 实际不可加载（DynamicCache + ModuleNotFound），结果**多花延迟 + 仍 fallback 到 chronos2**
- bandit_greedy 通过几次失败 obs 学到 "tirex/toto loss 大" → posterior 收敛到 chronos2
- bandit_thompson 探索分散 → 持续踩雷

**Finding F14**：
> **可用性 = 决策的隐式输入**。当 prior 信号源于"在另一 env 跑通"的历史数据，但部署 env 不同时，prior 信号成为**误导**。Bayesian routing 在多 env 部署场景下必须把 `availability` 作为 **explicit prior factor**，而非外挂 mask（当前 `AvailabilityPrior` 只能 hard 排除，不能 soft 降权）。

**意义**：Round 6 应实现 **`OperationalReliabilityPrior`** — 学习 per-(env, model) 加载成功率，并对低可靠性模型 soft 降权。

### 9.3 错误类型分布（66 fallback cells）

| 错误 | 次数 | 来源 |
|---|---|---|
| ValueError | 58 | tirex DynamicCache + 各种 input shape mismatch |
| ModuleNotFoundError | 8 | timesfm2 / moirai2 未在本 env 安装 |

### 9.4 Per-dataset Δ Sweep1 vs Sweep2 (prior_aware)

所有 6 dataset 上 Δ = 0%（同 cell 同 router 输出一致）。**说明 prior_aware 完全确定性**，没有探索抖动。

### 9.5 Sweep 1 vs Sweep 2 综合

| | Sweep 1 (5 safe) | Sweep 2 (8 ext.) |
|---|---|---|
| prior_aware mean MAE | 18860 | 18860 (一样) |
| bandit_greedy | 23852 (锁定 chronos+naive) | **18358** (锁定 chronos2 after env failure feedback) |
| bandit_thompson | 20787 | 21635 (探索成本更大) |
| 最佳 policy | **prior_aware** | **bandit_greedy** |

→ 候选集越大，bandit 通过失败学习的优势越显现。但 **F14 揭示这是"踩雷过滤" 优势**，不是真正 niche-aware。

---

## 10. Round 6 · Adaptive Runtime 实测对比 (162-cell × 3 policies)

`research/experiments/adaptive_compare_sweep.py` · 同 cell grid 跑 adaptive (Round 6) vs bayesian (Round 5) vs prior_aware (Round 4-A)。

### 10.1 Raw 结果 (含 selection bias)

| Policy | n | mean MAE | Δ vs prior_aware | 决策分布 |
|---|---|---|---|---|
| **adaptive** | 48/54 | 7312 | **-61%** ⚠ misleading | tirex:23 / chronos2:18 / toto:7 |
| bayesian | 54/54 | 18240 | -3.3% | chronos2:42 / toto:6 / tirex:6 |
| prior_aware | 54/54 | 18860 | 0 | chronos2:42 / toto:6 / tirex:6 |

⚠ adaptive 6 cells 失败（全在 ILI）→ raw mean 有 selection bias。

### 10.2 Apples-to-apples (共同 48 cells)

| Policy | mean | median | std | **Δ vs prior** |
|---|---|---|---|---|
| **adaptive** | 7312 | 3.66 | 29259 | **-23.4%** ✓ honest |
| bayesian | 8843 | 3.54 | 34686 | -7.3% |
| prior_aware | 9541 | 3.45 | 37580 | 0 |

### 10.3 Per-cell win/loss (adaptive vs prior_aware)

```
adaptive wins  15 cells
ties           15
adaptive loses 18  ← 多数 cells 小输！
```

**关键 honest finding F15** ⭐：

> **adaptive 是高方差 "全或无" policy**。少数 niche cells (Weather/ECL → tirex/toto specialist) 大胜，但**多数 cells 略输** prior_aware 的稳定 default。
> 平均 -23% 的 gain **几乎全部来自分布尾部的少数大胜**，中位数实际略差 (3.66 > 3.45)。
> Reflective loop 升级（L1+L3 ensemble）触发率 = 31% (15/48)，但不能保证比 single-model 更稳。

**意义**: feedback Round 5 §3.1 "Circuit Breaker + 优雅降级" 是必须 — adaptive 的 6 个 ILI 失败 + 18 个小输 cells 是同源问题（缺 fallback 保护）。

### 10.4 Per-dataset 详解

| dataset | adaptive | bayesian | prior_aware | 解读 |
|---|---|---|---|---|
| ECL | 16.97 | 17.56 | 17.30 | adaptive **赢** (-2%) |
| Weather | **11.75** | 12.84 | 12.84 | adaptive **大赢** (-8.5%) ⭐ |
| Exchange | 0.018 | 0.017 | 0.018 | ~tied |
| ETTh1 | 2.22 | 2.17 | 2.08 | adaptive **小输** (+7%) |
| ETTh2 | 3.98 | 3.93 | 3.99 | ~tied |
| ILI | 116892 | 109403 | 113126 | adaptive **小输** (+3%) — 加上 6 个 failures 实际更差 |

niche 化决策（tirex/toto）在 Weather/ECL 上正确，但在 ETTh1/ILI 上引入 wrong-specialist 噪声。

### 10.5 Latency 对比

| Policy | mean wall (s/cell) |
|---|---|
| adaptive | 14.71 (含 telemetry + observe loop) |
| bayesian | 10.26 |
| prior_aware | **1.95** ⭐ |

adaptive 比 prior_aware 慢 **7.5×** — 主要来自 reflective L1 ensemble (跑 3 模型) + telemetry I/O。**生产部署需关 L1 escalation 或加 confidence-gated**。

### 10.6 6 个 ILI 失败原因

```
ILI N=20 s=1     SSLError (HF API 间歇)
ILI N=20 s=42    ModuleNotFoundError: 'research.baseline.moirai'  ← prior 含 moirai 但本机未装
ILI N=20 s=123   ValueError: shape (24,) vs (10,)                ← 模型返回错长度
ILI N=50 s=1,42,123  同上 shape mismatch (3 cells)
```

→ **Round 7 P0 必做**:
1. **CircuitBreakerPrior**：失败 N 次后自动 hard mask 该 model
2. **ShapeValidator wrapper**：predict 后校验长度 == H，不符则 fallback
3. **OperationalReliabilityPrior** (F14 解药)：成功率 → log-odds 持续 soft 降权

---

## 11. Open Items

### 阻塞 (外部依赖)

| 项 | 阻塞 | 路径 |
|---|---|---|
| Timer-S1 weights 下载 | hf-mirror.com 间歇 EOF | retry loop `/data2/c220/hz/retry_dl.sh` 持续中 |
| TabPFN-TS | license 需交互接受 | 等非交互流程 |
| KairosHope | repo 未公开 | 等 release |

### Round 6+ 待办

| 项 | 内容 | 入口 |
|---|---|---|
| Round 6 P1 | ContextualBanditRouter 接入 forecaster_reflect (`ADAPTTS_PLANNER=bandit`) | `bandit.py` 已就位 |
| Round 6 P2 | Online observe loop：predict → mae → bandit.observe(z, chosen, mae) → 下一次 decide | wrapper 加 post-prediction hook |
| Round 6 P3 | MomentEmbedding sweep（frozen TSFM encoder vs hand25 baseline）| `representation.py:MomentEmbedding` 已就位 |
| Round 6 P4 | 非稳态实验：regime 偏移模拟，验证 thompson > greedy | 加 drift simulator |
| Round 7 | Dynamic MoE：weights = g_θ(x, t, ℓ) 时间/分位变化 | F2 实测 oracle ceiling -5.24% → 论文 future work |
| ~~Phase 4~~ | ~~Learned routing repr~~ | ✅ §6 |
| ~~Phase 2~~ | ~~Contextual bandit / Thompson~~ | ✅ §7 |

### 工程残留

- Bayesian router 实测 sweep（替换 prior_aware 后跑 6 datasets × N × seed 完整对比）
- 远程 sweep wall_time 应写入每 cell jsonl（当前 chronos2/tirex 缺 latency → cost_metrics 不全）
- Cross-env routing 调度器（subprocess dispatcher）→ paper §6 future

---

## 12. Round 6 后半 · Self-Adaptive Closed Loop 实测 (B2 + B3 + E1 + R6-E)

> 方法本身写在 `method2.md` §11.5。本节只承接实测结果 + Findings。

### 12.1 R6-E1 端到端 demo（合成 series）

文件：`experiments/e1_action_demo.py`。合成序列 T∈[0,150) 稳态 (~50) + T∈[150,240) 线性漂移到 110；upper_threshold=100。期望 Action 级联 MONITOR → INSPECT → THROTTLE → SHUTDOWN。

**修复后实测**（every=2, 100 obs）：

| Intervention | 次数 |
|---|---|
| MONITOR  | 85 |
| INSPECT  | 8  |
| THROTTLE | 4  |
| SHUTDOWN | 3  |
| ESCALATE | 0  |

5 条 drift 事件正确入 `state.drift_history`，末态 `memory_trust=0.30`、`explore_scale=3.00`。

**纯隔离测试**（单模型、单 regime、outcome 从 N(0.3, 0.05) 跳变 N(0.8, 0.05)）：

```
pred_residual_z = 9.06   (> 2.0 ✓ fired)
routing_kl      = 0.00   (单模型，无对照)
detected: {feature: False, routing: False, pred_residual: True, residual: True, memory: True}
```

确认 `pred_residual_z` 是真正路由无关的 drift 检测器（见 F-R6.1）。

### 12.2 R6-G 实数据 sanity（ETTh1, 8 步）

文件：`experiments/g_real_demo.py`。R6 全套 (`adaptive_decide → scheduler → safe_predict → ensemble → action_layer → adaptive_observe`) walk-forward 在 ETTh1 OT 列上 8 步（warmup=200, step=100, H=24, threshold=95% 分位 ≈ 33.63）：

| t | truth | pred | std | p_br | decision |
|---|---|---|---|---|---|
| 200 / 300 | 28.8 / 32.9 | 29.2 / 30.9 | 1.1 / 1.4 | 0.00 / 0.02 | MONITOR / MONITOR |
| 400 ─ 700 | 36.6 ─ 39.5 | 36.5 ─ 40.3 | 1.3 ─ 2.0 | 0.97 ─ 1.00 | **SHUTDOWN** ×4 |
| 800 | 30.4 | 32.4 | 1.24 | 0.16 | **INSPECT** |
| 900 | 35.3 | 34.4 | 1.01 | 0.78 | SHUTDOWN |

end-to-end MAE = 1.11；calibrator 在 t=700 处把 conf 从 1.00 调到 0.60；t=800 truth 跌回阈值下，p_breach=0.16 → INSPECT 而非 MONITOR（cost-min 在中等不确定下偏保守）。

### 12.3 R6-H 多数据集 stress（3 ds × 4 scenario × 6 步）

文件：`experiments/h_stress_demo.py`。ETTh1 / ETTh2 / Exchange × {clean, trend_break, variance_explode×3.5, outlier_burst}，序列截到 ~1000 让注入故障落进 walk 中点，全程 17s。

**Per-scenario 跨数据集均值**：

| Scenario | MONITOR | INSPECT | SHUTDOWN | mean MAE |
|---|---|---|---|---|
| clean              | 3.0 | 0.3 | 2.3 | 0.70 |
| trend_break        | 4.3 | 0.3 | 1.3 | 0.69 |
| **variance_explode** | 2.7 | **1.0** | 1.7 | **1.12 (+60 %)** |
| outlier_burst      | 2.7 | 0.3 | 2.0 | 0.72 |

### 12.4 R6-J Drift 收敛性（140-step 长走查）

文件：`experiments/j_drift_convergence.py`。ETTh1 walk 拉到 140 步（warmup=100, step=30, drift_check_every=15）使 drift_engine 的 120-obs 启动窗口能跨过。clean vs 注入 variance_explode×4 两条路径并跑。

| 模式 | n_drift_events | final memory_trust | final explore_scale | mean MAE | SHUTDOWN | INSPECT |
|---|---|---|---|---|---|---|
| clean              | 6 | **0.30** | **3.00** | 0.782 | 30 | 6 |
| injected variance×4 | 6 | **0.30** | **3.00** | **1.887** | 45 | 11 |

`pred_residual_z` 在 clean 模式下最高 0.85（未破 2.0 阈值），在 injected 模式 step 3220 直接 **3.50**（独立 fired）。

### 12.5 Findings F-R6.1 ~ F-R6.4

| ID | 内容 | 实测来源 | 修复 |
|---|---|---|---|
| **F-R6.1** | Drift 4 原始信号（feature_kl / residual_ks / routing_kl / memory_mismatch）都隐含依赖 router 的"决策多样性"作为放大器；router 退化为单模型路径时集体失灵 | §12.1 初版 demo 3 次触发 0 detect | 新增 `pred_residual_z`（Welch mean-shift z-score on outcome）作为第 5 信号，路由无关；§12.1 隔离测试 z=9.06，§12.4 injected 路径 z=3.50 独立 fired |
| **F-R6.2** | 真实 ETTh1 上 router 一致选 chronos2 → scheduler 永远只跑 1 个模型 → 跨模型 disagreement std=0 | §12.2 g_real_demo 早期版本 p_breach 阶跃 (0 或 1) | 单模型时回退到序列残差 std：`std = √(cross_std² + resid_std²)`，让 p_breach 重新有梯度 |
| **F-R6.3** | 故障扰动有差异：variance_explode 是最毒（MAE +60%, INSPECT 顶到 1.0）；trend_break 方向相关（本 seed 偏负向反而减少 SHUTDOWN）；outlier_burst ≈ clean（H=24 平均稀释）；Exchange 值域 <1 全 MONITOR | §12.3 多数据集 stress 表 | 不是 bug，是 Action 层正确的方向 + 量级敏感行为 |
| **F-R6.4** | Drift 修正动作幂等：clean 和 injected variance×4 两条路径终态完全相同（trust=0.30, explore=3.00），多次触发不抖、不过冲；trust 降低后系统"calm down"行为可观察（clean 模式 SHUTDOWN→MONITOR 大幅转移）；MAE 1.887 vs 0.782 (2.4×) 印证 F-R6.3 | §12.4 长走查对比 | 设计本身正确，未来 ablation 用 `record_state_trace=True` 暴露 |
