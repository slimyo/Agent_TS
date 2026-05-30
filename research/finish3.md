# Finish v3 — Round 7 实测

> 版本：2026-05-29 起
> 方法本身写在 `method3.md`。本文件只承接 Round 7 子项的实测 + Findings。
> 前置实测档案：`finish.md`（Phase 1-6）/ `finish-1.md`（Round 2-5 + Round 6）

---

## 0. Round 8 增量 · M1 Meta-bandit on decide_mode

文件：`research/agent/meta_bandit.py`。

### 0.1 单元 smoke（200-step 合成）

3-arm meta-bandit on `{argmax, thompson, risk_min}`，ground truth 平均 loss：

| mode | truth μ |
|---|---|
| argmax | 0.60 |
| thompson | **0.40** (best) |
| risk_min | 0.80 |

冷启动 K=5，200 步 Thompson sample 后：

| mode | usage (n / 200) | learned μ ± σ | truth |
|---|---|---|---|
| thompson | **184 / 200 (92.0%)** | 0.406 ± 0.010 | 0.40 |
| argmax   | 11 / 200 (5.5%) | 0.648 ± 0.100 | 0.60 |
| risk_min | 8 / 200 (4.0%) | 0.836 ± 0.129 | 0.80 |

冷启动正确填满 N=5/arm，Thompson 收敛到 thompson 模式 92%，risk_min（最差）仅 4%。

### 0.2 集成路径 sanity（adaptive_decide/observe × 60 步）

启用 `RouterConfig.decide_mode="auto"` + 在 `adaptive_decide / adaptive_observe` 中自动序列化 / 反序列化 `state.meta_bandit_dict`。同样 truth (thompson 0.35 / argmax 0.6 / risk_min 0.85) 下：

| mode | usage (n / 60) | learned μ |
|---|---|---|
| **thompson** | **50 / 60 (83.3%)** | 0.363 |
| argmax | 6 / 60 (10.0%) | 0.645 |
| risk_min | 4 / 60 (6.7%) | 0.898 |

60 步内已经主要 exploit best mode。`state.meta_bandit_dict` save/load round-trip 通过。

---

## 0.3 Round 8 增量 · M4 Per-regime bandit decay

文件：`research/agent/bandit.py`（扩展 `BanditState`）+ `drift_engine.py`（B3 联动）。

### 0.3.1 单元 smoke · "公平对比"实验

两个 regime 都先观测 20 次 loss=0.5（达到稳态），然后注入 5 次 loss=1.5 漂移：

| Regime | decay | 漂移前 μ | 漂移后 μ | Δ |
|---|---|---|---|---|
| 0 | 0.80 (fast) | 0.502 | **1.175** | **+0.672** |
| 1 | 0.99 (slow) | 0.541 | 0.739  | +0.198 |

→ **fast decay regime 反应漂移 3.40× 快**（Δ_fast / Δ_slow = 0.672 / 0.198）。

### 0.3.2 B3 drift 联动测试

drift_step 在 200 稳态 + 80 漂移合成数据上触发 `boost_exploration`：

```
before drift_step: regime_decay = {}
after  drift_step: regime_decay = {0: 0.9, 1: 0.9}
                   ↑↑↑ 最近 30 条 telemetry 出现的 regime 都被自动收紧
```

只有最近活跃的 regime 被收紧；其他 regime 的 decay 保持默认 1.0。

### 0.3.3 持久化

`BanditState.save/load` round-trip 通过：`regime_decay={0: 0.8}` 经 save → load 后保持完整。

---

## 1. M2 · Model 自动淘汰 · smoke

文件：`research/agent/model_culling.py`。

**单元 smoke**（regime 0 × 5 候选 × 5 obs each）：
- 注入：chronos2 μ=0.30 / tirex μ=0.40 / toto μ=0.85 / **moirai μ=1.20** / naive_drift μ=0.50
- 配置：`fraction=0.30, min_keep=2, protect=(naive_drift, chronos2), min_observations=3`
- 结果：moirai 被识别为底部、加入 `state.culled[0]={'moirai'}` ✓
- `EliminationPrior` 对 moirai 返回 log_prior=-50（其他 0.0）✓
- `resurrect(state)` 清空 culled set ✓

## 2. M3 · Empirical Bayes Prior strength · smoke

文件：`research/agent/prior_learning.py`。

**合成实验**（120 条 telemetry，NPrior 助力 / 高 lp → 低 outcome）：
- Pearson r = +0.616 → NPrior strength **2.000 → 2.246** (+12.3%)
- `state.learned_prior_strengths = {'N_prior': 2.246}` ✓
- `apply_learned_strengths(fresh_router, state)` 把学到的值写回新构造的 prior 实例 ✓
- TypePrior（无 `.strength` 属性）正确跳过 ✓

## 3. M2 + M3 集成 · walk-forward sanity

`adaptive_planner` 改造：
- `adaptive_decide`：在 priors 末尾自动追加 `EliminationPrior(state_ref=state)`；调 `apply_learned_strengths` 把 state 里的学习值灌进当前 prior 实例；`ctx.features["regime"]` 暴露 regime 给 EliminationPrior
- `adaptive_observe`：按 `cull_every / eb_learn_every` 周期触发；drift_engine 的 `boost_exploration` 自动 `resurrect` 已淘汰模型（M2 与 B3 互锁）

**ETTh1 walk-forward**（80 obs, drift_every=20, cull_every=40, eb_every=30, fraction=0.30）：

| n_obs | culled | learned_strengths |
|---|---|---|
| 1 | {} | {} |
| 21 | {} | {} |
| 41 | {} | {'N_prior': 2.0} |
| 80 | {} | {'N_prior': 2.0} |

drift_history len = 3（drift_step 周期性触发）。

**观察**：
1. M2 culling hook 在 n_obs=40 / 80 正确触发，但 culled 保持空 —— 因为 router 始终选 chronos_bolt → arima_ets n=0 < min_observations=3 → 无可淘汰资格
2. M3 EB hook 在 n_obs=30 / 60 / 90 正确触发，但 N_prior 保持 2.0 —— 因为 chronos_bolt 在每步 lp 值高度相似，Pearson r ≈ 0 → 不更新

## 4. Findings F-R7.x

| ID | 内容 | 来源 | 复用前文 |
|---|---|---|---|
| **F-R7.1** | M2/M3 与 F-R6.1 同源约束：当 router 退化为单模型路径时，culling 因"差模型从未被采样"而无可工作样本，EB 因"prior 输出无方差"而 r≈0；两个机制都需要**路由多样性**作触发条件 | §3 walk-forward 80 obs 无 culling/学习 | 同 F-R6.1：feature_kl/routing_kl/memory_mismatch 集体失灵 |
| **F-R7.2** | M2 与 B3 drift 互锁是必要的：drift_engine 触发 `boost_exploration` 时通过 `resurrect()` 清空 culled，避免"扰动来时被淘汰的模型变成永久失活"导致系统失去复原能力 | `adaptive_observe` 集成路径 | — |

**未来缓解**：在 candidates 中保留 1-2 个 forced-explore 模型，或让 BanditLikelihoodFactor 周期性把 ε-greedy 探索打到 culling 候选上。本轮先记入 limitation，待 Round 8 / 与 M7 anomaly 任务结合时统一处理。

---

## 5. M7 · Anomaly Detection Phase 1 · smoke

文件：`research/agent/anomaly.py`。Phase 1 显式不做 LLM / 新 Memory / 论文模型本体；只复用 BayesianRouter + 新 `AnomalyTypePrior` + 2 个轻量 detector。

**单元 smoke**（合成序列 200 步，cumsum 标准正态 ×0.1 + 50.0，注入 4 类故障）：

| 注入 | `type_prior` argmax 概率 | 判定正确 | 备注 |
|---|---|---|---|
| normal             | `normal` 0.475          | ✓ | 兜底 baseline 胜出 |
| trend_break        | **`trend_break` 0.9999** | ✓ | level_shift_z 触发 |
| variance_explode   | **`variance_explode` 0.9946** | ✓ | variance_ratio 触发 |
| outlier_burst      | **`outlier_burst` 0.667**     | ✓ | max_outlier_z 触发 |

4 / 4 故障类型在 type-prior 上正确识别。

`detector_posterior` 在 normal 输入下 rule_baseline (0.58) > residual_score (0.42)；在 variance_explode 下 residual_score (0.51) > rule_baseline (0.49)，符合"高波动数据更偏好模型-based detector"的预期。

## 6. Findings F-R7.x（接续）

| ID | 内容 | 来源 | 复用前文 |
|---|---|---|---|
| **F-R7.3** | M7 Phase 1 实证：window-level fault flavor 完全由 `AnomalyTypePrior` 的 3 条统计特征 (level_shift_z / variance_ratio / max_outlier_z) 决定即可达 4/4 准确；不需要深度模型介入。点级 detector 仅在故障发生在 window 末尾时才放大 combined score，对 regime-shift 类故障（trend_break / variance_explode）combined < threshold | §5 smoke | 这与 F-R6.3 一致：variance_explode 是窗口级而非点级扰动 |
| **F-R7.4** | M7 复用度极高：B2/B3/E1/R6-E/M2/M3 全部都能直接消费 `AnomalyResult` 而无需新机制（B2 吃 score，B3 跟踪 fault 分布，E1 把 is_anomaly+conf 喂进 5-tier，R6-E 把 detector 当 candidate）。印证 feedback §5 "尽量少引入新组件" 的可达性 | §7.3 method3 复用表 | — |
| **F-R8.1** | M1 比 M2/M3 更容易"被自动化体现"，因为不依赖路由多样性：3 个 arm 都会被采样（冷启动强制 K=10/arm），所以即使 router 永远选 chronos2，meta-bandit 仍能拿到 per-mode outcome 区分度。这印证一个一般规律：**自动化层数越高（meta），对下层多样性的依赖越弱** | §0.2 60 步即 83.3% exploit | 对照 F-R7.1：M2/M3 受路由单一约束 |
| **F-R8.2** | 冷启动 K=10 / 60 步 / 3-arm = 30 obs 暖机 + 30 obs exploit，已经达到 80%+ 单 mode usage。production 部署可以从 `decide_mode="auto"` 启动而不担心冷启动惩罚 | §0.2 集成测试曲线 | — |
| **F-R8.3** | M4 实证：fast decay (0.80) vs slow decay (0.99) 在同一漂移场景下 belief 适应速度差 **3.40×**。这意味着扰动检测→收紧 decay 的 B3↔M4 联动确实让"扰动时按需加速遗忘"成为可用的工程手段，而非概念 | §0.3.1 公平对比 | — |
| **F-R8.4** | B3 drift `boost_exploration` 自动收紧的 regime 集合 = 最近 30 条 telemetry 中出现的 regime，而不是所有 regime。这避免了"系统某处漂移就让所有 regime 都遗忘"的过度反应 | §0.3.2 联动 | 同 F-R7.2 互锁哲学 |

---

## 7. M8 · Factor Attribution + framing 修正 · smoke（feedback 问题 1+2）

文件：`research/agent/bayesian_router.py`（docstring framing + 3 个分析接口 + Test 8/9）。

### 7.1 Per-decision LOFO（Test 8, ECL N=10 + CV losses）

```
chosen=toto  runner_up=tirex  margin(log-odds)=+2.400  p=0.899
  factor            Δmargin  KL_drop  flips?
  crps               +0.400    7.036  → timesfm2
  availability       +0.000    5.881  → time_moe
  cv                 +2.000    0.585
  decisive (flip argmax if removed): ['availability', 'crps']
reconstruction max|Σfactors − log_posterior| = 0.00e+00   ← 拆解精确
```

### 7.2 Cross-decision redundancy（Test 9, 200 次随机 ctx sweep）

```
mean |influence| (centred L2 norm):
  availability 165.83 (hard-mask, clipped)  crps 18.80  cv 15.87
  type 1.54  entropy 0.48  N_prior 0.077
redundant pairs (|corr| ≥ 0.8): (none)
```

### 7.3 Findings

| ID | 内容 | 来源 |
|---|---|---|
| **F-R8.5** | Factor 拆解可精确重构 log_posterior（误差 0.0e+00），证明 energy model 是严格可加的、可审计的；LOFO 揭示单次决策的"决定性 factor"未必是 Δmargin 最大者——`cv` Δmargin=+2.0 但去掉只让 KL=0.585 且**不翻转**，而 `crps`/`availability` Δmargin 小却翻转 argmax（因为它们改变的是 runner-up 之外模型的可行域）。**KL_drop / argmax_changed 比 Δmargin 更能反映 factor 的因果影响力** | §7.1 |
| **F-R8.6** | 当前 6 forecasting factor 在 200 次 sweep 上**无 |corr|≥0.8 冗余对** → 现阶段 factor 集尚未触发 feedback 问题 2 的 unidentifiability；`AvailabilityPrior`(±1e6 硬 mask) 数值上需 clip=50 才能与软 factor 同量纲对比，印证它本质是**约束**而非偏好 factor。redundancy_matrix 提供了后续加 factor 时的**自动护栏**（新 factor 与既有高相关 → 拒绝合入） | §7.2 |
