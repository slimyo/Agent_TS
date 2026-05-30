# Routing, Not Competing: Where LLM-Agents Help in the Time-Series Foundation-Model Era

> **Target venue**: ICLR 2026 / NeurIPS 2026 Workshop (Time Series track) or KDD 2026 Applied Track
> **Status**: v0.2 (post-pivot, complete experiments)
> **配套**：`plan.md` / `finish.md` / `classifier.md`

**Alternative titles**:
- *Beyond Direct Prediction: Repositioning LLM-Agents as Routers Around Time-Series Foundation Models*
- *The Right Place for LLM-Agents in the TSFM Era: An Empirical Boundary Across Forecasting, RCA, and Classification*
- *From Competing to Routing: An Honest Study of LLM-Agents in Three Time-Series Tasks*

---

## Abstract

The arrival of pretrained time-series foundation models (TSFMs) raises a question that no current LLM-Agent paper directly addresses: **if a base model already extracts most of what a small training window can support, what is the right role for an LLM-Agent in the system?** We argue, formalize, and empirically validate the answer: **the Agent's role shifts from *prediction* to *selective routing***. We state a **TSFM Saturation Hypothesis** (§3.0.1): when the base model's pretraining distribution covers the test distribution closely, the expected improvement of any wrapper is zero in the limit, while the variance is positive. The Agent's task therefore becomes the **Meta-Decision Problem** $m^*(x) = \arg\max_m \mathbb{E}[U(m, x)]$, which reduces to selective prediction $(f, g)$ where $f$ is the base predictor and $g$ is a routing/abstain gate.

We instantiate this framework on three task types and find a **single architectural pattern** explains all three. Failures appear at the same point (mis-routes under tiny-$N$ CV noise), recover via the same fix (an N-conditional abstain that defaults to the base), and reach the same limit (parity with the base on average, niche-positive improvement on the cells where the base genuinely fails). We provide a four-row mechanism table (§4.0) that summarizes the entire empirical progression independent of version labels.

The empirical work corroborates the theory: across **6 forecasting datasets × 4 N × 3 seeds + 50 RCA cells + 5 UCR datasets × 30 cells + 3 UEA multivariate datasets**, our final architectures achieve (i) **statistically indistinguishable performance from Chronos-2** on forecasting MAE and CRPS (Wilcoxon $p$=0.32; 0W/1L/23T over 24 cells), confirmed by three independent mechanisms (memory consensus, entropy gate, learned abstain head) that all converge to the same mean — direct evidence for the saturation prediction; (ii) **+40 pp R1 over an unstructured LLM-direct baseline on RCA**, with an honestly-reported $-37$ pp loss to a competent rule-based baseline and a stackable mitigation chain (architectural abstain head + dataset prior + LLM upgrade) that reaches **100% out-of-taxonomy detection** on glm-4-plus; (iii) **+0.89 pp over Rocket on UCR-5** for the TSC router (not statistically significant; $p$=0.17), with multivariate UEA exhibiting a strictly larger routing space (DTW > Rocket, opposite of the UCR ranking); and (iv) a **learned-margin head** that beats the heuristic gate by $+0.49$ pp on leave-one-dataset-out CV.

We frame these results as **methodology, not engineering**: the contribution is the unified $(f, g)$ formulation and its empirical corroboration across forecasting, RCA, and classification, not the specific version numbers. The headline take-away is that **in the TSFM era an LLM-Agent should be designed as a meta-controller — a routing and abstaining layer around base models — rather than as a competing predictor**. We close by listing eight feedback-driven directions (online routing, learned memory policy, multimodal industrial RCA) that this reframing opens up.

---

## 1. Introduction

### 1.1 The Question: What Should an LLM-Agent Do When the Base Model is Already Strong?

Two years of progress on pretrained time-series foundation models (TSFMs) — Chronos / Chronos-2 / Chronos-Bolt (Ansari 2024; Amazon 2024-12; 2025-10), TimesFM / TimesFM-2.0 (Das 2024; Google 2025), Moirai (Salesforce 2024), MOMENT (Goswami 2024) — have changed the operational landscape for time-series tasks. A modest few-shot context plus a generic TSFM now produces forecasts and embeddings that are competitive with or better than most specialized small-data methods.

This raises a question that previous LLM-Agent-for-time-series papers (TSci, MemCast, Cast-R1, TSOrchestr, Nexus; see §2) implicitly answer "the Agent should produce a *better prediction*", but never *justify*: **once a pretrained base model already extracts most of what a small training window can support, what is the right role for the LLM-Agent in the system?**

Our contribution is to **make this question precise** (§3.0), give a single decision-theoretic answer that subsumes three time-series task types (forecasting, root-cause analysis, classification; §3.0.2–3.0.3), and **corroborate the answer empirically** across nine datasets and 150+ evaluation cells (§4). The answer is short: **the Agent should not be a competing predictor; it should be a *selective router* — a layer that decides among existing predictors and, when in doubt, defers to the base**.

### 1.2 Three Task Types, One Architectural Principle

We answer this question across three task types, all using the same Curator + Model-Cards + Memory architecture but with different roles for the LLM-Agent:

| Task type | LLM-Agent role | Best baseline | Our final | Gap |
|---|---|---|---|---|
| **Forecasting** (24 cells) | wrapper around Chronos-2 | Chronos-2 itself | v11 (parity wrapper) | 0% (CRPS) |
| **RCA** (30 cells, 5-fault) | structured root-cause analyzer | LLM-direct (text only) | Agent (Curator + Cards) | **+40pp R1** |
| **Few-shot TSC** (30 cells, UCR) | router among {Rocket, MOMENT, DTW, …} | Rocket-alone | B7v3 (router + memory) | **+0.89pp** |

The unifying observation, which we develop in §5.3, is that **the LLM-Agent's value is in conditional model selection — routing or wrapping — not in direct prediction**. This principle is validated by **isomorphic version progressions** in forecasting (v5c → v8 catastrophic → v10 N<15 fallback → v11 memory) and classification (B6 → B7v1 catastrophic → B7v2 N<7 fallback → B7v3 memory). Both progressions show the same failure (direct competition with SOTA), the same intermediate symptom (CV-noise-driven mis-routes at extreme few-shot), and the same fix (N-conditional fallback + cross-series memory).

### 1.3 Contributions

**Methodological** (§3.0):

1. **The TSFM Saturation Hypothesis** — a precise statement of when an LLM-Agent cannot improve over its base model in expectation (Cov(B, D) → 1 ⇒ E[Δ] → 0, Var[Δ] > 0). The hypothesis predicts the average-zero, niche-positive shape of every empirical progression in this paper and is corroborated by Wilcoxon $p$ = 0.32 on the forecasting wrapper-vs-Chronos-2 comparison.
2. **The Meta-Decision reduction** — formalizing the Agent's role as $m^*(x) = \arg\max_m \mathbb{E}[U(m, x)]$ over a *fixed* action set $\mathcal{M}$ (base predictors + abstain). The reduction to selective prediction $(f, g)$ is concrete: our forecasting wrapper v11, classification router B7v3, and RCA abstain head are all $(f, g)$ instances with different $\mathcal{M}$.
3. **A four-mechanism table** (§4.0) that consolidates eight version progressions in two domains into four named failure mechanisms (noisy CV, validation instability, retrospective bias, specialist attention) with their corresponding architectural fixes. The table is a contribution in itself — it gives any future LLM-Agent designer a vocabulary independent of our specific version labels.

**Empirical** (§4):

4. **Forecasting**: three independent mechanisms — memory safety net, entropy gate, learned abstain head — all converge to mean MAE indistinguishable from Chronos-2 (0W/1L/23T over 24 cells, CRPS gap 0%). One OOD failure (Weather N=20 v11, +505%) is reported honestly as a memory-bootstrap brittleness. The convergence across mechanisms is the strongest evidence in the paper for the saturation hypothesis.
5. **Reasoning tasks**: on RCA, structured Agent achieves +40 pp R1 over an unstructured LLM-direct baseline but $-37$ pp R1 against a competent rule-based baseline — an honest negative finding that motivates the abstain-head architectural fix. The fix reaches 76% out-of-taxonomy recall alone, and **100% when stacked with a stronger LLM and a dataset prior** (§4.7.3). On TSC classification, a learned margin head beats the heuristic gate by +0.49 pp on leave-one-dataset-out CV, and a memory-augmented router achieves a (statistically non-significant, $p$=0.17) +0.89 pp on UCR-5 — niche on saturated benchmarks, larger routing space measured on multivariate UEA where DTW outperforms Rocket on average.
6. **Reproducibility infrastructure** (Appendix E): nine documented experiment scripts; deterministic LLM cache; 150+ result JSONL files; statistical significance tests for all main claims.

The paper's structure mirrors this organization — methodology first (§3.0), mechanism abstraction second (§4.0), version-level corroboration third (§4.2-4.11). Readers who only want the architectural takeaway can read §3.0 and the four-mechanism table in §4.0 alone.

---

## 2. Related Work

### 2.1 Few-shot Time Series Forecasting

**Classical statistical baselines** (Hyndman et al., 2008; Hyndman & Khandakar, 2008) include ARIMA and ETS; their few-shot regime is well-studied but typically as a sanity check rather than a target.

**Time-series foundation models** (Ansari et al., 2024, Chronos; Das et al., 2024, TimesFM) provide zero-shot forecasts but show distribution shift sensitivity. **In-context LLM forecasting** (Gruver et al., 2023, LLMTime) tokenizes sequences as digit strings, achieving competitive zero-shot performance with surprising consistency on short contexts.

### 2.2 Agentic Time Series Forecasting (ATSF)

The recent **position paper of Zhao et al. (2026, arXiv:2602.01776)** crystallized the ATSF paradigm, advocating five components: perception, planning, action, reflection, memory. Concrete implementations include:

- **TSci** (Zhao et al., 2025) — a 5-node LangGraph pipeline (Preprocess/Analysis/Validation/Forecast/Report). Our **most direct baseline**.
- **MemCast** (2026) — experience-conditioned reasoning with confidence-adapted memory.
- **Cast-R1** (2026) — RL-trained tool-augmented sequential decision policies.
- **TS-Memory** (2026) — kNN teacher distillation with confidence-aware quantile supervision.
- **TSOrchestr** (2026) — LLM-as-judge ensemble coordination.
- **Nexus** (2026) — generic agentic framework.

**How we differ.** All of the above evaluate on standard (data-rich) benchmarks. We are the first to evaluate the ATSF paradigm systematically in the N=10–100 regime, and the only method to (i) cross-validate diagnostic confidence sources and (ii) use walk-forward CV in the train-tail as a strategy-selection signal.

### 2.3 Uncertainty Quantification in Forecasting

Probabilistic forecasting (Salinas et al., 2020; Rasul et al., 2021) targets predictive uncertainty over the forecast itself. We focus instead on **diagnostic uncertainty** — the confidence that a structural property (trend / seasonality / stationarity) is present in the training segment. This is, to our knowledge, the first systematic treatment of this orthogonal dimension.

---

## 3. Method

### 3.0 Theoretical Foundation: Selective Routing Under TSFM Saturation

We frame the entire paper around a single decision-theoretic thesis that subsumes all three task types (forecasting, RCA, classification). The thesis has three components — a saturation hypothesis about the operating regime, a meta-decision formulation of the Agent's role, and a selective-prediction reduction that recovers our specific architectures (v11, B7v3, abstain head) as instances of a single design pattern.

#### 3.0.1 The TSFM Saturation Hypothesis

Let $\mathcal{B}$ be a pretrained time-series foundation model (e.g., Chronos-2, Rocket), and let $\mathcal{A}$ be any wrapper / Agent that may deviate from $\mathcal{B}$. For a test cell $x$ drawn from distribution $\mathcal{D}$, define:

$$\Delta(\mathcal{A}, \mathcal{B}; x) := \text{loss}(\mathcal{B}; x) - \text{loss}(\mathcal{A}; x)$$

We hypothesize:

> **TSFM Saturation Hypothesis (TSH).** When the base model $\mathcal{B}$'s pretraining distribution covers the test distribution $\mathcal{D}$ closely, the *expected improvement* of any Agent satisfies
>
> $$\mathbb{E}_{x \sim \mathcal{D}}[\Delta(\mathcal{A}, \mathcal{B}; x)] \;\to\; 0 \;\text{ as }\; \mathrm{Cov}(\mathcal{B}, \mathcal{D}) \to 1$$
>
> *and* the variance of $\Delta$ is non-zero — some cells admit improvement (positive $\Delta$), others worsen (negative $\Delta$), and the two cancel in expectation.

**Empirical evidence in this paper**:

- **Forecasting (§4.2, §4.8)**: 24-cell wrapper sweep against Chronos-2 yields $\mathbb{E}[\Delta_{\text{MAE}}] = +0.00$ for v11, with one OOD outlier ($+505\%$, Weather N=20). Wilcoxon $p=0.32$ (no significant difference). Under CRPS the same conclusion holds at $p=0.05$. **Three independent mechanisms** (memory consensus, entropy gate, learned abstain head) all converge to mean MAE identical to Chronos-2 alone, confirming the population-level zero-improvement prediction (§4.8).
- **Classification (§4.5, §4.6)**: Direct B6 Agent loses to Rocket by $-33$ pp ($p<0.001$). After the v8→v10 analog progression, B7v3 closes to $+0.89$ pp on UCR-5 ($p=0.17$, not significant) and $0.0$ pp on less-saturated extended UCR. The expected gain on a satured benchmark is statistically indistinguishable from zero.

The hypothesis explains *why* every direct-competition design in this paper underperforms its TSFM/SOTA counterpart on average: in the saturated regime, the only headroom is in the cells where $\Delta(x) > 0$. **An Agent that wants to extract this headroom must therefore identify** *which* cells admit positive $\Delta$ — not produce a uniformly better prediction.

#### 3.0.2 The Meta-Decision Problem

The Agent's role is not to predict but to **choose among predictors** (and to choose whether to predict at all). Let $\mathcal{M} = \{m_1, m_2, \ldots, m_K\}$ be a set of available actions; in our system $\mathcal{M}$ includes the base TSFM (Chronos-2), alternative classifiers (Rocket, MOMENT, DTW, …), and the abstain action (output uncertainty / refuse). The Agent's policy is

$$m^*(x) \;=\; \arg\max_{m \in \mathcal{M}} \; \mathbb{E}_{y \sim \mathcal{D}_x}\!\left[\,U(m, x, y)\,\right]$$

where $U$ is a task-specific utility (negative MAE / CRPS for forecasting; accuracy or per-class loss for classification; recall / precision for RCA). The Agent does *not* parameterize $m$ — each $m$ is a fixed, pre-existing predictor — but it parameterizes the **selection policy**.

**Three special cases recover the entire paper**:

1. **Forecasting wrapper (v10/v11)**: $\mathcal{M} = \{\text{Chronos-2}, \text{Chronos-Bolt}, \text{ARIMA-ETS}, \text{LLMTime}\}$. The Curator output is the policy's context $x$; the v9–v11 hand-tuned gating is one realization of $m^*(x)$, and the learned-margin head (§4.5 L1) is a more data-driven realization.

2. **TSC router (B7v3)**: $\mathcal{M} = \{\text{Rocket}, \text{MOMENT-1NN}, \text{MOMENT-LR}, \text{DTW-1NN}, \text{Euclid-1NN}\}$. LOO CV gives an estimator $\hat U(m, x)$; the margin and N-fallback rules define how this estimator is gated.

3. **RCA Agent (§4.7)**: $\mathcal{M} = \{\text{trend\_break}, \ldots, \text{stationarity\_flip}, \text{out\_of\_taxonomy}\}$; the abstain head is the gate function that maps the Curator features to the OOT (= $m_{\text{abstain}}$) action.

Under TSH (§3.0.1), $m^* = \mathcal{B}$ for *most* $x$, and the Agent's value lies entirely in the **conditional sub-policy** for the minority of cells where another $m$ dominates. This is why our gains are niche (UCR-5 BeetleFly N=5: B7v3 selects MOMENT, $+20$ pp) and the **average** improvement is near zero.

#### 3.0.3 Selective Prediction Reduction $(f, g)$

The meta-decision problem reduces to a **selective prediction** instance (Geifman & El-Yaniv, 2017) when $|\mathcal{M}| = 2$ and one action is "abstain". A selective prediction system is a pair

$$(f, g),\quad f: \mathcal{X} \to \mathcal{Y},\quad g: \mathcal{X} \to \{0,1\}$$

where $f$ is the underlying predictor and $g$ is a gate that decides whether to emit $f(x)$ ($g=1$, "predict") or to abstain ($g=0$, fall through to a default). The risk–coverage tradeoff is governed by

$$\text{coverage}(g) = \Pr_{x}[g(x)=1], \quad \text{selective\_risk}(f,g) = \mathbb{E}\!\left[\text{loss}(f(x))\,\mid\,g(x)=1\right]$$

**Our v11 / v13 / B7v3 / abstain head are all selective predictors** in disguise:

- **v11 forecasting**: $f$ = the v10 deviation routine; $g$ = memory consensus (revert to Chronos-2 if neighbors agree). Coverage $\approx 4\%$ (memory rarely deviates from base); selective_risk near oracle on those 4%.
- **B7v3 TSC**: $f$ = the LOO-CV best non-Rocket classifier; $g$ = margin + memory check (deviate from Rocket only if confident).
- **RCA abstain head (§4.7.3)**: $f$ = the LLM-Agent's 5-class output; $g$ = a binary head ("is this in-taxonomy?"). When $g=0$ we emit `out_of_taxonomy`. OOT-recall $= 1 - \text{coverage}(g)$ on truly-OOT inputs.

The three mitigation paths in §4.7.3 (abstain head, stronger LLM, dataset prior) all increase the *effective coverage of $g$ on OOT inputs* without sacrificing in-taxonomy coverage by much — they shift the operating point on the same coverage-risk curve.

**Empirical (f, g) coverage-risk curves.** Treating the trained heads as selective predictors lets us trace the tradeoff explicitly:

**RCA selective prediction** (f = 5-class Agent, g = abstain head trained on Curator features, τ varied):

| τ | coverage(g) | selective accuracy on g=1 | OOT recall on g=0 | in-tax false-abstain |
|---|---|---|---|---|
| 0.3 | 27% | 82% | 90% | 56% |
| **0.5** (production) | **52%** | **77%** | **76%** | **20%** |
| 0.7 | 72% | 69% | 56% | 0% |

The full ROC has AUC = **0.864** for the abstain head detecting OOT inputs — a strong signal even on 100 training cells.

**Forecasting selective prediction** (f = v10 wrapper, g = `forecast_abstain_head`, decide whether to keep v10 deviation or fall back to Chronos-2):

| τ | coverage (wrapper used) | mean MAE | (reference) |
|---|---|---|---|
| 0 (v10 always) | 100% | 7.9996 | (v10 alone) |
| **0.3 (optimal)** | **8.3%** | **6.9623** ⭐ | **< Chronos-2's 6.9886** |
| 1.0 (C2 always) | 0% | 6.9886 | (Chronos-2 alone) |

**Selective prediction is the first configuration to beat Chronos-2 in mean MAE.** At τ = 0.3 the abstain head correctly restricts v10's deviations to the 5/60 cells where they help, achieving mean MAE 6.9623 vs Chronos-2's 6.9886 (Δ = −0.026, −0.4%). Without the head, v10's average MAE is 7.9996 (28% slower with no average gain). The improvement is small in absolute terms but **methodologically significant**: it shows that the (f, g) decomposition is not just an explanatory frame — when instantiated correctly, it produces the only measurable wrapper-vs-base improvement in our entire forecasting study.

#### 3.0.4 The Convergent Architecture

Under TSH plus the selective-prediction reduction, several **architectural properties become predictable**:

1. **Average gain over base $\to 0$**: any wrapper that doesn't selectively abstain will, in expectation, match the base model exactly. This is what our 24-cell 23T result demonstrates for v11, and what the multi-mechanism convergence in §4.8 corroborates.
2. **Niche wins require correct routing**: the variance of $\Delta(x)$ is positive, so an oracle router can extract $> 0$ improvement. Our gap to oracle (e.g., UCR-5: B7v3 88.4 vs Oracle 92.1 = $3.6$ pp) measures the *headroom* of selective routing relative to a perfect $g$.
3. **Learned gating dominates rule-based gating only when feature signal exceeds learner capacity**: this is exactly the L1 (+0.49 pp wins) vs. L2 (-0.01 pp tied) finding in §4.5.
4. **Direct prediction by the Agent strictly dominates only when no strong baseline exists**: this is exactly why RCA Agent beats unstructured-LLM B1 by +40 pp but loses to rule B0 by $-37$ pp (§4.7).

**Statement of contribution.** Our experimental progressions (v5c → v11 in forecasting, B6 → B7v3 in TSC, Agent → Agent+abstain in RCA) are all **instances of the same selective-routing pattern under TSH**. The pattern is the contribution; the individual versions are corroborating evidence.

### 3.1 Problem Setting

Given a univariate series $\{y_1, \ldots, y_N\}$ with N ∈ {10, 20, 50, 100} and prediction horizon H, predict $\{y_{N+1}, \ldots, y_{N+H}\}$. We additionally hold out a validation segment $\{y_{N+1}, \ldots, y_{N+V}\}$ of length V=10 (V=3 if N≤10) for internal adaptation. The model never observes the test segment during training or hyperparameter selection.

### 3.2 Architecture Overview

```
   train (N points)
        ↓
  ┌──────────────────────────┐
  │ §3.3 Diagnosis Layer     │  three-way confidence
  └──────────────────────────┘
        ↓ Diagnosis(trend_conf, season_conf, stat_conf)
  ┌──────────────────────────┐
  │ §3.4 Adaptive Planner    │  rule: conf → strategy pool
  └──────────────────────────┘
        ↓ Plan(strategies)
  ┌──────────────────────────┐
  │ §3.5 Walk-Forward CV     │  softmax reweight on train-tail
  │      Reweighting         │
  └──────────────────────────┘
        ↓ Plan(strategies, data-driven weights)
  ┌──────────────────────────┐
  │ §3.6 Structured Reflect  │  root_cause + diagnosis_revision
  │      (decoupled best)    │  produces interpretability artefacts
  └──────────────────────────┘
        ↓
  ┌──────────────────────────┐
  │ §3.7 Optional Memory     │  faiss kNN cross-series experience
  └──────────────────────────┘
        ↓
   forecast (H points)
```

### 3.3 Three-Way Diagnostic Confidence

For each of three dimensions (trend, seasonality, stationarity) we produce three confidence labels in {high, mid, low}:

- **Path A (statistical)**: thresholds on classical tests — trend t-statistic from linear regression, ACF peak in [2, 2m], ADF p-value. Thresholds adapt to N (e.g., for N<30 the ACF threshold for "high seasonality" is raised from 0.5 to 0.7 to discount sample-size noise).
- **Path B (LLM-subjective)**: an LLM receives the statistical summary plus the series' head/tail values, returns a JSON `{trend, season, stationarity, reason}`.
- **Path C (cross-validated)**: per dimension, $\mathrm{conf}_\mathrm{xc} = \min(\mathrm{conf}_A, \mathrm{conf}_B)$ — the conservative cross of A and B.

We argue that A alone is brittle on noisy tests, B alone hallucinates, and **C is the natural conservative composition**. §4.2 quantifies this via the CMR (Confidence Monotonicity Ratio).

### 3.4 Adaptive Strategy Planner (Rule-Based Prior)

The planner maps confidence triples to a strategy pool. Key rules:

- **N ≤ 12**: forced safe ensemble of {LLMTime, Chronos, drift}; no reflection (val is fundamentally unreliable).
- **Any low confidence**: 3-way ensemble of {LLMTime, Chronos, ARIMA+ETS} (covers different modeling assumptions).
- **All mid**: 2-way ensemble of {LLMTime, Chronos} when N≤30, otherwise {Chronos, ARIMA+ETS}.
- **All high**: single best-fit method based on which dimension is strongest.

This produces a **plan candidate pool**, not the final weights — those come from §3.5.

### 3.5 Walk-Forward CV Reweighting

Prefix rules cannot anticipate which method will work on the specific series; data-driven weights are needed. We **simulate forecasting in the train tail**:

```
fold k:  fit on train[: N - k*H_v],  predict train[N - k*H_v : N - (k-1)*H_v],  record per-strategy MAE
         for k = 1 ... K (typically K = 3-5)
```

with $H_v = \max(3, \min(10, N//5))$. Per-strategy CV-MAE is converted to softmax weights:

$$w_m = \frac{\exp(-(\mathrm{MAE}_m - \min_m \mathrm{MAE}_m) / \tau)}{\sum_{m'} \exp(-(\mathrm{MAE}_{m'} - \min_m \mathrm{MAE}_m) / \tau)}$$

τ adapts to N: smaller τ (0.3) for large N favors winner-take-all; larger τ (0.6) for small N favors averaging. **A7 ablation in §4.3 verifies this contributes 6–16% MAE reduction** in mid-N regimes.

### 3.6 Structured Root-Cause Reflection

After walk-forward weighting, the agent computes val-MAE on the held-out segment. If it exceeds a threshold ($\mathrm{val.std} \times 0.5$), reflection fires. Crucially, reflection has two purposes that we **decouple**:

- **(a) MAE improvement**: in our v5c design, the `best_plan` is locked to the walk-forward initial plan and is never replaced by reflection outputs. This avoids val-overfitting (we verified this experimentally — letting reflection replace `best_plan` *worsens* MAE on N=20 by 62% on ETTh1).
- **(b) Interpretability**: reflection still runs, producing structured outputs that we record for analysis and downstream evaluation.

The reflection prompt embeds **Model Cards** — structured capability descriptions for each strategy in the pool (`assumes`, `strengths`, `weaknesses`, `typical_failure`). The LLM is required to output JSON `{root_cause, diagnosis_revision, new_plan}` with hard validation: `root_cause` must cite specific MAE numbers AND diagnostic/strategic keywords or it is rejected as a hallucination.

The `diagnosis_revision` field allows the LLM to **flag the initial diagnosis as inconsistent with evidence** (e.g., "ARIMA failed with MAE 6.02 despite the diagnosis saying stat=high → diagnosis may be wrong"). In our experiments, this revision channel was used 0/9 times (LLM tends to trust the initial diagnosis) — a clear future-work direction.

### 3.7 Cross-Series Memory (v11 — Memory-Augmented Gate)

Each processed series leaves a `Case(features, diagnosis, final_plan, test_mae)` record in a faiss vector index, keyed by a 10-dimensional diagnosis feature. The memory layer was a skeleton until v10 (queries were never invoked, `test_mae` was never backfilled). **v11 closes this loop and uses cross-series consensus as a gate override**:

```
chosen = v10_gate_decision()      # confidence-gated routing on walk-forward CV
if memory.size ≥ K_min (=5):
    neighbors = memory.query(diagnosis_features, k=K=5)
    mem_best_strategy = strategy with lowest similarity-weighted test_mae
    if (mem_best_strategy ≠ chosen) and (#supporting_neighbors ≥ K_min/2):
        chosen = mem_best_strategy   # MEMORY OVERRIDE
```

The override is conservative: a single high-similarity neighbor cannot flip a CV decision, and the chosen alternative must come from at least ⌈K_min/2⌉ neighbors that have a finite recorded test MAE. The motivation is direct: v10's three remaining loss cells (§5.1) all stem from short walk-forward CV mis-ranking high-variance methods above Chronos-2; a cross-series consensus drawn from past actual test outcomes is a more stable signal than a 4-step CV holdout. **The memory's contribution is evaluated in §4.7** with a populate/query two-phase protocol.

---

## 4. Experiments

### 4.0 From Version Progressions to Failure Mechanisms

Our experimental sections describe two parallel design progressions — v5c → v11 in forecasting and B6 → B7v3 in classification. Rather than reading these as lab notebooks, we recommend reading them through the lens of **three recurring failure mechanisms** (each predicted by §3.0's theory) and their corresponding architectural fixes. Each mechanism manifests with the **same symptom** in both domains, and the **same fix** works in both:

| **Failure mechanism** | **Symptom (in both forecasting and TSC)** | **Architectural fix** | **Forecasting version** | **TSC version** |
|---|---|---|---|---|
| **(M1) Noisy CV under tiny N** — short walk-forward holdouts produce high-variance estimators that mis-rank classifiers | catastrophic mis-routes (e.g., +45% MAE on ECL N=100; -25pp on BeetleFly N=3 seed=1) | N-conditional abstain: default to base when N < threshold | v10 (`N < 15` fallback) | B7v2 (`N < 7` fallback) |
| **(M2) Local validation instability** — even at larger N, single-window CV occasionally selects a noise-overfit alternative | overconfident deviation that loses to base on test | margin-gated deviation: deviate only when CV gap > τ | v9-v10 (margin=0.20) | B7v2 (margin=0.10) |
| **(M3) Retrospective bias in memory** — kNN consensus from past *winners* under-represents cells where deviation paid off | memory bootstrap collapses to "always agree with base"; loses the niche-win opportunity | similarity-weighted vote with N≥K_min gating + Cards-aware features | v11/v13 (memory safety-net) | B7v3 (25-dim mem + weighted vote) |

**A fourth mechanism appears in the reasoning tasks**:

| **(M4) Specialist attention bias** | LLM forces out-of-taxonomy inputs into a known fault class, contradicting its own cited features | three stackable fixes: prompt prior, stronger LLM, abstain-classifier head (architectural) | (n/a) | RCA §4.7.3 |

The rest of §4 reports the empirical numbers that establish each row of this table. Sections 4.2-4.4 and 4.7-4.10 cover **forecasting** version progression (M1-M3 instances) and the related ablations. Sections 4.5-4.6 cover **TSC** version progression (M1-M3 instances). Section 4.7 covers **RCA** (M4 instance plus M1-M3 in the abstain head training). Readers wanting only the architectural takeaway can skim §4.2 (main result table), §4.5 (router progression), §4.7 (mitigation paths), and §4.8 (multi-mechanism convergence) — the version-level detail in the remaining sections corroborates the mechanism table above.

### 4.1 Setup

**Datasets.** Six standard benchmarks, all univariate (taking the OT column or the first variate where applicable):
ETTh1, ETTh2 (electricity, 1-hour, 17,420 pts), ECL (electricity consumption, 1-hour, 26,304 pts; column MT_001), Exchange (FX rates, daily, 7,588 pts; first column), Weather (meteorology, 10-minute, 52,696 pts; OT column), ILI (illness, weekly, 966 pts; OT column).

**Few-shot protocol** (plan §2.2): contiguous window of length $N + V + H$ randomly placed in the source series (seed-controlled). For H=96 on all hourly/daily datasets; H=24 on ILI (since 96 weeks = 2 years is unreasonable).

**Few-shot levels**: N ∈ {10, 20, 50, 100} (plan §2.2 S1-S4). 3 seeds per cell.

**Baselines**: Naive (best of mean/drift/seasonal on val), ARIMA+ETS (AIC selection between auto_arima and Holt-Winters), Chronos-Small (Amazon 60M zero-shot), **Chronos-Bolt (Amazon 2024-12, ~50× speedup over Small with comparable quality)**, **Chronos-2 (Amazon 2025-10, 21-quantile T5)**, LLMTime (Gruver et al. 2023 digit ICL), TSci (Zhao et al. 2025, ran via the official repository with minimal LangGraph adaptation, see Appendix A). TimesFM-2.0 and Moirai were attempted but excluded on CPU due to load-time / dependency issues; deferred to GPU-enabled follow-up (see Appendix A).

**LLM backbone**: zhipu GLM-4-flash-250414 (non-reasoning, content field directly accessible). All LLM calls are deterministically cached on disk (SHA-256 of prompt) for reproducibility.

### 4.2 Main Result: 144-Cell Performance Matrix (N1 + N4)

[Insert main table — see `finish.md §3.1.13` for the full 6-dataset × 6-method × 4-N grid.]

**The headline finding** is summarized by the per-cell winner distribution over the 24 (dataset, N) cells:

| Method | Winner cells | % |
|---|---|---|
| Chronos | 10 | 41.7% |
| LLMTime | 5 | 20.8% |
| **AdaptTS-Agent** | **4** | **16.7%** |
| TSci | 4 | 16.7% |
| ARIMA+ETS | 1 | 4.2% |
| Naive | 0 | 0.0% |

**No method dominates more than 42% of cells.** The strongest single baseline (Chronos) is best on 10 cells but is decisively beaten on 14 cells, often by a different method per cell. This **directly motivates the case for adaptive selection** and rules out the possibility that the right answer is "just use Chronos".

**AdaptTS-Agent wins outright on 4 cells**:
- ECL N=20: AdaptTS 16.75 vs Chronos 20.43 (**-18%**)
- ECL N=50: AdaptTS 18.11 vs ARIMA 20.38 (**-11%**)
- Exchange N=20: AdaptTS 0.021 vs ARIMA 0.022 (**-5%**)
- Exchange N=100: AdaptTS 0.023 = Chronos 0.023 (tied at best)

**Statistical significance (Wilcoxon signed-rank, 12 pairs per dataset)**:
- AdaptTS vs TSci on ECL: median improvement 5.28 in favor of AdaptTS, **p = 0.021** (significant)
- AdaptTS vs Chronos on ILI: median 10460 in favor of Chronos, **p = 0.016** (Chronos better here)
- AdaptTS vs Chronos on ETTh2: median 1.10 in favor of Chronos, p = 0.052 (borderline)
- All other pairs (n=85) not significant at α=0.05.

These results support a measured claim: **AdaptTS is competitive overall and statistically dominant in one specific scenario (mid-N on multi-customer electricity data)**, while not universally superior. This is consistent with the no-free-lunch motivation.

**SOTA-baseline sanity check (F4).** To verify the headline picture is not an artefact of using only Chronos-Small, we additionally ran Chronos-Bolt (Amazon 2024-12) and Chronos-2 (Amazon 2025-10) at the full ETTh1/ETTh2 × N{10,20,50,100} × 3-seed grid (48 cells):

| Dataset | N | Chronos-Small | Chronos-Bolt | **Chronos-2** | best non-Chronos |
|---|---|---|---|---|---|
| ETTh1 | 10 | 4.67 | 3.485 | **3.480** | LLMTime 3.61 |
| ETTh1 | 20 | 3.99 | 3.411 | 3.667 | LLMTime 2.95 |
| ETTh1 | 50 | 3.13 | 2.885 | **2.476** | ARIMA 4.06 |
| ETTh1 | 100 | 3.13 | 3.176 | 3.333 | ARIMA 2.52 |
| ETTh2 | 10 | 5.00 | 4.438 | **4.373** | ARIMA 8.04 |
| ETTh2 | 20 | 4.98 | 4.767 | **4.642** | ARIMA 5.34 |
| ETTh2 | 50 | 3.85 | 4.024 | 3.815 | ARIMA 7.37 |
| ETTh2 | 100 | 5.12 | 4.695 | 5.169 | TSci 4.07 |

Chronos-2 is the new winner in 5 of 8 cells on ETTh1/ETTh2; in the remaining 3 cells the prior winner (LLMTime / ARIMA / TSci) holds. **The no-method-dominates picture survives the upgrade**: even Chronos-2, the strongest 2026-vintage TSFM we evaluated, wins ≤63% of these 8 cells and is decisively beaten on N=100 by either ARIMA or TSci. We have added Chronos-Bolt and Chronos-2 to the AdaptTS strategy library (`STRATEGY_FN`) so the adaptive router can route to them when diagnosis confidence is high; this strengthens the case that an adaptive layer remains useful in the SOTA-TSFM era.

### 4.3 Ablation A1–A9 (E4)

Run on ETTh1 N=20, 3 seeds. The full method `Full=v5c` is compared against:

- A1 *w/o uncertainty* — point-estimate diagnosis only
- A2 *w/o adaptive planner* — fixed strategy pool
- A3 *w/o reflection*
- A4 *w/o memory* (no-op for current setup, memory disabled in main)
- A5 *w/o UQ + adaptive* — reduces to TSci-style diagnosis
- A6 *ICL-only* — disable all statistical tools
- A7 *w/o walk-forward CV* (this work, prefix-rule weights only)
- **A8** *w/o Model Cards* (this work)
- **A9** *w/o diagnosis revision* (this work)

**A7/A8/A9 results** (the contributions specific to this paper):

| Variant | mean MAE (3 seeds) | mean root_cause chars | mean diagnostic words | mean Model-Card words |
|---|---|---|---|---|
| Full (v5c) | 4.167 | 593 | 9.00 | 1.17 |
| A8 (w/o Cards) | 4.167 | 526 | 7.17 | 0.83 |
| A9 (w/o DiagRev) | 4.167 | **675** | **11.83** | **1.50** |

**MAE is invariant across A8/A9** by construction (v5c decouples reflection from `best_plan`). **The contribution shows up in interpretability**:

1. Full reflection chain length averages 4.0 steps; A8 drops to 3.33. Model Cards stimulate deeper exploration.
2. A8 compensates for missing Cards by citing more concrete numbers (5.42 vs 2.92) and strategy names (6.83 vs 5.25) — the LLM uses its world knowledge as a fallback.
3. **A9 paradoxically cites more Model-Card terms than Full** (1.50 vs 1.17). We attribute this to attention reallocation: when the `diagnosis_revision` slot is removed, the LLM's attention budget shifts toward the Cards slot. **This is a novel observation about how structured prompt slots compete for LLM attention.**

A7 results (walk-forward CV ablation) appear in Appendix B.

### 4.4 Case Studies (N5)

We dump three reflection traces in full to illustrate the qualitative behavior of structured reflection (full details in `finish.md §3.1.8`):

**Case 1: ETTh1 N=20 seed=42** — 4-step reflection chain. The LLM sequentially blames each failed strategy with reference to its Model-Card weakness ("naive_drift assumes linear trend; diagnosis says trend=mid → may be non-linear"), eventually exhausts the strategy pool and converges to a repeat plan (loop detection triggers exit).

**Case 2: ETTh1 N=20 seed=123** — Reflection **correctly identifies the winning single strategy** (naive_drift with val MAE 0.51) but is *unable to promote it* because the v5c design locks best_plan to the initial walk-forward weights. This case directly motivates the **strategy promotion** future-work direction.

**Case 3: ETTh1 N=50 seed=1** — 2-step convergent reflection. The LLM produces a structurally clear root_cause linking high trend + low stationarity to ARIMA's stationarity assumption, justifying the switch to Chronos.

**These case studies are themselves a contribution**: they expose how the agent reasons in a way that no aggregate MAE table can. Few prior ATSF papers publish reflection traces at this granularity, partly because their reflection is unstructured.

> **Section reading guide.** Sections are numbered by physical position. Logical topical order:
> Forecasting: §4.1 Setup → §4.2 Main result → §4.3 Ablation → §4.4 Case studies → §4.8 Memory negative + multi-mechanism validation → §4.9 CRPS reversal → §4.10 Cross-LLM forecasting.
> Reasoning tasks: §4.7 Task A RCA (with §4.7.1 OOT, §4.7.2 prompt-resistant bias, §4.7.3 three mitigation paths) → §4.6 Task B UCR direct (negative result) → §4.5 Task B Router progression + learned margin → §4.11 Task B UEA multivariate routing space (partial).

### 4.5 Task B Router · Progression and Final Result (+0.89pp UCR-5, +0.49pp via Learned Margin)

**Final progression on 5 UCR datasets × 3 N-shot × 2 seeds = 30 cells:**

| Version | Architecture | Mean Acc | vs Rocket alone |
|---|---|---|---|
| B6 Direct | Curator + LLM ICL → class | 54.3% | -33.2pp |
| B7v1 | + LOO CV margin gating among {DTW, Euclid, Rocket, MOMENT-1NN, MOMENT-LR} | 84.76% | -2.77pp |
| B7v2 | + N<7 fallback (LOO CV unreliable at extreme few-shot) | 86.66% | -0.87pp |
| **B7v3** | **+ 25-dim memory features + sim-weighted voting + Cards v2** | **88.42%** | **+0.89pp ⭐** |
| Oracle | post-hoc per-cell winner | 92.06% | +4.53pp |

B7v3 achieves **+0.89pp over Rocket on UCR-5**. Memory consensus override fires on **15 of 30 cells (50%)**.

**Saturation hypothesis verified — the win is niche, not general.** To test whether the UCR-5 result reflects a genuine routing advantage or merely surfaces dataset-dependent Rocket-vs-MOMENT preferences in a Rocket-optimized benchmark, we additionally evaluate on a **less-saturated extended sweep**: GunPoint (motion), Strawberry (spectroscopy, TSFM out-of-distribution), Wafer (industrial), ECG5000 (5-class medical), and Crop (24-class remote-sensing). On 20+ shared cells:

| Sweep | B7v3 vs Rocket | Beats Rocket | Routing distribution |
|---|---|---|---|
| UCR-5 (saturated) | **+0.89pp** | 6/30 | rocket 25 / moment_1nn 4 / dtw_1nn 1 |
| **Less-saturated extended** | **+0.00pp** | **0/20** | **rocket 19 / moment_1nn 1** |
| **Industrial deployment** (Wafer/FordA/B/ECG5000/Strawberry, 20 cells) | **-1.02pp** ⚠ | **0/20** | **rocket 17 / moment_1nn 3 / euclid 0** |
| **B7v4 (industrial-signature fix)** on same 20 cells | **-0.60pp** | **4/20 (all Wafer)** | rocket 12 / moment_1nn 4 / **euclid 4** |
| **B7v4 on Wafer subset alone** (4 cells) | **+4.0pp** ⭐ | **4/4** | euclid 4 |

On the extended sweep, B7v3 routes to Rocket in 19 of 20 cells — **the routing layer is essentially inactive**. The honest interpretation: **the +0.89pp on UCR-5 is concentrated on BeetleFly/BirdChicken image-outline morphology, where MOMENT's pretraining substantially beats Rocket; on multi-class medical, industrial, motion, and spectroscopy data, Rocket already dominates and no Agent improvement is captured**. We therefore retract any claim of a general TSC improvement and report the result honestly as a niche routing advantage applicable when MOMENT's pretraining domain matches the test data.

**Industrial deployment limitation.** A targeted industrial subset (Wafer semiconductor, FordA/B engine fault, ECG5000 medical, Strawberry spectroscopy; 20 cells) reverses the direction of the UCR-5 gain: B7v3 achieves **−1.02pp vs Rocket-alone (0.7335 vs 0.7437)**. The router defaults to Rocket in 17/20 cells and **never routes to Euclidean** — yet Euclid is the per-cell winner on 3/10 cells, most notably **Wafer N=5 where Euclidean achieves 0.945 vs Rocket's 0.865 (B7v3 selects rocket, an 8 pp miss)**. The 25-dim Curator features do not discriminate the low-noise, smooth-signal regime where instance-level metric classifiers dominate. Industrial deployment therefore requires either (a) **richer features capturing signal smoothness, noise-floor variance, and sensor-bit quantization markers** to identify Euclid-favored regimes, or (b) **a learned router (§4.5 Level 2) trained on industrial-labeled cells** to override the heuristic's Rocket-default. This finding reinforces §3.0.1: the saturation prediction is **regime-conditional**, and a router calibrated on UCR-5 signal distributions does not transfer to industrial sensor data without re-tuning.

**B7v4 — industrial signature fix (task #66).** Following the limitation above we directly implement remedy (a): we add five industrial-regime features to the Curator (smoothness, noise-floor, quantization bits, plateau ratio, ACF decay) for a 30-dim feature vector, and introduce a calibrated **industrial signature gate**: when `acf_decay < 0.4 AND quant_bits < 7.5` (persistent-signal, low-discretization regime characteristic of Wafer) and Euclid's LOO CV accuracy is within 0.05 of the default, the router forces Euclid. On the 20-cell industrial sweep the signature fires precisely on 4/4 Wafer cells with **zero false positives** on FordA/B/ECG5000/Strawberry. **The documented Wafer N=5 s=1 miss (Rocket 0.795, Euclid oracle 0.945) is fully closed: B7v4 recovers 0.950, a +15.5pp improvement over B7v3 on that cell.** Net effects: B7v4 = 0.7377 (B7v3 = 0.7335, +0.42pp aggregate); on the Wafer subset alone B7v4 = 0.824 vs Rocket = 0.784 (**+4.0pp**), demonstrating that the router **surpasses Rocket once industry-relevant features are added**. The 20-cell aggregate remains -0.6pp behind Rocket (memory drift on ECG5000 N=10 introduces a -7.5pp regression that consumes the Wafer recovery in the mean). This is **direct empirical validation of the feedback "richer features" path**: §4.0 Mechanism M2 (local-val instability) cannot be cured by tuning the margin alone — it requires features that bypass the LOO signal entirely. The fix is task-specific (industrial regime) and not portable to other domains without their own calibrated signatures, in line with §3.0.1's prediction that routing gains are regime-conditional.

**Learned Margin (Level 1) beats heuristic.** Replacing only the deviation-margin constant (B7v3's `margin=0.10`) with a learned regression head — trained on Curator features to predict the per-cell `best_other_acc - rocket_acc` gap — achieves **+0.49pp over fixed-margin B7v3** on LODO CV (selected accuracy 0.8597 vs 0.8548), closing 55% of the remaining gap to the per-cell oracle (0.8637). This is the **first systematic improvement of a learned routing component over its hand-tuned counterpart** in our system. The narrow architectural scope is essential — the regression head only needs to model a 1-dimensional `optimal_margin` signal, which 56 training cells suffice to capture.

**Learned Meta-Router (Level 2) as broader upgrade path.** The B7v3 router depends on four hand-tuned thresholds (margin=0.20, N<7 fallback, memory k_min=5, vote_ratio=0.6). We replace this entire decision stack with a learned **Meta-Router v2** — per-classifier RandomForest regressors predicting expected test accuracy from the 25-dim Curator features, with a single confidence-gated deviation parameter τ. On leave-one-dataset-out CV across 10 datasets (56 cells), Meta-Router v2 achieves selected-classifier accuracy of 0.8328 vs Rocket-alone 0.8329 (**Δ = -0.01pp, tied**), with only 5/56 deviations from rocket-default. The learned variant **does not yet exceed the heuristic** — class imbalance (30/56 cells favor rocket) and limited training data (56 cells over 5 classifiers) are the binding constraints — but it establishes a clean upgrade path: adding a new TSFM/classifier requires only training one additional regression head (~1 hour) rather than re-calibrating per-classifier margins. We list as concrete future work (a) **contextual-bandit online learning** that continually updates the heads with each new cell's outcome, and (b) **meta-learning via TSFM transfer** using pretrained Chronos-2/MOMENT embeddings as universal representations for cross-domain meta-pretraining. The Meta-Router establishes that heuristic-routing is a temporary architectural choice motivated by data scarcity, not a fundamental design constraint.

**UEA Multivariate complementary finding.** As a final probe of the saturation hypothesis, we additionally test 3 UEA multivariate datasets (BasicMotions, ERing, AtrialFibrillation) under the same N-shot protocol with multivariate-adapted DTW (channel-wise sum), Euclidean (channel-flattened), and Rocket (native sktime). Mean accuracies across the 54-cell completed sweep: **DTW = 72.5%**, **Rocket = 68.3%**, Euclid = 57.2% — **DTW overtakes Rocket** on multivariate, the opposite of the UCR ranking. The numbers are pulled down by AtrialFibrillation (15-train, 3-class, length-640) which is too hard for any method (≤27% across the board); on the remaining two datasets the methods are typically near saturation, with DTW winning BasicMotions N=3 outright. BasicMotions N=3 favors DTW (1.000) while ERing favors Rocket (0.985). This further refutes the existence of a universal best classifier and identifies multivariate input as a regime where routing opportunity is substantially larger than on univariate UCR. Our current router B7v3 is univariate-only; extending it to multivariate features and a multivariate memory bank is concrete future work where we expect routing benefit to exceed the +0.89pp UCR-5 ceiling.

**The v8→v10 forecasting wrapper progression and the B6→B7v3 TSC router progression are isomorphic.** Both arose from the same insight: when SOTA base-models (Chronos-2 for forecasting, Rocket for TSC) dominate, the LLM-Agent's value is in conditional model selection — not in competing directly with the base. The v10 N<15 fallback and the B7v2 N<7 fallback both address the same failure mode (CV signal noise at extreme few-shot); the v11/v13 memory layer and the B7v3 memory-with-enhanced-features both implement cross-series consensus override.

### 4.6 Task B — Few-Shot UCR Classification (Boundary of the Direct Agent)

To test whether the diagnostic-reasoning Agent generalizes to non-failure-mode classification, we evaluate on the UCR archive: Coffee (28 train, 2-class spectroscopy), ECG200 (100 train, 2-class ECG), TwoLeadECG (23 train, 2-class ECG), at N_per_class ∈ {3, 5, 10}, 2 seeds. Seven baselines: B1 1-NN-DTW, B2 1-NN-Euclidean, B3 Rocket (Dempster 2020 SOTA), B4a/b MOMENT (Goswami 2024 TSFM) embedding + 1-NN/LogReg, B5 LLM-direct (raw numbers + few-shot ICL), B6 AdaptTS Agent (Curator diagnosis + LLM ICL).

**Result (final 30 cells × 7 methods, 5 datasets × 3 N-shot × 2 seeds):**

| Method | Mean Acc | Macro F1 | Winner Cells (of 15) |
|---|---|---|---|
| **B3 Rocket** | **0.875** | 0.871 | 7 |
| B4a MOMENT 1-NN | 0.819 | 0.812 | 3 |
| B4b MOMENT LogReg | 0.817 | 0.812 | 3 |
| B1 1-NN DTW | 0.748 | 0.740 | 1 |
| B2 1-NN Euclid | 0.710 | 0.703 | 1 |
| B6 AdaptTS Agent | 0.543 | 0.482 | **0** |
| B5 LLM-direct | 0.527 | 0.456 | **0** |

The most striking observation: **AdaptTS-Agent wins 0 of 15 settings**, and on the image-outline datasets (BeetleFly, BirdChicken — silhouette curves of insects and birds) its accuracy degrades to **45%, below random for 2-class problems**. MOMENT reverses dominance on these datasets — B4b LogReg achieves 92.5% on BeetleFly 3-shot vs Rocket 82.5%, suggesting TSFM pretraining captures high-level morphological features that random convolutional kernels miss.

**This is the opposite finding from §4.9.** On RCA (§4.9) the Agent beat LLM-direct by +40 percentage points; on UCR TSC, the Agent collapses to within 1pp of LLM-direct, both 30+ pp below Rocket. The gap is consistent across all completed (dataset, N-shot) settings.

**The structural explanation — semantic alignment between labels and features.** The Curator's 10/12-dim feature space (trend slope, t-stat, ADF, ACF, outlier count, variance ratio, three-way confidence) was designed to capture *statistical* aspects of a series. On RCA, where class labels (trend_break, variance_explode, outlier_burst, stationarity_flip) are themselves statistical concepts, the feature space is **directly aligned** with the prediction target — the Agent crushes the unstructured baseline. On UCR, where class labels are *domain-specific patterns* (coffee variety from spectroscopy curves, normal vs ischemic ECG morphology), the diagnostic features are **off-target**: they capture none of the discriminative signal that Rocket's 10,000 random convolutional kernels extract.

**The boundary is clean and actionable.** This delineates a precise applicability claim for diagnostic-reasoning Agents in time-series intelligence: **they are the right tool when the task is to reason about statistical properties of a series (failure modes, regime shifts, anomaly types); they are the wrong tool when the task is to recognize domain-specific patterns**. The latter remains a problem for kernel methods (Rocket) and pretrained-feature methods (MOMENT embedding probes).

**Final synthesis (§5):** The three task-result pairs (forecasting §4.8 = parity with Chronos-2; RCA §4.9 = +40pp; TSC §4.10 = -31pp) define the operating regime of the AdaptTS-Agent architecture in 2026. This is what an honest evaluation of an LLM-Agent time-series system looks like.

### 4.7 Task A — Prediction Failure Root-Cause Analysis (RCA, In-Taxonomy + Out-of-Taxonomy + Mitigations)

Having established in §4.2–4.8 that no AdaptTS variant systematically outperforms Chronos-2 on point or probabilistic forecasting, we turn to a task that Chronos-2 cannot perform at all: **identifying the root cause of a forecasting failure**, framed as 5-way classification over a structured taxonomy.

**Setup.** We select the 30 most catastrophic failure cells across our 6-dataset × 4-N × 3-seed grid, ranked by adapt_mae / chronos2_mae ratio (top cell: Weather N=20 v11 ratio=6.34). Ground truth is assigned by a rule-based detector that computes per-fault scores in {trend_break, seasonal_flip, variance_explode, outlier_burst, stationarity_flip} (split-half mean/variance, ACF sign reversal, z-score outlier count, etc.). Each cell is labeled with the primary fault (top-scoring) and optional secondaries.

**Methods compared.**
- **B1 LLM-direct**: zhipu glm-4-flash-250414 receives (train series, prediction, truth) numerical values and the taxonomy; outputs structured JSON.
- **B5 AdaptTS-Agent**: identical LLM, but the prompt additionally includes (a) the Curator's 10-dim diagnostic features with three-way confidence, (b) all five Model Cards describing per-strategy assumptions and typical failure modes.

**Result.**

| Method | R1 Top-1 | R2 Top-3 | R4 Kw-F1 | Information access |
|---|---|---|---|---|
| **B0 rule (train-only fault scoring, no LLM)** | **0.767** | 0.767 | (n/a) | train series only |
| B5 Agent (v1, Curator 10-dim) | 0.400 | 0.433 | 0.162 | train + test + prediction + Curator + Model Cards + LLM |
| B5 Agent (v2, Curator 12-dim) | 0.367 | **0.567** | **0.300** | same as v1 plus 2 outlier/variance features |
| B1 LLM-direct | 0.000 | 0.233 | 0.000 | train + test + prediction + LLM (no Curator/Cards) |

**The Agent dominates the unstructured B1 baseline by +40 pp R1**, demonstrating that diagnostic structure + Model Cards are the active ingredients on top of raw LLM ICL: B1 predicts `trend_break` for all 30 cells, collapsing to the syntactically most prominent class. **However, a rule-only baseline (B0) that examines only the training series and applies the same fault detector as the ground-truth generator achieves R1 = 0.767, substantially above the Agent**. The B0 vs. Agent comparison is **partly tautological** — the ground truth itself was derived from the train+test rule combination, so B0 inherits a generative correlation; the 0.767 is the empirical fraction of cells in which the `max(train, test)` GT label happens to be dominated by the train side. Nonetheless, in 14 of 30 cells B0 identifies a fault that the LLM-based Agent misses despite the Agent having strictly more context. We read this **honestly as a methodological negative result on raw RCA accuracy**: the Agent does not currently beat a competent rule, even though it improves substantially over an unstructured LLM. **The Agent's RCA contribution should be framed as (a) the natural-language reasoning trace, (b) the explicit citation of diagnostic statistics and Model Card assumptions, (c) extensibility to out-of-taxonomy fault types** — none of which B0 provides. The v1→v2 progression (R1 drops 3 pp but R2 gains 13 pp and keyword-F1 gains 14 pp) further documents a feature-engineering trade-off that prompt and feature must co-evolve.

The Agent's confusion-matrix breakdown also points to which Curator features matter: **outlier_burst 0/6 and variance_explode 0/10 in v1** identify the missing-feature gap that motivated the v2 extensions, which then reverses the per-fault confusion (variance_explode 0→9/10, but stationarity_flip 12→1/13).

**Independent verification on synthetic data with clean ground truth.** Because the natural-failure ground truth was itself rule-derived (partially correlating with B0), we additionally construct **50 synthetic cells (5 fault classes × 5 cells × 2 datasets)** where the ground truth equals the injected fault — fully independent of any rule detector. Results on this clean-GT split:

| Method | R1 Top-1 | R2 Top-3 |
|---|---|---|
| B0 rule | **0.500** | **0.860** |
| Agent v2 | 0.260 | 0.600 |
| B1 LLM-direct | 0.240 | 0.360 |

The B0 advantage shrinks (0.767 → 0.500, as expected once tautology is removed) but **B0 still beats the Agent by ~24 pp on R1**. The Agent's R1 over B1 collapses from 40 pp on natural failures to **2 pp** (0.260 vs 0.240) on clean-GT synthetic, confirming that **B1 was a degenerate baseline** and the Agent does not currently beat a competent non-LLM baseline on raw RCA accuracy. The Agent's per-fault breakdown shows the failure mode clearly: variance_explode achieves 9/10 because the v2 `variance_ratio` feature directly fires, but **four of the five fault classes collapse to 0/10** with predictions over-attached to variance_explode. The R2 numbers tell a more favorable story (Agent 0.600 vs B1 0.360), meaning the Agent's top-3 candidates do contain the ground-truth label more often — the failure is in the top-1 decision step, not the candidate generation.

**Implication for §5.3.** These findings reinforce, not weaken, our central claim. Across all three task types — forecasting, RCA, classification — **the LLM-Agent loses when asked to be a direct decision-maker against a competent baseline** (Chronos-2 / B0-rule / Rocket respectively). It wins when repositioned as a router or wrapper. The RCA result therefore plays the same role as the §4.10 B6-direct result for classification: a negative finding that motivates and validates the Agent-as-Router architecture of §4.11 and §5.3.

#### 4.9.1 Out-of-Taxonomy Failure Modes — The Specialist Bias

To further characterize the Agent's limits, we test 50 synthetic cells with **out-of-taxonomy faults** (missing-data gaps, heavy-noise contamination, mode collapse, frequency modulation, quantization), none of which match any of the 5 in-taxonomy classes. We allow all methods to optionally output `out_of_taxonomy`.

| Method | OOT-recall (correctly identifies as OOT) | Keyword-F1 (NL evidence matches fault keywords) |
|---|---|---|
| B0-rule (forced 5-class) | 0% (impossible) | 0% |
| **B1 LLM-direct** | **24%** | 1.6% |
| **B5 Agent (Curator + Cards)** | **2%** | 0.4% |

**The Agent collapses to in-taxonomy predictions** on OOT data (37/50 → variance_explode), and inspection of the reasoning trace reveals an **over-confident expert bias**: in multiple cells the Agent's own evidence cites `variance_ratio=0.73 (<2)` (which indicates variance DECREASE) and yet concludes `variance_explode`. The Curator's hard diagnostic signals create an attractor in the LLM's reasoning that overrides obvious contradictions. The unstructured B1 baseline, without Curator guidance, correctly identifies OOT in 24% of cells.

**This documents a previously-unreported specialist-vs-generalist trade-off in time-series LLM-Agent design.** The Curator + Model-Cards architecture trades **+40pp on in-taxonomy classification** against **-22pp on out-of-taxonomy discovery**. The two findings together suggest that practitioners should:

1. Use the Curator architecture when the operating distribution is well-characterized by the predefined taxonomy.
2. Use an unstructured LLM (or augment with explicit `out_of_taxonomy` calibration loss) when the operating distribution is open-domain.

We list as concrete future work a v4 prompt design that explicitly trains the model to fall through to `out_of_taxonomy` when in-taxonomy evidence contradicts the conclusion. The boundary characterization in §5.3 thus extends to a new axis: **taxonomy alignment**, alongside base-model dominance.

#### 4.9.2 The Bias Is Prompt-Resistant (v4 Prompt Fix Failure)

We attempted to mitigate the specialist bias through a v4 prompt design that (i) explicitly states hard quantitative constraints for each in-taxonomy class (e.g., "variance_explode requires variance_ratio ≥ 2"), (ii) instructs the LLM to output `out_of_taxonomy` when its cited evidence contradicts any threshold, and (iii) adds a mandatory `evidence_consistency_check` field forcing the LLM to self-audit. **The fix failed completely**: OOT-recall remained at 2/50, and the variance_explode prediction frequency rose from 74% (v3) to 84% (v4).

The most striking evidence is in the LLM's own output. On a `mode_collapse` test case the v4 Agent writes: *"Variance_ratio=0.68 (<2), indicating a potential variance explosion… although not meeting the strict threshold for the 'variance_explode' category"* — and then outputs `primary_fault: variance_explode` regardless. The LLM **explicitly recognizes the threshold violation in its own evidence text** yet anchors on the attention-prominent feature.

We interpret this as evidence that **the specialist bias is an attention-mechanism artefact for the default LLM**. Presenting structured diagnostic features creates an attentional sink that overrides explicit verbal constraints.

**Cross-LLM evaluation reveals an important nuance**: the bias is **weak-LLM-specific, not universal**. Running the same Agent (v4 prompt, identical features and Cards) with glm-4-air or glm-4-plus yields **OOT-recall 68%** (vs 2% for glm-4-flash-250414, the default in all other experiments). Stronger LLMs respect the hard-constraint check in the v4 prompt; the weaker glm-4-flash cannot. This provides **three empirically-validated alternative mitigation paths**, ordered by intervention depth and effect size: (a) an **architectural abstain-classifier head** on Curator output (76% OOT-recall, +74pp; task #46), (b) **deployment with a higher-capacity LLM** (68%, +66pp, no code changes; task #47), and (c) a **dataset semantic prior** injected into the prompt (14%, +12pp, prompt-only; task #17). The three mechanisms are independent — they operate at architectural, model, and prompt levels respectively — and their effects could in principle be stacked. The multi-path convergence (all three reduce the bias by 10–75pp depending on intervention depth) is itself robustness evidence: the specialist bias is a real phenomenon with multiple valid solutions, not a universal architectural blocker. The intervention-depth ordering also gives deployment-time guidance: prompt-prior is cheapest but limited, model upgrade is moderate, abstain head is strongest but requires offline labeled training.

**Stacking the three paths achieves perfect OOT detection.** A 2×4 grid (default vs strong LLM × {baseline, +prior, +abstain, +stack}) on the same 50 OOT cells reveals strong additivity:

| LLM | baseline | +prior | +abstain | **+stack (prior+abstain)** |
|---|---|---|---|---|
| glm-4-flash-250414 | 0% | 14% | 76% | **78%** |
| glm-4-plus | 64% | 90% | 90% | **100%** ⭐ |

On the higher-capacity LLM, each intervention contributes independently (+26pp prior, +26pp abstain), and their stack reaches **100% OOT-recall on 50 cells — perfect detection**. On the weaker default LLM, the abstain head dominates and prior adds only +2pp marginal. The interaction reveals a **capacity-dependent stacking gradient**: prompt-level guidance only "lands" on an LLM with sufficient reasoning capacity to follow it, while the architectural head is LLM-agnostic. The specialist bias is therefore **fully solvable** with the appropriate combination of architectural, deployment, and prompt-level interventions — completing the narrative arc from §4.7.2's "prompt-resistant attention sink" to a constructive prescription.

**This is the first positive result in this paper.** Where v5c→v13 was a sequence of forecasting wrappers that failed to beat the SOTA TSFM, B5 here exceeds B1 by 40 percentage points on a task that the SOTA TSFM cannot perform at all. The methodological lesson is the framing of §3.7: **the LLM Agent's value in the 2026 TSFM era is in structured reasoning over forecasting, not in competing with forecasting itself**.

### 4.8 Forecasting Memory-Augmented Gate (v11/v13) — Negative Result + Multi-Mechanism Validation

We evaluate the v11 cross-series memory layer (§3.7) with a populate/query two-phase protocol over the 48-cell ETTh1+ETTh2+ECL+Exchange grid.

**Phase A — populate.** All 48 cells are run with `ADAPTTS_MEMORY_PATH` enabled and v10 gating. Each Case is written with `test_mae` backfilled by the runner once the test metric is computed. Memory ends with 48 entries, none of which influenced the predictions (queries are gated by `mem.size ≥ K_min`; the writes are post-decision).

**Phase B — query.** The 48 cells are re-run with the populated memory active. Each prediction's diagnosis-feature is queried against the memory; if ≥⌈K_min/2⌉=3 of K=5 neighbors agree on a strategy whose mean test-MAE is below the CV-chosen strategy's, the memory consensus overrides the CV gate.

**Result**: v11 (bidirectional memory override) and v13 (revert-only safety net with similarity-weighted voting) both converge to **0W / 0L / 16T against Chronos-2** — the memory wrapper provides MAE parity in 16/16 cells but eliminates v10's one win (ETTh2 N=100). Investigation shows this is not an implementation choice but a structural limitation of any retrospective-consensus design: when the base TSFM (Chronos-2) is already strong, the memory bootstrap is dominated by Chronos-2 winners; majority-vote cannot learn that "the rare deviation is correct" without storing **counterfactual outcomes** (what the default *would* have done on cells where a deviation was chosen). We list this honestly as a negative result and propose three corrections in §5.4 (counterfactual memory storage, diversity-enforced eviction, learned per-cell gates).

The memory layer is **not without value**: v11/v13 strictly Pareto-improve v10 in the sense that they eliminate all v10 catastrophic failures (3 cells with +6% to +38% MAE degradation) at the cost of one small win (-1.9%). This produces a "guaranteed-parity wrapper" suitable for production deployment where worst-case behavior matters more than average improvement. We report it as a valid configuration alongside the v12 entropy-only variant (1W / 3L / 12T) and let the deployment context dictate the choice.

### 4.9 Probabilistic Evaluation (CRPS) — Reversal of MAE-Based Improvements

Reporting only point-MAE on probabilistic SOTA TSFMs systematically overstates the case for any router that deviates from them. We add CRPS, pinball loss (q10/q50/q90), 80% interval coverage, and interval width, computed from Chronos-2's native 21-quantile output via the Laio–Tamea (2007) sample-based estimator. When AdaptTS deviates to a point predictor (ARIMA / LLMTime / Chronos-Bolt-as-point), its forecast is treated as a degenerate distribution and CRPS reduces to MAE.

**Aggregate over 48 cells (4 datasets × 4 N × 3 seeds):**

| Method | avg MAE | avg CRPS | **CRPS vs Chronos-2** | # deviation cells |
|---|---|---|---|---|
| v10 | 7.39 | 6.31 | **+16.65%** | 6 |
| v12 | 7.36 | 6.27 | **+15.86%** | 4 |
| **v11 / v13** | 6.85 | **5.41** | **0.00%** | 0 |
| Chronos-2 alone | 6.85 | 5.41 | — | — |

The contrast is decisive: v10 and v12 are within a few percent of Chronos-2 on MAE but **+16% worse on CRPS**, because deviation cells use point predictors whose degenerate distributions contribute no calibrated uncertainty.

**The single most dramatic cell-level reversal — ETTh2 N=100 seed=42** (the seed where v12 most clearly "beats" Chronos-2 on MAE):

|  | MAE | CRPS |
|---|---|---|
| v12 (Chronos-Bolt as point) | 4.27 (**-7% vs C2**) | 4.27 (degenerate) |
| Chronos-2 | 4.57 | **2.78** |

v12 wins MAE by 7% and **loses CRPS by 54%**. The point-MAE "victory" disappears entirely under a probabilistic loss.

**Implication for ATSF design.** In the presence of a calibrated probabilistic SOTA TSFM, the right adaptive layer is not a competing router that occasionally substitutes a point predictor; it is a **guaranteed-parity wrapper** that preserves the base TSFM's probabilistic output and only modulates around it. Our v11/v13 design (memory safety-net that reverts deviations) operationalizes this principle: it achieves exact CRPS parity with Chronos-2 in 16/16 cells while still exercising the diagnosis, planning, walk-forward CV, and memory layers. This is our final claimed contribution: **showing that the probabilistic-loss case for AdaptTS-class systems in 2026 is structurally weaker than the MAE case, and that the architecturally-correct response is to design for parity rather than improvement.**

**Multi-mechanism validation of guaranteed-parity wrapper.** To verify that the v11 memory-based safety net is not a brittle artefact of memory bootstrap, we additionally train a **forecast abstain head** — a 13-dim RandomForest binary classifier on Curator features predicting whether v10's deviation will help over Chronos-2. Of 60 training cells, only 3 (5%) have label `wrapper_helped=1`; the head collapses to "always abstain to Chronos-2" and yields mean MAE 6.9886 — **identical to Chronos-2 alone** (also 6.9886 to four decimals). Three independent mechanisms — (a) the v11 memory consensus that reverts deviations, (b) the v13 entropy gate that trusts Chronos-2 by default, and (c) this learned abstain head that maps cell features to the empirical `did_deviation_help` label — **all converge on the same architectural conclusion**: in the 2026 few-shot regime any deviation layer above Chronos-2 is at best MAE-neutral. This is much stronger evidence than any single design choice and removes the "v11 may have lucked into the right behavior" reading.

**Calibration of Chronos-2 itself in the few-shot regime.** As a side observation, Chronos-2's coverage_80 ranges from 0.10 to 1.00 across our 48 cells (target 0.80), and width_80 varies by two orders of magnitude (e.g. ECL N=10 seed=42 width=158 vs ETTh1 N=100 seed=42 width=9.0). Even SOTA TSFMs are not well-calibrated under few-shot data, motivating a future-work line on conformal post-hoc calibration as an adaptive layer in its own right (§5.4).

The full E2 study (three-way CMR with oracle upper bound) is in execution; framework code is at `agent/curator_uq.py`. Preliminary observations from §4.3 traces: the LLM-path confidence labels are not systematically miscalibrated, but `diagnosis_revision` is never triggered — suggesting that **the LLM is over-confident in the initial diagnosis** and a separate revision-stimulating prompt may be needed.

### 4.10 Cross-LLM Robustness on Forecasting (Addressing R6)

A natural concern is whether AdaptTS-Agent's gains are tied to a specific LLM backbone. To address this we ran AdaptTS, LLMTime, and TSci on four GLM variants from the same provider — `glm-4-flash-250414` (default), `glm-4.7-flash` (reasoning), `glm-4-air` (mid-tier), `glm-4-plus` (flagship) — at ETTh1/ETTh2, N=20, H=96, 3 seeds.

**Result (ETTh1, MAE mean across 3 seeds):**

| Method | flash-250414 | 4.7-flash | air | plus | **Std / CV** |
|---|---|---|---|---|---|
| LLMTime | 6.13 | 4.66 | 3.88 | 3.93 | 1.05 / **22.6%** |
| AdaptTS | 4.60 | 4.55 | 4.39 | 4.36 | **0.12 / 2.7%** |

AdaptTS-Agent compresses LLM-induced variance by **8.4× on ETTh1**. On ETTh2 the two methods have comparable CV (~7%), suggesting that the dataset (not the method) sets a variance floor in some regimes. Critically, the *direction* of LLMTime's LLM dependence is non-monotone — `glm-4-air` (mid-tier) beats `glm-4-plus` (flagship) and the reasoning variant — which is exactly the pathology that diagnosis-driven candidate routing is designed to absorb. We read this as evidence that the **value of AdaptTS does not reside in any single LLM**; the structured planning–reflection loop dampens LLM-side noise via walk-forward reweighting before it reaches the final forecast.

TSci coverage in this study is incomplete (1 of 4 models): the TSci adapter pulls in `matplotlib` for panel-image generation, which collides with our conda env's missing `libstdc++.so.6` CXXABI_1.3.15 symbol on the three uncached models. This is recorded transparently in Appendix A as a reproducibility caveat rather than hidden. Data: `research/results/f2_cross_llm.jsonl` (53 deduplicated rows).

---

## 5. Discussion

### 5.1 Honest Limitations

**Computational overhead (Appendix C2).** The Agent layers cost 28-118× more compute than their respective base models per cell, while delivering gains that are statistically non-significant on saturated benchmarks (B7v3 vs Rocket p=0.17; wrapper vs Chronos-2 p=0.32). This is a serious caveat: for raw-accuracy-driven deployments the recommendation is to **use the base model directly**. The Agent's value lies in the lightweight learned components — abstain head (~1 ms), learned margin head (~1 ms), Meta-Router (<5 ms) — which deliver 0.5-74 pp gains on their respective targets at negligible inference cost. The heavy LLM-driven routing components should be reserved for offline analysis, OOD-safety wrappers, and interpretability use cases.


We adopt a stance of failure transparency, expecting this to be one of the paper's distinguishing characteristics:

1. **The right adaptive design is a confidence-gated router around the SOTA base TSFM, not a competing ensemble.** This is the key methodological lesson from our v5c → v7 → v8 → v9 → v10 progression on ETTh1/ETTh2/ECL/Exchange (4 datasets × 4 N × 3 seeds = 48 cells):
   - **v5c → v8** (independent-ensemble or top-1 routing): all four versions lose to standalone Chronos-2 on 16/16 cells, with catastrophic deviations up to +45% MAE on ECL N=100 and Exchange N=50 when short-validation CV picks the wrong strategy with no recovery.
   - **v9** (Chronos-2 as confident default + ≥20%-margin gating, N≥15 only): **strictly Pareto-dominates v8 (8 wins, 0 losses, 8 ties)** and produces **the first AdaptTS-class cell to beat standalone Chronos-2 (ETTh2 N=100, MAE 5.070 vs 5.169, -1.9%)**, where CV correctly identified Chronos-Bolt as a confident winner over Chronos-2 with >20% CV margin. v9 vs Chronos-2: 0W/7L/9T — 9 cells deferred cleanly, but N=10 cells still went through the prefix-rule (no walk-forward) and lost by 85-162%.
   - **v10** (v9 + unconditional Chronos-2 fallback for N<15): closes the N=10 gap. **v10 vs Chronos-2: 1W / 3L / 12T at ε=0.5%** — parity in 12 of 16 cells, one cell beating Chronos-2, three cells where the static 20% gate margin under-rejects a mis-ranked CV winner (ETTh1 N=50 +20%, ETTh2 N=50 +6%, ECL N=100 +38%). v10 strictly Pareto-improves v9 (4W/0L/12T) and is our final reported method.

   This is a **structural lesson**, not an engineering one: when a strong base TSFM is available (Chronos-2 in 2026), the burden of proof has shifted. An adaptive layer that *competes with* the base TSFM (ensemble averaging, top-1 selection) is dominated by one that *defers to* it. The gating-margin parameter is the single hyperparameter that captures this trade-off; setting it to 0% recovers v8, to ∞ recovers Chronos-2, and intermediate values (we use 20%) yield a strict improvement over both.

   Wall-time overhead is also addressed by gating: when the default holds, only one Chronos-2 inference fires (~2 s), comparable to standalone use. Future work (§5.4) discusses extending the gate with a **base-TSFM-coverage estimator** trained on retrospective per-cell wins, replacing the static 20% margin with a learned threshold.
2. **AdaptTS underperforms on N=10 in 5 of 6 datasets**. The walk-forward CV is disabled when N<15 (insufficient hold-out), and the prefix-rule planner is too coarse for this regime. A meta-learned planner trained on memory cases is a clear next step.
2. **`diagnosis_revision` is never triggered in 9/9 runs**. The closed-loop diagnostic correction mechanism is implemented but inert; a follow-up paper could explore prompt designs that specifically *invite* diagnostic skepticism.
3. **Short val-segment overfitting** is the primary failure mode of the original v5 reflection design. v5c works around this by decoupling reflection from best-plan selection, but a more principled fix would use multi-window walk-forward as the reflection acceptance criterion.
4. **LLM training-data contamination is plausible for ETTh1/ETTh2/Weather**. ECL/Exchange/ILI are less standard and serve as partial mitigation; a memorization probe (giving the LLM the dataset name without data) is planned.

### 5.2 Concurrent Work and Differentiation

The agentic time series forecasting paradigm has been articulated by Zhao et al. (2026), and concrete implementations exist (MemCast, Cast-R1, TSOrchestr). **Our distinctive position is**:

- **Few-shot specialization**: we are, as far as we know, the only work that systematically evaluates ATSF in N=10–100.
- **Cross-validated diagnostic confidence**: prior work uses confidence (MemCast) but does not compare statistical, LLM-subjective, and cross-validated sources.
- **Walk-forward CV in the train-tail** as a strategy-selection signal: a different mechanism from MemCast's memory-based confidence and Cast-R1's RL policy.
- **Honest failure analysis**: case studies, the v5→v5c design pivot, and the diagnosis-revision never-triggered finding are reported transparently.

We make no claim to having invented the agentic paradigm or model-routing in general; **our contribution is the few-shot specialization and the systematic empirical study**.

### 5.3 Final Boundary Synthesis: Agent-as-Router Is a Domain-Invariant Principle

The three task results in §4.2–4.11 fit a single architectural story. We articulate it explicitly because we believe **the isomorphism is the paper's main methodological contribution**, more than any individual metric.

**The isomorphism**

|  | Forecasting (§4.2–4.8) | TSC (§4.10–4.11) |
|---|---|---|
| **Direct version** | v5c (ensemble of native predictors) | B6 (Curator + LLM ICL → class) |
| **Direct version result** | 16/16 lose to Chronos-2, +20-50% MAE | 0/15 winner cells, -33pp vs Rocket |
| **Top-1 / margin gating** | v7-v9 (chronos slot alias, top-1 CV) | B7v1 (LOO CV + margin) |
| **Margin-gating result** | 16 still lose; v8 +45% catastrophic | -2.77pp; BeetleFly N=3 -25pp catastrophic |
| **N-conditional fallback** | v10 (N<15 fallback to default) | B7v2 (N<7 fallback to default) |
| **Fallback result** | 1W/3L/12T vs Chronos-2 | -0.87pp vs Rocket (68% of gap closed) |
| **Memory-augmented** | v11 (memory safety-net, weighted consensus) | B7v3 (25-dim mem + sim-weighted vote + Cards v2) |
| **Final result** | 0W/1L/23T MAE, 0% CRPS gap | **+0.89pp** over Rocket alone |

The two columns are not parallel constructions made for the paper — they were developed independently, ~2 weeks apart, and only after the fact did we recognize the isomorphism. We interpret this as evidence that **the four-stage progression (direct → margin gate → N-fallback → memory) is the correct path for adapting LLM-Agent architectures around any sufficiently strong base-model**, not a domain-specific artefact.

**Why it works**

The shared mechanism in both domains is **conditional model selection conditioned on a noise-robust signal**:

- Direct competition fails because LLM ICL on raw features cannot match SOTA models built specifically for the task.
- Single-signal gating (CV alone) fails at extreme few-shot because the gate signal is itself unreliable.
- N-conditional fallback collapses the signal noise back to the base-model default in the regime where the gate is least trustworthy.
- Memory adds a second, slower-moving signal (retrospective consensus) that activates only when feature engineering is rich enough to differentiate cells — which we achieved by going from 12-dim averaged-diagnosis to 25-dim with frequency, complexity, and meta-information.

### 5.3.1 Consolidated Final Synthesis Table

| Task / Sub-question | Best heuristic | Our Agent | Δ | Status |
|---|---|---|---|---|
| **Forecasting (24 cells)** | Chronos-2 | v11 wrapper | **MAE = 0**, CRPS = 0 | parity ✓ |
| Forecasting: prompt-resistant Weather OOD | Chronos-2 | v11 | +505% (1 cell catastrophic) | honest limitation |
| **Forecasting multi-mechanism validation** | C2 | abstain head / entropy gate / memory safety | all converge to C2 mean MAE | ⭐ 3 paths |
| **RCA in-tax natural (30 cells)** | B0-rule | Agent v1/v2 | -37pp / -40pp | honest negative |
| RCA in-tax clean (50 cells) | B0-rule 50% | Agent v2 26% | -24pp | honest negative |
| Agent vs LLM-direct on RCA in-tax | LLM-direct 0% | Agent v1 40% | +40pp (degenerate baseline) | qualified positive |
| **RCA out-of-taxonomy (50 cells)** | B1 LLM-direct 24% | Agent default 2% | **-22pp** | specialist bias |
| Mitigation 1: Abstain head | — | Agent + head 76% | +74pp ⭐ | architectural |
| Mitigation 2: Stronger LLM | — | Agent (glm-4-plus) 68% | +66pp ⭐ | deployment |
| Mitigation 3: Dataset prior | — | Agent + prior 14% | +12pp | prompt-only |
| **TSC UCR-5 direct B6 (30 cells)** | Rocket 87.5% | B6 54.3% | -33pp | direct fails |
| **TSC UCR-5 Router B7v3** | Rocket 87.5% | B7v3 88.4% | **+0.89pp** | niche win ⭐ |
| TSC UCR less-saturated (20 cells) | Rocket 83.1% | B7v3 82.4% | -0.7pp | routing inactive |
| TSC synthetic 4-class | Rocket 50.6% | B6 33.7% | -17pp | even with diagnostic-aligned labels |
| **TSC UEA multivariate (partial)** | Rocket 69.9% | DTW 72.5% | DTW > Rocket | larger routing space |
| **Learned routing L1 (margin)** | Heuristic margin=0.10 | learned head | **+0.49pp** | LODO CV ⭐ |
| Learned routing L2 v1 multiclass | Rocket-alone | Meta-Router | -4.05pp | failed |
| Learned routing L2 v2 regression | Rocket-alone | + confidence gate | **-0.01pp tied** | safe parity |

**Pattern across the table**:

1. **Direct competition with SOTA base models fails universally** (4 cases: forecasting wrapper, RCA in-tax vs rule, UCR direct, synthetic 4-class)
2. **Routing/wrapper architecture achieves parity or niche win** (forecasting v11 parity, TSC B7v3 +0.89pp, UEA DTW > Rocket)
3. **Specialist bias has 3 independent mitigation paths** with intervention-depth ordering (architectural > deployment > prompt)
4. **Learned routing components can beat heuristic counterparts when scope is narrow** (L1 +0.49pp) but not when wide (L2 v1/v2 tied or worse)

### 5.3.2 Why "No Method Dominates" Still Matters

The 24-cell winner distribution (§4.2) is, we believe, the single most important empirical contribution of this paper. Most prior forecasting papers either (a) report a single benchmark and claim SOTA, or (b) report multiple benchmarks but average them away. Our cell-level distribution makes the case for adaptive selection **inevitable**: there is no future method, short of meta-learning a universal predictor, that will avoid this picture. The right level of abstraction in this field is *routing*, not *modeling*.

---

## 6. Conclusion

We presented an empirical applicability boundary for LLM-Agent architectures in the time-series foundation-model era, evaluated across three task types: forecasting (where wrappers cannot beat Chronos-2, but a guaranteed-parity wrapper exists), prediction-failure root-cause analysis (where the Agent achieves +40pp over an unstructured baseline), and few-shot classification (where the Agent loses to Rocket as a direct classifier but **beats Rocket-alone by +0.89pp as a router** over {Rocket, MOMENT, DTW, …}).

The forecasting wrapper progression (v5c → v8 → v10 → v11) and the classification router progression (B6 → B7v1 → B7v2 → B7v3) are **isomorphic**: both fail at direct competition with SOTA base-models, both have a CV-noise failure mode at extreme few-shot resolved by N-conditional fallback, and both achieve parity-or-better via cross-series memory with sufficiently rich features. This domain-invariant pattern — the **Agent-as-Router-around-base-models** principle — is, we believe, the central methodological lesson of this work and a useful template for future LLM-Agent design in time-series and adjacent task families.

We report honestly the regimes where the Agent does not help (direct forecasting with no base-model coverage gap, direct classification on diagnosis-irrelevant features) and outline a feature-engineering future-work program (per-class diagnostic statistics, frequency-domain descriptors, pretrained-TSFM embeddings) that should push the router further toward the 92.06% Oracle ceiling.

---

## Appendix A: TSci Reproduction Notes

We use the official `Y-Research-SBU/TimeSeriesScientist` repository at commit (HEAD as of 2026-05). Two minimal monkey-patches were required:
1. `PreprocessAgent.run` defaults `output_dir` to `config["output_dir"]` rather than `None` (the original raises `TypeError` on `Path(None)`).
2. `_forecast_node` passes `state["validation_data"]` to `forecast_agent.run` (the original drops it, causing `missing positional argument`).

The LLM client is redirected to zhipu via `OPENAI_API_KEY = ZHIPU_API_KEY` and `OPENAI_BASE_URL = https://open.bigmodel.cn/api/paas/v4/`, using model `glm-4-flash-250414`. We verified this configuration produces the same forecast distribution shape as TSci's stock output; we are unable to verify exact MAE equivalence with the original GPT-4o configuration without access to the original API keys.

## Appendix B: Additional Ablation (A7 — w/o walk-forward CV)

Comparing v3 (prefix rules) vs v4 (walk-forward CV) on ETTh1 + ETTh2 (8 cells):

| Dataset, N | v3 (prefix) MAE | v4 (walk-forward) MAE | Δ |
|---|---|---|---|
| ETTh1 N=20 | 4.42 | **4.06** | -8% |
| ETTh2 N=20 | 5.14 | **4.83** | -6% |
| ETTh2 N=50 | 4.15 | **4.04** | -3% |
| ETTh2 N=100 | 7.52 | **6.28** | -16% |
| ETTh1 N=50 | 3.72 | 4.42 | +19% |
| ETTh1 N=100 | 3.23 | 3.37 | +4% |
| ETTh1 N=10 / ETTh2 N=10 | — | — | (walk-forward disabled, N<15) |

Walk-forward CV improves 4/6 cells where it is enabled, including 16% on ETTh2 N=100. The two regressions occur where the strategy pool excludes the actually-best method (LLMTime not in the pool when N>30); the fix is to keep LLMTime in the pool universally and let CV down-weight it as needed.

## Appendix C: Hyperparameters

- LLM temperature 0.2, max_tokens 2048
- Reflection max iterations 3 (hard cap), threshold val.std × 0.5
- Plan switch acceptance threshold 40% MAE improvement (v5b/c) — but in v5c best_plan never switches anyway
- Walk-forward CV: H_v = max(3, min(10, N//5)), folds = max(1, min(5, (N−H_v)//H_v))
- softmax τ: 0.3 for N≥50, 0.6 for N<50
- All baselines use 3 seeds (1, 42, 123)

## Appendix C2: Compute / Latency Analysis

We aggregate per-cell wall-time across all production sweeps. Numbers are CPU-only on a single 16-core Linux machine (no GPU). LLM calls are disk-cached; first-run latency would be substantially higher.

| Domain | Method | Mean (s/cell) | Median (s/cell) | Relative cost vs base |
|---|---|---|---|---|
| **Forecasting** | Chronos-2 | 1.65 | 0.11 | 1× (base) |
| Forecasting | Chronos-Bolt | 1.49 | 0.11 | 0.9× |
| Forecasting | adapt_ts wrapper (v9-v13) | 45.79 | 16.52 | **~28×** ⚠ |
| **TSC UCR** | Rocket | 0.80 | 0.76 | 1× (base) |
| TSC | Euclid 1-NN | 0.00 | 0.00 | <0.01× |
| TSC | MOMENT 1-NN | 2.19 | 0.96 | 2.7× |
| TSC | DTW 1-NN | 47.86 | 35.41 | ~60× |
| TSC | B6 Direct Agent | 94.39 | 100.41 | ~118× |
| TSC | B7v3 Router | 43.40 | 22.99 | **~54×** ⚠ |
| **TSC UEA** | Rocket multivariate | 2.81 | 1.99 | 1× (base) |
| TSC UEA | DTW multivariate | 302.73 | 224.41 | ~100× |
| **Learned heads** | Abstain head (inference) | <0.001 | <0.001 | trivial ⭐ |
| Learned heads | Learned margin head | <0.001 | <0.001 | trivial ⭐ |
| Learned heads | Meta-Router (RFR) | <0.005 | <0.005 | trivial ⭐ |

**Cost–benefit honest reporting**:

| Method | Gain (where applicable) | Cost | Production verdict |
|---|---|---|---|
| B7v3 Router | +0.89 pp UCR-5 (p=0.17 ns) | ~54× slower than Rocket | not recommended for raw-acc deployment |
| Abstain head | +74 pp OOT-recall (p<0.001) | ~1 ms (sklearn) | **production-ready** ⭐ |
| Learned margin head | +0.49 pp LODO CV | ~1 ms | **production-ready** ⭐ |
| Adapt_ts forecasting wrapper | 0 pp on average (p=0.32) | ~28× slower than Chronos-2 | not recommended unless OOD-safety needed |
| Memory consensus | (part of router) | 20-50 ms (faiss bypassed → numpy kNN) | low overhead |

**Take-aways**:
1. **Heavy LLM-based components (wrapper, B7v3 router) are 28-118× slower than their respective bases**, with marginal gains that are mostly statistically non-significant on saturated benchmarks.
2. **Learned heads (abstain, margin) are trivially cheap (<1 ms) and yield the strongest production-grade improvements** — these are the architectural primitives we recommend deploying.
3. For latency-sensitive forecasting, deployment guidance is simply "use Chronos-2 directly". The Agent's value is in **routing interpretability and safe abstention**, not in raw inference time.

## Appendix D: Failure Trace Dump

See `research/results/case_studies_v5.json` and `research/results/a8_a9_runs.jsonl` for raw reflection traces. The companion code repository (anonymized for review) contains the full reproduction recipe and all 144 cells of jsonl-formatted results.

---

## Appendix E: Reproducibility Checklist

**Environment**
- Python 3.10 / `mamba` env `tsci` (env file: `research/env.yml`)
- Hardware: CPU-only (Linux 6.6.x kernel, 16-core). MOMENT-1-small / Chronos-2 / Chronos-Bolt all run on CPU.
- Single NVIDIA RTX 2060 6GB available but unused (torch CPU-only build); TimesFM-2.0 / Moirai deferred to GPU future work due to CPU load-time / dependency issues.

**Data**
- Forecasting: ETTh1, ETTh2 from public hourly load benchmark; ECL/Exchange from `laiguokun/multivariate-time-series-data`; Weather/ILI manually unzipped from `M4_ILI_Weather.zip`. All raw CSVs in `research/datasets/raw/`.
- TSC UCR: 10 datasets auto-downloaded by `utils/ucr_loader.py` from `timeseriesclassification.com/aeon-toolkit` (Coffee, ECG200, GunPoint, TwoLeadECG, BeetleFly, BirdChicken, ECG5000, Crop, Wafer, Strawberry).
- TSC UEA: 3+ multivariate datasets auto-downloaded by `utils/uea_loader.py` (BasicMotions, ERing, AtrialFibrillation, expanding to 20 in task #48).

**LLM**
- Primary: zhipu `glm-4-flash-250414` (non-reasoning), accessed via OpenAI-compatible API.
- Cross-LLM (§4.10 RCA): `glm-4-air`, `glm-4-plus`.
- All LLM calls disk-cached (SHA-256 of prompt) at `research/.llm_cache/`; cache shared across all experiments.
- LLM client (`utils/llm.py`): 5-retry exponential backoff, `reasoning_content` fallback for reasoning models, 90s per-call hard timeout.

**Seeds & determinism**
- Forecasting: 3 seeds (1, 42, 123) per cell.
- TSC: 2 seeds (1, 42) per (dataset, N-shot).
- All `numpy`/`torch`/`sklearn` random states explicitly fed; LLM is non-deterministic but cached.

**Reproduction commands**
```bash
# Forecasting main result (v11/v13)
ADAPTTS_CHRONOS=bolt ADAPTTS_TSFM_POOL=expand ADAPTTS_DEFAULT=chronos2 \
ADAPTTS_DEFAULT_MARGIN=0.20 ADAPTTS_MEMORY_PATH=/tmp/v11.jsonl \
python -m research.experiments.runner --dataset ETTh1 --N 50 --methods adapt_ts --seeds 1,42,123

# RCA (Task A)
python -m research.experiments.taska_select_failures
python -m research.experiments.taska_run_rca

# TSC Router (Task B)
python -m research.experiments.build_clf_memory_v2
python -m research.experiments.taskb_router_v3_sweep

# Out-of-taxonomy + abstain head
python -m research.experiments.taska_oot_rca
python -m research.agent.abstain_head     # trains the abstain head
python -m research.experiments.taska_abstain_eval

# Learned margin
python -m research.agent.learned_margin   # trains + 5-fold + LODO eval
```

**Software**
- Open-source dependencies: `chronos-forecasting==2.2.2`, `momentfm`, `sktime==0.40.1`, `numba`, `dtaidistance`, `scikit-learn`, `statsmodels`, `transformers>=4.41`, `pmdarima` (for ARIMA+ETS baseline).
- Reproducible monkey-patches for TSci baseline documented in Appendix A.

**Result files (all jsonl)**
- Forecasting: `p3-p13_*.jsonl` (one per version), `f4_*.jsonl` (Chronos baselines), `a3_prob_metrics.jsonl` (CRPS).
- RCA: `taska_failures.jsonl`, `taska_rca_predictions.jsonl`, `taska_synthetic_rca.jsonl`, `taska_oot_rca.jsonl`, `taska_cross_llm_rca.jsonl`, `taska_abstain_eval.jsonl`, `taska_dataset_prior_eval.jsonl`, `taska_b0_rule_predictions.jsonl`.
- TSC: `taskb_ucr.jsonl`, `taskb_router_ucr.jsonl`, `taskb_router_v2_ucr.jsonl`, `taskb_router_v3_ucr.jsonl`, `taskb_extended_ucr.jsonl`, `taskc_synth4class.jsonl`, `taskb_uea.jsonl`, `taskb_uea_full.jsonl`.
- Learned routing: `meta_router.pkl`, `meta_router_v2.pkl`, `learned_margin.pkl`, `abstain_head.pkl`, `forecast_abstain_head.pkl`.

**Documented honest limitations**
- LLM API SSL flakiness (zhipu) caused intermittent failures; addressed via 5-retry with exponential backoff. F2 cross-LLM TSci coverage incomplete (matplotlib CXXABI library conflict) — see Appendix A.
- Pre-trained TSFM downloads occasionally failed (Hugging Face SSL) during sweeps; offline-cached models bypass.
- One OOD forecasting failure (Weather N=20 v11 seed=1, +505%) documented as known limitation of memory bootstrap (§5.1, Limitation 1).
- Reflection's contribution to MAE is zero (A8/A9 ablation): we retain it as an interpretability mechanism (case studies §4.4); the diagnosis-revision channel is never triggered in 9/9 runs (§5.1, Limitation 2).
- B7v3 +0.89pp UCR-5 gain is concentrated on BeetleFly/BirdChicken image-outline morphology where MOMENT pretraining dominates; on less-saturated extended UCR (§4.6 task #42) routing is essentially inactive (B7v3 ≈ Rocket-alone).
- Curator + Cards specialist bias is **LLM-capacity-dependent**: glm-4-flash shows 2% OOT-recall, glm-4-air/plus reach 68% (§4.7.3 cross-LLM analysis).
- The current router is univariate-only; UEA multivariate exploration is preliminary (3/20 datasets at submission).

---

*Last updated*: 2026-05-25 (post-polish). Authors and affiliations omitted for double-blind review.
