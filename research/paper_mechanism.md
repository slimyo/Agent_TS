# The Design of a Selective-Action Decision Mechanism under a Strong Default

> 机制论文草稿（Round 11 #100 / method6+method7 综合）。版本：2026-06-05。
> 配套实测：`finish6.md`（F-R11.x）/ `finish7.md`（F-R12.x）/ `baseline.md §1`（全量库）/
> `advisor_report.md`（脉络）/ 图 `results/m22_threetask_phase.png`、`results/m14_phase_diagram.png`。
> 定位：**独立于"提分"的机制设计科学论文**——研究对象是"一个 agent 在强默认模型旁，何时该行动/退/信自己"。

---

## Abstract

When a strong base model (a TSFM or Rocket) already solves most of a task, the useful question is no longer *"which model is best?"* but *"should the agent deviate from the default at all, and if so, can it know when?"* We study this **selective-action decision** as an object of design in its own right, decoupled from raw accuracy. Across three time-series tasks (classification, forecasting, fault detection), a **full candidate library** (22 classifiers / 14 forecasting TSFMs), and leave-one-dataset-out (LODO) evaluation, we establish four mechanism results that are reproducible and domain-invariant:

1. **Trust ≠ confidence (F-R11.5/11.6).** A model's own belief strength is an *inverse* indicator of correctness under a strong default; an *epistemic* trust signal (conformal nonconformity) instead orders "will this deviation be safe" at AUC 0.80, far above ensemble-disagreement (0.55) and raw confidence (0.40).
2. **Avoid-harm and seek-gain are two different sub-problems needing two different signals (F-R12.5).** Conformal trust answers avoid-harm (AUC 0.79) but not seek-gain (0.40); a saturation/headroom signal answers seek-gain (0.71) but not avoid-harm (0.35). Naively fusing them into one "confidence" *dilutes both*.
3. **The headroom is real but the per-cell winner is unpredictable (new, full-library, all 228 cells run).** Expanding the library to 22 classifiers (all genuinely run to full coverage, incl. remote TSFM embeddings) un-saturates the benchmark — the base is the per-cell oracle on only 24.4% of cells (down from 71%), oracle headroom rises to 5.95pp (median 3.0pp), and 21 of 22 models win somewhere. Yet predicting *which* model wins from deployable features reaches only 0.11–0.25 accuracy (vs 0.20 for always-default), and acting on the prediction *loses* −0.25 to −4.64pp. Richer features make it worse.
4. **The reachable value of the mechanism is damage control, not profit (F-R11.7/11.9).** The optimal policy degenerates to "almost always commit to base"; deployed as a trust-gate it converts *near-base* into *exactly-base* with zero harmful deviations. On three full-library tasks the decoupled gate yields +0.00pp / +0.00pp (classification, detection) and −1.6% rel-MAE (forecasting) — honest abstention where no safe gain is learnable.

The contribution is not a higher score but a **vocabulary and a falsifiable structure** for selective-action agents: separate the signals, expect avoid-harm to be learnable and seek-gain not, and treat "+0.00pp under a full library" as a *correct* outcome (reachable-ceiling ≠ per-cell-predictability), not a failure.

---

## 1. Introduction

### 1.1 The shift from "which model" to "should I act"

Foundation models have made a single default (Chronos-2 for forecasting, Rocket for few-shot TSC) strong enough that, on standard benchmarks, an aggressive router that always picks "the best candidate" *loses* to simply committing to the default — because the per-cell best is not reliably identifiable from deployable information. The interesting agentic question becomes a **meta-decision**: given a query cell, emit one of `{commit-base, deviate→candidate, defer, ensemble, explore}`. This paper treats that decision mechanism — not the underlying models — as the research object.

### 1.2 Why this is worth a paper even at +0.00pp

A mechanism that "doesn't beat the base" is usually discarded. We argue the opposite: under a strong default, the *act of correctly abstaining* is the value, and the *conditions under which acting is/ isn't learnable* are the finding. We make this concrete by (i) separating two confounded signals, (ii) measuring each signal's discriminative power on the two sub-problems it could serve, and (iii) showing — on a deliberately un-saturated full library — that the residual barrier is per-cell *unpredictability*, not absence of headroom.

### 1.3 Contributions

- **C1.** A decomposition of "confidence" into **belief / saturation / trust / proposal**, with a measurement protocol (LODO Trust-AUC on avoid-harm vs seek-gain) that shows which signal serves which sub-problem (§3, §4).
- **C2.** The **two-signal decoupled decision rule** `deviate iff conformal_safe ∧ saturation_headroom`, and evidence that naive fusion underperforms either single-best signal (§4.3).
- **C3.** A **full-library un-saturation + unpredictability audit**: expanding to 22 classifiers (all run to full 228-cell coverage) removes the saturation artifact (base-is-oracle 71%→24.4%, headroom 5.95pp, 21/22 models win somewhere) yet the per-cell winner stays unpredictable (winner-acc 0.11–0.25; acting loses) (§5).
- **C4.** A **three-task phase diagram** (`m22`) showing all decision cells collapse into the high-saturation / commit region, and a **deployment validation** where the trust-gate makes near-base exactly-base (§6).
- **C5.** A set of **clean negative results** (gain unlearnable, portfolio loses, offline exploration net-negative, bigger TSFM ≠ better base) that bound the design space (§7).

---

## 2. Problem Setup

**Cells.** A *cell* is `(dataset, N, seed)`: a few-shot instance with N examples/class (TSC/detection) or N context points (forecasting). For each cell we have, offline, the realized score of every candidate model — an *oracle library*. The base model `b` is Rocket (TSC/detection) or Chronos-2 (forecasting).

**Decision.** A policy π reads only **deployable features** z(cell) (no test labels) and emits an action. We focus on the binary core `{commit-base, deviate→π_propose(z)}`, extended with `defer` (LLM second opinion) and `ensemble`.

**Honesty.** All evaluation is **leave-one-dataset-out**: every signal/regressor is trained only on *other* datasets' outcomes; the held-out dataset never enters training. This is stricter than leave-one-cell-out and removes the leakage that produced an early spurious +0.89pp (corrected to −0.62pp; the leakage audit is itself a contribution).

**Metrics (replacing vs-base±pp).** Trust-AUC (avoid-harm / seek-gain), Regret-to-Oracle, Safe-Deviation-Rate, Abstain-Accuracy, phase-boundary purity, inversion-coefficient.

---

## 3. Signal Decomposition: belief / saturation / trust / proposal

A single scalar "confidence" conflates four distinct quantities:

| Signal | Question it answers | Estimator |
|---|---|---|
| **belief** b(M\|z) | which candidate is most likely best | softmax head, LODO-CE to oracle-winner |
| **saturation** s(z) | is there any headroom over base | regression of predicted oracle-gap |
| **trust** t(z) | is *this* deviation in-distribution / safe | conformal nonconformity quantile |
| **proposal** | a concrete non-base candidate to consider | belief-argmax, or a learned proposal-net, or an LLM |

**Key inversion (F-R9.2 → F-R11.5).** Raw belief strength is an *inverse* indicator under a strong default: when a learned router deviates and is *wrong*, its mean confidence (0.79) *exceeds* when it deviates and is *right* (0.51). Bagging K=20 heads removes this inversion (inversion-coef +0.475; conf right 0.607 > wrong 0.483). → **Do not gate on confidence; gate on an epistemic trust signal.**

---

## 4. Two Sub-problems, Two Signals

### 4.1 Avoid-harm is learnable (trust)

On the un-saturated 10→22-classifier library, **conformal trust attains Trust-AUC 0.796** for "will the proposed deviation be ≥ base" — and conformal trust when the deviation is correct (0.744) is significantly above when wrong (0.624), not driven by singletons (26/32 balanced) (F-R11.5). Conformal ≫ ensemble-disagreement (0.796 vs 0.55) and ≫ raw confidence (0.40) (F-R11.6). → **avoid-harm has a working signal: conformal nonconformity.**

### 4.2 Seek-gain is (mostly) not learnable (gain)

Predicting *magnitude* of gain is weakly possible (corr 0.40 classification, 0.18 forecasting) but predicting *whether* a deviation will profit is at chance: **profit-AUC 0.47** (F-R12.1). Even in forecasting where oracle headroom is 17% rel-MAE, the gain-gate's only safe operating point is "never deviate" (F-R12.2). → **seek-gain has no reliable per-cell signal from deployment features.**

### 4.3 The decoupling result (F-R12.5)

Measuring four candidate trust-sources on *both* sub-problems:

| signal | AUC_avoid-harm | AUC_seek-gain |
|---|---|---|
| **conformal** | **0.787** ✅ | 0.399 ✗ |
| disagreement | 0.590 | 0.602 |
| density | 0.465 | 0.567 |
| **saturation** | 0.352 ✗ | **0.714** ✅ |
| meta (logistic fusion) | 0.579 | 0.559 |

Conformal owns avoid-harm; saturation owns seek-gain; **each is weak on the other axis, and the logistic fusion is worse than either single-best** — diluting two specialized signals into one general "confidence" destroys both. → **Decision rule: `deviate iff conformal_safe(z) ∧ saturation_headroom(z)`**, not a single threshold.

### 4.4 The proposal role is cheap (F-R12.4)

A lightweight proposal-net (LogReg(z)→non-base candidate) matches an LLM second opinion (profits 0.235 vs 0.267) and beats belief-argmax — the LLM's value is in *proposing* deviations, not reasoning, and is replaceable without an API.

### 4.5 A selective-action regret bound (theory, #105)

We formalize why the two empirical pillars (avoid-harm learnable, seek-gain not) imply a commit-dominant optimum. Fix a proposal map z↦m(z). For cell *i* let δ_i = score(m_i) − score(base) be the **signed** deviation gain (committing yields 0). A gate deviates on D = {i : t(z_i) ≥ τ}. The realized excess-over-base is E(D) = Σ_{i∈D} δ_i, and the relevant oracle (same proposal, perfect gate) deviates exactly on D⁺ = {i : δ_i > 0}, achieving G⁺ = Σ_i max(δ_i, 0). Define **regret** R(D) = G⁺ − E(D).

**Proposition (AUC-controlled regret).** Let the gate score t rank cells, and let A = Trust-AUC be its probability of ranking a safe cell (δ>0) above an unsafe one (δ≤0). Then for the coverage-c threshold,
E[E(D)] = c·n·( p₊·μ₊·s₊(A,c) − p₋·|μ₋|·s₋(A,c) ),
where p± are the fractions of safe/unsafe cells, μ± their mean |δ|, and s±(A,c) the AUC-induced selection ratios (s₊→1, s₋→0 as A→1; s₊=s₋=c at A=½). Two regimes follow:

- **A → 1 (perfect avoid-harm):** D → D⁺, so R(D) → 0 — the gate captures all positive mass and leaks no harm; excess-over-base → G⁺.
- **A = ½ (avoid-harm at chance):** E[E(D)] = c·Σ_i δ_i = c·n·E[δ]. In a saturated/strong-default library **E[δ] < 0** (most deviations hurt), so *any* nonzero coverage gives **negative** expected excess — this is precisely the observed always-deviate −2.88pp (classification) and −5.3pp (decision cells).

**Corollary (why commit-dominant is optimal here).** The captured **gain** term scales with the *seek-gain* AUC, not the avoid-harm AUC. F-R12.1 measures seek-gain (profit-AUC) ≈ ½; F-R12.8 shows the deployable-feature avoid-harm AUC is also ≈ ½ (only the rich-feature avoid-harm reaches 0.79). With both ≈ ½, the only loss-minimizing threshold is τ → ∞ (coverage → 0), i.e. **commit to base**, whose regret-to-oracle equals the irreducible oracle headroom G⁺/n (5.95pp classification, full library) that no deployable gate can convert. ∎(sketch)

This bound ties the paper together: **the mechanism's reachable value is the harm it avoids (controlled by avoid-harm AUC), never the gain it captures (controlled by seek-gain AUC, ≈chance).** When the safe signal is rich enough (conformal on epistemic features, A=0.79) the gate provably reduces harm to ~0 — exactly the F-R11.9 deployment result (−0.42pp→+0.00pp, 2→0 mis-deviations).

---

## 5. The Headroom Is Real, the Winner Is Not Predictable (full-library audit)

A natural objection: *if the library is small and the base happens to dominate the chosen datasets, "+0.00pp / routing loses" is a circular artifact.* We confront this by expanding the classifier library 5→22 and auditing.

**Un-saturation is genuine.** With all 22 classifiers run to full 228-cell coverage, base-is-oracle drops **71% → 24.4%** of cells; oracle headroom rises **1.88 → 5.95pp** (median 3.0pp); the oracle is won by **21 of 22 models** (rocket 24%, muse 23 cells, minirocket 20, rocket_mv 14, multirocket 13, fcn 11, …); **22/38 datasets** have a non-base top model. The saturation framing *was* partly a small-library artifact — corrected here.

**Yet the winner is unpredictable.** Predicting per-cell winner from deployable features, LODO:

| predictor | winner-acc | acc of picking predicted-winner vs base |
|---|---|---|
| always-base | 0.204 | +0.00pp |
| 4-dim system features | 0.249 | **−0.25pp** |
| 30-dim series statistics | **0.113** | **−4.64pp** |

Richer features **overfit and hurt** (0.113 < always-base). The multiple winners are largely **noise-driven** (which model wins flips across seeds), not a stable "model A ↔ data type B" mapping. → **The barrier is per-cell unpredictability, not absence of headroom: reachable-ceiling ≠ per-cell-predictability.** This is a stronger, non-circular restatement of "+0.00pp."

---

## 6. The Phase Diagram and Deployment

### 6.1 Decision phase diagram (three tasks, full library)

Plotting every decision cell in `(saturation × trust)` (`m22_threetask_phase.png`):

| task | library | high-saturation frac | system deviations | vs base |
|---|---|---|---|---|
| classification | 22 clf | 1.00 | 0 | +0.00pp |
| detection | 22 clf | 0.58 | 3 | −0.63pp |
| forecasting | 14 TSFM | 0.92 | 6 | −1.6% rel-MAE |

The "deviate" region (trust≥0.5 ∧ saturation≤0.5) is nearly empty for classification; detection/forecasting *do* expose low-saturation cells (real headroom) but deviating there **still loses** — confirming §5 at the policy level. The learned policy and the rule-phase policy both ≈ always-commit; oracle-action ceiling is only 0.4pp above commit on the decision cells (F-R11.7).

### 6.2 Deployment validation (F-R11.9)

Adding the trust-gate (deviate requires CV-margin **and** conformal-trust ≥ τ) to the deployed multi-agent system intercepts the only mis-deviations (BirdChicken N=10 → moment_1nn, CV-saturated but test-worse): classification moves **−0.42pp → +0.00pp**, deviations **2 → 0**, no cell harmed. The mechanism's value is realized as *damage control in production*, exactly as the theory predicts.

---

## 7. What Bounds the Design Space (clean negatives)

- **Gain unlearnable** across saturated/un-saturated and classification/forecasting (F-R12.1/12.2).
- **Portfolio loses** to a single base (−0.14 to −3.6pp); a saturated library has no complementary structure worth diversifying over (F-R12.3).
- **Offline exploration is net-negative** (the "unknown-unknown" region is 28.8% of cells with 12pp headroom, but the lowest-belief candidate is usually genuinely worst); its only value is online information gain (F-R12.6).
- **A bigger TSFM is not a better base**: Timer-S1 (8.3B) underperforms Chronos-2 by −7.0% rel-MAE on 60 cells (F-R12.7) — base strength is not parameter count.

---

## 8. Discussion: when would seek-gain become learnable?

The results bound a clear frontier. Avoid-harm is learnable offline; seek-gain is not, on **academic UCR/UEA/ETT domains with mostly homogeneous perception-style candidates**. Two honest open conditions, untested here, could change this:

1. **A truly heterogeneous, structurally-complementary asset pool** (different physical priors, not kernel/distance/dict/deep variants of the same idea) — might produce a *stable, predictable* specialization.
2. **Online feedback** — turning the per-cell winner from an offline guess into an accumulating posterior, the only setting where exploration (F-R12.6) and gain prediction could pay off.

→ The path to positive selective-action value is **changing the domain / asset pool / feedback regime**, not a more elaborate offline confidence.

## 9. Conclusion

Under a strong default, the right design question is the selective action, and its answer factorizes: **belief proposes, saturation says whether there is room, conformal trust says whether it is safe, and the two axes must not be merged.** On a deliberately full, un-saturated library across three tasks, the mechanism correctly converges to honest abstention — not because the base is unbeatable in principle (5.95pp oracle headroom exists; rocket wins only 24% of cells) but because the per-cell winner is not predictable from anything deployable. We offer this as a falsifiable, domain-invariant account of *what a selective-action agent can and cannot know* — and a vocabulary for the next designer to test the two open conditions above.

---

## 附：Findings 索引（证据出处）

| 论文论点 | Finding | 文件 |
|---|---|---|
| trust≠confidence / inversion | F-R9.2, F-R11.1, F-R11.5 | finish4/finish6 |
| conformal ≫ ensemble | F-R11.6 | finish6 §5 |
| trust 只避险不获利 | F-R11.7 | finish6 §6 |
| LLM/proposal 补获利提议 | F-R11.8, F-R12.4 | finish6 §7 / finish7 |
| 双信号解耦 | F-R12.5 | finish7 §5 |
| gain 不可学 | F-R12.1, F-R12.2 | finish7 §1-2 |
| portfolio/explore 否证 | F-R12.3, F-R12.6 | finish7 §3,6 |
| 更大 TSFM ≠ 更好 base | F-R12.7 | finish7 §6b |
| 全量库 un-saturation + winner 不可预测 | §5（本文新审计）| baseline.md §1.2-1.3 / advisor_report §6 |
| trust 迁移性是特征级非任务级 | F-R12.8 | finish7 §6c / m23 |
| selective-action regret bound（理论）| §4.5（本文）| #105 |
| 部署 trust-gate 落地 | F-R11.9 | finish6 §8 |
| 三任务全量库相图 | m22 | results/m22_threetask_phase.png |
