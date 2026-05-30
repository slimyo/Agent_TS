顶会风格架构图 · AI 生成提示词集

  ▎ 风格基准：NeurIPS / ICML / ICLR 2024-25 paper figure 通用美学。
  ▎ 适配工具：DALL-E 3 / SDXL / Midjourney v6（推荐前两个，对长 prompt 理解准）。
  ▎ 长宽比建议：16:9（PPT 横屏） 或 4:3。

  通用 style suffix（每条 prompt 末尾拼接以保证一致风格）：

  Style: NeurIPS / ICML conference paper figure aesthetic. Vector-clean flat design,
  no 3D bevels, no glossy reflections. White background. Restrained palette:
  muted cool blues (#3A6FA0, #6BA4D8) for blocks, warm orange (#E08E3C) for the
  feedback loop, dark gray (#333) text, light gray (#E5E5E5) dividers. Crisp
  Iosevka or Inter typeface, slight letter-spacing. Every arrow labeled.
  LaTeX-style math where shown. Avoid: photo-realism, ornate borders, drop shadows,
  emoji, neon, gradients beyond 2 stops, Chinese characters (use English labels).

  ---
  Figure 1 · 主架构图（hero）
  
  用途：method 节 / ppt slide 5 主图。

  A horizontal architecture diagram of a unified Bayesian time-series routing
  framework, suitable for a NeurIPS 2025 paper. Wide 16:9 canvas, left-to-right
  data flow in three vertical bands.

  LEFT BAND — Input:
  Three stacked horizontal mini-charts labelled "Vibration", "Electricity Load",
  "Temperature", each a 1-D curve in light gray. Below them a small label "x ∈ R^L".
  An arrow leads right.

  MIDDLE BAND — Encoder + Regime:
  Top block titled "Curator Agent (statistical + LLM)", small clipboard icon,
  outputs a 25-cell colored feature vector labelled "d ∈ R^25".
  Below, a wider block titled "Embedding f_φ" — shows three stacked options
  "MOMENT-1 / Chronos-2-enc / Hand-crafted" with one highlighted. Arrow output
  "z ∈ R^d_emb".
  Below that, a Voronoi-style cluster diagram with 6 colored cells labelled
  "Regime r(z) via k-means", a small black dot showing "new x" landing in
  regime r₃.

  RIGHT BAND — Bayesian Router + Library:
  A large central rounded box titled "Bayesian Router" containing two sub-columns:
    - "Prior factors": stack of 6 horizontal bars labelled
      "Availability / CRPS / Regime / Type / N / Entropy / Industrial".
    - "Likelihood factors": stack of 2 bars labelled "CV / Memory / Representation".
  Above the box: equation displayed prominently:
    "p(M_k | x, h, t) ∝ exp( Σ log π_k^(i)(z) + Σ log L_k^(j)(z, h, t) )"
  Below the box, three short arrows labelled "argmax / Thompson / risk-min"
  fork out, then merge into a single chosen-model arrow.

  The chosen-model arrow points to a model library shelf on the far right:
  a 4×3 grid of square cards each labelled with a model name
  (Chronos-2, TiRex, Toto, TimesFM-2, Moirai, Moirai-2, Time-MoE, Sundial,
  Timer-S1, Rocket, WEASEL, MOMENT). One card is highlighted with a checkmark.

  Bottom right: small downstream output icons in a row:
  "Forecasting (curve with confidence band)" /
  "Classification (3 color buckets)" /
  "Anomaly + RCA (gear with warning)".

  DASHED ORANGE FEEDBACK ARROW: curves from the forecast output back to a small
  cylinder labelled "BanditState (μ_r,k, σ_r,k)" attached to the Router box,
  with arrow caption "observe(z, chosen, loss)".

  Title at top: "Universal Bayesian Adaptive Routing for Time-Series".

  [+ style suffix above]

  ---
  Figure 2 · BayesianRouter 详图（zoom）
  
  用途：method 节方法详解 slide。

  A zoomed-in detail diagram of a single block called "Bayesian Router", in the
  style of a NeurIPS paper figure. Square canvas, single-block focus.

  Outer rounded rectangle, title "BayesianRouter" with subtitle
  "unified probabilistic decision rule".

  Inside, three vertical lanes:

  Lane 1 (left) — "Priors π_k(z)":
  Vertical stack of 6 rounded rectangles, each containing a small equation:
    • AvailabilityPrior:  log π = -∞ · 1[k ∉ env]
    • CRPSPrior:           log π = log(1/L_val(k))
    • RegimePrior:         log π = log Π_k(regime(z))
    • TypePrior:           log π = log α · 1[k ∈ POINT]
    • NPrior:              log π = β · (1 - N/N₀) · 1[k = default]
    • EntropyPrior:        log π = -γ · H_C2 · 1[k = default]
  A "Σ" symbol below sums them.

  Lane 2 (middle) — "Likelihoods L_k(z, h)":
  Vertical stack of 3 rectangles:
    • CVLikelihood:        log L = -loss_k / σ²
    • MemoryLikelihood:    log L = log Σ_n sim(z, z_n) / (1 - acc_n[k] + ε)
    • RepresentationLikelihood:  log L = log Σ_n sim_z-space · 1/loss_n[k]
  A "Σ" symbol below.

  Lane 3 (right) — "Posterior + Decide":
  Top: equation "p(M_k | x, h) = softmax(Σ priors + Σ likelihoods)".
  Below: a 3-way switch box with three exits labelled:
    argmax(p)   |   sample(p)   |   argmin(μ + λσ)
  Three corresponding arrows leave the box to the right.

  Lane 1 and Lane 2 arrows merge at the "p(M_k)" node.

  Crisp paper-figure typesetting, all equations in serif math font, prose in
  sans-serif. Use the muted blue/orange/gray palette.

  [+ style suffix above]

  ---
  Figure 3 · Regime Manifold 可视化
  
  用途：Phase 4 章节 / 论文 §4.

  A scientific visualization of a learned regime manifold for time-series routing,
  NeurIPS-paper style. 16:9 canvas.

  Left half: a 2D scatter plot, titled "Learned regime manifold (MOMENT embedding,
  K=8)". Background tessellated into 8 colored Voronoi regions in muted pastel
  tones. Inside each region, scattered small dots (~6 per region). Each dot color
  matches its dataset of origin (legend: Weather=blue, ECL=orange, Exchange=green,
  ETTh1=purple, ETTh2=red, ILI=teal). Some regions are monochrome (purity 100%),
  others are mixed colors (purity ~50%).

  Centroids: small black ×-marks at cluster centers, labelled r₀ through r₇.

  Right half: a horizontal bar chart titled "Per-regime prior π_k", 8 rows
  (one per regime). Each row stacks 5 colored segments labeled
  "chronos2 / TiRex / Toto / Time-MoE / Sundial" with their proportion.
  Highlight rows where a single model exceeds 0.5 prior — e.g.
  "r₅: chronos2 = 0.58 (Weather-dominant)" or
  "r₂: TiRex = 0.38 (Exchange niche)".

  Top center caption: "Regime ⊃ dataset: cross-dataset clusters reveal
  shared latent regimes; intra-dataset clusters reveal within-dataset
  heterogeneity."

  [+ style suffix above]

  ---
  Figure 4 · 在线 Bandit 闭环时间线
  
  用途：Phase 2 章节 / 介绍 online adaptation.

  A horizontal temporal flow diagram in NeurIPS-paper style, showing the
  contextual bandit online loop. Wide 16:9 canvas.

  A long horizontal axis labelled "time t" at the bottom. Five evenly-spaced
  "episode" snapshots stacked left to right, each rendered as a small column:

  Column header for each: "t = t_i" (i = 1, 2, ..., 5).

  Inside each column, top to bottom:
    1. Small line-graph icon "series x_t arrives".
    2. Small embedding icon "z_t = f_φ(x_t)".
    3. Mini-bar chart "Bayesian posterior over models" with one bar
       highlighted — the sampled choice. The highlighted bar shifts model
       across episodes to illustrate exploration → exploitation transition.
    4. Outcome: "ℓ_t observed" with a number under it.

  Below all five columns, a horizontal "BanditState belief band" showing how
  μ_k tightens and σ_k shrinks across t for one specific model, drawn as a
  shrinking Gaussian curve under each snapshot.

  Top right: equation
    "μ_{t+1} = (decay·n_t·μ_t + ℓ_t) / (decay·n_t + 1)".

  ├────────┼───────────────────────────────────────────────────┼────────┤
  │ P2     │ method.md 加 [DEPRECATED] 头                      │ 1 min  │
  ├────────┼───────────────────────────────────────────────────┼────────┤
  │ P2     │ finish-1 §8 / §9 压缩重复部分                     │ 30 min │
  ├────────┼───────────────────────────────────────────────────┼────────┤
  │ P3     │ 加 results 子目录 README 解释 jsonl 命名          │ 15 min │
  └────────┴───────────────────────────────────────────────────┴────────┘

  ---