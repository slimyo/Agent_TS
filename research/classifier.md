# Classifier Agent · 框架与现状

> 文档时间：2026-05-24
> 作用：把 pivot 后两个分类任务（RCA、TSC）的 agent 设计、模块清单、baseline、结果统一记录。配合 `plan.md §十六` + `finish.md §3.1.29-3.1.32` 看。

---

## 1. Pivot 上下文

2026-05-24 起，研究方向从"在 forecasting 击败 Chronos-2"转向"做 C2 完全不能做的 reasoning 任务"。

**核心观察**：6 数据集 × 4 N × 3 seeds = 72 cells 后，v5c→v13 任何 wrapper 都无法 systematic 击败 Chronos-2（v11/v13 = C2 严格 0W/1L/23T，CRPS 平），但 **同套架构（Curator + Reflection + Model Cards + Memory）在 RCA 任务上 Agent +40pp 击败 LLM-direct baseline**。结论：**LLM Agent 的价值在 reasoning，不在 forecasting**。

两个 reasoning 任务：

| Task | 输入 | 输出 | TSFM 能做吗 |
|---|---|---|---|
| **TaskA · RCA** | (train series, prediction, truth) | primary_fault ∈ {trend_break, seasonal_flip, variance_explode, outlier_burst, stationarity_flip} + supporting evidence | ❌ |
| **TaskB · TSC** | (N labeled series per class) + 1 query | class label + confidence | ❌（除 fine-tune） |

---

## 2. Agent 总体架构

```
┌─────────────── Input ───────────────┐
│  series x ∈ ℝ^L (+ optional context) │
└──────────────────┬───────────────────┘
                   │
       ┌───────────▼────────────┐
       │  Curator (curator_uq)  │  10/12-dim 诊断 + 三路置信度（stat/llm/xc）
       │  Diagnosis dataclass   │  v2: + outlier_count_z3 + variance_ratio
       └───────────┬────────────┘
                   │
   ┌───────────────┼───────────────┐
   │               │               │
   ▼               ▼               ▼
┌──────┐    ┌─────────────┐   ┌─────────────────┐
│Memory│    │ Model Cards │   │ Task-specific   │
│query │    │ (5 张)      │   │ context         │
│(kNN) │    │             │   │ (taxonomy, etc) │
└──┬───┘    └──────┬──────┘   └────────┬────────┘
   │               │                   │
   └───────────────┼───────────────────┘
                   │
       ┌───────────▼────────────┐
       │  LLM (zhipu glm-4-     │
       │   flash-250414)        │  Structured JSON prompt
       │  Reasoning + JSON out  │  Validated parser + fallback
       └───────────┬────────────┘
                   │
       ┌───────────▼────────────┐
       │  Output:               │
       │  RCA → fault label     │
       │  TSC → class + conf    │
       └────────────────────────┘
```

**复用四层**（来自 forecasting 时代的 AdaptTS 架构）：

| 层 | 模块 | RCA 用途 | TSC 用途 |
|---|---|---|---|
| ①诊断 | `agent/curator_uq.py` | 12-dim 诊断特征喂 LLM | 同 — 作为类比推理特征空间 |
| ②反思 | `agent/forecaster_reflect.py` (`_reflect`) | RCA root_cause 输出的 prompt 基础 | 暂未用（TSC 是单次 prediction） |
| ③Cards | `agent/model_cards.py` | 5 张策略 Card 引导 LLM 引用模型假设 | 同（弱相关） |
| ④Memory | `agent/memory.py` | （暂未用，task #25 拟启用） | （暂未用，可扩 case retrieval） |

---

## 3. 分类 Agent 实现清单（按 task）

### 3.1 TaskA · RCA Agent (`agent/rca.py`, 175 行)

**两条路径**：

```python
def agent_rca(train, val, test, prediction, dataset, N, seed, H,
              strategy, adapt_mae, c2_mae, season_m) -> dict
    # B5 Agent path
    1. diag = Curator.diagnose(train, season_m)
    2. cards = render_cards_block(5 strategies)
    3. prompt = AGENT_RCA_PROMPT.format(
         diag_text, cards, fault taxonomy,
         train head/tail, test head/tail, pred head/tail,
         adapt_mae, c2_mae, ratio
       )
    4. response = chat_cached([{"role":"user","content":prompt}])
    5. parsed = _parse_rca_json(response)
    6. parsed.primary_fault = _validate_fault(...)  # 规范化
    return {primary_fault, secondary_faults, supporting_evidence,
            hypothesized_repair, _raw, _path="agent"}

def b1_direct_rca(train, test, prediction, ..., strategy, adapt_mae, c2_mae) -> dict
    # B1 baseline - LLM 直接看数字，无 diagnosis 无 Cards
    prompt = B1_DIRECT_PROMPT.format(train/test/pred head&tail + taxonomy)
    ...
```

**5-fault taxonomy**（`utils/fault_taxonomy.py`）：

| ID | Fault | 诊断信号 | rule-based detector 阈值 |
|---|---|---|---|
| F1 | `trend_break` | split-half mean shift > k·std | k=2.0 → 1.0 |
| F2 | `seasonal_flip` | ACF lag=m 符号前后翻转 | abs(a_early - a_late) |
| F3 | `variance_explode` | late_std/early_std > k | k=2.0 → 1.0 |
| F4 | `outlier_burst` | MAD-based z > 3 计数 ≥3 | n_out/3.0 → 1.0 |
| F5 | `stationarity_flip` | (mean_diff + var_diff)/2 | join score |

`assign_ground_truth(train, test, season_m)` 在 train ∪ test 上各自打分取 max → primary + secondary。

### 3.2 TaskB · TSC Agent (`agent/tsc_classifier.py`, 130 行)

**两条 LLM 路径 + 4 经典/TSFM baselines**：

```python
def b5_llm_direct(X_train, y_train, X_test, llm_model=None) -> np.ndarray
    # 给 LLM K 个 (series 摘要, label) + query series → predict class
    for each q in X_test:
        prompt = B5_PROMPT.format(
            K=len(X_train), classes=sorted(set(y_train)),
            train_examples=[summary(X_train[i], label=y_train[i])],
            query_series=summary(q),
        )
        cls, _ = _parse_class(chat_cached([...]))
    return preds

def b6_agent(X_train, y_train, X_test, season_m=1, llm_model=None) -> np.ndarray
    # 1) Curator 给每个 training series 算 12-dim 诊断
    train_diags = [diagnose(x, season_m=season_m) for x in X_train]
    # 2) Query 也算诊断
    # 3) LLM 在诊断特征空间做类比推理
    for each q in X_test:
        q_diag = diagnose(q, season_m=season_m)
        prompt = B6_AGENT_PROMPT.format(
            classes=...,
            train_diag=[(diag_dict, label)],
            query_diag=q_diag,
        )
        cls, _ = _parse_class(...)
    return preds
```

**Series summary 给 LLM 看**：`mean, std, first[:6], last[:6]` —— 压缩到 ~200 tokens 避免上下文爆炸。

**Diagnosis-to-dict for prompt**：
```python
{n, mean, std, trend_slope, trend_tstat, adf_pvalue,
 acf_peak_lag, acf_peak_value,
 trend_conf=xc, season_conf=xc, stat_conf=xc}  # 11 keys, 12-dim Curator
```

**JSON 输出强制 schema**：
```json
{
  "class": <int>,
  "confidence": <0-1>,
  "supporting_neighbors": [<idx>...]  // B6 only
  "reason": "<1-2 句>"
}
```

`_parse_class` 三层 fallback：JSON parse → 字符串匹配 → 正则提 `"class": <int>` → 兜底返回 classes[0]。

---

## 4. 完整 Baseline 矩阵（TSC）

| ID | 名 | 类型 | 模块 | 强项 |
|---|---|---|---|---|
| B1 | 1-NN DTW | classical | `baseline/tsc_classical.py` | 全局形状对齐，经典强 baseline |
| B2 | 1-NN Euclid | classical | 同 | 最快 baseline |
| **B3** | **MiniRocket / Rocket** | **kernel SOTA** | 同 | **fine-grained temporal pattern**，UCR 主导 |
| **B4a** | **MOMENT 1-NN** | **TSFM probe** | `baseline/moment_classifier.py` | shape/morphology（BeetleFly/BirdChicken 反超 Rocket）|
| B4b | MOMENT LogReg | TSFM probe | 同 | 同 |
| B5 | LLM-direct | LLM ICL | `agent/tsc_classifier.py` | (baseline) |
| **B6** | **AdaptTS Agent** | **LLM + Curator** | 同 | **statistical-diagnostic class** (RCA-like) |

---

## 5. 当前结果（截至 2026-05-24）

### 5.1 RCA 结果（30 catastrophic failure cells, finish §3.1.29/31）

| Variant | Metric | B5 Agent | B1 LLM-direct | Δ |
|---|---|---|---|---|
| **v1 Curator (10-dim)** | R1 Top-1 | **40.0%** | 0.0% | **+40pp** ⭐ |
| | R2 Top-3 | 43.3% | 23.3% | +20pp |
| | R4 Kw-F1 | 16.2% | 0.0% | +16pp |
| **v2 Curator (12-dim, +outlier+var)** | R1 Top-1 | 36.7% | 0.0% | -3pp vs v1 |
| | **R2 Top-3** | **56.7%** | 23.3% | **+13pp** vs v1 ⭐ |
| | **R4 Kw-F1** | **30.0%** | 0.0% | **+14pp** vs v1 ⭐ |

**v1→v2 trade-off**：variance_explode 0/10→9/10（修好），但 stationarity_flip 12/13→1/13（LLM 过度依赖新特征）。R2/R4 大涨说明 Agent 解释质量翻倍但 R1 strict 略降。

**B1 collapse**：全 30 case 预测 `trend_break` — paper §5.1 motivation："without diagnostic structure LLM collapses to a single class"。

### 5.2 TSC UCR 结果（210 cells × 5 datasets × 3 N-shot × 2 seeds, finish §3.1.32）

| Method | Mean Acc | Macro F1 | Winner cells (/15) |
|---|---|---|---|
| **B3 Rocket** | **87.5%** | 0.871 | 7 |
| B4a MOMENT 1-NN | 81.9% | 0.812 | 3 |
| B4b MOMENT LogReg | 81.7% | 0.812 | 3 |
| B1 1-NN DTW | 74.8% | 0.740 | 1 |
| B2 1-NN Euclid | 71.0% | 0.703 | 1 |
| **B6 AdaptTS Agent** | **54.3%** | 0.482 | **0** |
| **B5 LLM-direct** | **52.7%** | 0.456 | **0** |

**Agent collapse**：BirdChicken/BeetleFly 10-shot **45%** ← 低于 random 50%。

**MOMENT 反超 Rocket** on image-outline (BeetleFly 3-shot: B4b 92.5% vs Rocket 82.5%)。

### 5.3 boundary characterization（已稳定，task #27 in-flight 验证中）

| Task | Class label 性质 | Agent 表现 | 解释 |
|---|---|---|---|
| **RCA** | statistical-diagnostic 概念 (trend_break, variance, outlier, stationarity) | **+40pp wins** | Curator 特征 = class label 代理 |
| **UCR TSC** | domain-specific patterns (coffee, ECG, image-outline) | **-33pp loses** | 诊断特征与标签无语义映射 |
| **Synthetic 4-class** (task #27, in-flight) | **statistical concepts 注入** (normal/trend_break/seasonal_break/outlier_burst) | **预期 wins** ⭐ | 反向 cross-check |

如果 task #27 验证 Agent ≥ Rocket，则论文 §5 **boundary 三角闭合**：Agent 表现差异是 **structural with class-label-feature semantic alignment**，与具体 task 无关。

---

## 6. 模块代码清单（已实现，2026-05-24）

```
research/
├── agent/
│   ├── curator_uq.py            ← v2 12-dim Diagnosis（task #29）
│   ├── model_cards.py           ← 5 张策略卡（forecasting 旧）
│   ├── memory.py                ← faiss Memory (forecasting 旧，TSC 待用)
│   ├── rca.py                   ← TaskA Agent (B5) + LLM-direct (B1)
│   └── tsc_classifier.py        ← TaskB Agent (B6) + LLM-direct (B5)
├── baseline/
│   ├── tsc_classical.py         ← B1 DTW / B2 Euclid / B3 Rocket
│   └── moment_classifier.py     ← B4a 1-NN / B4b LogReg (MOMENT-1-small)
├── utils/
│   ├── fault_taxonomy.py        ← 5-fault rule-based detector (RCA GT)
│   ├── inject_fault.py          ← 4-fault injector (task #27 synthetic)
│   └── ucr_loader.py            ← UCR archive loader + N-shot 抽样
└── experiments/
    ├── taska_select_failures.py ← 选 30 个 catastrophic + auto GT (task #24)
    ├── taska_run_rca.py         ← RCA Agent vs B1 跑分
    ├── taska_eval.py            ← R1/R2/R4 + confusion matrix
    ├── taskb_run.py             ← UCR 5 数据集 × 3 N × 2 seeds × 7 methods (task #26)
    └── taskc_synth4class.py     ← 合成 4-class fault 分类 (task #27 in-flight)
```

**结果文件**：
```
results/
├── taska_failures.jsonl               ← 30 cells + GT
├── taska_rca_predictions.jsonl        ← Agent + B1 v2 (latest)
├── taska_rca_predictions_v1.jsonl     ← v1 备份
├── taskb_ucr.jsonl                    ← 210 cells × 7 methods
└── taskc_synth4class.jsonl            ← in-flight
```

---

## 7. 下一步路线（按 ROI 排序）

| 优先 | task | 预期 | 工程量 |
|---|---|---|---|
| **P0** | **task #27 完成 + 分析**（in-flight） | 验证 boundary 三角 | 0（等结果）|
| **P0** | **task #25 RCA synthetic + v3 ensemble** | R1 → 55%+ clean GT | 1 天 |
| **P1** | task #28 论文双轨整理（结果完整后定 title） | 投稿就绪 | 2-3 天 |
| P2 | v3 Curator: ensemble v1+v2 投票 | 修 RCA 零和 trade-off | 半天 |
| P2 | TSC Memory.query 启用 | retrieve K cases by diag → ICL | 1 天 |
| P3 | TaskA RCA 加 multi-modal panel (B5 Curator LLM 升级) | feedback 推荐 | 2 天 |
| P3 | 5-class / 24-class UCR (ECG5000 / Crop) | 验证 multi-class boundary | 1 天 |

---

## 8. 关键设计决策（论文 § 3 Method 撰写参考）

1. **同套 Agent 架构，task-specific prompt**：Curator + Cards 复用；reasoning prompt 按 task 重写 → "task-agnostic features, task-specific reasoning"
2. **结构化 JSON 输出 + 三层 fallback parser**：避免 LLM 自由文本无法量化评估
3. **少样本 ICL，不 fine-tune**：保持 base TSFM/LLM 不可训前提（feedback alignment）
4. **B1 LLM-direct 作为 unstructured baseline**：把 "diagnosis-provided" 的边际效果隔离测量
5. **Memory 暂用 forecasting 时代的 faiss IndexFlatIP**：分类需要 case retrieval（kNN labels）是天然 use-case，待 task #25 启用

---

## 9. 已知 limitations / future work

1. **TSC Agent 在 image-outline 域失败** → 需要 task-adapted features（如 shapelet 或 raw morphology）—— 不是 Curator 的设计目标
2. **R1 v1→v2 trade-off** → ensemble vote 或 prompt 调整待实证
3. **LLM 调用成本**：210 UCR cells × 20 query × 2 method = 8400 LLM 调用 → cache 命中关键
4. **rule-based ground truth taxonomy 噪声**：30 case 自动标 GT 与 LLM 判断可能 misalign → task #25 synthetic 有 clean GT
5. **Memory layer 未在 TSC 启用**：retrieval-augmented ICL 是论文 §5.4 future work 一项
