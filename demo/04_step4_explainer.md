# Demo 第 4 步：Forecaster Agent

> 对应 `04_forecaster.py`，TSci 论文里的第三个 Agent。
> 读完你会理解：(1) 为什么集成往往打败单个模型，(2) 三种集成策略各自的适用场景，(3) 为什么"集成权重不能看 test"是数据泄漏的红线。

---

## 1. 这一步做什么？

| 输入 | 处理 | 输出 |
|---|---|---|
| `planner_state.json`（top-k 调好参模型） | 重训 → 预测 → LLM 选集成策略 → 应用 → test 评估 | `forecaster_state.json` + `forecaster_panel.png` |

---

## 2. 为什么要重训（train+val 合并）？

Planner 阶段为了"公平地比较候选模型"，所有模型只在 `train` 上拟合，用 `val` 打分。
但选定的超参一旦定下来，**最后部署用的模型应该见过尽可能多的数据**——尤其是最近的 14 天验证集，往往含有最近的趋势/季节信息。

所以 Forecaster 第一步是：

```
fit(train+val, best_params)   →   forecast(h)
```

这是机器学习里常见的 "refit on full" 套路。

> **注意**：refit 是用 `best_params`（已选好的超参），**不是重新搜索**。所有"调参"决策必须在 Planner 阶段封死，Forecaster 不能再碰超参，否则 val 信息会泄漏到最终预测。

---

## 3. 三种集成策略的本质

### 3.1 `single_best`

```python
yhat = preds[argmin(val_mape)]
```

**何时用**：某个模型 val MAPE 远低于其它（论文给的标准是相差 > 30%）。
**优点**：简单；如果赢家真的稳，误差最小。
**风险**：如果赢家在 test 上恰好遇到分布偏移，整个预测崩盘。

### 3.2 `performance_weighted`（性能加权平均）

```python
w_i ∝ (val_mape_i + ε)^(-β)
w_i ← (1-λ) · w_i  +  λ · (1/k)        # shrinkage
yhat = Σ w_i · pred_i
```

**何时用**：差不多的几个模型，没有压倒性赢家。
**为什么要 shrinkage**：纯按 `1/MAPE^β` 会把所有权重砸到最强者（β 大时），这跟 single_best 没区别。`λ=0.1` 让每个模型至少保留 10% 权重，是抗"评估方差"的手段。
**β 怎么选**：
- β=1：温和，弱模型也能贡献；
- β=2：强者更强；
- β=3：接近 winner-takes-most。

### 3.3 `robust_aggregation`（鲁棒聚合）

```python
yhat = median([pred_i for i in 1..k])
# 或 trimmed mean
```

**何时用**：候选 k≥3 且模型彼此分歧大（个别模型可能完全跑偏）。
**原理**：中位数 / 截尾均值天然抗极端值。
**代价**：当所有模型都对时，中位数不如加权平均精确。

> 我们这次只有 2 个候选，`robust_aggregation` 退化成普通平均，意义不大。LLM 大概率会选 `performance_weighted` 或 `single_best`。

---

## 4. 数据泄漏红线：权重必须在看 test 前定

ML 流水线最容易出错的就是这条。"用 test 数据调过的任何参数都不该出现在最终预测里"——包括集成权重。

我们的代码强制：

```python
weights = derive_from_val_mape_only(...)    # ① 权重定下来
yhat = combine(preds, weights)              # ② 才算集成
test_mape = mape(test, yhat)                # ③ 最后才看 test
```

如果允许"先算 test MAPE，挑表现最好的策略"——那就是经典的 **post-hoc winner picking**，最终上线性能必然会回落。

> TSci 论文专门强调这点。Forecaster 的 LLM 决策**只能看 val MAPE**，prompt 里也明确不给 test 数字。

---

## 5. 评估什么？

我们对比 4 个数字：

| 指标 | 含义 |
|---|---|
| 各模型 `test_mape` | 单模型在 test 上的真实表现 |
| ensemble `test_mape` | 集成在 test 上的表现 |
| ensemble vs best | 如果集成 < 单模型最佳，说明集成有用 |
| ensemble vs val_mape | 如果 test_mape 远高于 val_mape，说明 test 上有分布偏移 |

---

## 6. 输出 JSON 形状

```json
{
  "decision": {"strategy": "performance_weighted", "beta": 2.0, "reason": "..."},
  "weights": {"holt_winters": 0.78, "arima": 0.22},
  "test_mape_per_model": {"holt_winters": 3.1, "arima": 4.5},
  "ensemble_test_mape": 3.0,
  "ensemble_forecast": [...14 个数字...],
  "test_index": ["2024-12-18", ...],
  "test_truth": [...真值...],
  "panel_path": "forecaster_panel.png"
}
```

下游 Reporter 直接读这个 JSON 就能写报告，不用再跑模型。

---

## 7. 跑起来

```bash
mamba activate tsci
cd /home/hz/code/agent_ts/demo
python 04_forecaster.py
```

无新依赖。预期产物：
- 7 段终端输出
- `forecaster_panel.png`：历史 + 各模型预测 + 集成 + 真值
- `forecaster_state.json`

**自查**：
1. LLM 选了哪种集成策略？理由是什么？是否合理？
2. 集成 test MAPE 是否比最强单模型更低？如果不是，可能是因为我们只有 2 个候选+模型差距明显（HW 远胜 ARIMA），集成被弱者拖累——这正是 `single_best` 的适用场景。
3. 看 `forecaster_panel.png`：HW 和 ARIMA 的预测形态有什么不同？

跑通后告诉我，进入 Step 5：Reporter Agent，把全流程产物汇总成最终报告。
