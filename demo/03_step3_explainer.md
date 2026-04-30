# Demo 第 3 步：Planner Agent

> 对应 `03_planner.py`，TSci 论文里的第二个 Agent。
> 读完你会理解：(1) 为什么把"选模型"也交给 LLM，(2) 时序数据怎么科学地切分 train/val/test，(3) 超参搜索为什么只搜小网格就够。

---

## 1. 这一步做什么？

| 输入 | 处理 | 输出 |
|---|---|---|
| `curator_state.json`（含 Q + A） | LLM 选 2-3 个候选 → 各自小网格超参搜索 → 按 val MAPE 排序 | `planner_state.json`（top-k 调好参的模型） |

数据流：

```
curator_state.json    +    cleaned series
        │                       │
        └──► LLM (glm-4.7-flash) 看 Q+A 选模型
                    │
                    ▼
        [arima, holt_winters, ridge_lag, ...]
                    │
                    ▼
        每个模型在 param_grid 上做 fit → forecast → MAPE on val
                    │
                    ▼
        排序 → planner_state.json
```

---

## 2. 模型库 M：为什么是这 4 个？

我们故意选**不同家族**的代表，避免冗余：

| 模型 | 家族 | 强项 | 弱项 |
|---|---|---|---|
| `naive_seasonal` | 朴素基线 | 强周期、零参数、永远能跑 | 没有趋势 / 渐变能力 |
| `arima` | 经典统计 | 平稳/弱季节序列、有理论基础 | 强季节性吃力 |
| `holt_winters` | 指数平滑 | 趋势 + 加性季节性兼顾 | 复杂非线性表现一般 |
| `ridge_lag` | ML 回归 | 灵活捕捉滞后/星期效应 + 正则化 | 需要构造特征 |

**为什么不放 PatchTST / N-BEATS？** 它们是神经网络，要 GPU 或长训练时间，对入门 demo 是噪声。等你 Step 6 用 LangGraph 串起来再决定要不要加。

**`naive_seasonal` 的关键作用**：它是**永远能跑、永远不报错**的基线。如果某个复杂模型的 val MAPE 还不如 naive，说明那个模型这个数据集压根不该用——这是一个非常好的 sanity check。

---

## 3. 数据切分：时序为什么不能随机划分？

普通 ML 常用 `train_test_split(shuffle=True)`，**时序绝对不能**：随机洗牌会让"未来"信息泄漏到训练集。

我们的切法（按时间严格顺序）：

```
| ←─────────  337 天 train ─────────→ | 14d val | 14d test |
```

- **train**：拟合参数；
- **val**：选超参 / 比较候选模型（这一步暴露在 LLM 给的策略下）；
- **test**：完全锁住，**Planner 不许碰**，留给 Step 4 的 Forecaster 做最终评估。

> 这样设计还能防一种隐蔽错误：如果 Planner 在 test 上调参，集成权重就含 test 信息，最后报告的"测试 MAPE"是虚假的低。**TSci 论文专门强调这点**——集成权重必须在测试前就冻结。

---

## 4. LLM 怎么选模型？

`PLANNER_SYSTEM` 提示词要求 LLM 看 Q + A 输出 JSON：

```json
{
  "selected": ["holt_winters", "ridge_lag"],
  "reason_per_model": {
    "holt_winters": "A 显示明显趋势 + 周季节性...",
    "ridge_lag": "Q 中 trend_slope=0.049 表明..."
  }
}
```

**Prompt 工程的两个关键点**：

1. **强制引用 Q 或 A 的字段**：避免 LLM 凭"看起来合理"就乱选。要求"理由必须引用具体字段"是把它的注意力锚到证据上。
2. **限定数量 2-3 个**：太多会导致超参搜索时间爆炸，也增加 Forecaster 集成的难度；只选 1 个又失去集成的意义。

**没有视觉**：这一步不需要图，因为 A 已经把图的结论文字化了。Curator 已经把"看图"这个能力封装好，下游 Agent 只读结构化字段。

---

## 5. 超参搜索：为什么每个模型只 3-4 组？

工业实践里超参搜索可以跑成千上万组，但 demo 阶段：

- **预算有限**：每组要 fit + forecast，慢的模型（HW、ARIMA）一组 1-3 秒；
- **过拟合风险**：val 只有 14 个点，搜得太细会让 val MAPE 排名不稳定；
- **诊断价值更高**：3-4 组就足够回答"模型对这个超参敏感吗？"，比"找到全网最优"重要。

我们的网格设计：

| 模型 | 关键超参 | 网格点数 |
|---|---|---|
| naive_seasonal | period | 1（只有 7） |
| arima | order | 4 |
| holt_winters | trend / damped | 3 |
| ridge_lag | alpha | 3 |

**评分指标用 MAPE**，因为：
- 量纲无关（百分比），跨数据集可比；
- 对零值不友好——但我们的合成序列没零值，OK；
- 直观（"平均预测误差占真值 5%"）。

---

## 6. 兜底设计

LLM 偶尔会胡来，代码里有 3 道防线：

1. **过滤非法 key**：`[m for m in pick["selected"] if m in MODEL_LIBRARY]`，模型库没有的名字直接丢；
2. **空选回退**：如果 LLM 一个合法的都没选出来，自动回退到全库搜索；
3. **拟合失败容错**：每个 fit/forecast 都包 `try/except`，失败的配置 score 设 inf，不阻断流程。

这是 production agent 的标配——**永远不要假设 LLM 输出"对"**，尤其在多步流水线里，一处 bug 会污染下游所有 Agent。

---

## 7. 输出 JSON 的形状

`planner_state.json`：

```json
{
  "task": {"horizon_test": 14, "horizon_val": 14, "period": 7, ...},
  "candidates": [
    {
      "model": "holt_winters",
      "best_params": {"trend": "add", "damped_trend": true},
      "val_mape": 3.214,
      "reason": "A 显示...",
      "all_trials": [[{"trend":"add", "damped_trend": false}, 3.51], ...]
    },
    ...
  ]
}
```

**为什么把 `all_trials` 也存了？** 给后面的 Reporter Agent 用——最终报告需要展示"我们尝试了哪些超参，分别多差"，这样人类用户才能信任 Planner 的选择。

---

## 8. 跑起来

```bash
mamba activate tsci
mamba install -c conda-forge scikit-learn -y     # 第一次跑要装
cd /home/hz/code/agent_ts/demo
python 03_planner.py
```

**预期输出**：
- 5 个段落（加载状态 → 数据切分 → LLM 选 → 超参搜索 → 写文件）
- `planner_state.json`：top-k 调好参的候选

**自查**：
1. LLM 选的 2-3 个模型是哪几个？理由是否真的引用了 A 的字段？
2. `naive_seasonal` 的 val_MAPE 通常作为基线——其它模型都比它好吗？如果某个比它差，意味着什么？
3. 如果你把 `H_VAL` 改成 7（更短），val_MAPE 排名会变吗？这就是"评估方差"问题——这也是你研究方向（"小样本下的置信度"）的另一个切面。

跑通后告诉我，进入 Step 4：Forecaster Agent，把 top-k 模型集成起来给最终预测。
