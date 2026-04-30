# Demo 第 6 步：LangGraph 串成多 Agent 工作流

> 对应 `06_langgraph_pipeline.py`。这是把前 5 步从"4 个独立脚本"升级为**真正的多 Agent 系统**的关键一步。
> 读完你会理解：(1) 为什么需要框架，(2) LangGraph 帮你做了什么，(3) 状态对象、节点、边是怎么对应到实际代码的。

---

## 1. 之前 5 步缺什么？

回顾我们的运行方式：

```
python 02_curator_visual.py
python 03_planner.py
python 04_forecaster.py
python 05_reporter.py
```

每一步**用 JSON 文件**做"状态传递"：上一步写盘 → 下一步读盘。这能跑，但缺：

| 缺失 | 后果 |
|---|---|
| 显式的状态 schema | 中间字段加一个、改个名，下游静默崩 |
| 错误恢复 | 任意一步炸 → 手动重跑 + 丢中间结果 |
| 可观测性 | 不知道哪一步慢、哪个 LLM 调用花了多少 token |
| 控制流灵活性 | 想加"Forecaster 失败 → 退回 Planner 选别的模型" → 要写一堆 if/else |
| 并行能力 | 候选模型超参搜索本可并行，纯脚本要自己写线程池 |

LangGraph 把这些做成基础设施。

---

## 2. LangGraph 的三个核心概念

### 2.1 State（状态）

一个 `TypedDict`（或 `pydantic.BaseModel`）描述工作流里"任何时刻可能存在"的所有字段：

```python
class PipelineState(TypedDict, total=False):
    raw_series: pd.Series          # Curator 写
    Q: dict                        # Curator 写
    candidates: list               # Planner 写
    ensemble_test_mape: float      # Forecaster 写
    report_path: str               # Reporter 写
    ...
```

**关键约定**：
- 节点函数返回 dict，dict 里的键会**合并**进 state（不是替换整个 state）；
- 没写到的字段保持原值；
- `total=False` 表示所有字段都可选，便于增量填充。

> 这正是 TSci 论文里 `C = {Q, V, A}` 的工程化形态——只是把它扩展到了流水线全程。

### 2.2 Node（节点）

一个普通函数 `(state) -> partial_state_update`：

```python
def node_curator(state: PipelineState) -> dict:
    raw = make_synthetic_series()
    Q = compute_quality_vector(raw)
    return {"raw_series": raw, "Q": Q, ...}
```

我们的 4 个节点完全**复用前 5 步写好的函数**——LangGraph 只是个调度器，不强迫你重写业务逻辑。

### 2.3 Edge（边）

边定义"上一步完成后下一步是谁"。最简单的形态是直线：

```python
g.add_edge(START, "curator")
g.add_edge("curator", "planner")
g.add_edge("planner", "forecaster")
g.add_edge("forecaster", "reporter")
g.add_edge("reporter", END)
```

更复杂的形态（**未来扩展**）：

| 边类型 | 用途示例 |
|---|---|
| `add_conditional_edges` | "如果 ensemble_test_mape > 5，回到 planner 选别的模型" |
| 自循环 | "Reporter 写完后让 LLM 自评，不达标就重写" |
| 并行扇出 + 合并 | 候选模型超参搜索同时跑多个 |

这些都是改图，不动节点。

---

## 3. 为什么 schema 必须显式？

`TypedDict` 看起来是冗余——前 5 步用 dict 不也跑了吗？

但显式 schema 给你三件事：

1. **IDE 自动补全**：在节点里写 `state["Q"]["n"]` 时编辑器知道类型；
2. **静态检查**：`mypy` / `pyright` 能查"forecaster 用了一个 planner 没写的字段"；
3. **可观测性**：LangGraph Studio / LangSmith 能直接展示每个节点输入输出的 schema diff。

> 这跟单元测试是同一个思想——**让错误尽早暴露**。

---

## 4. 可视化：图就是文档

`StateGraph` 编译后可以导出 Mermaid PNG：

```python
png = app.get_graph().draw_mermaid_png()
```

我们把它存成 `pipeline_graph.png`。看着这张图比读 200 行 README 更快地传达系统结构。

> 装了 `graphviz` 才能画。如果失败也无所谓，主流程不依赖。

---

## 5. 现在你可以做的扩展

到 Step 6 你已经有一个完整骨架，往下任何 TSci 论文里的"高级特性"都是**改图，不重写**：

### A. 反思机制（论文 Future Work）
```python
g.add_conditional_edges(
    "forecaster",
    lambda s: "reporter" if s["ensemble_test_mape"] < 5 else "planner",
)
```
预测不达标自动回退选别的模型。

### B. 多变量预测（论文 Future Work）
PipelineState 加一个 `extra_vars: dict[str, pd.Series]`，Curator/Planner 节点改造成多变量版本。

### C. 置信度传播（你的研究方向）
PipelineState 加 `Q_confidence: dict`、`A_confidence: dict`，让 Curator 输出每个判断的置信度（基于样本量、p 值），Planner 节点根据置信度调整候选权重——这正是 TSci 论文最后留下的研究入口。

---

## 6. 跑起来

```bash
mamba activate tsci
mamba install -c conda-forge langgraph -y
# 或: pip install langgraph
cd /home/hz/code/agent_ts/demo
python 06_langgraph_pipeline.py
```

> 如果 conda-forge 没有 langgraph（视版本而定），改用 `pip install langgraph`。两种都行，LangGraph 是纯 Python。

**预期产物**：
- 4 段执行日志（每个节点一段）+ 总览
- `final_report.md`、`curator_panel.png`、`forecaster_panel.png`（同前几步）
- `pipeline_graph.png`（如果装了 graphviz）

**自查**：
1. 终端日志里 4 个节点是否按 curator → planner → forecaster → reporter 顺序执行？
2. 最终的 `ensemble_test_mape` 和直接跑 step 1-5 是否大致一致？（合成数据有 seed，应几乎相同）
3. 试着在 `node_planner` 故意 `raise RuntimeError("test")`，看 LangGraph 怎么报错——你会发现它会保留 curator 已经写入的 state，方便你定位问题。

---

## 7. 至此 demo 全程完结

你已经从"零基础"走到"能用 LangGraph 跑一个 4 Agent 时序预测流水线"。下一步建议方向：

- **复现 TSci 论文实验**：把 `make_synthetic_series` 换成真实 ETT/Weather 数据集，看精度能否对得上论文。
- **延伸研究**：实现 §5.C 的"置信度传播"，写一个能在小样本下退化为更保守决策的版本。这就是一篇可投会议的小论文起点。
- **加一个 Agent**：给系统加一个 `Validator` 节点，专门做"集成预测的回测"和"分布偏移检测"，把这个流水线变成 5 Agent。

到这一步你已经不再是"Agent 入门"了——你是在用 Agent 解决一个真实的研究问题。
