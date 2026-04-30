# 00 · 入门：什么是 Agent？怎么从零造一个？

> 本文档是**纯入门讲义**，不依赖任何代码，读完后你应该能用自己的话回答："Agent 比 LLM 多了什么？我要写一个 Agent，至少需要哪几块？"
> 看完再去读 `01_curator_minimal.py`，会觉得"啊，原来是这样拼起来的"。

---

## 1. 一句话定义

> **Agent = LLM + 工具 + 控制循环 + 状态。**

把这四个词拆开看：

| 组件 | 作用 | 类比 |
|---|---|---|
| **LLM** | 思考、决策、生成自然语言 | 大脑 |
| **工具 (tools)** | 做 LLM 做不准的事：算数、查 DB、画图、发请求 | 手 / 计算器 |
| **控制循环 (loop)** | "想 → 做 → 看结果 → 再想" 直到任务完成 | 你做事的过程 |
| **状态 (state)** | 跨步骤共享的内存：已经做过什么、还差什么 | 草稿纸 |

**少了任何一块都不算 Agent**：
- 只调一次 `llm("帮我做X")` → 是**单次推理**，不是 Agent；
- LLM + 工具但只调用一轮 → 是**函数调用 (function calling)**，是 Agent 的最简形态；
- 多轮但没状态 → 是**多轮对话**，不是 Agent；
- 全套都有但只能聊天没工具 → 是**Chatbot**。

---

## 2. 为什么需要 Agent？LLM 自己不够用吗？

LLM 有三个硬伤：

1. **不会精确计算**：让它算 `mean([1.2, 3.4, 5.6, ...])` 200 个数，它会瞎编一个看起来合理的数。
2. **没有副作用能力**：它不能真的去读你的文件、调你的数据库、发 HTTP 请求。
3. **上下文窗口有限**：你不能把 100 万行日志全塞进 prompt。

Agent 的解决思路很简单：**硬数学和副作用交给代码（工具），软判断交给 LLM**。

举个例子，TSci 的 Curator 要"诊断时序数据质量"：

| 子任务 | 谁来做 | 为什么 |
|---|---|---|
| 算均值、方差、缺失数、IQR | Python 代码 | LLM 算不准 |
| 看到这些统计量后，"该用插值还是删除？" | LLM | 需要常识判断，规则写不完 |
| 真去做插值 | Python 代码 | LLM 改不了数据 |
| 把统计量和策略一起记下来给下游用 | 状态对象 | LLM 没有持久记忆 |

这就是一个最小 Agent 的形态。

---

## 3. 一个 Agent 由哪几块代码构成？

实战中，**每一个 Agent 至少有 5 个模块**：

```
┌──────────────────────────────────────────┐
│  ① 输入 / 输出 schema （契约）           │
│  ② 工具集 (tools)                         │
│  ③ Prompt 模板（角色 / 指令 / 示例）      │
│  ④ LLM 调用封装（含模型、解析、重试）     │
│  ⑤ 控制循环（决定下一步做什么）           │
└──────────────────────────────────────────┘
```

下面逐块讲。

### 3.1 输入 / 输出 schema

Agent 不应该返回"一段聊天文字"，而应该返回**机器可解析的结构化对象**（通常是 JSON 或 Pydantic）。

```python
# 输入：原始时序
# 输出：{"missing_strategy": ..., "outlier_strategy": ..., "reason": ...}
```

**为什么重要？** 因为下游 Agent / 代码要消费它。如果 Curator 返回"我建议你大概用插值吧也许"，Planner 没法用。结构化是多 Agent 协作的前提。

### 3.2 工具集

工具就是普通 Python 函数，签名清晰、副作用可控：

```python
def compute_quality_vector(series: pd.Series) -> dict: ...
def apply_strategy(series: pd.Series, strategy: dict) -> pd.Series: ...
```

LLM 不直接调用它们（在我们这版最简 demo 里），而是：
- 你先把工具结果算好，塞进 prompt；
- LLM 看着结果做判断；
- 你再根据 LLM 的判断调下一个工具。

**进阶版**（function calling / tool use）会让 LLM 自己决定调哪个工具——但那是第二步。先把这步做扎实。

### 3.3 Prompt 模板

至少包含三段：

1. **System prompt**：定角色 + 输出格式约束
   > "你是时序数据预处理专家，仅输出 JSON，字段为..."
2. **User message**：实际任务输入（工具结果）
   > "Q = {n: 200, missing_count: 8, ...}"
3. **(可选) Few-shot examples**：给 1-2 个示例，对小模型尤其有效。

写 prompt 的三个铁律：
- **明确输出格式**：JSON 字段名要一字不差地写出来；
- **枚举允许的取值**：`missing_strategy ∈ {linear_interpolation, ffill, ...}`，否则 LLM 会自创新值；
- **温度调低**（`temperature=0.2`）：决策类任务不要让模型发挥。

### 3.4 LLM 调用封装

一个 production-ready 的封装至少要做：

| 关心的点 | 怎么做 |
|---|---|
| 换服务方便 | 走 OpenAI 兼容协议，只换 `base_url` |
| 输出一定是 JSON | `response_format={"type": "json_object"}` |
| 解析失败时重试 | try / except + 重新请求一次 |
| 超时 / 限流 | `timeout=30, max_retries=2` |
| 成本可控 | 选小模型、降低 `max_tokens` |

我们在 Step 1 还没加重试和限流，因为先要"跑通"，避免一次性引入太多概念。

### 3.5 控制循环

这是 Agent 最体现"智能"的地方。最简单的循环只有一轮：

```
读数据 → 算工具 → 调 LLM → 解析 → 应用 → 结束
```

更复杂的形态：

- **ReAct**（Reason + Act）：LLM 输出 `Thought → Action → Observation`，看 Observation 后决定下一个 Action，直到输出 `Finish`。
- **Reflect**：LLM 做完一次后，让它"再看一眼自己的结果"，找问题再修。
- **Plan-and-Execute**：先让 LLM 列计划，再逐项执行。

**Step 1 的 demo 是退化的 ReAct**——只有一个 Action，没有循环。Step 2 之后我们会引入"看可视化图再判断"这种二轮决策。

---

## 4. 状态对象：多 Agent 协作的关键

单 Agent 时状态可能就是几个变量。**多 Agent 时必须显式建模一个共享状态**：

```python
@dataclass
class CuratorState:
    raw_series: pd.Series          # 原始数据 D
    quality_vector: dict           # Q
    strategy: dict                 # π
    cleaned_series: pd.Series      # D̃
    visuals: dict                  # V (后续 step 加)
    structure_report: dict         # A (后续 step 加)
```

TSci 论文里这就是 `C = {Q, V, A}`。Planner 拿到 `CuratorState` 后只读它的字段，不需要重跑前面的步骤。

**为什么不能用全局变量？** 因为 Agent 的状态需要：
- **可序列化**（存盘 / 跨进程传）；
- **可追溯**（debug 时知道哪一步出错）；
- **不可篡改**（每个 Agent 只该写自己负责的字段）。

`dataclass` / `pydantic.BaseModel` / LangGraph 的 `TypedDict` 都是常见选择。Step 6 我们会用 LangGraph。

---

## 5. 一个最小 Agent 的"打卡清单"

写一个新 Agent 之前，把这 6 个问题答一遍：

1. ✅ 这个 Agent 的**输入**是什么？类型？
2. ✅ 这个 Agent 的**输出**是什么？字段？
3. ✅ 哪些子任务交给**代码（工具）**？哪些交给 **LLM**？
4. ✅ Prompt 里 system / user 各写什么？输出格式怎么约束？
5. ✅ 控制循环是几轮？什么时候终止？
6. ✅ 怎么把它的输出塞进**共享状态**给下游？

回到 `01_curator_minimal.py`，你会发现这 6 个问题都有明确答案——这就是为什么它是一个（虽然简陋的）Agent，而不是一段普通脚本。

---

## 6. 主流 Agent 框架地图（建议先用裸代码理解原理再上框架）

| 框架 | 一句话定位 | 适合场景 |
|---|---|---|
| **裸 OpenAI SDK** | 自己写循环，全透明 | 入门 / 本 demo 前期 |
| **LangChain** | 工具链 + Prompt 模板生态最大 | 单 Agent + RAG |
| **LangGraph** | 用有向图描述多 Agent 工作流 | TSci 这种多 Agent 系统 ⭐ |
| **AutoGen (微软)** | 多 Agent 对话模式 | Agent 互相聊天的场景 |
| **CrewAI** | 类似 AutoGen，更"角色化" | 角色分工明显的任务 |
| **OpenAI Agents SDK** | OpenAI 官方，function calling 优先 | 紧绑 OpenAI 生态 |

**我们的策略**：Step 1-5 用裸 SDK 把每个 Agent 写清楚 → Step 6 用 LangGraph 串起来。这样你能理解"框架到底帮你省了什么"。

---

## 7. 最后：Agent 的常见误区

1. **过度依赖 LLM**：能用代码算的指标就用代码，别让 LLM 算。
2. **Prompt 不约束格式**：等 LLM 心情好才输出 JSON，会被坑死。
3. **没有错误恢复**：LLM 偶尔会输出非法 JSON / 编造字段值，必须有重试 + 校验。
4. **状态泄漏**：把上一轮整段对话历史塞进下一轮，token 爆炸。共享状态要"瘦"。
5. **一上来就用框架**：不知道框架在替你做什么，bug 永远修不动。

读到这里，可以打开 `README.md` 走 Step 1，再回头对照本文。
