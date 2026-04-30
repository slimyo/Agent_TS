> 👉 **第一次来 / 不知道下一步做什么？请先看 [`TODO.md`](./TODO.md)**，那是项目唯一看板。
> 👉 **完全没接触过 Agent？先读 [`00_what_is_an_agent.md`](./00_what_is_an_agent.md)** 再回来。

# Demo 第 1 步：最小化 Curator Agent

> 本文档面向**刚入门 Agent**的读者。我们一步步把 TSci 论文里的第一个模块（Curator，数据管家）做成一个能跑的最小原型，然后扩展。看完你会理解：
>
> 1. "Agent" 到底比"调一次 LLM"多了什么；
> 2. 怎么把 LLM 接到一段真实的时序数据上；
> 3. 为什么 TSci 要先做数据诊断、再选模型。

---

## 📌 用户需求 & 项目路线图（每次回到这个 demo 先读这块）

**最终目标**：实现 TSci 论文中的完整多 Agent 系统（Curator → Planner → Forecaster → Reporter）。

**学习风格约束**：
- 用户是 Agent **入门者**，每一步都要逐行解释做了什么、为什么这么做；
- 优先做**最小可运行原型**，先跑通再加复杂度；
- LLM **优先使用开源 / 免费 API**（默认 SiliconFlow，备选 DashScope / DeepSeek / 本地 Ollama），不使用 OpenAI 官方 / Anthropic 等付费服务；
- 每一步都附 markdown 说明文档，方便下次回来对照。

**整体路线图（按顺序推进）**：

| 步骤 | 文件 | 内容 | 状态 |
|---|---|---|---|
| Step 1 | `01_curator_minimal.py` | Curator 第 1 子步：质量向量 Q + LLM 给清洗策略 π + 应用 π | ✅ 当前在做 |
| Step 2 | `02_curator_visual.py` | Curator 第 2-3 子步：画 STL/ACF/PACF + 多模态 LLM 看图 | ⏳ |
| Step 3 | `03_planner.py` | Planner Agent：模型库 + LLM 选模型 + 超参数搜索 | ⏳ |
| Step 4 | `04_forecaster.py` | Forecaster Agent：top-k 集成 + 加权策略 | ⏳ |
| Step 5 | `05_reporter.py` | Reporter Agent：聚合产物输出报告 | ⏳ |
| Step 6 | `06_langgraph_pipeline.py` | 用 LangGraph 把四个 Agent 串成有状态工作流 | ⏳ |

---

## 0. 这一步要做什么？

TSci 的 Curator Agent 完整功能分三步：

1. **质量诊断**：算统计量、找缺失值、找异常值，再让 LLM 推荐预处理策略。
2. **多模态可视化**：画图（时序总览 / STL / ACF / PACF）。
3. **结构画像**：让 LLM 看图说话，给出趋势 / 季节性 / 平稳性结论。

**本 demo 只做第 1 步的"骨架"**：
读一段时序 → 用 Python 算统计量 → 把统计量喂给 LLM → LLM 输出 JSON 格式的预处理建议。

> 第 2、3 步留到 demo 后续。先把"和 LLM 对话拿到结构化输出"这件事跑通，因为这是所有 Agent 框架的基本功。

---

## 1. 什么是 Agent？为什么不直接 `llm("帮我分析这个序列")`？

一个最朴素的定义：

> **Agent = LLM + 工具 + 控制循环 + 状态。**

- **LLM**：负责"思考"和"决策"。
- **工具（tools）**：LLM 自己算不准的东西交给确定性代码做，比如 `numpy.mean`、画图、调数据库。
- **控制循环**：不是问一次答一次，而是"读数据 → 调工具 → 给 LLM 看结果 → LLM 决定下一步 → ……"直到任务结束。
- **状态（state）**：跨步骤共享的对象，比如 TSci 里的 `C = {Q, V, A}` 就是 Curator 写给后续 Agent 看的状态。

**为什么不直接 prompt？** 因为 LLM **不擅长精确计算**（算均值、做 ADF 检验都会瞎编），但**擅长在结构化信息上做判断**。所以 Agent 的精髓是：**"硬数学交给代码，软判断交给 LLM"**。

本 demo 里：
- 工具 = `numpy/pandas` 算的统计量；
- LLM = 看统计量后输出 JSON 策略；
- 控制循环 = 这一版只有"算 → 问 → 解析"一轮（最简单的 ReAct 退化形式）；
- 状态 = 一个 `dict`，对应论文里的 `Q`。

---

## 2. 准备 LLM API

我们用 **智谱 BigModel** 的 `glm-4.7-flash`（**完全免费**），理由：

- 完全免费、不限速可日常用；
- 接口与 OpenAI 兼容（base_url + 标准 chat.completions），换服务只改 `base_url` 和 `model` 两个字符串；
- 国内直连不需要代理。

步骤：

1. 打开 https://open.bigmodel.cn/ 注册登录；
2. 控制台 → API Keys → 新建一个 key（形如 `xxx.xxx`）；
3. 复制 key 写进 `.env`（参见 `.env.example`）。

**切换其它服务**：直接改 `.env` 里的 `PROVIDER` 字段（代码里的 `PROVIDERS` 字典已写好五种 provider 的 base_url 和默认模型）：

| PROVIDER 值 | 对应服务 | 默认模型 | 备注 |
|---|---|---|---|
| `zhipu` | 智谱 BigModel | `glm-4.7-flash` | **默认**，完全免费 |
| `siliconflow` | SiliconFlow（硅基流动） | `Qwen/Qwen2.5-7B-Instruct` | 注册送额度 |
| `dashscope` | 阿里 DashScope | `qwen-turbo` | 有免费额度 |
| `deepseek` | DeepSeek | `deepseek-chat` | 极便宜，非完全免费 |
| `ollama` | 本地 Ollama | `qwen2.5:7b` | 完全免费，需 `ollama pull qwen2.5:7b` |

如果想换默认模型，再在 `.env` 加一行 `MODEL=...` 覆盖即可。

---

## 3. 安装依赖

```bash
cd /home/hz/code/agent_ts/demo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` 里只有四个包：`openai`（调 LLM）、`pandas`、`numpy`、`python-dotenv`（读 .env）。

---

## 4. 跑起来

```bash
python 01_curator_minimal.py
```

预期输出（示意）：

```
=== Step 1: 加载数据 ===
shape=(200,), 缺失值=8, 类型=float64

=== Step 2: 计算质量向量 Q ===
{
  "n": 200,
  "missing_count": 8,
  "missing_ratio": 0.04,
  "mean": 12.34,
  "std": 3.21,
  "min": 4.5,
  "max": 22.7,
  "trend_slope": 0.018,
  "outlier_count_iqr": 5
}

=== Step 3: 调 LLM 拿预处理策略 π ===
[LLM thinking…]

=== Step 4: 解析 LLM 输出 ===
{
  "missing_strategy": "linear_interpolation",
  "outlier_strategy": "clip_iqr",
  "reason": "缺失率 4% 较低且分布零散，线性插值能保留趋势；IQR 检出少量异常值，裁剪比删除更稳妥。"
}

=== Step 5: 执行变换得到 D̃ ===
干净数据 shape=(200,), 缺失值=0
```

---

## 5. 代码逐段讲解

打开 `01_curator_minimal.py`，对照下面看：

### 5.1 `make_client()` —— 选 LLM 服务

读 `.env` 的 `PROVIDER` 字段，从代码顶部的 `PROVIDERS` 字典里查出 `base_url` 和默认 `model`，再读对应的 `*_API_KEY` 环境变量。返回一个 `OpenAI` 客户端 + 模型名。

> **为什么用 `openai` 这个库连 SiliconFlow / DeepSeek？** 因为这些服务都实现了 OpenAI 兼容接口（路径、参数、返回格式都一样），换服务只改 `base_url` 即可。这是当前国内 LLM 生态的事实标准。

### 5.2 `compute_quality_vector(series)` —— 论文里的 `Q`

用 pandas / numpy 算一组**确定性指标**：长度、缺失数 / 缺失率、均值、方差、min/max、线性趋势斜率（`np.polyfit` 一次拟合）、IQR 异常值数（落在 `[Q1-1.5·IQR, Q3+1.5·IQR]` 之外）。

> **为什么不让 LLM 算？** LLM 看到 200 个浮点数算均值会错。这一步必须用代码。

### 5.3 `ask_llm_for_strategy(client, model, q)` —— 论文里的 `π`

把 `Q` 序列化成 JSON 塞进 prompt，要求 LLM **只输出 JSON**，字段固定为 `missing_strategy`、`outlier_strategy`、`reason`。

关键技巧三个：

1. **System prompt 定角色**：「你是时序数据预处理专家，只输出 JSON」。
2. **`response_format={"type": "json_object"}`**：很多 OpenAI 兼容服务支持，强制输出合法 JSON，不再需要正则去抠。如果你换的服务不支持，删掉这一行也能跑（解析时再加 try）。
3. **枚举允许的取值**：在 prompt 里明确写出 `missing_strategy ∈ {linear_interpolation, ffill, bfill, mean, drop}`，避免 LLM 自创新值。
4. **`temperature=0.2`**：决策类任务调低温度，输出更稳定可复现。

### 5.4 `apply_strategy(series, strategy)` —— 论文里的 `D → D̃`

按 LLM 给的策略，用 pandas 实际做插值 / 裁剪。这一步又回到确定性代码。注意 `clip_iqr` 用 `Series.clip` 做的是**裁剪**（保留行数），`drop` 才是真删。

### 5.5 主流程

`Q = compute_quality_vector(...)` → `π = ask_llm_for_strategy(Q)` → `D̃ = apply_strategy(D, π)`，对应论文 Curator 第 1 子步。

---

## 6. 这一步的"Agent 味"在哪里？

很淡，但已经有雏形：

- **角色化**：System prompt 让 LLM 扮演一个"数据预处理专家"，而不是通用聊天机器人。
- **工具调用的雏形**：Python 函数 `compute_quality_vector` 是工具，LLM 不直接处理原始序列，只看工具结果。
- **结构化交接**：LLM 输出 JSON，下游代码可机器解析——这是多 Agent 系统能拼起来的前提。

下一版（demo 第 2 步）我们会加：

1. **多轮**：让 LLM 在看到第一次预处理结果后，再决定要不要二次清洗（"反思"机制）；
2. **可视化工具**：把 STL 分解图喂给多模态 LLM，做论文第 2、3 步；
3. **状态对象**：用一个 `CuratorState` 类把 `Q, V, A` 都装起来，准备和 Planner Agent 对接。

---

## 7. 学习自查

跑完后试着回答：

1. 如果把 `01_curator_minimal.py` 里的 LLM 调用整段删掉，剩下的代码还能不能完成预处理？能的话，加上 LLM 解决了什么原本没解决的问题？
2. 如果数据只有 30 个点，IQR 算出的 `outlier_count_iqr` 还可信吗？这正是 TSci 论文末尾说的"置信度缺失"问题——你已经站在研究切入点上了。
3. 试着把 `model` 改成 `Qwen/Qwen2.5-72B-Instruct`（如果额度允许），观察 `reason` 字段质量的变化。

跑通后告诉我，我们进入 demo 第 2 步：加可视化 + 多模态视觉推理。

---

## 8. 常见报错排查

| 报错 | 原因 | 解法 |
|---|---|---|
| `RuntimeError: 请在 .env 里设置 SILICONFLOW_API_KEY` | 没把 `.env.example` 复制成 `.env`，或 key 没填 | `cp .env.example .env` 后编辑 |
| `openai.BadRequestError: response_format` | 当前 provider/模型不支持 JSON mode | 删掉 `response_format` 那行；或换支持的模型 |
| `json.JSONDecodeError` | LLM 没乖乖输出 JSON（小模型常见） | 换更大模型，比如 `Qwen/Qwen2.5-32B-Instruct` |
| `ConnectionError` (Ollama) | 本地没启 ollama 服务 | `ollama serve` 后再 `ollama pull qwen2.5:7b` |
