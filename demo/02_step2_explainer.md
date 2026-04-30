# Demo 第 2 步：Curator 加可视化 + 多模态 LLM 看图

> 本文档对应 `02_curator_visual.py`，是 Step 1 的延伸。
> 读完你会理解：(1) 为什么 Curator 要画图，(2) 多模态 LLM 怎么"读图"，(3) 状态对象 `C = {Q, V, A}` 是怎么拼起来的。

---

## 1. 这一步要做什么？

回忆 TSci 论文里 Curator 的三个子步骤：

| 子步骤 | 输出 | Step 1 | Step 2 |
|---|---|---|---|
| ① 质量诊断 + 预处理 | `Q, π, D̃` | ✅ | ✅（复用） |
| ② 多模态可视化生成 | `V` |  | ✅ |
| ③ 时序结构画像 | `A = {trend, seasonality, stationarity}` |  | ✅ |

**Step 2 的关键能力**：让 LLM "看图说话"——不再只看统计数字，而是看完整的 STL 分解图、ACF/PACF 图后，给出对趋势/季节性/平稳性的判断。

最终我们组装出共享状态对象 `C = {Q, V, A}`，下游 Planner 可以直接消费。

---

## 2. 为什么要画图给 LLM 看？

你可能问：既然有 Q（统计向量），为什么还要画图？

**统计量是有损压缩，图是无损呈现**：
- 数字 `trend_slope=0.05` 告诉你"有正趋势"，但不告诉你**趋势是稳定线性还是分段斜率突变**；
- 数字 `outlier_count=2` 告诉你"有 2 个异常值"，但不告诉你**它们是孤立尖峰还是连续段偏离**；
- ACF 的 7 阶峰值是数字，但**多个 lag 的相对衰减形态**只有图能直观传达。

多模态 LLM 的优势：它可以一次"扫"完整张图，做整体判断，远比把数字一个个塞进 prompt 高效。

> 这也是 TSci 消融实验显示**去掉数据分析模块 MAE 上升 28.3%** 的原因。

---

## 3. 6 个子图都画了什么？

`plot_curator_panel()` 生成一张 PNG，含 6 个子图：

| 子图 | 内容 | 看什么 |
|---|---|---|
| **A 总览** | 原始 + 14 天滚动均值 + 滚动标准差 | 整体形态、是否同方差 |
| **B STL trend** | 趋势分量 | 单调？拐点？分段？ |
| **C STL seasonal** | 季节分量（period=7） | 周期性是否稳定 |
| **D STL residual** | 残差 | 是否近似白噪声 |
| **E ACF** | 自相关 | 长记忆？周期峰？ |
| **F PACF** | 偏自相关 | AR 阶数线索 |

**为什么是 STL 不是经典 `seasonal_decompose`？** STL（Seasonal-Trend decomposition using LOESS）对异常值更鲁棒、能处理变化的季节性振幅。`robust=True` 让它再加一层迭代加权。

---

## 4. 多模态 LLM 调用：图怎么塞进去？

OpenAI 兼容协议规定多模态消息是这样的：

```python
messages = [
    {"role": "system", "content": "你是..."},
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0..."}},
            {"type": "text", "text": "请基于这张图给出 JSON..."},
        ],
    },
]
```

关键点：
- `content` 从字符串变成**数组**，每项是一个"内容块"（content block）；
- `image_url` 既可以是公网 URL（要求模型方能访问），也可以是 base64 data URL（更稳，所有模型都吃）；
- 文本块和图像块可以**任意顺序、多张图混排**。

我们用 base64 data URL 是因为：
1. 不依赖外网；
2. 不暴露用户图片；
3. 智谱、SiliconFlow 的 VLM 都支持。

代价是 token 消耗会比传 URL 多（base64 比二进制大约 33%），但 demo 场景无所谓。

---

## 5. 提示词工程：怎么逼 VLM 输出严格 JSON？

视觉模型比纯文本模型**更爱絮叨**——动不动就给一段散文分析。我们的 prompt 用了三招：

1. **System 里逐图列清单**：明确告诉它每个子图（A-F）是什么，模型才知道怎么对应；
2. **Schema 全字段枚举**：所有字段值都给定枚举集（`"increasing"|"decreasing"|"none"`），并注明类型；
3. **`reason` 字段强制引用子图**：写 `"<引用具体子图作为证据，例如 '子图B显示...'>"`，把模型的注意力拉回图上，不让它编。

代码层还有两道兜底：
- 优先 `response_format={"type":"json_object"}`；视觉模型不支持时退回普通模式；
- 普通模式下 `json.loads` 失败时用正则抠 `{...}`。

---

## 6. 状态对象：第一次显式写出来

到 Step 1 时我们用 dict 隐式表示状态。Step 2 开始引入显式持久化：

```python
state = {
    "Q": q,
    "V_path": "curator_panel.png",
    "A": structure,
    "strategy_pi": strategy,
}
# 写入 curator_state.json
```

为什么不用 `dataclass`/`pydantic`？因为这一步还在"原型"阶段，dict + JSON 序列化最直白。Step 6 引入 LangGraph 时再升级成 `TypedDict`/`pydantic.BaseModel`。

> **关键设计原则**：状态对象要**可序列化**，所以 `V` 存的是 PNG 路径字符串，不是 `bytes` 或 `np.ndarray`。后续 Planner 读这个 JSON 时只需从 `V_path` 加载图就行。

---

## 7. 为什么这一步换了合成数据？

`sample_data.csv` 只有 41 个点：
- STL 至少要 `n > 2*period`，period=7 时只剩 5 个完整周期，分解很噪；
- ACF/PACF 在 < 50 点时置信带太宽，几乎没法判断显著性。

所以 Step 2 内置 `make_synthetic_series()` 生成 365 天合成序列：
- 线性趋势（斜率 0.05/天）
- 周季节性（振幅 3，周期 7 天）
- 高斯噪声（σ=0.6）
- 注入 4 个缺失 + 2 个明显异常

这正是 TSci 论文末尾提到的"小样本下置信度缺失"问题的反面教材——**先用充分数据让 demo 跑得漂亮，再回过头看小样本时为什么会出问题**，就是你的研究切入点。

---

## 8. 跑起来

```bash
mamba activate tsci
cd /home/hz/code/agent_ts/demo
mamba install -c conda-forge statsmodels matplotlib scipy -y   # 一次性
python 02_curator_visual.py
```

预期产物（同目录下）：
- `curator_panel.png`：6 合 1 诊断面板
- `curator_state.json`：完整 Curator 状态 `C = {Q, V_path, A, strategy_pi}`

**自查问题**：
1. VLM 给出的 `seasonal_period` 是 7 吗？为什么？
2. 把合成序列的 `seasonal` 振幅改成 0（在代码里），再跑一次，VLM 还能正确判断"无季节性"吗？
3. 如果你把 dpi 从 110 调到 50，图变模糊后 VLM 的 `reason` 质量是否下降？这就是"分辨率 vs token 成本"的 tradeoff。

跑通后告诉我，进入 Step 3：Planner Agent 接收 `curator_state.json`，做模型选型 + 超参搜索。
