# TODO · 项目进度看板

> **每次回到这个项目，只需看这一个文件**。
> 上半部分是 Claude 已经做完的事，下半部分是你（用户）需要执行的命令。

---

## ✅ Claude 已完成（截至 2026-04-29）

### 文档
- [x] `README.md` —— Step 1 详细讲解 + 项目路线图（顶部表格记录 Step 1-6 进度）
- [x] `00_what_is_an_agent.md` —— **入门讲义**：什么是 Agent，5 个组成模块，6 问打卡清单（建议先读）
- [x] `TODO.md` —— 本文件，项目看板
- [x] `env.md` —— 环境/代理配置笔记（mamba + WSL 代理 + NO_PROXY 白名单）
- [x] `02_step2_explainer.md` —— Step 2 详解：为什么要画图、6 子图含义、多模态 prompt 工程、状态对象
- [x] `03_step3_explainer.md` —— Step 3 详解：模型库选型、时序切分、LLM 选模型 prompt、超参搜索预算
- [x] `04_step4_explainer.md` —— Step 4 详解：refit 的意义、三种集成策略、数据泄漏红线
- [x] `05_step5_explainer.md` —— Step 5 详解：LLM 只写散文 + 代码负责数字、limitations 的研究价值
- [x] `06_step6_explainer.md` —— Step 6 详解：LangGraph 三核心概念（State/Node/Edge）+ 后续扩展方向

### 代码
- [x] `01_curator_minimal.py` —— Curator Agent 最小可运行版本：算质量向量 Q → LLM 给清洗策略 π → 应用策略 ✅ 已跑通
- [x] `02_curator_visual.py` —— Curator 可视化版：合成 365 天序列 → 6 合 1 诊断图 → glm-4.6v-flash 看图 → 输出 A
- [x] `03_planner.py` —— Planner Agent：读 Curator 状态 → LLM 选 2-3 候选 → 小网格超参搜索 → 写 planner_state.json
- [x] `04_forecaster.py` —— Forecaster Agent：refit on train+val → LLM 选集成策略 → test 评估 → 写 forecaster_state.json
- [x] `05_reporter.py` —— Reporter Agent：聚合三份 state → LLM 写 5 段散文 → 代码组装 markdown 报告
- [x] `06_langgraph_pipeline.py` —— 用 LangGraph 把 4 节点串成有状态工作流（TypedDict + StateGraph）
- [x] `.env.example` —— 5 个 LLM provider 切换模板（含智谱 vision_model）
- [x] `requirements.txt` —— Step 1+2 全部依赖：openai/pandas/numpy/dotenv/httpx[socks]/matplotlib/statsmodels/scipy
- [x] `sample_data.csv` —— 40 行测试时序，含 3 个缺失值 + 2 个明显异常值

### 记忆系统（Claude 内部，不在仓库里）
- [x] 已落盘：项目目标 / 用户入门状态 / LLM 偏好（开源免费优先），下次新对话能直接拾起

---

## 🔧 你需要做的（按顺序）

### 步骤 A · 创建 conda/mamba 虚拟环境（已完成 ✅）

环境 `tsci` 已创建。下次新开终端记得先 `mamba activate tsci`。

<details>
<summary>原始命令（备查）</summary>

> **为什么要虚拟环境？** 你系统里可能装了别的项目的 pandas / openai，版本可能冲突。独立环境给本项目一个沙盒，删了不影响系统。
>
> **本项目环境管理工具：mamba**（你的偏好）。系统已检测到 `mamba 2.5.0` + `conda 26.3.2`，直接可用。
>
> **当前状态**：环境 `tsci` 还没创建。

```bash
# 1. 创建一个名为 tsci 的环境（Python 3.10，跟系统对齐）
mamba create -n tsci python=3.10 -y

# 2. 激活（每次新开终端都要重新激活）
mamba activate tsci
# 激活成功后命令行最前面会出现 (tsci) 前缀

# 3. 进入项目目录
cd /home/hz/code/agent_ts/demo

# 4. 安装项目依赖（这几个包用 pip 装即可，conda-forge 也有但 pip 更快）
pip install -r requirements.txt

# 5. 验证
python -c "import openai, pandas, numpy, dotenv; print('OK')"
```

**常用操作**：
- 退出环境：`mamba deactivate`
- 列出所有环境：`mamba env list`
- 删除本环境（重置时用）：`mamba env remove -n tsci`
- 检查是否在 tsci 里：`which python` —— 输出含 `envs/tsci/bin/python` 即对
- VSCode：`Ctrl+Shift+P` → `Python: Select Interpreter` → 选 `envs/tsci/bin/python`

**为什么不用 `conda install` 而用 `pip install`？** 这几个依赖（openai/pandas/numpy/python-dotenv）在 PyPI 上版本更新更快，conda-forge 偶尔滞后。conda 环境里混用 pip 是被官方支持的常见做法，只要不在同一个包来回切换就行。

**后续 step 如果引入需要 C 扩展或系统库的包**（比如 `statsmodels`、`scikit-learn`、`graphviz`），优先用 `mamba install -c conda-forge <pkg>`，二进制依赖处理得比 pip 干净。

</details>

---

### 步骤 B · 配置 LLM API key（已完成 ✅）

`.env` 已为你写好，使用智谱 **GLM-4.7-Flash**（完全免费），key 已填入。

如想换其它 provider：
- 智谱 → SiliconFlow / DashScope / DeepSeek / Ollama
- 改 `.env` 里的 `PROVIDER=zhipu` 为目标名字 + 取消对应 key 行的注释，**代码不用改**。

---

### 步骤 C · 跑通 Step 1（已完成 ✅，2026-04-29）

输出验证：缺失率 7.3% → LLM 选 `linear_interpolation`，2 个异常值 → 选 `clip_iqr`，清洗后 `min=1.025, max=34.025, 缺失=0`，符合预期。

---

### 步骤 D · 跑通 Step 2（已完成 ✅，2026-04-29）

输出验证：
- 365 天合成序列 → 1.1% 缺失 → 线性插值 + IQR 截断
- VLM (`glm-4.6v-flash`) 判断 trend=increasing / seasonality=yes / **period=7** ✅ / non_stationary
- `reason` 正确引用子图 B/C/A 作为证据
- 产物：`curator_panel.png`（126 KB）+ `curator_state.json`

---

### 步骤 E · 跑通 Step 3（已完成 ✅，2026-04-29）

输出验证：
- LLM 选 `holt_winters` + `arima`，理由都正确引用了 Q/A 字段
- 超参搜索：HW 最优 `{trend:add, damped:False}` val_MAPE=**2.573**；ARIMA 最优 `(2,1,2)` val_MAPE=**3.903**
- 产物：`planner_state.json`

---

### 步骤 F · 跑通 Step 4（已完成 ✅，2026-04-29）

输出验证：
- LLM 选 `single_best`，理由："HW (2.5727) 与 ARIMA (3.9032) 差距 51.65% > 30%" —— 30% 规则被正确执行
- 集成权重：HW=1.0 / ARIMA=0.0
- test MAPE：**HW=2.159**（比 val 2.573 还低，泛化好）/ ARIMA=5.546（比 val 恶化）/ Ensemble=2.159
- 产物：`forecaster_panel.png` + `forecaster_state.json`

**亮点**：LLM 的 single_best 判断被 test 验证为正确——若用 perf_weighted 把 ARIMA 拉进来反而会让 ensemble 变差。

---

### 步骤 G · 跑通 Step 5（已完成 ✅，2026-04-29）

`final_report.md` 已生成（用 VSCode `Ctrl+Shift+V` 预览）。

---

### 步骤 H · 跑通 Step 6 👈 **现在做这步（最后一步）**

**先装 LangGraph**：

```bash
mamba activate tsci
pip install langgraph
# 或: mamba install -c conda-forge langgraph -y（如版本不全，回退到 pip）
```

**再跑**：

```bash
cd /home/hz/code/agent_ts/demo
python 06_langgraph_pipeline.py
```

**预期产物**：
- 4 段节点执行日志 + 总览
- 同 step 1-5 的 png/json/md 文件被重新生成
- 可选：`pipeline_graph.png`（装了 graphviz 时）

**自查**（详见 `06_step6_explainer.md` §6）：
1. 4 个节点按 curator → planner → forecaster → reporter 顺序执行了吗？
2. 最终 `ensemble_test_mape` 和你之前跑 step 4 拿到的 ~2.16 是否一致？
3. 看完 `06_step6_explainer.md` §5，想想后续要不要做"置信度传播"——这是你的研究切入点。

跑通后整个 demo 完结。

---

### 步骤 D · （可选）阅读理解

跑代码前/后，建议读：
1. `00_what_is_an_agent.md` —— 理解 Agent 抽象
2. `README.md` 第 5 节 "代码逐段讲解" —— 对照源码读
3. 仓库根目录 `TimeSeriesScientist（TSci）.md` 第二节 —— 看 Curator 在论文里的定位

完成 Step 1 后回答 README §7 的 3 道自查题，能答上就可以推进。

---

## 🐛 遇到问题排查

| 现象 | 可能原因 | 处理 |
|---|---|---|
| `mamba: command not found` | shell 没初始化 | `source ~/Downloads/ENTER/etc/profile.d/conda.sh && conda activate base` |
| `mamba activate` 报 `CommandNotFoundError` | shell hook 没装 | `mamba init bash` 然后重开终端 |
| `pip install` 卡很慢 | 国内网络 | 加镜像：`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple` |
| `RuntimeError: 请在 .env 里设置 ...` | key 没填 / .env 没创建 | 重做步骤 B |
| `openai.AuthenticationError` | key 错了 | 检查 SiliconFlow 控制台 |
| `json.JSONDecodeError` | 小模型没乖乖输出 JSON | 把 `.env` 加一行 `MODEL=Qwen/Qwen2.5-32B-Instruct` |

更多见 `README.md` §8。

---

## 🗺️ 全局路线图（每个 step 完成后回来勾掉）

- [x] **Step 1** 最小 Curator（统计 + LLM 策略 + 清洗）—— `01_curator_minimal.py` ✅ 跑通
- [x] **Step 2** Curator 加可视化（STL / ACF / PACF）+ glm-4.6v-flash 看图 —— `02_curator_visual.py` ✅ 跑通
- [x] **Step 3** Planner Agent（模型库 + LLM 选模型 + 超参搜索）—— `03_planner.py` ✅ 跑通 (HW=2.57, ARIMA=3.90)
- [x] **Step 4** Forecaster Agent（top-k 集成 + 加权策略）—— `04_forecaster.py` ✅ 跑通 (HW single_best, test=2.16)
- [x] **Step 5** Reporter Agent（聚合产物输出报告）—— `05_reporter.py` ✅ 跑通
- [x] **Step 6** LangGraph 串成有状态工作流 —— `06_langgraph_pipeline.py` ⏳ 待你跑（最后一步）
- [ ] **Step 3** Planner Agent（模型库 + LLM 选模型 + 超参搜索）
- [ ] **Step 4** Forecaster Agent（top-k 集成 + 加权策略）
- [ ] **Step 5** Reporter Agent（聚合产物输出报告）
- [ ] **Step 6** 用 LangGraph 把四个 Agent 串成有状态工作流
