# research/ · TSci-Demo Multi-Agent TS Routing

> **目标导向 README**。方法/框架 → `method.md`；实测日志 + 实验现象 + 可理论化 finding → `finish.md`（Phase 1-6）+ `finish-1.md`（Round 2-4 优化 + 实验现象记录）；外部 review feedback → `feedback.md`（每轮覆盖）。

---

## 项目目标

### 总目标（长期）

构建 **统一的时序基础模型自适应路由框架**（Universal Time-Series Adaptive Routing）：

> 给定任意时序任务（forecasting / classification / anomaly / imputation）和一组候选 base models（含 zero-shot TSFM + 经典 baseline），在**不重新训练**的前提下，**自动**根据 series 的 regime / domain / N / uncertainty 选出最优 model 或 model 子集组合，并在性能 / 推理成本 / 部署约束之间做 explicit trade-off。

为什么：当前 TSFM 数量爆炸（每月新发），但**没有任何单一模型 dominates**。固定模型 = 错过 niche；naive ensemble = 平均化 + 拖累 CRPS；hand-coded router = 维护爆炸。系统级答案 = **probabilistic adaptive routing**。

### 当前阶段目标（论文 v1）

聚焦 forecasting + TSC 双轨：

1. ✅ 完成 12 forecasting + 11 TSC baseline library 落地
2. ✅ 完成 4-层 prior 栈（static π_k + type + N + availability）
3. ✅ 完成 L0/L1/L2 分层 + BMA posterior + 反事实记忆 + ε-greedy
4. ✅ 完成 remote sweep 实测 4 个新 TSFM vs C2（5-way oracle = -5.24%）
5. ✅ 记录 11 个可理论化 finding（finish-1 §9）
6. 🔄 P1 risk-sensitive + P2 cost-aware 评估扩展（**当前**）
7. ⏳ Round 5: heuristic stack → Bayesian unification（不增机制，纯重 frame）
8. ⏳ Round 6: contextual bandit / Thompson Sampling 替 ε-greedy
9. ⏳ Round 7: dynamic MoE + learned routing representation

### 关键工程约束（不可妥协）

- **不增 hand heuristic**（feedback Round 4 警告）：新功能必须 frame 为 prior / posterior / risk 项
- **每个 finding 必须有可证伪命题**（finish-1 §9 草稿）
- **每个新机制必须可消融**（独立开关 + 默认关）
- **远程模型必须可被 prior 自动 down-weight**（不依赖人工 override）

---

## 不在范围内（避免 scope creep）

| 不做 | 为什么 |
|---|---|
| 实时 online learning loop | 当前是 batch sweep，转 online 需要 deployment loop |
| 训练自定义 TSFM | 我们只 route 现成 zero-shot 模型 |
| Multivariate full support | A2 已在 backlog，但 multivariate routing 是 ortho 维度 |
| 异常检测 / imputation 评估 | F11 之后可扩，但当前 paper 不写 |
| KairosHope / TabPFN-TS 等 blocked 模型 | 外部依赖未释放 |

---

## 路线图

| Round | 时间 | 关键产物 | 状态 |
|---|---|---|---|
| Phase 1-3 | 2026-Q1 | 4-model wrapper + N-fallback | ✅ finish §3.1.1-25 |
| Phase 4-5 | 2026-Q1 | Forecasting boundary 立 paper §4 | ✅ finish §3.1.26-28 |
| Phase 6 | 2026-Q2 | RCA / TSC / B7 router | ✅ finish §3.1.29-34 |
| Round 2 | 2026-Q2 | Gated Residual / Soft Router NEGATIVE | ✅ finish-1 §2 |
| Round 3 | 2026-Q2 | Library 扩 (TiRex / Toto / Mantis / WEASEL / remote) | ✅ finish-1 §3-4 |
| Round 4-A | 2026-05-27 | feedback Items 2-4 + L0/L1/L2 + quantile pool + ε-greedy | ✅ finish-1 §5-7 |
| Round 4-B | 2026-05-27 | 远程 sweep + 现象 finding + P1/P2 评估 | **🔄** |
| Round 5 | 2026-Q3 | Bayesian unification | ⏳ |
| Round 6 | 2026-Q3 | Contextual bandit | ⏳ |
| Round 7 | 2026-Q4 | Dynamic MoE + learned representation | ⏳ |

---

## 快速开始

```bash
mamba activate tsci
cd /home/hz/code/agent_ts

# Prior-aware 端到端 forecasting
ADAPTTS_PLANNER=prior_aware python -c "
from research.agent.forecaster_reflect import forecast_with_reflection
from research.agent.curator_uq import diagnose
from research.utils.data_loader import load_series
from research.utils.splitter import few_shot_split
series, meta = load_series('ECL')
sp = few_shot_split(series, N=100, H=96, seed=42)
diag = diagnose(sp.train, season_m=meta.season_m)
pred, trace = forecast_with_reflection(
    train=sp.train, val=sp.val, H=96, diag=diag,
    season_m=meta.season_m, dataset='ECL'
)
print(trace.final_plan)
"

# Risk-sensitive eval over current 4-way sweep
python -m research.utils.risk_metrics

# Cost-aware eval
python -m research.utils.cost_metrics
```

---

## 目录结构

```
research/
├── README.md              # 本文件：目标 + 路线图 + 范围
├── method.md              # 方法自顶向下技术文档（16 节）
├── plan.md / TODO.md      # 总规划 + 当前看板
├── finish.md              # Phase 1-6 实测日志
├── finish-1.md            # Round 2-4 优化 + 实验现象 + 理论 finding
├── feedback.md            # 当前 round 外部 review（每轮覆盖）
├── classifier.md          # TSC 分支独立设计
│
├── baseline/              # 12 forecasting + 11 TSC baselines
├── agent/                 # Curator / Planner / Memory / Quantile Ensemble
├── utils/                 # data / features / metrics / risk / cost
├── experiments/           # 各 sweep runner
├── scripts/               # 远程 sweep / smoke
└── results/               # *.jsonl 实测数据
```

---

## 文件读取指引

| 想知道 | 看哪里 |
|---|---|
| 我们在做什么、为什么、路线图 | 本 README |
| 系统怎么搭的、API / 流程 | `method.md` |
| 历史实测细节 / Phase 1-6 数据 | `finish.md` |
| 最近优化 + 远程 sweep + 可理论化现象 | `finish-1.md` |
| 当前 review 提出的改进点 | `feedback.md`（每轮覆盖） |
| 实验进度 / 任务看板 | `TODO.md` |
| 论文写作版本 | `paper_draft.md` |
