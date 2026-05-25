# research/ · AdaptTS-Agent 论文实验

按 `plan.md` 五组实验（E1~E5）执行。本目录与 `../demo/` 的关系：demo 是"教学版最小 4-Agent 流水线"，research 是"按论文规范、可对比可复现的实验工程"。

## 目录结构

```
research
├── plan.md            # 实验总规划（不要改，所有决策的源头）
├── TODO.md            # 可执行清单 / 进度看板
├── README.md          # 本文件
├── utils/             # 数据加载 / 切割 / 指标
├── baseline/          # B1~B5 基线封装（统一接口 predict(train, val, H) -> y_hat）
├── agent/             # AdaptTS-Agent 实现（复用 demo 的 4 个节点 + 增加四层机制）
├── experiments/       # runner 主入口 + 各实验脚本
├── datasets/raw/      # 原始数据缓存（git ignore）
└── results/           # 实验结果 jsonl + 图表
```

## 快速开始

```bash
mamba activate tsci
cd /home/hz/code/agent_ts

# Phase 0 验收命令（跑通 Naive 基线 → 产出 results/p0_naive.jsonl）
python -m research.experiments.runner \
    --dataset ETTh1 --N 20 --H 96 \
    --methods naive --seeds 1,42,123
```

## 基线统一接口

所有方法（baseline + agent）实现同一函数签名，便于 runner 平等调用：

```python
def predict(train: np.ndarray,
            val:   np.ndarray,
            H:     int,
            seed:  int = 42,
            **kwargs) -> np.ndarray:
    """返回长度为 H 的预测序列。"""
```

## 与 demo 的复用关系

- demo 里 `03_planner.py` / `04_forecaster.py` 的统计模型库（ARIMA / HoltWinters / Ridge）→ Phase 1 包成 baseline.B2
- demo 里 LangGraph 流水线 → Phase 3 改造为 AdaptTS-Agent（加置信度 + 反思 + 记忆）
