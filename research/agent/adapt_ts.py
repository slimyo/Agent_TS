"""Phase 3.4 · AdaptTS-Agent 统一入口（plan §一三层 + 记忆延后）。

接口与所有 baseline 一致：predict(train, val, H, seed, season_m) -> np.ndarray
内部组合：
  1) Curator UQ：诊断 + 三路置信度
  2) Adaptive Planner：置信度→策略组合
  3) Forecaster + Reflect：跑、评估、反思（≤3 次）

层四（记忆）后置到 Phase 3.5；本文件保留 hook（trace 输出可用作记忆库的输入）。
"""
from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from research.agent.curator_uq import diagnose
from research.agent.forecaster_reflect import forecast_with_reflection
from research.agent.memory import Case, Memory, case_features


def predict(train: np.ndarray, val: np.ndarray, H: int,
            seed: int = 42, season_m: int = 1, **kwargs) -> np.ndarray:
    """AdaptTS-Full 主入口。"""
    # 层一：诊断
    diag = diagnose(train, season_m=season_m)
    # 层二+三：自适应选择 + 反思（forecast_with_reflection 内部完成）
    y_hat, trace = forecast_with_reflection(
        train=train, val=val, H=H,
        diag=diag, season_m=season_m,
        max_reflect=kwargs.get("max_reflect", 3),
        val_mae_threshold=kwargs.get("val_mae_threshold", None),
        conf_source=kwargs.get("conf_source", "xc"),
        use_walk_forward=kwargs.get("use_walk_forward", True),
        use_model_cards=kwargs.get("use_model_cards", True),
        allow_diagnosis_revision=kwargs.get("allow_diagnosis_revision", True),
        enable_promotion=kwargs.get("enable_promotion", False),
        promotion_improve_frac=kwargs.get("promotion_improve_frac", 0.30),
    )
    # trace 暴露在模块级（runner 可选 dump）
    predict.last_trace = trace  # type: ignore[attr-defined]

    # 记忆写入（事后）：MEMORY_PATH 环境变量启用，未设则跳过，保持当前跑分不变
    mem_path = os.environ.get("ADAPTTS_MEMORY_PATH")
    if mem_path:
        mem = Memory(mem_path, k_cap=int(os.environ.get("ADAPTTS_MEMORY_CAP", "1000")))
        feat = case_features(diag)
        mem.add(Case(
            feature=feat.tolist(),
            diag=asdict(diag),
            final_plan={"strategies": trace.final_plan.strategies,
                        "combine": trace.final_plan.combine,
                        "weights": list(trace.final_plan.weights)},
            test_mae=None,   # 由 backfill_test_mae 在 test 评估后回填
            meta=dict(kwargs.get("meta", {})),
        ))
        # 模块级保存 ref，便于 backfill
        predict._last_mem = mem  # type: ignore[attr-defined]
    return y_hat


def backfill_test_mae(test_mae: float) -> bool:
    """v11 闭环：runner 在算完 test metrics 后调用此函数回填刚写入的 case test_mae。"""
    mem = getattr(predict, "_last_mem", None)
    if mem is None:
        return False
    return mem.update_last_test_mae(test_mae)
