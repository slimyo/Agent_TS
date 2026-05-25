"""B4 · TSci (Zhao et al., arXiv 2510.01538) 适配层。

策略：把 (train, val) 拼成临时 csv，让 TSci 自己切 1 个 slice 跑全流程，
然后从 ensemble_predictions 取 H 步预测。

LLM 走 zhipu glm-4-flash-250414（非 reasoning，与 LangChain ChatOpenAI 兼容）。
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_TSCI_ROOT = Path(__file__).resolve().parents[1] / "external" / "TimeSeriesScientist"
_TSCI_PKG = _TSCI_ROOT / "time_series_agent"
_PATCHED = False


def _ensure_patched():
    """惰性初始化：加 sys.path、env、monkey-patch、重定向 LLM 到 zhipu。"""
    global _PATCHED
    if _PATCHED:
        return
    # 1) LLM env
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / "demo" / ".env")
    os.environ["OPENAI_API_KEY"] = os.environ["ZHIPU_API_KEY"]
    os.environ["OPENAI_BASE_URL"] = "https://open.bigmodel.cn/api/paas/v4/"
    # 2) sys.path
    if str(_TSCI_PKG) not in sys.path:
        sys.path.insert(0, str(_TSCI_PKG))
    # 3) monkey-patch: graph 漏传 output_dir / validation_data
    from agents.preprocess_agent import PreprocessAgent
    _orig_pre = PreprocessAgent.run

    def _patched_pre(self, data, output_dir=None):
        if output_dir is None:
            output_dir = self.config.get("output_dir", "/tmp/tsci_out")
        return _orig_pre(self, data, output_dir)
    PreprocessAgent.run = _patched_pre

    from graph.agent_graph import TimeSeriesAgentGraph

    def _patched_fc(self, state):
        result = self.forecast_agent.run(
            state["selected_models"],
            state["best_hyperparameters"],
            state["validation_data"],
            state["test_data"],
            output_dir=self.config.get("output_dir", "/tmp/tsci_out"),
        )
        state["forecast_result"] = result
        return state
    TimeSeriesAgentGraph._forecast_node = _patched_fc
    _PATCHED = True


def predict(train: np.ndarray, val: np.ndarray, H: int,
            seed: int = 42, season_m: int = 1, **_) -> np.ndarray:
    _ensure_patched()
    from config.default_config import DEFAULT_CONFIG
    from graph.agent_graph import TimeSeriesAgentGraph

    # 准备临时 csv：date + OT 两列；TSci 自带切片器需要 date 列
    series = np.concatenate([train, val, np.zeros(H)])  # test 段用 0 占位（TSci 切片需要）
    n = len(series)
    dates = pd.date_range("2020-01-01", periods=n, freq="h")
    out_dir = tempfile.mkdtemp(prefix="tsci_")
    csv_path = Path(out_dir) / "input.csv"
    pd.DataFrame({"date": dates, "OT": series}).to_csv(csv_path, index=False)

    cfg = DEFAULT_CONFIG.copy()
    cfg.update({
        "llm_model": "glm-4-flash-250414",
        "num_slices": 1,
        "input_length": len(train) + len(val),
        "horizon": H,
        "data_path": str(csv_path),
        "date_column": "date",
        "value_column": "OT",
        "output_dir": out_dir,
        "debug": False,
        "verbose": False,
    })

    g = TimeSeriesAgentGraph(config=cfg, model=cfg["llm_model"], debug=False)
    res = g.run()
    # res["aggregated_results"]["ensemble_predictions"] 是个 list[dict]，取首 slice
    agg = res["aggregated_results"]
    ens = agg["ensemble_predictions"]
    if isinstance(ens, list) and ens:
        preds = ens[0].get("predictions") or ens[0].get("simple_average")
    elif isinstance(ens, dict):
        preds = ens.get("predictions") or ens.get("simple_average")
    else:
        raise RuntimeError(f"unexpected ensemble_predictions shape: {type(ens)}")
    y_hat = np.asarray(preds, dtype=np.float64)
    if len(y_hat) != H:
        # 长度不符则前 H 步截断或 last-value 填充
        if len(y_hat) > H:
            y_hat = y_hat[:H]
        else:
            y_hat = np.concatenate([y_hat, np.full(H - len(y_hat), y_hat[-1])])
    return y_hat
