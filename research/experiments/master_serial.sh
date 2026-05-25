#!/usr/bin/env bash
# 严格串行 master：避免 CPU 竞争 + 进程混乱
set -uo pipefail
cd "$(dirname "$0")/../.."

echo "[$(date +%H:%M)] === master serial start ==="

# 1) v6 ETTh1 N=100 剩余 3 cell
echo "[$(date +%H:%M)] v6 ETTh1 N=100 ..."
mamba run -n tsci python -m research.experiments.runner \
    --dataset ETTh1 --N 100 --H 96 --methods adapt_ts_v6 \
    --seeds 1,42,123 --out research/results/p6_adapt_v6_etth1.jsonl 2>&1 \
    | grep -E '"method"|wrote' | tail -4 || true

echo "[$(date +%H:%M)] v6 done: $(wc -l < research/results/p6_adapt_v6_etth1.jsonl) rows"

# 2) F2 cross-LLM resume（去重后剩 ~46 cells）
echo "[$(date +%H:%M)] F2 cross-LLM resume ..."
mamba run -n tsci python -u -m research.experiments.f2_cross_llm 2>&1 \
    | grep -E "MAE=|done|FAIL|skip" | tail -50 || true

echo "[$(date +%H:%M)] F2 done: $(wc -l < research/results/f2_cross_llm.jsonl) rows"
echo "[$(date +%H:%M)] === master serial finished ==="
