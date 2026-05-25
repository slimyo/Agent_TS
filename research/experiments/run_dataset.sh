#!/usr/bin/env bash
# 单数据集全套基线 + AdaptTS sweep（参数化版）
# 用法: bash run_dataset.sh <DATASET>
# 例如: bash run_dataset.sh ECL  /  bash run_dataset.sh Exchange
set -uo pipefail
cd "$(dirname "$0")/../.."

DS="${1:?need dataset name}"
DSL=$(echo "$DS" | tr '[:upper:]' '[:lower:]')
OUT_FAST=research/results/p5_${DSL}_fast.jsonl
OUT_LLM=research/results/p5_${DSL}_llm.jsonl
OUT_TSCI=research/results/p5_${DSL}_tsci.jsonl
OUT_ADAPT=research/results/p5_${DSL}_adapt.jsonl
LOG=research/results/p5_${DSL}.log

mkdir -p research/results
echo "[$DS] start at $(date)" | tee -a "$LOG"

for N in 10 20 50 100; do
  for M in naive arima_ets chronos; do
    mamba run -n tsci python -m research.experiments.runner \
        --dataset "$DS" --N "$N" --H 96 \
        --methods "$M" --seeds 1,42,123 --out "$OUT_FAST" 2>&1 \
    | tee -a "$LOG" | grep -E '"method"|wrote' || true
  done
done

for N in 10 20 50 100; do
  for S in 1 42 123; do
    mamba run -n tsci python -m research.experiments.runner \
        --dataset "$DS" --N "$N" --H 96 \
        --methods llmtime --seeds "$S" --out "$OUT_LLM" 2>&1 \
    | tee -a "$LOG" | grep -E '"method"|wrote' || true
  done
done

for N in 10 20 50 100; do
  for S in 1 42 123; do
    mamba run -n tsci python -m research.experiments.runner \
        --dataset "$DS" --N "$N" --H 96 \
        --methods tsci --seeds "$S" --out "$OUT_TSCI" 2>&1 \
    | tee -a "$LOG" | grep -E '"method"|wrote' || true
  done
done

for N in 10 20 50 100; do
  mamba run -n tsci python -m research.experiments.runner \
      --dataset "$DS" --N "$N" --H 96 \
      --methods adapt_ts --seeds 1,42,123 --out "$OUT_ADAPT" 2>&1 \
  | tee -a "$LOG" | grep -E '"method"|wrote' || true
done

echo "[$DS] done at $(date)" | tee -a "$LOG"
for f in "$OUT_FAST" "$OUT_LLM" "$OUT_TSCI" "$OUT_ADAPT"; do
  wc -l "$f" 2>/dev/null | tee -a "$LOG"
done
