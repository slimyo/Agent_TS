#!/usr/bin/env bash
# Phase 4.2 · ETTh2 全套基线 + AdaptTS
# 6 方法 × 4 N × 3 seeds = 72 cell；naive/arima/chronos 快；llmtime/tsci/adapt 含 LLM
set -uo pipefail
cd "$(dirname "$0")/../.."

OUT_FAST=research/results/p4_etth2_fast.jsonl
OUT_LLM=research/results/p4_etth2_llm.jsonl
OUT_TSCI=research/results/p4_etth2_tsci.jsonl
OUT_ADAPT=research/results/p4_etth2_adapt.jsonl
LOG=research/results/p4_etth2.log

mkdir -p research/results
echo "[etth2] start at $(date)" | tee -a "$LOG"

# 阶段 1：快速基线（naive/arima/chronos）
for N in 10 20 50 100; do
  for M in naive arima_ets chronos; do
    echo "=== $(date '+%H:%M:%S') $M N=$N ===" | tee -a "$LOG"
    mamba run -n tsci python -m research.experiments.runner \
        --dataset ETTh2 --N "$N" --H 96 \
        --methods "$M" --seeds 1,42,123 --out "$OUT_FAST" 2>&1 \
    | tee -a "$LOG" | grep -E '"method"|wrote' || true
  done
done

# 阶段 2：LLMTime（慢，含缓存）
for N in 10 20 50 100; do
  for S in 1 42 123; do
    echo "=== $(date '+%H:%M:%S') llmtime N=$N seed=$S ===" | tee -a "$LOG"
    mamba run -n tsci python -m research.experiments.runner \
        --dataset ETTh2 --N "$N" --H 96 \
        --methods llmtime --seeds "$S" --out "$OUT_LLM" 2>&1 \
    | tee -a "$LOG" | grep -E '"method"|wrote' || true
  done
done

# 阶段 3：TSci（每 cell ~70s 已有 cache）
for N in 10 20 50 100; do
  for S in 1 42 123; do
    echo "=== $(date '+%H:%M:%S') tsci N=$N seed=$S ===" | tee -a "$LOG"
    mamba run -n tsci python -m research.experiments.runner \
        --dataset ETTh2 --N "$N" --H 96 \
        --methods tsci --seeds "$S" --out "$OUT_TSCI" 2>&1 \
    | tee -a "$LOG" | grep -E '"method"|wrote' || true
  done
done

# 阶段 4：AdaptTS v3
for N in 10 20 50 100; do
  echo "=== $(date '+%H:%M:%S') adapt_ts N=$N ===" | tee -a "$LOG"
  mamba run -n tsci python -m research.experiments.runner \
      --dataset ETTh2 --N "$N" --H 96 \
      --methods adapt_ts --seeds 1,42,123 --out "$OUT_ADAPT" 2>&1 \
  | tee -a "$LOG" | grep -E '"method"|wrote' || true
done

echo "[etth2] done at $(date)" | tee -a "$LOG"
for f in "$OUT_FAST" "$OUT_LLM" "$OUT_TSCI" "$OUT_ADAPT"; do
  wc -l "$f" 2>/dev/null | tee -a "$LOG"
done
