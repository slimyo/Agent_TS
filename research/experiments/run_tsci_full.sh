#!/usr/bin/env bash
# Phase 4 · TSci 全量跑：ETTh1 × N∈{10,20,50,100} × 3 seeds
# 后台运行，预计 ~5 小时（zhipu glm-4-flash-250414，单 cell ~25 min）。
# 失败的 cell 单独打印；jsonl 持续 append，断点可重启。

set -uo pipefail
cd "$(dirname "$0")/../.."

OUT=research/results/p4_tsci_etth1.jsonl
LOG=research/results/p4_tsci_etth1.log
DATASET=ETTh1
H=96

mkdir -p research/results
echo "[run_tsci_full] start at $(date)" | tee -a "$LOG"

for N in 10 20 50 100; do
    for S in 1 42 123; do
        echo "=== $(date '+%H:%M:%S') N=$N seed=$S ===" | tee -a "$LOG"
        mamba run -n tsci python -m research.experiments.runner \
            --dataset "$DATASET" --N "$N" --H "$H" \
            --methods tsci --seeds "$S" --out "$OUT" 2>&1 \
        | tee -a "$LOG" \
        | grep -E '"method"|wrote|Error|Traceback' || true
    done
done

echo "[run_tsci_full] finished at $(date)" | tee -a "$LOG"
wc -l "$OUT" | tee -a "$LOG"
