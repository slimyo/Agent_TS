#!/usr/bin/env bash
# task #41 autopilot · 后台串行执行：等 v2 → build mem v2 → B7v3 → 额外实验
set -uo pipefail
cd "$(dirname "$0")/../.."
source /root/Downloads/ENTER/etc/profile.d/conda.sh
conda activate tsci
export HF_HUB_OFFLINE=1

LOG=research/results/autopilot_phase64b.log
echo "=== AUTOPILOT START $(date) ===" | tee -a "$LOG"

# Step 1: Wait for v2 sweep (taskb_router_v2_sweep) to reach 30 rows
echo "[STEP 1] wait v2 sweep done..." | tee -a "$LOG"
while true; do
    LINES=$(wc -l < research/results/taskb_router_v2_ucr.jsonl 2>/dev/null || echo 0)
    if [ "$LINES" -ge 30 ]; then
        echo "  v2 done ($LINES rows)" | tee -a "$LOG"
        break
    fi
    sleep 30
done

# Step 2: Build enhanced memory bank v2
echo "[STEP 2] build_clf_memory_v2 ..." | tee -a "$LOG"
python -m research.experiments.build_clf_memory_v2 2>&1 | tail -30 | tee -a "$LOG"

# Step 3: Run B7v3 sweep (with enhanced memory + weighted vote + Cards v2 + N-fallback)
echo "[STEP 3] B7v3 sweep ..." | tee -a "$LOG"
python -m research.experiments.taskb_router_v3_sweep 2>&1 | tee -a "$LOG"

# Step 4: Run RCA v3 with extended Curator features (re-runs cached)
echo "[STEP 4] re-run RCA v3 with v2 Curator (cached, fast verify)..." | tee -a "$LOG"
python -m research.experiments.taska_run_rca 2>&1 | tail -10 | tee -a "$LOG"
python -m research.experiments.taska_eval 2>&1 | tee -a "$LOG"

# Step 5: Aggregate B7v3 summary
echo "[STEP 5] aggregate B7v3 vs v1 vs Rocket vs Oracle ..." | tee -a "$LOG"
python -c "
import json
from collections import defaultdict, Counter

def load(p):
    try: return [json.loads(l) for l in open(p)]
    except FileNotFoundError: return []

v3 = load('research/results/taskb_router_v3_ucr.jsonl')
v2 = load('research/results/taskb_router_v2_ucr.jsonl')
v1 = load('research/results/taskb_router_ucr.jsonl')
all_b = load('research/results/taskb_ucr.jsonl')

# Oracle and Rocket aligned
oracle = {}
rocket = {}
for r in all_b:
    k = (r['dataset'], r['N_per_class'], r['seed'])
    oracle[k] = max(oracle.get(k, -1), r['acc'])
    if r['method'] == 'B3_rocket': rocket[k] = r['acc']

def agg(rows):
    if not rows: return None
    accs = [r['acc'] for r in rows]
    return sum(accs)/len(accs), len(accs)

print('\n=== Summary ===')
for name, rows in [('B7v3', v3), ('B7v2', v2), ('B7v1', v1)]:
    if rows:
        m, n = agg(rows)
        print(f'  {name}: mean={m:.4f} n={n}')

if rocket:
    print(f'  Rocket alone: mean={sum(rocket.values())/len(rocket):.4f}')
if oracle:
    print(f'  Oracle:       mean={sum(oracle.values())/len(oracle):.4f}')

# v3 routing distribution
if v3:
    chosen_dist = Counter(r['chosen_classifier'] for r in v3)
    print(f'  v3 routing: {dict(chosen_dist)}')
    mem_overrides = sum(1 for r in v3 if r.get('mem_winner'))
    print(f'  v3 memory overrides triggered: {mem_overrides}/{len(v3)}')
" 2>&1 | tee -a "$LOG"

echo "=== AUTOPILOT END $(date) ===" | tee -a "$LOG"
