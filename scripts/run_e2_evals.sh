#!/usr/bin/env bash
# E2 evals, reordered for the 2:30 pm deadline (2026-11-09): format check and the pre-named primary (fixed slot)
# for every arm first, then one random control, then the extras (other random controls, bare-letter D1).
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="uv run --no-sync python"
DEV=cuda:0
step () { echo "STEP $* $(date +%T)"; }
EV=(--checkpoint meta_instruct --allow-tf32 --batch-size 32 --ignore-preflight-problems --anchors pruned
    --turns single --letter-spellings "{L}, {L}" --device $DEV)
wacc () { $PY scripts/check_written_accuracy.py --tag "$1_$2" --prune-indices "results/meta_instruct/btap/$1/$2.json" \
            --device $DEV > "logs/e2_wacc_$1_$2.log" 2>&1; step "wacc $1 $2: $(tail -1 logs/e2_wacc_$1_$2.log)"; }
prefix () { $PY scripts/run_sycophancy_eval.py "${EV[@]}" --answer-prefix "The correct answer is (" \
            --prune-indices "results/meta_instruct/btap/$1/$2.json" --out-dir "runs/e2_prefix_$1_$2" \
            --results-dir "results/meta_instruct/btap/e2_eval/prefix_$1_$2" > "logs/e2_prefix_$1_$2.log" 2>&1; step "prefix $1 $2"; }
bare () { $PY scripts/run_sycophancy_eval.py "${EV[@]}" \
            --prune-indices "results/meta_instruct/btap/$1/$2.json" --out-dir "runs/e2_bare_$1_$2" \
            --results-dir "results/meta_instruct/btap/e2_eval/bare_$1_$2" > "logs/e2_bare_$1_$2.log" 2>&1; step "bare $1 $2"; }
for arm in e2b_s0 e2c_s0 e2e_s0 e2d_s0 e2b_s1 e2c_s1 e2e_s1; do wacc $arm selected; done
wacc e2b_s0 random
for arm in e2b_s0 e2c_s0 e2e_s0 e2b_s1 e2c_s1 e2e_s1 e2d_s0; do prefix $arm selected; done
prefix e2b_s0 random
step "PRIMARY DONE"
wacc e2c_s0 random; wacc e2e_s0 random
prefix e2c_s0 random; prefix e2e_s0 random
for arm in e2b_s0 e2c_s0 e2e_s0 e2d_s0; do bare $arm selected; done
step "ALL DONE"
