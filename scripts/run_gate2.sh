#!/usr/bin/env bash
# B-TAP Gate 2 pipeline on a 2-GPU box (PRUNING_PLAN.md steps C-E). Run from the repo root after build_pruning_sets.py.
#   bash scripts/run_gate2.sh            all steps
# Each GPU gets a balanced share of layers (backward cost grows with the number of layers above a group).
set -euo pipefail
PY="uv run --no-sync python"
EVAL_FLAGS=(--checkpoint meta_instruct --allow-tf32 --batch-size 32 --ignore-preflight-problems --letter-spellings "{L}, {L}" --anchors pruned)  # same reading as Gate 1 (fork #22)
score_seed () {  # $1 = seed
  $PY scripts/run_btap.py score --seed "$1" --layers 0-7,24-31 --device cuda:0 > "logs/btap_score_s$1_a.log" 2>&1 &
  $PY scripts/run_btap.py score --seed "$1" --layers 8-23 --device cuda:1 > "logs/btap_score_s$1_b.log" 2>&1 &
  wait
  $PY scripts/run_btap.py select --seed "$1" --device cuda:0 > "logs/btap_select_s$1.log" 2>&1
  echo "STEP seed $1 scored and selected"
}
mkdir -p logs
echo "STEP C start $(date +%T)"
score_seed 0
echo "STEP D start $(date +%T)"
D=results/meta_instruct/btap/seed0
$PY scripts/run_sycophancy_eval.py "${EVAL_FLAGS[@]}" --prune-indices $D/selected.json --device cuda:0 > logs/gate2_pruned.log 2>&1 &
$PY scripts/run_sycophancy_eval.py "${EVAL_FLAGS[@]}" --prune-indices $D/random.json --device cuda:1 > logs/gate2_random.log 2>&1 &
wait
echo "STEP D evals done $(date +%T)"
$PY scripts/run_btap.py skills --tag original --device cuda:0 > logs/skills_original.log 2>&1 &
$PY scripts/run_btap.py skills --tag pruned_seed0 --prune-indices $D/selected.json --device cuda:1 > logs/skills_pruned.log 2>&1 &
wait
$PY scripts/run_btap.py skills --tag random_seed0 --prune-indices $D/random.json --device cuda:0 > logs/skills_random.log 2>&1
echo "STEP E start $(date +%T)"
score_seed 1
score_seed 2
echo "STEP ALL DONE $(date +%T)"
