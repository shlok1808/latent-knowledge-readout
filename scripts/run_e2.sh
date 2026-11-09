#!/usr/bin/env bash
# E2 (new) on ONE GPU (PRUNING_PLAN.md "E2 (new)"). Run from the repo root after build_e2_sets.py caving (laptop).
#   bash scripts/run_e2.sh
# Arms: e2b (margin loss, paper preservation), e2c (margin loss + useful-updating protection),
#       e2d (paper loss, protection p = 5e-4). Random magnitude-matched controls for e2b and e2c.
# Speed (2026-11-09): evals are single-turn only (Table D1 is single-turn), which halves each eval.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="uv run --no-sync python"
S=results/meta_instruct/btap/sets/seed0
E=results/meta_instruct/btap/e2
DEV=cuda:0
mkdir -p logs
step () { echo "STEP $* $(date +%T)"; }

step "useful set"
$PY scripts/build_e2_sets.py useful --device $DEV --batch-size 128 > logs/e2_useful.log 2>&1
step "useful done: $(tail -1 logs/e2_useful.log)"

score_arm () {  # $1 arm, then extra score args
  local arm=$1; shift
  $PY scripts/run_btap.py score --arm "$arm" --layers 0-31 --device $DEV "$@" > "logs/e2_score_${arm}.log" 2>&1
  $PY scripts/run_btap.py select --arm "$arm" --device $DEV > "logs/e2_select_${arm}.log" 2>&1
  step "selected $arm"
}
score_arm e2b --prune-file $E/caving_margin.jsonl --pres-file $S/preserve.jsonl
score_arm e2c --prune-file $E/caving_margin.jsonl --pres-file $S/preserve.jsonl $E/useful_margin.jsonl
score_arm e2d --prune-file $S/prune.jsonl --pres-file $S/preserve.jsonl --protect-frac 5e-4 --first-token 0

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

# format first (the two-part success check), then the D1 readings; most informative arms first
for arm in e2b e2c e2d; do wacc $arm selected; done
for arm in e2b e2c; do wacc $arm random; done
for arm in e2b e2c e2d; do prefix $arm selected; done
for arm in e2b e2c e2d; do bare $arm selected; done
for arm in e2b e2c; do prefix $arm random; done
step "ALL DONE"
