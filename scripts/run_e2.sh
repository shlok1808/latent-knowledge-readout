#!/usr/bin/env bash
# E2 (new) on ONE GPU (PRUNING_PLAN.md "E2 (new)" + "E2 changes after the design critic"). Run from the repo root.
#   bash scripts/run_e2.sh          (assumes results/meta_instruct/btap/e2/useful_margin.jsonl exists)
# Arms per seed S in {0,1}: e2b_sS (margin loss, paper preservation), e2c_sS (+ useful-updating protection, split),
#   e2e_sS (contrastive: caving minus useful updating). e2d_s0 (paper loss, p = 5e-4). Random controls for seed-0 b/c/e.
# Speed: evals single-turn only (Table D1 is single-turn).
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="uv run --no-sync python"
E=results/meta_instruct/btap/e2
DEV=cuda:0
mkdir -p logs
step () { echo "STEP $* $(date +%T)"; }

[ -s results/meta_instruct/btap/e2/useful_margin.jsonl ] || { echo "useful_margin.jsonl missing: run build_e2_sets.py useful first"; exit 1; }
$PY scripts/check_e2_useful.py > logs/e2_useful_check.log 2>&1; step "useful check: $(cat logs/e2_useful_check.log | tr '\n' ' ')"
for S in 0 1; do
  $PY scripts/build_e2_sets.py caving --seed $S --device $DEV > logs/e2_caving_s$S.log 2>&1
  step "caving set seed $S: $(tail -1 logs/e2_caving_s$S.log)"
done

score_arm () {  # $1 arm name, $2 seed, then extra score args
  local arm=$1 seed=$2; shift 2
  $PY scripts/run_btap.py score --arm "$arm" --seed "$seed" --layers 0-31 --device $DEV "$@" > "logs/e2_score_${arm}.log" 2>&1
  $PY scripts/run_btap.py select --arm "$arm" --seed "$seed" --device $DEV > "logs/e2_select_${arm}.log" 2>&1
  step "selected $arm"
}
for S in 0 1; do
  P=results/meta_instruct/btap/sets/seed$S/preserve.jsonl
  score_arm e2b_s$S $S --prune-file $E/caving_margin_seed$S.jsonl --pres-file $P
  score_arm e2c_s$S $S --prune-file $E/caving_margin_seed$S.jsonl --pres-file $P $E/useful_margin.jsonl --pres-split
  score_arm e2e_s$S $S --prune-file $E/caving_margin_seed$S.jsonl --neg-prune-file $E/useful_margin.jsonl --pres-file $P
done
score_arm e2d_s0 0 --prune-file results/meta_instruct/btap/sets/seed0/prune.jsonl \
  --pres-file results/meta_instruct/btap/sets/seed0/preserve.jsonl --protect-frac 5e-4 --first-token 0

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

# Gate 0 (format) first for every arm, then the pre-named primary reading (fixed slot), then secondaries
for arm in e2b_s0 e2c_s0 e2e_s0 e2d_s0 e2b_s1 e2c_s1 e2e_s1; do wacc $arm selected; done
for arm in e2b_s0 e2c_s0 e2e_s0; do wacc $arm random; done
for arm in e2b_s0 e2c_s0 e2e_s0 e2b_s1 e2c_s1 e2e_s1 e2d_s0; do prefix $arm selected; done
for arm in e2b_s0 e2c_s0 e2e_s0; do prefix $arm random; done
for arm in e2b_s0 e2c_s0 e2e_s0 e2d_s0; do bare $arm selected; done
step "ALL DONE"
