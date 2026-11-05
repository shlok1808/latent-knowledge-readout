#!/usr/bin/env bash
# Gate 2 diagnostics after run_gate2.sh: written-reply neutral accuracy (original / pruned / random) and the pruned +
# random evals under ALL letter spellings ("A", " A", "(A"), the main-run reading (fork #22: own findings need both).
set -euo pipefail
PY="uv run --no-sync python"
D=results/meta_instruct/btap/seed0
until grep -q "STEP ALL DONE" ~/gate2.log; do sleep 10; done
echo "CHECKS start $(date +%T)"
( $PY scripts/check_written_accuracy.py --tag original --device cuda:0 > logs/wacc_original.log 2>&1
  $PY scripts/check_written_accuracy.py --tag pruned_seed0 --prune-indices $D/selected.json --device cuda:0 > logs/wacc_pruned.log 2>&1
  $PY scripts/check_written_accuracy.py --tag random_seed0 --prune-indices $D/random.json --device cuda:0 > logs/wacc_random.log 2>&1
  echo "CHECKS written accuracy done $(date +%T)"
  $PY scripts/run_sycophancy_eval.py --checkpoint meta_instruct --allow-tf32 --batch-size 32 --ignore-preflight-problems \
      --anchors pruned --prune-indices $D/selected.json --device cuda:0 \
      --out-dir runs/allspell_pruned_seed0 --results-dir results/meta_instruct/btap/eval_seed0_selected_allspellings \
      > logs/gate2_pruned_allspell.log 2>&1 ) &
$PY scripts/run_sycophancy_eval.py --checkpoint meta_instruct --allow-tf32 --batch-size 32 --ignore-preflight-problems \
    --anchors pruned --prune-indices $D/random.json --device cuda:1 \
    --out-dir runs/allspell_random_seed0 --results-dir results/meta_instruct/btap/eval_seed0_random_allspellings \
    > logs/gate2_random_allspell.log 2>&1 &
wait
echo "CHECKS ALL DONE $(date +%T)"
