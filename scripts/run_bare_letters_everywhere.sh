#!/usr/bin/env bash
# Clean bare-letter rerun (lab bet #22): the full Meta Instruct run with ONLY "A"/" A" counted as letters,
# for every step (no-claim answer, right/wrong groups, doubt prompts, log-odds). Then commit and push results.
# Run on the GPU machine from the repo root, inside tmux:  bash scripts/run_bare_letters_everywhere.sh
set -uo pipefail
BRANCH=sycophancy-eval-id8yhs
OUT=results/meta_instruct/bare_letters_everywhere

git fetch origin "$BRANCH" && git checkout -B "$BRANCH" "origin/$BRANCH" || { echo "git checkout failed"; exit 1; }
mkdir -p "$OUT"

uv run --no-sync python scripts/run_sycophancy_eval.py --checkpoint meta_instruct --allow-tf32 --batch-size 8 \
  --ignore-preflight-problems --letter-spellings "{L}, {L}" \
  --out-dir runs/sycophancy_eval_meta_instruct_bare_everywhere \
  --results-dir "$OUT" 2>&1 | tee "$OUT/run.log"
echo "EXIT=${PIPESTATUS[0]}" | tee -a "$OUT/run.log"

git add "$OUT"
git commit -m "Bare letters everywhere: results (lab bet #22), pushed from Lambda"
for wait in 2 4 8 16 0; do
  git pull --rebase -q origin "$BRANCH" && git push origin "HEAD:$BRANCH" && { echo PUSHED; exit 0; }
  [ "$wait" -gt 0 ] && sleep "$wait"
done
echo "PUSH FAILED: results are committed locally in $OUT"
