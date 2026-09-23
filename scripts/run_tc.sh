#!/usr/bin/env bash
# Chained, resumable TextCraft pipeline for one config (collect -> MC -> extract ->
# dataset -> probes [all depths, depth>=2] -> figures). Each step is skipped when its
# output is complete, so re-running after any interruption resumes where it stopped.
# Progress lines go to logs/runner_status.md (merged into STATUS.md by Claude).
#
#   tmux new -d -s tc17 "bash scripts/run_tc.sh configs/ms_tc_1p7b.json runs/ms-tc-1p7b results/multistep/textcraft_qwen3-1.7b"
set -uo pipefail
CFG=$1; RUN=$2; OUT=$3
cd ~/agent-failure-probes
export PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 PATH=$HOME/envs/vllm/bin:$HOME/.local/bin:$PATH
mkdir -p logs "$OUT"
NAME=$(basename "$RUN")
LOG=logs/$NAME.log
note() { echo "- $(date -u +%F\ %H:%M) UTC [$NAME] $*" | tee -a logs/runner_status.md; }
step() {  # step <name> <cmd...>
  local name=$1; shift
  note "start $name"
  if "$@" >> "$LOG" 2>&1; then note "done $name"; else note "FAILED $name (see $LOG)"; exit 1; fi
}
soft() {  # analysis steps: log failure but keep the GPU pipeline going
  local name=$1; shift
  note "start $name"
  if "$@" >> "$LOG" 2>&1; then note "done $name"; else note "FAILED(soft) $name (see $LOG)"; fi
}
VPY=~/envs/vllm/bin/python; PY=.venv/bin/python

analyze() {  # analyze <tag>
  soft dataset_$1   $PY -m mstep.dataset --run "$RUN"
  soft probes_$1    $PY -m mstep.probes --run "$RUN" --out "$OUT/all_depths"
  soft figures_$1   $PY -m mstep.figures --out "$OUT/all_depths"
  soft probes_d2_$1  $PY -m mstep.probes --run "$RUN" --out "$OUT/depth_ge2" --subset "depth>=2"
  soft figures_d2_$1 $PY -m mstep.figures --out "$OUT/depth_ge2"
}
step collect  $VPY -m mstep.rollout collect --config "$CFG" --run "$RUN"
soft summary  $PY scripts/ms_summary.py "$RUN"
step extract  $PY -m mstep.extract --run "$RUN"
analyze nomc          # FINAL/validity/progress results first; MC labels are NaN here
step mc       $VPY -m mstep.rollout mc --run "$RUN"
analyze withmc        # re-run with MC-value labels filled in
note "PIPELINE COMPLETE"
