#!/usr/bin/env bash
# v2 of run_tc.sh: after extraction, MC (GPU) runs IN PARALLEL with the no-MC analysis
# (CPU), so the A100 is not idle while probes train. Resumable like v1.
#   bash scripts/run_tc2.sh configs/ms_tc_1p7b.json runs/ms-tc-1p7b results/multistep/textcraft_qwen3-1.7b
set -uo pipefail
CFG=$1; RUN=$2; OUT=$3
cd ~/agent-failure-probes
export PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 PATH=$HOME/envs/vllm/bin:$HOME/.local/bin:$PATH
mkdir -p logs "$OUT"
NAME=$(basename "$RUN"); LOG=logs/$NAME.log
note() { echo "- $(date -u +%F\ %H:%M) UTC [$NAME] $*" | tee -a logs/runner_status.md; }
step() { local n=$1; shift; note "start $n"; if "$@" >> "$LOG" 2>&1; then note "done $n"; else note "FAILED $n (see $LOG)"; exit 1; fi; }
soft() { local n=$1; shift; note "start $n"; if "$@" >> "$LOG" 2>&1; then note "done $n"; else note "FAILED(soft) $n (see $LOG)"; fi; }
VPY=~/envs/vllm/bin/python; PY=.venv/bin/python; JOBS=${JOBS:-20}
analyze() {
  soft dataset_$1    $PY -m mstep.dataset --run "$RUN"
  soft probes_$1     $PY -m mstep.probes --run "$RUN" --out "$OUT/all_depths" --jobs $JOBS
  soft figures_$1    $PY -m mstep.figures --out "$OUT/all_depths"
  soft probes_d2_$1  $PY -m mstep.probes --run "$RUN" --out "$OUT/depth_ge2" --subset "depth>=2" --jobs $JOBS
  soft figures_d2_$1 $PY -m mstep.figures --out "$OUT/depth_ge2"
}
step collect $VPY -m mstep.rollout collect --config "$CFG" --run "$RUN"
soft summary $PY scripts/ms_summary.py "$RUN"
step extract $PY -m mstep.extract --run "$RUN"
( step mc $VPY -m mstep.rollout mc --run "$RUN" ) &
MCPID=$!
analyze nomc
wait $MCPID || { note "MC failed; skipping with-MC analysis"; exit 1; }
analyze withmc
note "PIPELINE COMPLETE"
