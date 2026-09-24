#!/usr/bin/env bash
# Generic resumable pipeline for a non-TextCraft env or model (Phase 4 / Phase 5):
# collect -> extract -> dataset -> probes -> figures. MC is off by default (budget plan
# in STATUS.md); set WITH_MC=1 to add it. Shares the GPU via MSTEP_GPU_UTIL (default 0.40).
#   tmux new -d -s alf17 "bash scripts/run_env.sh configs/ms_alfworld_1p7b.json runs/ms-alfworld-1p7b results/multistep/alfworld_qwen3-1.7b"
set -uo pipefail
CFG=$1; RUN=$2; OUT=$3
cd ~/agent-failure-probes
export PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 PATH=$HOME/envs/vllm/bin:$HOME/.local/bin:$PATH
export MSTEP_GPU_UTIL=${MSTEP_GPU_UTIL:-0.40}
mkdir -p logs "$OUT"
NAME=$(basename "$RUN"); LOG=logs/$NAME.log
note() { echo "- $(date -u +%F\ %H:%M) UTC [$NAME] $*" | tee -a logs/runner_status.md; }
step() { local n=$1; shift; note "start $n"; if "$@" >> "$LOG" 2>&1; then note "done $n"; else note "FAILED $n (see $LOG)"; exit 1; fi; }
soft() { local n=$1; shift; note "start $n"; if "$@" >> "$LOG" 2>&1; then note "done $n"; else note "FAILED(soft) $n (see $LOG)"; fi; }
VPY=~/envs/vllm/bin/python; PY=.venv/bin/python
[ -f ~/.wandb_env ] && source ~/.wandb_env
N=$($PY -c "import json;print(json.load(open('$CFG')).get('n_servers',1))")
if [ "$N" -gt 1 ]; then bash scripts/env_servers.sh start_n $($PY -c "import json;print(json.load(open('$CFG'))['env'])") $N >> "$LOG" 2>&1
else bash scripts/env_servers.sh >> "$LOG" 2>&1; fi
step collect $VPY -m mstep.rollout collect --config "$CFG" --run "$RUN"
soft summary $PY scripts/ms_summary.py "$RUN"
[ "${WITH_MC:-0}" = 1 ] && step mc $VPY -m mstep.rollout mc --run "$RUN"
step extract $PY -m mstep.extract --run "$RUN" --token-budget ${EXTRACT_BUDGET:-32768}
soft dataset $PY -m mstep.dataset --run "$RUN"
soft probes  $PY -m mstep.probes --run "$RUN" --out "$OUT" --jobs ${JOBS:-12}
soft figures $PY -m mstep.figures --out "$OUT"
soft wandb   $PY -m mstep.wb "$OUT" "$NAME"
note "PIPELINE COMPLETE"
