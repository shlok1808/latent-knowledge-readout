#!/usr/bin/env bash
# Phase 4 candidate pilots (10 tasks x 8, Qwen3-1.7B) sharing the GPU with the TextCraft run.
# Waits until >= 20 GB GPU memory is free, then pilots each env with vLLM capped at 0.40.
# Log: logs/env_pilots.log; summaries appended to logs/runner_status.md.
set -uo pipefail
cd ~/agent-failure-probes
export PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 PATH=$HOME/envs/vllm/bin:$HOME/.local/bin:$PATH MSTEP_GPU_UTIL=0.40
exec >> logs/env_pilots.log 2>&1
note() { echo "- $(date -u +%F\ %H:%M) UTC [env-pilots] $*" | tee -a logs/runner_status.md; }
[ -x scripts/env_servers.sh ] && bash scripts/env_servers.sh
# vLLM's startup profiling counts other processes' allocations, so never start while another
# engine is initialising: wait for free memory to be stable, and retry on failure.
sleep 180
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge 20000 ]; do sleep 60; done
for ENV in ${ENVS:-alfworld babyai sciworld}; do
  RUN=runs/ms-$ENV-1p7b-pilot
  note "start pilot $ENV"
  ok=0
  for try in 1 2 3; do
    if timeout 3600 ~/envs/vllm/bin/python -m mstep.rollout collect --config configs/ms_${ENV}_1p7b.json --run $RUN --pilot; then ok=1; break; fi
    note "pilot $ENV attempt $try failed; retrying in 120 s"; sleep 120
  done
  if [ $ok = 1 ]; then
    note "done pilot $ENV: $(.venv/bin/python scripts/ms_summary.py $RUN | tr '\n' ' ')"
  else
    note "FAILED pilot $ENV (see logs/env_pilots.log)"
  fi
done
note "env pilots complete"
