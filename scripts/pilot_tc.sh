#!/usr/bin/env bash
# Pilot: ~10 tasks per depth end-to-end (collect -> MC -> extract). Log: logs/pilot_tc.log
set -euo pipefail
cd ~/agent-failure-probes
export PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 PATH=$HOME/envs/vllm/bin:$HOME/.local/bin:$PATH
RUN=runs/ms-tc-1p7b-pilot
exec >> logs/pilot_tc.log 2>&1
echo "== pilot start $(date -u)"
~/envs/vllm/bin/python -m mstep.rollout collect --config configs/ms_tc_1p7b.json --run $RUN --pilot
~/envs/vllm/bin/python -m mstep.rollout mc --run $RUN
.venv/bin/python -m mstep.extract --run $RUN
echo "== pilot done $(date -u)"
