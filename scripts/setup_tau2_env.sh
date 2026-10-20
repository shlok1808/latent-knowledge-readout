#!/usr/bin/env bash
# τ²-bench venv (~/envs/tau2): tau2-bench @ b7ea907 + everything its gym path and our
# collector import (found by smoke-testing 2026-09-24: websockets, gymnasium, CPU torch).
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
cd ~ && { test -d tau2-bench || git clone -q https://github.com/sierra-research/tau2-bench.git; }
cd tau2-bench && git checkout -q b7ea907
[ -d ~/envs/tau2 ] || uv venv -q ~/envs/tau2 --python 3.12
VIRTUAL_ENV=~/envs/tau2 uv pip install -q -e . transformers requests wandb websockets gymnasium
VIRTUAL_ENV=~/envs/tau2 uv pip install -q torch --index-url https://download.pytorch.org/whl/cpu
VIRTUAL_ENV=~/envs/tau2 uv pip install -q --no-deps -e ~/agent-failure-probes
cd ~/agent-failure-probes && PYTHONHASHSEED=0 ~/envs/tau2/bin/python -c "import tau2.gym.gym_agent, mstep.rollout, mstep.tau2env, litellm; print('TAU2 ENV OK')"
