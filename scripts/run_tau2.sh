#!/usr/bin/env bash
# τ²-bench pipeline on one A100: vLLM OpenAI server (Qwen3-8B) serving BOTH the agent (exact
# token ids via our ServerEngine) and τ²'s user simulator (litellm), then collect -> stop server
# -> extract -> dataset -> probes -> figures -> W&B.  Pilot by default; FULL=1 for the full split.
#   tmux new -d -s tau "bash scripts/run_tau2.sh"
set -uo pipefail
cd ~/agent-failure-probes
export PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 PATH=$HOME/envs/vllm/bin:$HOME/.local/bin:$PATH
[ -f ~/.wandb_env ] && source ~/.wandb_env
CFG=configs/ms_tau2_qwen3_8b.json
if [ "${FULL:-0}" = 1 ]; then RUN=runs/ms-tau2-qwen3-8b; PIL=""; else RUN=runs/ms-tau2-qwen3-8b-pilot; PIL="--pilot"; fi
OUT=results/multistep/tau2_qwen3-8b/$(basename $RUN)
NAME=$(basename $RUN); LOG=logs/$NAME.log; mkdir -p logs $OUT
note() { echo "- $(date -u +%F\ %H:%M) UTC [$NAME] $*" | tee -a logs/runner_status.md; }
step() { local n=$1; shift; note "start $n"; if "$@" >> "$LOG" 2>&1; then note "done $n"; else note "FAILED $n (see $LOG)"; exit 1; fi; }
soft() { local n=$1; shift; note "start $n"; if "$@" >> "$LOG" 2>&1; then note "done $n"; else note "FAILED(soft) $n (see $LOG)"; fi; }
TPY=~/envs/tau2/bin/python; PY=.venv/bin/python
if ! curl -fsS -m 3 127.0.0.1:8000/v1/models >/dev/null 2>&1; then
  note "starting vLLM server"
  tmux new -d -s vllmsrv "~/envs/vllm/bin/vllm serve Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 --served-model-name Qwen/Qwen3-8B --dtype bfloat16 --gpu-memory-utilization 0.88 --max-model-len 32768 --enable-prefix-caching --port 8000 2>&1 | tee -a logs/vllm_server.log"
  until curl -fsS -m 3 127.0.0.1:8000/v1/models >/dev/null 2>&1; do sleep 10; done
  note "vLLM server up"
fi
step collect $TPY -m mstep.rollout collect --config $CFG --run $RUN $PIL
soft summary $PY scripts/ms_summary.py $RUN
tmux kill-session -t vllmsrv 2>/dev/null; sleep 15   # free the GPU for extraction
step extract $PY -m mstep.extract --run $RUN --token-budget ${EXTRACT_BUDGET:-24576}
soft dataset $PY -m mstep.dataset --run $RUN
soft probes  $PY -m mstep.probes --run $RUN --out $OUT --jobs ${JOBS:-24}
soft figures $PY -m mstep.figures --out $OUT
soft wandb   $PY -m mstep.wb $OUT $NAME
note "PIPELINE COMPLETE"
