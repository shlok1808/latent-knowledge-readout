#!/usr/bin/env bash
# After pilots_7b.sh finishes: resume any 7B pilot that crashed (< expected episodes) with the
# keep-alive fix in mstep/httpenv.py, then write a corrected PHASE 4 GATE SUMMARY.
# The first summary counted a crashed ScienceWorld pilot (1/160 episodes) as "0/20 mixed".
set -uo pipefail
cd ~/agent-failure-probes
export PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 PATH=$HOME/envs/vllm/bin:$HOME/.local/bin:$PATH MSTEP_GPU_UTIL=0.85
exec >> logs/pilots_7b.log 2>&1
note() { echo "- $(date -u +%F\ %H:%M) UTC [pilots-7b-rerun] $*" | tee -a logs/runner_status.md; }
mixed() { .venv/bin/python -c "
import json,glob,collections
t=collections.defaultdict(set)
for f in glob.glob('$1/episodes/*.json'):
    e=json.load(open(f)); t[e['task_id']].add(e['success'])
print(sum(len(v)==2 for v in t.values()))"; }
n() { ls $1/episodes 2>/dev/null | wc -l; }
note "armed: waiting for pilots_7b.sh to finish"
while pgrep -f "scripts/pilots_7b.sh" >/dev/null; do sleep 60; done
for pair in "sciworld 160" "babyai 160" "alfworld 160"; do
  set -- $pair; RUN=runs/ms-$1-q25-7b-pilot
  [ -f $RUN/manifest.json ] || continue
  if [ "$(n $RUN)" -lt $2 ]; then
    note "resuming $1 pilot ($(n $RUN)/$2 episodes)"
    bash scripts/env_servers.sh restart $1 >/dev/null 2>&1
    if ~/envs/vllm/bin/python -m mstep.rollout collect --config configs/ms_$1_q25_7b.json --run $RUN --pilot; then
      note "done $1 pilot: $(.venv/bin/python scripts/ms_summary.py $RUN | tr '\n' ' ')"
    else note "FAILED again: $1 pilot ($(n $RUN)/$2 episodes; see logs/pilots_7b.log)"; fi
  fi
done
S="TextCraft d2-3 mixed $(mixed runs/ms-tc-q25-7b-pilot)/40 ($(n runs/ms-tc-q25-7b-pilot)/320 eps)"
for e in sciworld babyai alfworld; do R=runs/ms-$e-q25-7b-pilot; [ -d $R ] && S="$S; $e mixed $(mixed $R)/20 ($(n $R)/160 eps)"; done
note "CORRECTED PHASE 4 GATE SUMMARY: $S. MC still on hold."
