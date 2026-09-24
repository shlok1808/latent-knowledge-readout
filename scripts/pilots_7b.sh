#!/usr/bin/env bash
# Qwen2.5-7B-Instruct pilots (Shlok 2026-09-24), launched automatically:
#   1. wait until the 1.7B TextCraft extraction has finished (acts/extraction.json, all rows done);
#   2. wait until no other vLLM collect is running;
#   3. verify the 7B fits and read its layer count from config (logged);
#   4. pilot TextCraft depths 2-3 (20 tasks PER depth x 8) and ScienceWorld (20 tasks x 8), util 0.85;
#   5. if ScienceWorld has < 6/20 mixed-outcome tasks, also pilot BabyAI and ALFWorld (20 x 8 each);
#   6. write one summary line with all env results.
# MC stays on hold (runs/ms-tc-1p7b/mc_scope.json) throughout; nothing here releases it.
# Log: logs/pilots_7b.log; milestones -> logs/runner_status.md
set -uo pipefail
cd ~/agent-failure-probes
export PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 PATH=$HOME/envs/vllm/bin:$HOME/.local/bin:$PATH
exec >> logs/pilots_7b.log 2>&1
note() { echo "- $(date -u +%F\ %H:%M) UTC [pilots-7b] $*" | tee -a logs/runner_status.md; }
mixed() { .venv/bin/python -c "
import json,glob,collections
t=collections.defaultdict(set)
for f in glob.glob('$1/episodes/*.json'):
    e=json.load(open(f)); t[e['task_id']].add(e['success'])
print(sum(len(v)==2 for v in t.values()))"; }
pilot() {  # pilot <config-stem> <run-name>
  local CFG=configs/ms_$1.json RUN=runs/$2
  note "start pilot $2"
  if ~/envs/vllm/bin/python -m mstep.rollout collect --config $CFG --run $RUN --pilot; then
    note "done pilot $2: $(.venv/bin/python scripts/ms_summary.py $RUN | tr '\n' ' ')"
  else
    note "FAILED pilot $2 (see logs/pilots_7b.log)"
  fi
}
note "armed (v2: 20 tasks/depth TextCraft; BabyAI+ALFWorld fallback if ScienceWorld < 6/20 mixed): waiting for 1.7B extraction + free GPU"
until .venv/bin/python -c "import json,sys; d=json.load(open('runs/ms-tc-1p7b/acts/extraction.json')); sys.exit(0 if d.get('all_rows_done') else 1)" 2>/dev/null; do sleep 60; done
note "1.7B extraction complete"
while pgrep -f "mstep.rollout collect --config configs/ms_" >/dev/null; do sleep 60; done
note "GPU free of other collectors"
.venv/bin/python - <<'PY' || { note "FAILED 7B fit check; stopping (see logs/pilots_7b.log)"; exit 1; }
import json, torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig
rev = "a09a35458c702b33eeacc393d103063234e8bc28"
snapshot_download("Qwen/Qwen2.5-7B-Instruct", revision=rev)
c = AutoConfig.from_pretrained("Qwen/Qwen2.5-7B-Instruct", revision=rev)
free, total = torch.cuda.mem_get_info()
need = 7.62e9 * 2  # bf16 weights
print(f"Qwen2.5-7B: layers={c.num_hidden_layers} hidden={c.hidden_size} kv_heads={c.num_key_value_heads}; "
      f"weights {need/2**30:.1f} GiB; GPU free {free/2**30:.1f}/{total/2**30:.1f} GiB")
assert free > need + 6 * 2**30, "not enough free GPU memory for 7B weights + KV cache"
json.dump({"layers": c.num_hidden_layers, "hidden": c.hidden_size, "free_gib": free / 2**30},
          open("logs/q25_7b_fit.json", "w"))
PY
note "7B fit ok: $(cat logs/q25_7b_fit.json)"
export MSTEP_GPU_UTIL=0.85
bash scripts/env_servers.sh restart sciworld >/dev/null 2>&1
pilot tc_q25_7b ms-tc-q25-7b-pilot
pilot sciworld_q25_7b ms-sciworld-q25-7b-pilot
SW=$(mixed runs/ms-sciworld-q25-7b-pilot)
SUMMARY="ScienceWorld mixed ${SW:-?}/20"
if [ "${SW:-0}" -lt 6 ]; then
  note "ScienceWorld 7B pilot has ${SW:-0}/20 mixed (< 6 gate): running BabyAI and ALFWorld 7B pilots"
  bash scripts/env_servers.sh restart babyai >/dev/null 2>&1
  bash scripts/env_servers.sh restart alfworld >/dev/null 2>&1
  pilot babyai_q25_7b ms-babyai-q25-7b-pilot
  pilot alfworld_q25_7b ms-alfworld-q25-7b-pilot
  SUMMARY="$SUMMARY; BabyAI mixed $(mixed runs/ms-babyai-q25-7b-pilot)/20; ALFWorld mixed $(mixed runs/ms-alfworld-q25-7b-pilot)/20"
else
  note "ScienceWorld 7B pilot passes the gate (${SW}/20 mixed); BabyAI/ALFWorld 7B pilots not needed"
fi
note "7B pilots complete. PHASE 4 GATE SUMMARY: TextCraft d2-3 mixed $(mixed runs/ms-tc-q25-7b-pilot)/40; $SUMMARY. MC still on hold awaiting Shlok's decision"
