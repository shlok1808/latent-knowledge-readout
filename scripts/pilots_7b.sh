#!/usr/bin/env bash
# Qwen2.5-7B-Instruct pilots (Shlok 2026-09-24), launched automatically:
#   1. wait until the 1.7B TextCraft extraction has finished (acts/extraction.json, all rows done);
#   2. wait until no other vLLM collect is running (the 1.7B ScienceWorld pilot finishes first);
#   3. verify the 7B fits and read its layer count from config (logged);
#   4. pilot TextCraft depths 2-3 (20 tasks x 8) and ScienceWorld (20 tasks x 8) at gpu util 0.85.
# MC stays on hold (runs/ms-tc-1p7b/mc_scope.json) throughout; nothing here releases it.
# Log: logs/pilots_7b.log; milestones -> logs/runner_status.md
set -uo pipefail
cd ~/agent-failure-probes
export PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 PATH=$HOME/envs/vllm/bin:$HOME/.local/bin:$PATH
exec >> logs/pilots_7b.log 2>&1
note() { echo "- $(date -u +%F\ %H:%M) UTC [pilots-7b] $*" | tee -a logs/runner_status.md; }
note "armed: waiting for 1.7B extraction + free GPU"
until .venv/bin/python -c "import json,sys; d=json.load(open('runs/ms-tc-1p7b/acts/extraction.json')); sys.exit(0 if d.get('all_rows_done') else 1)" 2>/dev/null; do sleep 60; done
note "1.7B extraction complete"
while pgrep -f "mstep.rollout collect --config configs/ms_sciworld_1p7b" >/dev/null; do sleep 60; done
while pgrep -f "run_env_pilots.sh" >/dev/null; do sleep 60; done
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
for pair in "tc_q25_7b ms-tc-q25-7b-pilot" "sciworld_q25_7b ms-sciworld-q25-7b-pilot"; do
  set -- $pair; CFG=configs/ms_$1.json; RUN=runs/$2
  note "start pilot $2"
  if ~/envs/vllm/bin/python -m mstep.rollout collect --config $CFG --run $RUN --pilot; then
    note "done pilot $2: $(.venv/bin/python scripts/ms_summary.py $RUN | tr '\n' ' ')"
  else
    note "FAILED pilot $2 (see logs/pilots_7b.log)"
  fi
done
note "7B pilots complete; MC still on hold awaiting Shlok's decision"
