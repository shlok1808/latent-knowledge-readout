#!/usr/bin/env bash
# One-shot preflight on the Lambda box: main venv, both models, the three candidate
# AgentGym envs (each in its own venv, served over HTTP as upstream intends), and a
# smoke test of each. Idempotent; safe to re-run. Log: ~/preflight.log
# Usage (on the box, inside tmux):  HF_TOKEN=... bash agent-failure-probes/scripts/preflight_box.sh
set -uo pipefail
exec > >(tee -a ~/preflight.log) 2>&1
REPO=~/agent-failure-probes
AG=$REPO/vendor/AgentGym
export PATH="$HOME/.local/bin:$PATH"
ok()   { echo "PREFLIGHT OK   $*"; }
fail() { echo "PREFLIGHT FAIL $*"; }
step() { echo; echo "===== $* ($(date -u +%H:%M:%S))"; }

step "system"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
nvidia-smi | grep -o "CUDA Version: [0-9.]*"
df -h ~ | tail -1; nproc; free -g | head -2

step "uv + python"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.13 3.10 >/dev/null && ok "uv pythons" || fail "uv python install"

step "java (ScienceWorld needs a JVM)"
if ! command -v java >/dev/null; then
  sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq && \
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openjdk-17-jre-headless >/dev/null
fi
java -version 2>&1 | head -1 && ok java || fail java

step "main venv (py3.13, torch cu128)"
cd $REPO
[ -d .venv ] || uv venv -q .venv --python 3.13
source .venv/bin/activate
uv pip install -q torch --index-url https://download.pytorch.org/whl/${TORCH_CUDA:-cu128} && \
uv pip install -q -e '.[dev]' matplotlib pandas scipy && ok "main deps" || fail "main deps"
python -c 'import torch;print("torch",torch.__version__,"cuda",torch.cuda.is_available(),torch.cuda.get_device_name(0))'
PYTHONHASHSEED=0 python -m pytest -q 2>&1 | tail -2
deactivate

step "models"
mkdir -p ~/.cache/huggingface && (umask 077; [ -n "${HF_TOKEN:-}" ] && echo "$HF_TOKEN" > ~/.cache/huggingface/token)
source $REPO/.venv/bin/activate
python - <<'EOF'
from huggingface_hub import snapshot_download
for repo, rev in [("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"), ("Qwen/Qwen3-8B", None)]:
    p = snapshot_download(repo, revision=rev)
    print("PREFLIGHT OK   model", repo, rev or "main", p)
EOF
python - <<'EOF'
import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi
rev = HfApi().model_info("Qwen/Qwen3-8B").sha
print("Qwen3-8B resolved revision", rev)
t = time.time()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", revision=rev)
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", revision=rev, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
ids = tok.apply_chat_template([{"role": "user", "content": "Say hi."}], add_generation_prompt=True, enable_thinking=False, return_tensors="pt").cuda()
out = m.generate(ids, max_new_tokens=8, do_sample=False)
print("8B gen:", repr(tok.decode(out[0, ids.shape[1]:])), "layers", len(m.model.layers),
      "peak GiB %.1f" % (torch.cuda.max_memory_allocated() / 2**30), "load+gen s %.0f" % (time.time() - t))
print("PREFLIGHT OK   qwen3-8b fits and generates")
EOF
deactivate

serve() {  # serve <name> <port> <module-cmd...>; waits for GET / to answer
  local name=$1 port=$2; shift 2
  pkill -f "port $port" 2>/dev/null; sleep 1
  nohup "$@" --host 127.0.0.1 --port $port > ~/env-$name.log 2>&1 &
  for i in $(seq 1 90); do curl -fsS 127.0.0.1:$port/ >/dev/null 2>&1 && return 0; sleep 2; done
  tail -20 ~/env-$name.log; return 1
}
post() { curl -fsS -X POST "127.0.0.1:$1$2" -H 'Content-Type: application/json' -d "$3"; }

step "env: sciworld"
E=~/envs/sciworld; [ -d $E ] || uv venv -q $E --python 3.10
VIRTUAL_ENV=$E uv pip install -q -e $AG/agentenv-sciworld && \
serve sciworld 36001 $E/bin/sciworld && {
  post 36001 /create '{}' ; echo
  post 36001 /reset '{"id":0,"data_idx":0}' | head -c 400; echo
  post 36001 /step '{"id":0,"action":"look around"}' | head -c 600; echo
  ok sciworld; } || fail sciworld

step "env: alfworld"
E=~/envs/alfworld; [ -d $E ] || uv venv -q $E --python 3.10
export ALFWORLD_DATA=~/.cache/alfworld
VIRTUAL_ENV=$E uv pip install -q alfworld==0.3.3 && VIRTUAL_ENV=$E uv pip uninstall -q opencv-python; \
VIRTUAL_ENV=$E uv pip install -q opencv-python-headless -e $AG/agentenv-alfworld && \
{ [ -d $ALFWORLD_DATA/json_2.1.1 ] || $E/bin/alfworld-download; } && du -sh $ALFWORLD_DATA && \
serve alfworld 36002 $E/bin/alfworld && {
  post 36002 /create '{}' | head -c 300; echo
  post 36002 /reset '{"id":0,"game":0,"world_type":"Text"}' | head -c 500; echo
  post 36002 /step '{"id":0,"action":"look"}' | head -c 400; echo
  ok alfworld; } || fail alfworld

step "env: babyai"
E=~/envs/babyai; [ -d $E ] || uv venv -q $E --python 3.10
VIRTUAL_ENV=$E uv pip install -q -e $AG/agentenv-babyai && \
serve babyai 36003 $E/bin/babyai && {
  post 36003 /create '{}' ; echo
  post 36003 /reset '{"id":0,"data_idx":0}' | head -c 500; echo
  post 36003 /step '{"id":0,"action":"turn left"}' | head -c 400; echo
  ok babyai; } || fail babyai

step "disk after preflight"
df -h ~ | tail -1; du -sh ~/.cache/huggingface ~/.cache/alfworld 2>/dev/null
echo; echo "PREFLIGHT DONE $(date -u)"
