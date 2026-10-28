#!/usr/bin/env bash
# Instance side: set up a fresh Lambda instance for this project. Idempotent; safe to re-run.
# Usually run by scripts/lambda_up.sh. Manual run, from the repo root: bash scripts/bootstrap.sh
# How the whole setup works: the research plugin's SETUP.md
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== Claude billing"
if [ -n "${ANTHROPIC_API_KEY:-}" ] || grep -qs ANTHROPIC_API_KEY ~/.bashrc ~/.profile ~/.bash_profile; then
  echo "WARNING: ANTHROPIC_API_KEY is set on this instance. Claude Code would bill the API, not your Max plan."
  echo "         Remove it from the shell and from ~/.bashrc / ~/.profile before starting a session."
else
  echo "ok: ANTHROPIC_API_KEY not set"
fi

echo "== Tools"
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# .mcp.json calls ${HOME}/.local/bin/uv; SSH sessions don't read ~/.profile, so also put uv on the system PATH
mkdir -p "$HOME/.local/bin"; [ -x "$HOME/.local/bin/uv" ] || ln -sf "$(command -v uv)" "$HOME/.local/bin/uv"
sudo ln -sf "$HOME/.local/bin/uv" /usr/local/bin/uv
command -v tmux >/dev/null || { sudo apt-get update -qq && sudo apt-get install -y -qq tmux; }

echo "== Python env: project + Scribe (uv.lock, Python from .python-version; torch CUDA wheels are ~2.5 GB)"
uv sync

echo "== Scribe kernel idle limit"
# Scribe's default (60 min) kills an idle kernel and its loaded model. 0 is buggy: it kills every minute.
mkdir -p ~/.jupyter
python3 - "${SCRIBE_IDLE_MINUTES:-720}" <<'PY'
import json, os, sys
path = os.path.expanduser("~/.jupyter/jupyter_server_config.json")
try:
    config = json.load(open(path)) if os.path.exists(path) else {}
except ValueError:  # unreadable config: keep it aside rather than abort the bootstrap
    os.replace(path, path + ".bak")
    print("note: unreadable", path, "moved to .bak")
    config = {}
config.setdefault("ScribeServerApp", {})["auto_shutdown_minutes"] = int(sys.argv[1])
json.dump(config, open(path, "w"), indent=1)
print(path, "->", config["ScribeServerApp"])
PY

echo "== Hugging Face"
[ -s ~/.cache/huggingface/token ] && echo "ok: HF token present" \
  || echo "WARNING: no HF token. Gated models (Gemma, Llama) and 'hf upload' will fail. Run 'hf auth login' on the laptop, then re-run lambda_up.sh"

echo "== git"
git config --global user.email >/dev/null || echo "WARNING: git identity not set (lambda_up.sh sets it)"
git ls-remote -q origin HEAD >/dev/null && echo "ok: GitHub reachable" || echo "WARNING: can't reach origin; pushes at wrap-up will fail"

echo "== GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
uv run --no-sync python -c "import sys, torch, scribe; print('python', sys.executable); print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
# Torch built for a newer CUDA than the driver supports imports fine but can't see the GPU. Fail loudly.
uv run --no-sync python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null \
  || { echo "ERROR: torch can't see the GPU. Its CUDA build is likely newer than the driver (see the torch index in pyproject.toml)."; exit 1; }

touch ~/.research-instance  # marks this box as an instance: turns on the research plugin's unpushed-work warning
echo "== Ready. In Desktop: environment 'lambda', folder $(pwd)"
