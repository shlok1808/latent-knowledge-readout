# latent-knowledge-readout

Studying what happens to what a language model knows at the moment it acts against it.

## What is here

- `failure_probes/`: pipeline that runs a model in an environment, logs rollouts, and reads activations.
- `tests/`: unit tests (`pytest -q`).
- `configs/debug.json`: small debug configuration used as the CLI default.
- `vendor/AgentGym`: pinned environment code (git submodule).

## Earlier project (archived)

The earlier multi-step probe-transfer project is preserved in full on branch `archive/stage-transfer-v1` (tag `stage-transfer-v1`), and `main` as it was before cleanup is tagged `main-before-clean`.

## Setup

```bash
git clone --recurse-submodules https://github.com/shlok1808/latent-knowledge-readout.git
cd latent-knowledge-readout
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

Python 3.11+. CUDA, Apple MPS, or CPU.
