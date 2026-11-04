# latent-knowledge-readout

Studying what happens to what a language model knows at the moment it acts against it.

## What is here

- `knowledge_readout/syco/`: replication of Orgad et al. 2610.28325 (sycophancy as in-context updating) across Llama-3.1-8B training stages. Spec: `papers/2610.28325.md`, exact templates: `papers/2610.28325_templates.md`.
- `knowledge_readout/common.py`: JSON/JSONL helpers.
- `configs/syco_*.json`: checkpoints and cue templates.
- `tests/`: unit tests on a tiny offline model (`pytest -q`).

## Earlier project (archived)

The earlier multi-step probe-transfer project is preserved in full on branch `archive/stage-transfer-v1` (tag `stage-transfer-v1`), and `main` as it was before cleanup is tagged `main-before-clean`. The TextCraft pipeline that sat on main afterwards (`actor.py`, `environment.py`, `pipeline.py`, `cli.py`, `vendor/AgentGym`) was removed on 2026-11-03 and is preserved on branch `archive/textcraft-pipeline-final`.

## Setup

```bash
git clone https://github.com/shlok1808/latent-knowledge-readout.git
cd latent-knowledge-readout
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

Python 3.11+. CUDA, Apple MPS, or CPU.
