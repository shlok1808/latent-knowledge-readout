# Agent failure probes

Research code for studying what happens inside an LLM at the moment it acts against something it internally represents (a knowing-doing gap): is the knowledge lost, suppressed, cancelled, outcompeted, or bypassed?

**Status: new direction, setup stage.** No experiments for this direction have been run in this repo yet.

## What is here

- `failure_probes/`: the reusable pipeline from the earlier project. It runs a model in an environment, logs complete rollouts and exact model-input tokens, replays saved tokens to read activations, and fits probes with whole tasks held out.
- `tests/`: unit tests for the environment, replay and analysis code (`pytest -q`).
- `configs/debug.json`: the small TextCraft debug configuration used as the CLI default.
- `vendor/AgentGym`: pinned environment code (a git submodule).

## What carries over from the earlier project

The same-task control (compare attempts at the same task, so task difficulty is not mistaken for failure), the shortcut-baseline audit, a text baseline that cannot see the future, and pre-registration before looking at results.

## Earlier project (archived)

The multi-step probe-transfer project (TextCraft, ALFWorld, τ²-bench, pre-registered verdicts, pitch and handoff notes) is preserved in full:

- branch `archive/stage-transfer-v1`, tag `stage-transfer-v1`
- branches `multistep-gap` and `stage2-determinism`

Check out the tag to reproduce any of it. `main` contains none of that work.

## Setup

```bash
git clone --recurse-submodules https://github.com/shlok1808/agent-failure-probes.git
cd agent-failure-probes
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

Python 3.11+. CUDA, Apple MPS, or CPU.
