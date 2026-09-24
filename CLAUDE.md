Read HANDOFF.md, TASKS.md, STATUS.md before doing anything.

When a step doesn't need my input, keep going; put status notes in the same message as your next action. Stop and ask only when you can't continue without me, or before anything destructive: deleting data or prior results, force-pushing, or changing anything outside this repo. Never offer to continue; just continue.

## Run context (multistep-gap, started 2026-09-23)

- Goal: evidence for/against "the standard single-step probe recipe does not transfer to multi-step agentic tasks", and how it fails. Not AUC maximisation.
- Read `TASKS.md` (checklist, source of truth for what's next), `STATUS.md` (latest state), `ASSUMPTIONS.md` (every unrequested definition), `AGENTS.md` (pipeline invariants).
- Prior results in `runs/stage2-002`, `results/stage2-002` are read-only. New work goes in `runs/ms-*` and `results/multistep/`.
- `failure_probes/*.py` is hash-guarded by the stage-2 pipeline; new code lives in `mstep/` so the old run stays resumable/auditable.
- GPU box: `scripts/lambda.sh {ip,ssh,sync,pull,status}`; keys in `~/.kyc-secrets`, SSH key `~/.ssh/lambda_key`. Long jobs in named tmux sessions on the box, logs under `~/agent-failure-probes/logs/`.
- Repo `shlok1808/agent-failure-probes` was made **private** by Shlok on 2026-09-24; pushing to it is now OK. It is still Lin's unpublished direction: never push anywhere public, never change repo visibility. No `Co-Authored-By` lines.
- Budget: 40 box-hours total (instance wall-clock). Same-episode stage control is a primary analysis.
- Bigger model is Qwen2.5-7B-Instruct (not Qwen3-8B). MC is depth-2 only and on HOLD (runs/ms-tc-1p7b/mc_scope.json on the box) until Shlok releases it.
