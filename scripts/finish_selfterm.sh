#!/usr/bin/env bash
# Runs ON a GPU box (tmux `finish`). When <run>'s pipeline completes (or hard-fails): upload the
# run's episodes/manifest/progress + results + logs to the private W&B project as an artifact
# (no activations: regenerable), wait GRACE seconds so the Mac watcher can pull a local copy,
# then terminate THIS instance via the Lambda API (key in ~/.lambda_api_key, mode 600).
#   tmux new -d -s finish "bash scripts/finish_selfterm.sh ms-tau2-qwen3-8b results/multistep/tau2_qwen3-8b/ms-tau2-qwen3-8b"
set -uo pipefail
RUN=$1; OUT=$2; GRACE=${GRACE:-900}
cd ~/agent-failure-probes
note() { echo "- $(date -u +%F\ %H:%M) UTC [finish:$RUN] $*" | tee -a logs/runner_status.md; }
note "armed: upload to W&B + self-terminate after pipeline completes"
until grep -qE "\[$RUN\] (PIPELINE COMPLETE|FAILED [a-z_]+ )" logs/runner_status.md; do sleep 60; done
# Also wait for any extra marker files (e.g. the pre-registered Phase 3X DONE marker).
for f in ${WAIT_FILES:-}; do until [ -f "$f" ]; do sleep 60; done; note "extra marker present: $f"; done
[ -f ~/.wandb_env ] && source ~/.wandb_env
if .venv/bin/python - <<PY
import wandb, pathlib
run = wandb.init(entity="agentic-interp", project="agent-failure-probes-multistep", name="$RUN:data", job_type="data")
art = wandb.Artifact("$RUN-data", type="dataset")
r = pathlib.Path("runs/$RUN")
art.add_dir(str(r / "episodes"), name="episodes")
for f in ("manifest.json", "progress.jsonl"):
    if (r / f).exists(): art.add_file(str(r / f))
if pathlib.Path("$OUT").exists(): art.add_dir("$OUT", name="results")
art.add_dir("logs", name="logs")
run.log_artifact(art).wait()
print("uploaded", art.name)
run.finish()
PY
then note "uploaded episodes/results/logs to W&B artifact $RUN-data"
else note "W&B upload FAILED; NOT self-terminating (data only on this box)"; exit 1; fi
note "waiting ${GRACE}s for a Mac pull, then self-terminating"
sleep $GRACE
curl -fsS -u "$(cat ~/.lambda_api_key):" -X POST https://cloud.lambda.ai/api/v1/instance-operations/terminate \
  -H "Content-Type: application/json" -d "{\"instance_ids\":[\"$(cat ~/.self_instance_id)\"]}" >/dev/null \
  && note "self-terminate requested" || note "self-terminate FAILED"
