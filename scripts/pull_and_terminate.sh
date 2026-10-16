#!/usr/bin/env bash
# Runs on the Mac. Wait for <run>'s "PIPELINE COMPLETE" on the box, pull its episodes (no
# activations), results and logs, verify episode counts, THEN terminate the box.
# A hard "FAILED <step>" line stops the watcher without terminating (data pulled, box kept).
#   bash scripts/pull_and_terminate.sh .lambda_instance_id_alf ms-alfworld-q25-7b
set -uo pipefail
IDF=$1; RUN=$2
cd ~/agent-failure-probes
export LAMBDA_ID_FILE=$IDF
source ~/.kyc-secrets
say() { echo "[$(date -u +%H:%M) UTC] [$RUN] $*" | tee -a logs/pull_and_terminate.log; }
say "watching"
while :; do
  st=$(scripts/lambda.sh ssh "grep -E '\[$RUN\] (PIPELINE COMPLETE|FAILED [a-z_]+ )' ~/agent-failure-probes/logs/runner_status.md | tail -1" 2>/dev/null)
  case "$st" in *"PIPELINE COMPLETE"*) break ;; *"FAILED "*) FAIL=1; break ;; esac
  sleep 120
done
H=ubuntu@$(scripts/lambda.sh ip | tr -dc '0-9.')
R="ssh -i $HOME/.ssh/lambda_key -o IdentitiesOnly=yes"
mkdir -p runs logs/box_$RUN
rsync -az -e "$R" --exclude acts "${H}:agent-failure-probes/runs/$RUN" runs/ \
 && rsync -az -e "$R" --exclude '*.lock' "${H}:agent-failure-probes/results/" results/ \
 && rsync -az -e "$R" "${H}:agent-failure-probes/logs/" logs/box_$RUN/ \
 && rsync -az -e "$R" "${H}:agent-failure-probes/runs/$RUN/acts/index.json" "${H}:agent-failure-probes/runs/$RUN/acts/extraction.json" runs/$RUN/acts/ 2>/dev/null
local_n=$(ls runs/$RUN/episodes | wc -l | tr -d ' ')
box_n=$(scripts/lambda.sh ssh "ls ~/agent-failure-probes/runs/$RUN/episodes | wc -l" 2>/dev/null | tr -dc 0-9)
say "pulled: $local_n local vs $box_n on box"
if [ "${FAIL:-0}" = 1 ]; then say "pipeline hard-FAILED ($st); data pulled, box KEPT for debugging"; exit 1; fi
if [ -n "$box_n" ] && [ "$local_n" = "$box_n" ]; then
  scripts/lambda.sh terminate && say "TERMINATED after verified pull"
else
  say "count mismatch; box KEPT"
fi
