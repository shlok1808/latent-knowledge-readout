#!/usr/bin/env bash
# Runs ON box 1 (tmux `to2`). Moves the depth>=2 analyses to the CPU helper box and back:
#  1. stop box 1's depth>=2 jobs (fast lane B, d2x) and hold depth_ge2's lock so tc2 doesn't redo them;
#  2. copy code + depth>=2 activations + per-round table to box 2, start scripts/cpu2_run.sh there in tmux;
#  3. poll until done, pull results back, mark DONE_all, release the lock, draw figures;
#  4. box 2 terminates itself via the Lambda API (key stored only on box 2, mode 600).
set -uo pipefail
IP2=$1
cd ~/agent-failure-probes
S="ssh -i $HOME/.ssh/box2_ed25519 -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"
O=results/multistep/textcraft_qwen3-1.7b/depth_ge2; R=runs/ms-tc-1p7b
note() { echo "- $(date -u +%F\ %H:%M) UTC [cpu2] $*" | tee -a logs/runner_status.md; }
note "orchestrator start (helper box $IP2)"
# 1. stop depth>=2 work on box 1 and hold the lock for tc2
tmux kill-session -t fast 2>/dev/null; tmux kill-session -t d2x 2>/dev/null
sleep 2; pkill -f 'mstep.probes.*depth>=2' ; sleep 3
sleep 999999 & HOLD=$!
echo $HOLD > $O/probes_all.lock
rm -f $O/probes_3X_*.lock
note "stopped box-1 depth>=2 jobs; holding $O/probes_all.lock (pid $HOLD)"
# 2. ship code + data
rsync -az -e "$S" --exclude .venv --exclude runs --exclude logs --exclude results --exclude writeup --exclude __pycache__ ./ ubuntu@$IP2:agent-failure-probes/
$S ubuntu@$IP2 "mkdir -p agent-failure-probes/$R/acts/by_layer/depthge2 agent-failure-probes/$O"
rsync -az -e "$S" $R/rounds.pkl ubuntu@$IP2:agent-failure-probes/$R/
rsync -az -e "$S" $R/acts/extraction.json ubuntu@$IP2:agent-failure-probes/$R/acts/
rsync -a -e "$S" $R/acts/by_layer/depthge2/last_L*.npy ubuntu@$IP2:agent-failure-probes/$R/acts/by_layer/depthge2/
rsync -az -e "$S" $O/phase2_cells.csv $O/phase2_baselines.csv $O/phase3_layers.json ubuntu@$IP2:agent-failure-probes/$O/
# full-set layer files only need to exist (subset files are already built): empty placeholders
$S ubuntu@$IP2 "cd agent-failure-probes/$R/acts/by_layer && for l in \$(seq -w 0 28); do touch last_L\$l.npy; done && ls depthge2 | wc -l"
note "data copied; starting helper run"
$S ubuntu@$IP2 "tmux new -d -s run 'bash ~/agent-failure-probes/scripts/cpu2_run.sh; exec bash'"
# 3. wait, pull back
until $S ubuntu@$IP2 'test -f ~/cpu2_done' 2>/dev/null; do sleep 60; done
$S ubuntu@$IP2 'tail -4 ~/cpu2_run.log' | tr '\n' ' ' | xargs -0 -I{} echo "- $(date -u +%F\ %H:%M) UTC [cpu2] helper: {}" >> logs/runner_status.md
rsync -az -e "$S" --exclude '*.lock' ubuntu@$IP2:agent-failure-probes/$O/ $O/
date -u +%F\ %T > $O/DONE_all
kill $HOLD; rm -f $O/probes_all.lock
note "results pulled into $O; DONE_all written, lock released"
.venv/bin/python -m mstep.figures --out $O >> logs/cpu2_figs.log 2>&1 && note "depth>=2 figures done" || note "FAILED depth>=2 figures (see logs/cpu2_figs.log)"
# 4. helper terminates itself
$S ubuntu@$IP2 'curl -fsS -u "$(cat ~/.lambda_api_key):" -X POST https://cloud.lambda.ai/api/v1/instance-operations/terminate -H "Content-Type: application/json" -d "{\"instance_ids\":[\"$(cat ~/.self_instance_id)\"]}" >/dev/null && echo terminate-requested' && note "helper box termination requested" || note "WARNING: helper box self-terminate FAILED — terminate afp-cpu-2 manually"
