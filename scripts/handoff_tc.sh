#!/usr/bin/env bash
# Wait until tc17 (run_tc.sh v1) finishes collection, then replace it with run_tc2.sh.
cd ~/agent-failure-probes
until grep -q "\[ms-tc-1p7b\] done collect" logs/runner_status.md; do sleep 20; done
sleep 5
tmux kill-session -t tc17
sleep 5
pkill -f "mstep.extract --run runs/ms-tc-1p7b" ; sleep 5
echo "- $(date -u +%F\ %H:%M) UTC [handoff] tc17 (v1) stopped after collect; starting tc2 (MC parallel to analysis)" >> logs/runner_status.md
tmux new -d -s tc2 "bash scripts/run_tc2.sh configs/ms_tc_1p7b.json runs/ms-tc-1p7b results/multistep/textcraft_qwen3-1.7b; exec bash"
