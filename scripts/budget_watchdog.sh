#!/usr/bin/env bash
# Box-side budget guard. Box-hours = this instance's uptime + hours already spent on
# earlier instances (OFFSET_H). At CAP_H - 0.5 it stops all GPU jobs and writes a
# note; Claude (or Shlok) then pulls results and terminates the instance from the Mac.
#   tmux new -d -s watchdog "bash scripts/budget_watchdog.sh 40 0.1"
CAP_H=${1:-40}; OFFSET_H=${2:-0}
cd ~/agent-failure-probes; mkdir -p logs
while :; do
  up=$(awk '{print $1/3600}' /proc/uptime)
  used=$(python3 -c "print(round($up + $OFFSET_H, 2))")
  echo "$(date -u +%F\ %H:%M) box_hours=$used cap=$CAP_H" > logs/budget.txt
  if python3 -c "import sys; sys.exit(0 if $used >= $CAP_H - 0.5 else 1)"; then
    echo "- $(date -u +%F\ %H:%M) UTC BUDGET STOP at $used box-hours (cap $CAP_H): killing GPU jobs" >> logs/runner_status.md
    pkill -f "mstep\." ; pkill -f "vllm" ; exit 0
  fi
  sleep 300
done
