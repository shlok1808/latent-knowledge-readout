#!/usr/bin/env bash
# Start the three AgentGym env servers (Phase 4 candidates), each in its own tmux
# session, exactly as preflight_box.sh launched them. Idempotent.
cd ~
start() {  # start <name> <port> <cmd...>
  local name=$1 port=$2; shift 2
  if curl -fsS 127.0.0.1:$port/ >/dev/null 2>&1; then echo "$name already answering on $port (existing server kept)"; return; fi
  if tmux has-session -t envsrv_$name 2>/dev/null; then echo "envsrv_$name session exists but port $port silent; see ~/env-$name.log"; return; fi
  # ScienceWorld opens sockets/files per env (py4j + JVM); the default 1024 fd limit
  # was exhausted at 80 concurrent envs ("Too many open files", 2026-09-23).
  tmux new -d -s envsrv_$name "ulimit -n 65536; $* --host 127.0.0.1 --port $port 2>&1 | tee -a ~/env-$name.log"
  for i in $(seq 1 90); do curl -fsS 127.0.0.1:$port/ >/dev/null 2>&1 && { echo "envsrv_$name up on $port"; return; }; sleep 2; done
  echo "envsrv_$name FAILED to answer on $port"; tail -20 ~/env-$name.log
}
declare -A PORT=([sciworld]=36001 [alfworld]=36002 [babyai]=36003)
declare -A CMD=([sciworld]="$HOME/envs/sciworld/bin/sciworld"
                [alfworld]="env ALFWORLD_DATA=$HOME/.cache/alfworld $HOME/envs/alfworld/bin/alfworld"
                [babyai]="$HOME/envs/babyai/bin/babyai")
if [ "${1:-}" = start_n ] || [ "${1:-}" = restart_n ]; then
  # start_n|restart_n <env> <N>: N parallel servers on ports 36100..36100+N-1 (alfworld).
  name=$2; n=$3
  for k in $(seq 0 $((n - 1))); do
    port=$((36100 + k))
    if [ "$1" = restart_n ]; then
      pids=$(lsof -t -i TCP:$port -s TCP:LISTEN 2>/dev/null); [ -n "$pids" ] && kill $pids
      tmux kill-session -t envsrv_${name}_$k 2>/dev/null
    fi
  done
  [ "$1" = restart_n ] && sleep 3
  for k in $(seq 0 $((n - 1))); do start ${name}_$k $((36100 + k)) ${CMD[$name]} & done
  wait
  exit
fi
if [ "${1:-}" = restart ]; then
  # restart <name>: kill whatever holds the port (the ALFWorld server has no /close, so
  # its envs leak; the rollout engine calls this between chunks), then start fresh.
  name=$2; port=${PORT[$name]}
  pids=$(lsof -t -i TCP:$port -s TCP:LISTEN 2>/dev/null)
  [ -n "$pids" ] && kill $pids && sleep 3
  tmux kill-session -t envsrv_$name 2>/dev/null
  start $name $port ${CMD[$name]}
  exit
fi
for name in sciworld alfworld babyai; do start $name ${PORT[$name]} ${CMD[$name]}; done
