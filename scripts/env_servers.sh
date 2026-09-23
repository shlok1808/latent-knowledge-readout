#!/usr/bin/env bash
# Start the three AgentGym env servers (Phase 4 candidates), each in its own tmux
# session, exactly as preflight_box.sh launched them. Idempotent.
cd ~
start() {  # start <name> <port> <cmd...>
  local name=$1 port=$2; shift 2
  if curl -fsS 127.0.0.1:$port/ >/dev/null 2>&1; then echo "$name already answering on $port (existing server kept)"; return; fi
  if tmux has-session -t envsrv_$name 2>/dev/null; then echo "envsrv_$name session exists but port $port silent; see ~/env-$name.log"; return; fi
  tmux new -d -s envsrv_$name "$* --host 127.0.0.1 --port $port 2>&1 | tee -a ~/env-$name.log"
  for i in $(seq 1 90); do curl -fsS 127.0.0.1:$port/ >/dev/null 2>&1 && { echo "envsrv_$name up on $port"; return; }; sleep 2; done
  echo "envsrv_$name FAILED to answer on $port"; tail -20 ~/env-$name.log
}
declare -A PORT=([sciworld]=36001 [alfworld]=36002 [babyai]=36003)
declare -A CMD=([sciworld]="$HOME/envs/sciworld/bin/sciworld"
                [alfworld]="env ALFWORLD_DATA=$HOME/.cache/alfworld $HOME/envs/alfworld/bin/alfworld"
                [babyai]="$HOME/envs/babyai/bin/babyai")
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
