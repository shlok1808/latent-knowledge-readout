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
start sciworld 36001 ~/envs/sciworld/bin/sciworld
start alfworld 36002 env ALFWORLD_DATA=$HOME/.cache/alfworld ~/envs/alfworld/bin/alfworld
start babyai 36003 ~/envs/babyai/bin/babyai
