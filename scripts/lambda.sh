#!/usr/bin/env bash
# Lambda Cloud helper for the multistep-gap experiments. Run from the repo root on the Mac.
# Keys come from ~/.kyc-secrets (LAMBDA_API_KEY, HF_TOKEN); never echoed.
#
#   scripts/lambda.sh ip          # print the instance IP (waits until active)
#   scripts/lambda.sh ssh [cmd]   # run a command on the box
#   scripts/lambda.sh sync        # rsync the repo (code + configs, not runs/) to the box
#   scripts/lambda.sh pull <dir>  # rsync <dir> from the box back into the same path here
#   scripts/lambda.sh status      # instance status
#   scripts/lambda.sh terminate   # stop billing (asks nothing; only call deliberately)
set -euo pipefail
[ -f "$HOME/.kyc-secrets" ] && source "$HOME/.kyc-secrets"
API="https://cloud.lambda.ai/api/v1"
KEY="$HOME/.ssh/lambda_key"   # unencrypted, registered as afp-lambda-key; lambda_final needs a passphrase
ID_FILE="${LAMBDA_ID_FILE:-.lambda_instance_id}"   # per-box: LAMBDA_ID_FILE=.lambda_instance_id_alf
REMOTE_DIR="agent-failure-probes"
api() { curl -fsS -u "${LAMBDA_API_KEY:?set LAMBDA_API_KEY}:" "$@"; }
id() { cat "$ID_FILE"; }
field() { api "$API/instances/$(id)" | python3 -c "import json,sys;d=json.load(sys.stdin)['data'];print(d.get('$1') or '-')"; }
ip() {
  while :; do
    s=$(field status); a=$(field ip)
    [ "$s" = active ] && [ "$a" != - ] && { echo "$a"; return; }
    [ "$s" = terminated ] && { echo "instance terminated" >&2; exit 1; }
    sleep 15
  done
}
rsh() { ssh -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 "ubuntu@$(ip)" "$@"; }

case "${1:-}" in
ip) ip ;;
status) field status ;;
ssh) shift; rsh "$@" ;;
sync)
  rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=accept-new" \
    --exclude .venv --exclude runs --exclude __pycache__ --exclude '*.egg-info' \
    --exclude .pytest_cache --exclude writeup --exclude '.lambda_instance*' \
    ./ "ubuntu@$(ip):$REMOTE_DIR/"
  ;;
pull)
  d="${2:?dir}"; mkdir -p "$d"
  rsync -az -e "ssh -i $KEY" "ubuntu@$(ip):$REMOTE_DIR/$d/" "$d/"
  ;;
terminate)
  api -X POST "$API/instance-operations/terminate" -H 'Content-Type: application/json' \
    -d "{\"instance_ids\":[\"$(id)\"]}" >/dev/null && echo "terminate requested for $(id)"
  ;;
*) sed -n 2,10p "$0"; exit 1 ;;
esac
