#!/usr/bin/env bash
# Laptop side: get a fresh Lambda instance ready for this project, then run bootstrap.sh on it.
# Usage: scripts/lambda_up.sh <instance-ip> [branch]      (branch defaults to the current one)
# How the whole setup works: ~/claude-mods/research/SETUP.md
set -euo pipefail

IP="${1:?usage: scripts/lambda_up.sh <instance-ip> [branch]}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
NAME="$(basename "$REPO_DIR")"
BRANCH="${2:-$(git -C "$REPO_DIR" branch --show-current)}"
SLUG="$(git -C "$REPO_DIR" remote get-url origin | sed -E 's#^(https://github.com/|git@github.com:)##; s#\.git$##')"
SSH_CONFIG="$HOME/.ssh/config"

echo "== 1/5 Point 'Host lambda' at $IP"
touch "$SSH_CONFIG" && chmod 600 "$SSH_CONFIG"
if grep -Eq '^Host[[:space:]]+lambda[[:space:]]*$' "$SSH_CONFIG"; then
  python3 - "$SSH_CONFIG" "$IP" <<'PY'
import re, sys
path, ip = sys.argv[1], sys.argv[2]
text = open(path).read()
block = re.search(r"(?msi)^Host[ \t]+lambda[ \t]*\n(.*?)(?=^Host[ \t]|\Z)", text)
body = re.sub(r"(?mi)^(\s*HostName\s+).*$", lambda m: m.group(1) + ip, block.group(1))
open(path, "w").write(text[: block.start(1)] + body + text[block.end(1):])
PY
else
  cat >> "$SSH_CONFIG" <<EOF

Host lambda
  HostName $IP
  User ubuntu
  # Launch instances with Lambda key 'afp-lambda-key' (= lambda_key, no passphrase). 'final' (= lambda_final) has a
  # passphrase: scripts and Desktop can't type it until you run once: ssh-add --apple-use-keychain ~/.ssh/lambda_final
  IdentityFile ~/.ssh/lambda_key
  IdentityFile ~/.ssh/lambda_final
  IdentitiesOnly yes
  IgnoreUnknown UseKeychain
  UseKeychain yes
  AddKeysToAgent yes
  StrictHostKeyChecking accept-new
  ServerAliveInterval 60
EOF
fi
[ "$(ssh -F "$SSH_CONFIG" -G lambda | awk '/^hostname /{print $2}')" = "$IP" ] || { echo "Host lambda still doesn't point at $IP; fix ~/.ssh/config by hand"; exit 1; }
ssh -o ConnectTimeout=15 lambda true || { echo "Can't reach lambda at $IP. Is the instance booted (3-15 min)?"; exit 1; }

echo "== 2/5 Copy global CLAUDE.md and personal skills (SSH sessions read the instance's ~/.claude)"
ssh lambda 'mkdir -p ~/.claude'
rsync -a "$HOME/.claude/CLAUDE.md" lambda:.claude/CLAUDE.md
rsync -aL --delete "$HOME/.claude/skills/" lambda:.claude/skills/

echo "== 3/5 GitHub access for clone and push (token lives only on the instance)"
# Prefer a token scoped to the research repos, stored once in the keychain:
#   security add-generic-password -s lambda-github-token -a "$USER" -w <token>
if TOKEN="$(security find-generic-password -s lambda-github-token -w 2>/dev/null)"; then
  echo "using the scoped token from the keychain"
else
  TOKEN="$(gh auth token)"
  echo "WARNING: using your full gh token (all repos). See SETUP.md to scope it."
fi
printf '%s\n' "$TOKEN" | ssh lambda 'umask 077; read -r t; printf "https://x-access-token:%s@github.com\n" "$t" > ~/.git-credentials; git config --global credential.helper store'
ssh lambda "git config --global user.name '$(git config user.name)'; git config --global user.email '$(git config user.email)'"
# HF token: from `hf auth login`, else HF_TOKEN in ~/.kyc-secrets (the file mechanistic-kyc/scripts/lambda.sh uses)
HF="$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)"
[ -n "$HF" ] || HF="$( [ -f "$HOME/.kyc-secrets" ] && . "$HOME/.kyc-secrets"; printf '%s' "${HF_TOKEN:-}")"
if [ -n "$HF" ]; then
  printf '%s\n' "$HF" | ssh lambda 'umask 077; mkdir -p ~/.cache/huggingface; read -r t; printf "%s" "$t" > ~/.cache/huggingface/token'
else
  echo "WARNING: no HF token on the laptop (run 'hf auth login'). Gated models and 'hf upload' will fail on the instance."
fi

echo "== 4/5 Clone or update $SLUG ($BRANCH)"
ssh lambda "set -e
if [ -d ~/$NAME/.git ]; then
  git -C ~/$NAME fetch -q origin && git -C ~/$NAME checkout -q $BRANCH && git -C ~/$NAME pull -q --ff-only
else
  git clone -q --branch $BRANCH https://github.com/$SLUG.git ~/$NAME
fi"
# .claude/ is not in git; the project settings enable the Scribe MCP, so copy them over
if [ -f "$REPO_DIR/.claude/settings.json" ]; then
  ssh lambda "mkdir -p ~/$NAME/.claude"
  rsync -a "$REPO_DIR/.claude/settings.json" "lambda:$NAME/.claude/settings.json"
else
  echo "WARNING: no .claude/settings.json on the laptop; Scribe MCP won't be enabled on the instance."
fi

echo "== 5/5 Bootstrap on the instance"
ssh -t lambda "cd ~/$NAME && bash scripts/bootstrap.sh"

echo "Done. In Desktop: new session -> environment 'lambda' -> folder ~/$NAME"
