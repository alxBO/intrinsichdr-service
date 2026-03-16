#!/bin/bash
# Deploy IntrinsicHDR to a Vast.ai instance from your local machine.
#
# Usage:
#   ./remote_deploy.sh <user@host> [ssh-options...]
#
# Examples:
#   ./remote_deploy.sh root@203.0.113.5 -p 12345
#   ./remote_deploy.sh root@203.0.113.5 -p 12345 -i ~/.ssh/vastai_key
#
# Streams remote logs until the service is ready, then detaches.
# The service keeps running in tmux session "intrinsichdr" on the remote.
# Reattach with: ssh <host> -t tmux attach -t intrinsichdr

set -euo pipefail

SERVICE_NAME="intrinsichdr-service"
REPO_URL="https://github.com/alxBO/intrinsichdr-service.git"
TMUX_SESSION="intrinsichdr"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <user@host> [ssh-options...]"
    echo "  e.g. $0 root@203.0.113.5 -p 12345"
    echo "  e.g. $0 root@203.0.113.5 -p 12345 -i ~/.ssh/vastai_key"
    exit 1
fi

SSH_ARGS=("$@")

# Vast.ai instances are ephemeral — skip host key checks
SSH_OPTS=(-o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)

echo "=== Remote Deploy: $SERVICE_NAME ==="
echo "Target: ${SSH_ARGS[0]}"
echo ""

ssh "${SSH_OPTS[@]}" "${SSH_ARGS[@]}" bash -s <<REMOTE
set -e

SERVICE="$SERVICE_NAME"
REPO="$REPO_URL"
SESSION="$TMUX_SESSION"
LOG="/tmp/deploy_\${SERVICE}.log"

# --- Clone or update ---
if [ ! -d ~/"\$SERVICE" ]; then
    echo "[\$SERVICE] Cloning..."
    git clone --recurse-submodules "\$REPO" ~/"\$SERVICE"
else
    echo "[\$SERVICE] Updating..."
    cd ~/"\$SERVICE"
    git pull --ff-only || true
    git submodule update --init
fi

# --- Kill previous session ---
tmux kill-session -t "\$SESSION" 2>/dev/null || true
rm -f "\$LOG"
touch "\$LOG"

# --- Start deploy in tmux ---
tmux new-session -d -s "\$SESSION" \\
    "cd ~/\$SERVICE/service && ./deploy_vastai.sh 2>&1 | tee \$LOG; echo DEPLOY_FAILED >> \$LOG"

# --- Stream logs until service is up or deploy fails ---
tail -f "\$LOG" | sed '/Uvicorn running/q; /DEPLOY_FAILED/q'

# --- Report result ---
if grep -q "Uvicorn running" "\$LOG"; then
    echo ""
    echo "=========================================="
    echo "  \$SERVICE is running!"
    echo "  tmux session: \$SESSION"
    echo "  Reattach:  tmux attach -t \$SESSION"
    echo "=========================================="
else
    echo ""
    echo "Deploy failed. Reattach to inspect:"
    echo "  tmux attach -t \$SESSION"
    exit 1
fi
REMOTE

echo ""
echo "Detached. Service running on remote."
