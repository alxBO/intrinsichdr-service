#!/bin/bash
# Deploy IntrinsicHDR to a remote GPU instance from your local machine.
#
# Usage:
#   ./remote_deploy.sh [--workdir DIR] <user@host> [ssh-options...]
#
# Options:
#   --workdir DIR   Remote directory to clone into (default: ~)
#
# Examples:
#   ./remote_deploy.sh root@203.0.113.5 -p 12345
#   ./remote_deploy.sh root@203.0.113.5 -p 12345 -i ~/.ssh/vastai_key
#   ./remote_deploy.sh --workdir /workspace root@203.0.113.5 -p 12345
#
# Streams remote logs until the service is ready, then detaches.
# The service keeps running in tmux session "intrinsichdr" on the remote.
# Reattach with: ssh <host> -t tmux attach -t intrinsichdr

set -euo pipefail

SERVICE_NAME="intrinsichdr-service"
REPO_URL="https://github.com/alxBO/intrinsichdr-service.git"
TMUX_SESSION="intrinsichdr"
REMOTE_WORKDIR="~"

# Parse our options (before SSH args)
while [ $# -gt 0 ]; do
    case "$1" in
        --workdir) REMOTE_WORKDIR="$2"; shift 2 ;;
        *) break ;;
    esac
done

if [ $# -lt 1 ]; then
    echo "Usage: $0 [--workdir DIR] <user@host> [ssh-options...]"
    echo "  e.g. $0 root@203.0.113.5 -p 12345 -i ~/.ssh/vastai_key"
    echo "  e.g. $0 --workdir /workspace root@203.0.113.5 -p 12345"
    exit 1
fi

SSH_ARGS=("$@")

# Ephemeral instances — skip host key checks
SSH_OPTS=(-o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)

echo "=== Remote Deploy: $SERVICE_NAME ==="
echo "Target: ${SSH_ARGS[0]}"
echo "Workdir: $REMOTE_WORKDIR"
echo ""

ssh "${SSH_OPTS[@]}" "${SSH_ARGS[@]}" bash -s <<REMOTE
set -e

SERVICE="$SERVICE_NAME"
REPO="$REPO_URL"
SESSION="$TMUX_SESSION"
LOG="/tmp/deploy_\${SERVICE}.log"

WORK="$REMOTE_WORKDIR"

# --- Clone or update ---
if [ ! -d "\$WORK/\$SERVICE" ]; then
    echo "[\$SERVICE] Cloning into \$WORK/\$SERVICE..."
    mkdir -p "\$WORK"
    git clone --recurse-submodules "\$REPO" "\$WORK/\$SERVICE"
else
    echo "[\$SERVICE] Updating \$WORK/\$SERVICE..."
    cd "\$WORK/\$SERVICE"
    git pull --ff-only || true
    git submodule update --init
fi

# --- Kill previous session ---
tmux kill-session -t "\$SESSION" 2>/dev/null || true
rm -f "\$LOG"
touch "\$LOG"

# --- Start deploy in tmux ---
tmux new-session -d -s "\$SESSION" \\
    "cd \$WORK/\$SERVICE/service && ./deploy_vastai.sh 2>&1 | tee -a \$LOG; echo DEPLOY_FAILED >> \$LOG"

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
