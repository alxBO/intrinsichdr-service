#!/bin/bash
# Stream logs from a running IntrinsicHDR service on a remote Vast.ai instance.
#
# Usage:
#   ./remote_logs.sh <user@host> [ssh-options...]
#   ./remote_logs.sh <user@host> -p 12345 --tail 50
#
# Options (must come after ssh options):
#   --tail N    Show last N lines then follow (default: follow from now)
#   --dump      Print full log and exit (no follow)
#   --attach    Attach to the tmux session directly (interactive)

set -euo pipefail

SERVICE_NAME="intrinsichdr-service"
TMUX_SESSION="intrinsichdr"

MODE="follow"
TAIL_LINES=0
SSH_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --tail)   MODE="follow"; TAIL_LINES="$2"; shift 2 ;;
        --dump)   MODE="dump"; shift ;;
        --attach) MODE="attach"; shift ;;
        *)        SSH_ARGS+=("$1"); shift ;;
    esac
done

if [ ${#SSH_ARGS[@]} -lt 1 ]; then
    echo "Usage: $0 <user@host> [ssh-options...] [--tail N | --dump | --attach]"
    exit 1
fi

LOG="/tmp/deploy_${SERVICE_NAME}.log"

case "$MODE" in
    attach)
        echo "Attaching to tmux session '$TMUX_SESSION'... (detach with Ctrl-B D)"
        ssh -o ConnectTimeout=10 -t "${SSH_ARGS[@]}" "tmux attach -t $TMUX_SESSION"
        ;;
    dump)
        ssh -o ConnectTimeout=10 "${SSH_ARGS[@]}" "cat $LOG 2>/dev/null || echo 'No log file found.'"
        ;;
    follow)
        if [ "$TAIL_LINES" -gt 0 ] 2>/dev/null; then
            echo "=== Last $TAIL_LINES lines + live follow (Ctrl-C to stop) ==="
            ssh -o ConnectTimeout=10 "${SSH_ARGS[@]}" "tail -n $TAIL_LINES -f $LOG 2>/dev/null || echo 'No log file found.'"
        else
            echo "=== Live logs (Ctrl-C to stop) ==="
            ssh -o ConnectTimeout=10 "${SSH_ARGS[@]}" "tail -f $LOG 2>/dev/null || echo 'No log file found.'"
        fi
        ;;
esac
