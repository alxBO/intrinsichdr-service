#!/bin/bash
# Run IntrinsicHDR service locally on Mac
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$SCRIPT_DIR/backend"
export MAX_MEGAPIXELS=50
export JOB_TTL_HOURS=24
echo "Starting IntrinsicHDR service on http://localhost:8004"
echo "Models will auto-download on first run (~500 MB total)"
exec uvicorn app.main:app --host 0.0.0.0 --port 8004 --workers 1
