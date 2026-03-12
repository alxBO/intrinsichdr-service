#!/bin/bash
# Run IntrinsicHDR service locally on Mac
#
# Prerequisites:
#   pip install -r backend/requirements.txt
#
# SingleHDR linearization weights must be in service/weights/:
#   - dequantization.pt
#   - linearization.pt
#   - invemor.txt
#
# To convert from TF checkpoints (on a machine with TF installed):
#   cd backend && python convert_linearization_weights.py \
#     --ckpt_deq /path/to/ckpt_deq/model.ckpt \
#     --ckpt_lin /path/to/ckpt_lin/model.ckpt \
#     --output_dir ../weights
#   curl -sL https://raw.githubusercontent.com/alex04072000/SingleHDR/master/invemor.txt \
#     -o ../weights/invemor.txt
#
# Or copy them from singlehdr-service:
#   cp ../singlehdr-service/service/weights/basic/dequantization.pt weights/
#   cp ../singlehdr-service/service/weights/basic/linearization.pt weights/
#   cp ../singlehdr-service/vendor/SingleHDR/invemor.txt weights/

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Check linearization weights
WEIGHTS_DIR="$SCRIPT_DIR/weights"
if [ ! -f "$WEIGHTS_DIR/dequantization.pt" ] || [ ! -f "$WEIGHTS_DIR/linearization.pt" ]; then
    echo "ERROR: SingleHDR linearization weights not found in $WEIGHTS_DIR/"
    echo "See comments in this script for setup instructions."
    exit 1
fi
if [ ! -f "$WEIGHTS_DIR/invemor.txt" ]; then
    echo "Downloading invemor.txt..."
    mkdir -p "$WEIGHTS_DIR"
    curl -sL "https://raw.githubusercontent.com/alex04072000/SingleHDR/master/invemor.txt" \
        -o "$WEIGHTS_DIR/invemor.txt"
fi

cd "$SCRIPT_DIR/backend"
export MAX_MEGAPIXELS=50
export JOB_TTL_HOURS=24
echo "Starting IntrinsicHDR service on http://localhost:8004"
echo "IntrinsicHDR models will auto-download on first run (~500 MB total)"
echo "SingleHDR linearization weights: $WEIGHTS_DIR"
exec uvicorn app.main:app --host 0.0.0.0 --port 8004 --workers 1
