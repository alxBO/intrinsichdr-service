#!/bin/bash
# Deploy IntrinsicHDR on a Vast.ai GPU instance
#
# === Vast.ai instance setup ===
#
# 1. Choose a GPU instance (RTX 3090+ with >= 12 GB VRAM recommended)
# 2. Use a PyTorch template image (e.g. pytorch/pytorch:2.x-cuda12.x-runtime)
# 3. In "Docker options", add:  -p 8004:8004
# 4. Set disk space to at least 10 GB
#
# === On the instance ===
#
# SSH in, then:
#   git clone --recurse-submodules <repo-url>
#   cd intrinsichdr-service/service
#   ./deploy_vastai.sh
#
# === Access ===
#
# Option A: Click "Open" on the instance card
# Option B: Use direct IP:port from "IP Port Info" popup

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== IntrinsicHDR Vast.ai Deployment ==="
echo ""

# 0. Ensure submodule is checked out
SUBMOD_DIR="$REPO_DIR/vendor/IntrinsicHDR"
if [ ! -f "$SUBMOD_DIR/inference.py" ]; then
    echo "[0/3] Initializing git submodule..."
    rm -rf "$SUBMOD_DIR"
    cd "$REPO_DIR" && git submodule update --init
fi

# 1. Install Python dependencies
echo "[1/3] Installing dependencies..."
pip install -q -r "$SCRIPT_DIR/backend/requirements.txt"

# 2. Download and convert SingleHDR linearization weights if needed
WEIGHTS_DIR="$SCRIPT_DIR/weights"
if [ ! -f "$WEIGHTS_DIR/dequantization.pt" ] || [ ! -f "$WEIGHTS_DIR/linearization.pt" ] || [ ! -f "$WEIGHTS_DIR/invemor.txt" ]; then
    echo "[2/3] Downloading and converting SingleHDR linearization weights..."
    pip install -q gdown tensorflow-cpu

    # Download SingleHDR checkpoints from Google Drive
    gdown --id 1e9vP8YPEjGcvXCa0Bfqwxw7qks7dH-VE -O /tmp/ckpt.zip
    unzip -q -o /tmp/ckpt.zip -d /tmp/ckpt_raw
    rm /tmp/ckpt.zip

    # Find the checkpoint root (handles nested or flat zip structures)
    CKPT_ROOT=$(find /tmp/ckpt_raw -name "ckpt_deq" -type d -print -quit | xargs dirname)
    echo "Found checkpoints at: $CKPT_ROOT"

    # Convert only dequantization + linearization (not hallucination/refinement)
    cd "$SCRIPT_DIR/backend"
    python convert_linearization_weights.py \
        --ckpt_deq "$CKPT_ROOT/ckpt_deq/model.ckpt" \
        --ckpt_lin "$CKPT_ROOT/ckpt_lin/model.ckpt" \
        --output_dir "$WEIGHTS_DIR"

    # Download invemor.txt (PCA basis for inverse CRF) from SingleHDR repo
    if [ ! -f "$WEIGHTS_DIR/invemor.txt" ]; then
        echo "Downloading invemor.txt..."
        curl -sL "https://raw.githubusercontent.com/alex04072000/SingleHDR/master/invemor.txt" \
            -o "$WEIGHTS_DIR/invemor.txt"
    fi

    rm -rf /tmp/ckpt_raw
    echo "Linearization weights ready."
else
    echo "[2/3] Linearization weights already present, skipping download."
fi

# 3. Start the service
# IntrinsicHDR models auto-download via torch.hub on first run (~500 MB total)
echo "[3/3] Starting service on port 8004..."
echo ""

if [ -n "$VAST_TCP_PORT_8004" ]; then
    echo "Direct access: http://$(hostname -I | awk '{print $1}'):$VAST_TCP_PORT_8004"
fi
echo "Local: http://0.0.0.0:8004"
echo ""

cd "$SCRIPT_DIR/backend"
export MAX_MEGAPIXELS=50
export JOB_TTL_HOURS=24

exec uvicorn app.main:app --host 0.0.0.0 --port 8004 --workers 1
