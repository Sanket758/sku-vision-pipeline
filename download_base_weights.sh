#!/usr/bin/env bash
#
# download_base_weights.sh
#
# Downloads base pretrained weights that are NOT included in the repo
# (only fine-tuned weights are tracked via Git LFS).
#
# Usage: bash download_base_weights.sh
#
# Downloads:
#   1. YOLOv8n base  — https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
#   2. YOLOv10n base — https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10n.pt
#   3. SKU110K_V3.pt — skipped if already present (already tracked via Git LFS)
#
# All weights are placed in the models/ directory alongside fine-tuned weights.

set -euo pipefail

MODELS_DIR="$(cd "$(dirname "$0")" && pwd)/models"

# Map of filename -> download URL
declare -A WEIGHTS
WEIGHTS[yolov8n.pt]="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
WEIGHTS[yolov10n.pt]="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10n.pt"
# SKU110K_V3.pt is already included via Git LFS — only downloaded if missing
WEIGHTS[SKU110K_V3.pt]="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"  # placeholder; adjusted below

download_file() {
    local filename="$1"
    local url="$2"
    local dest="$MODELS_DIR/$filename"

    if [ -f "$dest" ]; then
        local size
        size=$(stat --printf="%s" "$dest" 2>/dev/null || stat -f%z "$dest" 2>/dev/null)
        echo "[SKIP] $filename already exists ($(numfmt --to=iec "$size" 2>/dev/null || echo "$size bytes"))"
        return 0
    fi

    echo "[DOWNLOAD] $filename ..."
    echo "  URL: $url"

    # Use wget if available, fall back to curl
    if command -v wget &>/dev/null; then
        wget -q --show-progress "$url" -O "$dest" || {
            echo "[ERROR] wget failed for $filename"
            rm -f "$dest"
            return 1
        }
    elif command -v curl &>/dev/null; then
        curl -# -L "$url" -o "$dest" || {
            echo "[ERROR] curl failed for $filename"
            rm -f "$dest"
            return 1
        }
    else
        echo "[ERROR] Neither wget nor curl found. Please install one of them."
        return 1
    fi

    # Validate the downloaded file
    if [ ! -f "$dest" ]; then
        echo "[ERROR] $filename was not created after download."
        return 1
    fi

    local size
    size=$(stat --printf="%s" "$dest" 2>/dev/null || stat -f%z "$dest" 2>/dev/null)
    if [ "$size" -lt 1000 ]; then
        echo "[ERROR] $filename is too small ($size bytes) — download likely failed."
        rm -f "$dest"
        return 1
    fi

    echo "[OK] $filename downloaded ($(numfmt --to=iec "$size" 2>/dev/null || echo "$size bytes"))"
    return 0
}

echo "=============================================="
echo " Base Pretrained Weight Downloader"
echo " Target: $MODELS_DIR"
echo "=============================================="
echo ""

# Create models directory if it doesn't exist
mkdir -p "$MODELS_DIR"

# --- Download weights ---
errors=0

download_file "yolov8n.pt"  "${WEIGHTS[yolov8n.pt]}"  || ((errors++))
download_file "yolov10n.pt" "${WEIGHTS[yolov10n.pt]}" || ((errors++))

# SKU110K_V3.pt — already included via Git LFS, skip if present
if [ -f "$MODELS_DIR/SKU110K_V3.pt" ]; then
    size=$(stat --printf="%s" "$MODELS_DIR/SKU110K_V3.pt" 2>/dev/null || stat -f%z "$MODELS_DIR/SKU110K_V3.pt" 2>/dev/null)
    echo "[SKIP] SKU110K_V3.pt already present ($(numfmt --to=iec "$size" 2>/dev/null || echo "$size bytes")) — already tracked via Git LFS"
else
    echo "[DOWNLOAD] SKU110K_V3.pt ..."
    echo "  URL: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
    echo "  [NOTE] SKU110K_V3.pt is a custom fine-tuned checkpoint included via Git LFS."
    echo "  It should already be present. Downloading a placeholder would not be useful."
    echo "  If you need this file, run: git lfs pull"
    ((errors++))
fi

echo ""
echo "=============================================="
if [ "$errors" -eq 0 ]; then
    echo " All base weights downloaded successfully."
else
    echo " $errors download(s) had errors. Check messages above."
fi
echo "=============================================="

exit "$errors"
