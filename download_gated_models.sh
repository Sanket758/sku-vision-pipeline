#!/usr/bin/env bash
#
# download_gated_models.sh
#
# Downloads DINOv3 ViT-S/16+ pretrained weights from Meta's gated model
# repository. These weights are available only after signing the research
# agreement at https://github.com/facebookresearch/dinov3.
#
# Usage:
#   ./download_gated_models.sh
#
# The script is idempotent: it skips the download if the expected model file
# is already present and passes validation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/models/dinov3"
MODEL_FILE="${MODEL_DIR}/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
MIN_EXPECTED_BYTES=$((100 * 1024 * 1024))   # 100 MB
MAX_EXPECTED_BYTES=$((120 * 1024 * 1024))   # 120 MB

# ---------------------------------------------------------------------------
#  Helper: pretty-print sizes
# ---------------------------------------------------------------------------
bytes_to_mb() {
    echo "$(awk "BEGIN { printf \"%.0f\", $1 / (1024*1024) }")"
}

# ---------------------------------------------------------------------------
#  Check if model already exists and is valid
# ---------------------------------------------------------------------------
if [ -f "$MODEL_FILE" ]; then
    ACTUAL_SIZE=$(stat -c%s "$MODEL_FILE" 2>/dev/null || stat -f%z "$MODEL_FILE" 2>/dev/null)
    if [ "$ACTUAL_SIZE" -ge "$MIN_EXPECTED_BYTES" ] && [ "$ACTUAL_SIZE" -le "$MAX_EXPECTED_BYTES" ]; then
        echo "[OK] DINOv3 ViT-S/16+ weights found and valid ($(bytes_to_mb $ACTUAL_SIZE) MB). Nothing to do."
        exit 0
    else
        echo "[WARN] Existing file size ($(bytes_to_mb $ACTUAL_SIZE) MB) is outside expected range (100-120 MB)."
        echo "       Re-downloading ..."
        rm -f "$MODEL_FILE"
    fi
fi

# ---------------------------------------------------------------------------
#  Ensure target directory exists
# ---------------------------------------------------------------------------
mkdir -p "$MODEL_DIR"

# ---------------------------------------------------------------------------
#  Print instructions
# ---------------------------------------------------------------------------
echo "======================================================================="
echo " DINOv3 ViT-S/16+ Pretrained Weights"
echo "======================================================================="
echo ""
echo "These weights are provided by Meta (Facebook) under a research license."
echo "You must accept the terms before downloading."
echo ""
echo "Steps:"
echo "  1. Open https://github.com/facebookresearch/dinov3"
echo "  2. Follow the link to the official model release"
echo "  3. Sign the research agreement if prompted"
echo "  4. Obtain the direct download URL for:"
echo "     dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
echo ""
echo "Once you have the URL, run the download manually:"
echo ""
echo "    wget -O \"${MODEL_FILE}\" \"<PASTE_URL_HERE>\""
echo ""
echo "Or, set the environment variable below and re-run this script:"
echo ""
echo "    export DINO_VITS16_URL=\"https://dl.fbaipublicfiles.com/dinov3/...\""
echo "    ./download_gated_models.sh"
echo ""

# ---------------------------------------------------------------------------
#  Try automated download if URL is provided via environment variable
# ---------------------------------------------------------------------------
if [ -n "${DINO_VITS16_URL:-}" ]; then
    echo "[INFO] DINO_VITS16_URL is set. Attempting download ..."
    wget -O "$MODEL_FILE" "$DINO_VITS16_URL" || {
        echo "[ERROR] Download failed. Check the URL or your network connection."
        exit 1
    }

    # Validate
    ACTUAL_SIZE=$(stat -c%s "$MODEL_FILE" 2>/dev/null || stat -f%z "$MODEL_FILE" 2>/dev/null)
    if [ "$ACTUAL_SIZE" -ge "$MIN_EXPECTED_BYTES" ] && [ "$ACTUAL_SIZE" -le "$MAX_EXPECTED_BYTES" ]; then
        echo "[OK] Download successful. File size: $(bytes_to_mb $ACTUAL_SIZE) MB."
        echo "Model saved to: ${MODEL_FILE}"
        exit 0
    else
        echo "[ERROR] Downloaded file size ($(bytes_to_mb $ACTUAL_SIZE) MB) is outside the expected range (100-120 MB)."
        rm -f "$MODEL_FILE"
        exit 1
    fi
else
    echo "[INFO] DINO_VITS16_URL not set. Skipping automated download."
    echo ""
    echo "After downloading manually, re-run this script to validate the file."
    exit 0
fi
