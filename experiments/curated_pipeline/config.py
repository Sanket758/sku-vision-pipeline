"""Configuration for the curated SKU labeling pipeline."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = Path(__file__).resolve().parent

DATA_DIR = PIPELINE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
CROPS_DIR = DATA_DIR / "crops"
EXPORTS_DIR = DATA_DIR / "exports"
YOLO_EXPORT_DIR = EXPORTS_DIR / "yolo"
RETRIEVAL_EXPORT_DIR = EXPORTS_DIR / "retrieval"

REGISTRY_FILE = PIPELINE_DIR / "sku_registry.json"
SESSIONS_DIR = DATA_DIR / "sessions"

# ── Detection ──────────────────────────────────────────────────────────────
SKU110K_MODEL_PATH = str(PROJECT_ROOT / "models" / "SKU110K_V3.pt")
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.45
DETECTION_IMAGE_SIZE = 1280

# ── Embeddings ─────────────────────────────────────────────────────────────
DINOV3_MODEL_PATH = str(
    PROJECT_ROOT / "models" / "dinov3" / "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
)
DINOV3_MODEL_NAME = "dinov3_vits16plus"
DINOV3_DIM = 384

MOBILENET_INPUT_SIZE = 224
MOBILENET_DIM = 1280

# Fusion weights for combined similarity
DINOV3_WEIGHT = 0.55
MOBILENET_WEIGHT = 0.45

# ── Matching ───────────────────────────────────────────────────────────────
HIGH_CONFIDENCE = 0.88      # auto-suggest zone
LOW_CONFIDENCE = 0.65       # below this → propose new SKU
POSITIONAL_BONUS = 0.10     # same-shelf-row bonus
POSITIONAL_ROW_EPS = 0.05   # DBSCAN eps for y-center grouping (normalized)

# ── Review Server ──────────────────────────────────────────────────────────
REVIEW_HOST = "127.0.0.1"
REVIEW_PORT = 8765

# ── Categories ─────────────────────────────────────────────────────────────
CATEGORIES = ["Beverages", "Chocolates", "Confectionery", "Dairy", "PCP", "Frozen", "Household", "Other", "PrivateLabel"]
