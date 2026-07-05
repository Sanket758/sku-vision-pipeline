"""Build Dataset/master_annotations.jsonl — the single source of truth for all label decisions.

Merges:
  - Dataset/detections.json       (crop bounding boxes from DETR)
  - <ocr_cache>/ocr_results.jsonl   (OCR text suggestions, optional)
  - Dataset/curated_subset/cluster_to_sku.json  (cluster→SKU mapping, optional)
  - Dataset/sku_catalogue/sku_manifest.csv       (valid SKU IDs)

Output: Dataset/master_annotations.jsonl — one JSON object per crop per line.

Usage:
    python experiments/shared/build_master_annotations.py
    python experiments/shared/build_master_annotations.py --ocr experiments/annotation_tool/data/ocr_results.jsonl
    python experiments/shared/build_master_annotations.py --force  # rebuild from scratch
"""

import argparse
import json
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared import hardware_utils as hw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DETECTIONS_PATH = PROJECT_ROOT / "Dataset" / "detections.json"
MASTER_PATH = PROJECT_ROOT / "Dataset" / "master_annotations.jsonl"
SKU_MANIFEST_PATH = PROJECT_ROOT / "Dataset" / "sku_catalogue" / "sku_manifest.csv"
CLUSTER_TO_SKU_PATH = PROJECT_ROOT / "Dataset" / "curated_subset" / "cluster_to_sku.json"
DEFAULT_OCR_PATH = PROJECT_ROOT / "Dataset" / "ocr_cache" / "ocr_results.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(description="Build master annotation file from detection + OCR + cluster data")
    parser.add_argument("--ocr", type=str, default=None,
                        help=f"Path to OCR results JSONL (default: {DEFAULT_OCR_PATH})")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild from scratch (default: skip if master exists)")
    parser.add_argument("--max-crops", type=int, default=None,
                        help="Limit crops for debugging (default: all)")
    return parser.parse_args()


def load_detections(path: Path) -> dict:
    """Load detections.json; returns {crop_id: {orig_img, box_2d, score, orig_size}}."""
    if not path.exists():
        print(f"ERROR: {path} not found. Run detect_incremental.py first.")
        sys.exit(1)
    with open(path) as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} crop entries from {path.name}")
    return data


def load_ocr_results(path: Path) -> dict[str, dict]:
    """Load OCR JSONL; returns {crop_id: {text, confidence, source}}.

    Each line: {fname, n, texts: [{text, confidence}], max_confidence, routing, source}
    We take the highest-confidence text from each crop entry.
    """
    if not path.exists():
        print(f"  (no OCR results at {path})")
        return {}

    results = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            fname = entry.get("fname", "")
            if not fname:
                continue

            # Pick the best text from the texts list
            texts = entry.get("texts", [])
            best_text = None
            best_conf = 0.0
            for t in texts:
                conf = t.get("confidence", 0.0)
                if conf > best_conf:
                    best_conf = conf
                    best_text = t.get("text", "")

            if best_text and best_conf > 0:
                results[fname] = {
                    "ocr_suggested": best_text.strip(),
                    "ocr_confidence": round(best_conf, 4),
                    "ocr_source": entry.get("source", "unknown"),
                }

    print(f"  Loaded {len(results)} OCR results from {path.name}")
    return results


def load_cluster_to_sku(path: Path) -> dict:
    """Load cluster→SKU mapping JSON.

    Format: {"cluster_042_size_442": {"sku_id": "colgate_max_white", "confidence": "high", "verified_by": "manual_review"}}
    Returns {cluster_dir_name: sku_id} for all entries.
    """
    if not path.exists():
        print(f"  (no cluster→SKU mapping at {path})")
        return {}

    with open(path) as f:
        data = json.load(f)

    mapping = {}
    for cluster_dir, info in data.items():
        if isinstance(info, dict) and "sku_id" in info:
            mapping[cluster_dir] = info["sku_id"]
        elif isinstance(info, str):
            # Simple {cluster_dir: sku_id} format
            mapping[cluster_dir] = info

    print(f"  Loaded {len(mapping)} cluster→SKU mappings from {path.name}")
    return mapping


def load_sku_manifest(path: Path) -> set[str]:
    """Load valid SKU IDs from sku_manifest.csv."""
    if not path.exists():
        print(f"  WARNING: {path} not found — SKU validation disabled")
        return set()

    skus = set()
    with open(path) as f:
        header = f.readline().strip().split(",")
        sku_idx = 0  # first column is sku_id
        for line in f:
            parts = line.strip().split(",")
            if parts:
                skus.add(parts[0].strip())

    print(f"  Loaded {len(skus)} valid SKU IDs from {path.name}")
    return skus


def build_cluster_to_crop_mapping(cluster_dir: Path) -> dict[str, str]:
    """Build {crop_filename: cluster_dir_name} mapping from curated_subset directory structure.

    Scans Dataset/curated_subset/cluster_*/ for image files and maps them back to their cluster.
    """
    if not cluster_dir.exists():
        return {}

    mapping = {}
    for cluster in sorted(cluster_dir.iterdir()):
        if not cluster.is_dir() or not cluster.name.startswith("cluster_"):
            continue
        for f in cluster.iterdir():
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                mapping[f.name] = cluster.name

    print(f"  Built {len(mapping)} crop→cluster mappings from curated_subset/")
    return mapping


def build_annotation(
    crop_id: str,
    det_entry: dict,
    ocr_results: dict,
    crop_to_cluster: dict,
    cluster_to_sku: dict,
    valid_skus: set[str],
) -> dict:
    """Build a single master annotation entry from all input sources."""
    # Source image: strip "Dataset/" prefix to get store/relative path
    orig_img = det_entry.get("orig_img", "")
    if orig_img.startswith("Dataset/"):
        source_image = orig_img[len("Dataset/"):]
    else:
        source_image = orig_img

    box = det_entry.get("box_2d", [])
    # Ensure box is [x1, y1, x2, y2]
    if len(box) == 4:
        bbox = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
    else:
        bbox = []

    score = det_entry.get("score", 0.0)
    orig_size = det_entry.get("orig_size", [])

    # OCR suggestion
    ocr_info = ocr_results.get(crop_id, {})
    ocr_suggested = ocr_info.get("ocr_suggested")
    ocr_confidence = ocr_info.get("ocr_confidence")

    # Cluster assignment
    cluster_name = crop_to_cluster.get(crop_id)
    cluster_id = None
    if cluster_name:
        # Extract numeric cluster ID from "cluster_042_size_442"
        try:
            parts = cluster_name.split("_")
            cluster_id = int(parts[1])
        except (IndexError, ValueError):
            pass

    # SKU from cluster→sku mapping, or from cluster name if cluster dir was renamed to SKU
    sku = None
    sku_confidence = "none"

    # Check explicit cluster→SKU mapping
    if cluster_name and cluster_name in cluster_to_sku:
        mapped_sku = cluster_to_sku[cluster_name]
        if mapped_sku in valid_skus or not valid_skus:
            sku = mapped_sku
            sku_confidence = "medium"  # cluster-based, not verified

    return {
        "crop_id": crop_id,
        "source_image": source_image,
        "bbox": bbox,
        "detection_score": round(score, 4),
        "orig_size": orig_size,
        "split": "unassigned",
        "sku": sku,
        "sku_confidence": sku_confidence,
        "verified": False,
        "ocr_suggested": ocr_suggested,
        "ocr_confidence": ocr_confidence,
        "cluster_id": cluster_id,
        "notes": "",
    }


def main():
    args = parse_args()
    t0 = time.time()

    print("=" * 60)
    print(hw.hw_summary())
    print("Master Annotation Builder")
    print("=" * 60)

    # Skip if already built (idempotent)
    if MASTER_PATH.exists() and not args.force:
        line_count = sum(1 for _ in open(MASTER_PATH))
        print(f"Master annotation file exists at {MASTER_PATH} ({line_count} lines)")
        print("Use --force to rebuild from scratch.")
        return

    # Resolve OCR path
    ocr_path = Path(args.ocr) if args.ocr else DEFAULT_OCR_PATH

    # Load inputs
    print("\nLoading inputs...")
    detections = load_detections(DETECTIONS_PATH)
    ocr_results = load_ocr_results(ocr_path)
    cluster_to_sku = load_cluster_to_sku(CLUSTER_TO_SKU_PATH)
    valid_skus = load_sku_manifest(SKU_MANIFEST_PATH)

    # Build crop→cluster mapping from directory structure
    curated_dir = PROJECT_ROOT / "Dataset" / "curated_subset"
    crop_to_cluster = build_cluster_to_crop_mapping(curated_dir)

    # Build annotations
    print(f"\nBuilding {len(detections)} master annotations...")
    written = 0
    with open(MASTER_PATH, "w") as out:
        for idx, (crop_id, det_entry) in enumerate(detections.items()):
            if args.max_crops and idx >= args.max_crops:
                break

            annotation = build_annotation(
                crop_id=crop_id,
                det_entry=det_entry,
                ocr_results=ocr_results,
                crop_to_cluster=crop_to_cluster,
                cluster_to_sku=cluster_to_sku,
                valid_skus=valid_skus,
            )
            out.write(json.dumps(annotation, ensure_ascii=False) + "\n")
            written += 1

    elapsed = time.time() - t0
    print(f"\nMaster annotation file built: {MASTER_PATH}")
    print(f"  Total entries: {written}")
    print(f"  With OCR suggestion: {sum(1 for r in ocr_results.values() if r.get('ocr_suggested'))}")
    print(f"  With cluster assignment: {len(crop_to_cluster)}")
    print(f"  With SKU mapping: {len(cluster_to_sku)}")
    print(f"  Time: {elapsed / 60:.1f} min")
    print("Done.")

    # Print summary stats
    print("\n--- Quick Stats ---")
    print(f"  Total detections: {written}")
    ocr_count = sum(1 for v in ocr_results.values() if v.get("ocr_suggested"))
    print(f"  Crops with OCR text: {ocr_count}")
    print(f"  Crops in curated clusters: {len(crop_to_cluster)}")
    print(f"  Valid SKUs in manifest: {len(valid_skus)}")


if __name__ == "__main__":
    main()
