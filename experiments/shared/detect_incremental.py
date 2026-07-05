"""Incremental YOLOv5 detection: process only new images, append crops,
export YOLO-format labels for detection training.

Usage:
    python detect_incremental.py                         # Detect new images
    python detect_incremental.py --conf 0.25              # Lower confidence
    python detect_incremental.py --force                  # Re-process all (ignore tracking)
    python detect_incremental.py --yolo_only              # Re-generate YOLO labels from existing metadata
"""

import argparse
import json
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared import hardware_utils as hw
from shared.yolo_detector import YOLODetector

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIRS = [
    PROJECT_ROOT / "Dataset" / "kaufland",
    PROJECT_ROOT / "Dataset" / "netto",
    PROJECT_ROOT / "Dataset" / "aldi",
    PROJECT_ROOT / "Dataset" / "lidl",
]
CROP_DIR = PROJECT_ROOT / "Dataset" / "processed_retrieval"
META_PATH = PROJECT_ROOT / "Dataset" / "detections.json"
TRACKING_PATH = PROJECT_ROOT / "Dataset" / "processed_images.json"
YOLO_IMG_DIR = PROJECT_ROOT / "Dataset" / "processed_yolo" / "images"
YOLO_LBL_DIR = PROJECT_ROOT / "Dataset" / "processed_yolo" / "labels"


def parse_args():
    parser = argparse.ArgumentParser(description="Incremental YOLOv5 + YOLO label export")
    parser.add_argument("--conf", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    parser.add_argument("--force", action="store_true", help="Re-process ALL images (ignore tracking)")
    parser.add_argument("--yolo_only", action="store_true", help="Skip detection, re-gen YOLO labels from existing metadata")
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation (flips + multi-scale) for more robust detections")
    return parser.parse_args()


def load_tracking():
    if TRACKING_PATH.exists():
        with open(TRACKING_PATH) as f:
            return json.load(f)
    return {"processed_paths": [], "last_run": None}


def save_tracking(tracking):
    tracking["last_run"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(TRACKING_PATH, "w") as f:
        json.dump(tracking, f, indent=2)


def load_existing_metadata():
    if META_PATH.exists():
        with open(META_PATH) as f:
            return json.load(f)
    return {}


def save_metadata(meta):
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)


def find_new_images(tracking, force=False):
    image_extensions = {".jpg", ".jpeg", ".png"}
    all_images = sorted(
        p for raw_dir in RAW_DIRS for p in raw_dir.rglob("*")
        if p.suffix.lower() in image_extensions
    )

    processed_set = set(tracking.get("processed_paths", []))
    if force:
        new_images = all_images
        print(f"Force mode: processing ALL {len(all_images)} images")
    else:
        new_images = [p for p in all_images if str(p) not in processed_set]
        print(f"Total images: {len(all_images)}, already processed: {len(processed_set)}, new: {len(new_images)}")

    return new_images


def export_yolo_labels(meta, source_images):
    """Generate YOLO-format label files from detection metadata."""
    YOLO_IMG_DIR.mkdir(parents=True, exist_ok=True)
    YOLO_LBL_DIR.mkdir(parents=True, exist_ok=True)

    # Group metadata by source image
    src_to_dets = {}
    for crop_name, entry in meta.items():
        src = entry["orig_img"]
        if src not in src_to_dets:
            src_to_dets[src] = []
        src_to_dets[src].append(entry)

    # For each source image, write YOLO label file
    label_count = 0
    for src_rel, dets in src_to_dets.items():
        src_path = PROJECT_ROOT / src_rel
        if not src_path.exists():
            continue

        img_stem = src_path.stem
        w, h = dets[0]["orig_size"] if dets else (1, 1)

        # Copy source image to YOLO dir (if not already there)
        dst_img = YOLO_IMG_DIR / src_path.name
        if not dst_img.exists():
            import shutil
            shutil.copy2(src_path, dst_img)

        # Write label file
        label_path = YOLO_LBL_DIR / f"{img_stem}.txt"
        with open(label_path, "w") as f:
            for det in dets:
                box = det["box_2d"]
                x_center = ((box[0] + box[2]) / 2) / w
                y_center = ((box[1] + box[3]) / 2) / h
                bw = (box[2] - box[0]) / w
                bh = (box[3] - box[1]) / h
                f.write(f"0 {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")

        label_count += 1

    total_labels = len(list(YOLO_LBL_DIR.glob("*.txt")))
    total_images = len(list(YOLO_IMG_DIR.glob("*.jpg"))) + len(list(YOLO_IMG_DIR.glob("*.png")))
    print(f"\nYOLO labels exported:")
    print(f"  Images in {YOLO_IMG_DIR}: {total_images}")
    print(f"  Labels in {YOLO_LBL_DIR}: {total_labels}")
    print(f"  Added/updated: {label_count}")


def detect_new(args, new_images, tracking, meta):
    """Run YOLOv5 on new images, append crops + metadata."""
    from PIL import Image

    print(f"\nLoading YOLOv5 SKU-110K model...")
    detector = YOLODetector.get_instance()
    if args.device:
        detector.model.to(args.device)

    CROP_DIR.mkdir(parents=True, exist_ok=True)

    total_new_crops = 0
    skipped = 0
    processed_paths = list(tracking.get("processed_paths", []))

    for img_idx, img_path in enumerate(new_images):
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [{img_idx + 1}/{len(new_images)}] Skipping {img_path.name}: {e}")
            skipped += 1
            processed_paths.append(str(img_path))
            continue

        detections = detector.detect(image, conf=args.conf, augment=args.tta)
        n_detections = len(detections)

        img_crops = 0
        for det_idx, det in enumerate(detections):
            x1, y1, x2, y2 = det["box"]

            crop = image.crop((x1, y1, x2, y2))
            crop_name = f"{img_path.stem}_det{det_idx:03d}.jpg"

            # Skip if crop already exists (from previous run)
            crop_path = CROP_DIR / crop_name
            if crop_path.exists():
                continue

            crop.save(crop_path, quality=92)

            meta[crop_name] = {
                "orig_img": str(img_path.relative_to(PROJECT_ROOT)),
                "box_2d": det["box"],
                "score": det["score"],
                "orig_size": [image.width, image.height],
            }
            img_crops += 1
            total_new_crops += 1

        processed_paths.append(str(img_path))

        print(f"  [{img_idx + 1}/{len(new_images)}] {img_path.name}: {n_detections} detections, {img_crops} new crops")

    # Update tracking
    tracking["processed_paths"] = list(set(processed_paths))
    save_tracking(tracking)
    save_metadata(meta)

    print(f"\nDetection complete.")
    print(f"  New images processed: {len(new_images) - skipped}")
    print(f"  New crops saved: {total_new_crops} to {CROP_DIR}")
    print(f"  Total crops in metadata: {len(meta)}")
    return total_new_crops


def main():
    args = parse_args()
    print("=" * 60)
    print(hw.hw_summary())
    print("=" * 60)

    t0 = time.time()

    tracking = load_tracking()
    meta = load_existing_metadata()

    if args.yolo_only:
        print("\nYOLO-only mode: exporting labels from existing metadata...")
        export_yolo_labels(meta, [])
        print(f"\nTotal time: {(time.time() - t0) / 60:.1f} min")
        return

    new_images = find_new_images(tracking, force=args.force)

    if not new_images:
        print("\nNo new images to process.")
        print(f"\nExporting YOLO labels from existing metadata...")
        export_yolo_labels(meta, [])
        print(f"\nTotal time: {(time.time() - t0) / 60:.1f} min")
        return

    total_crops = detect_new(args, new_images, tracking, meta)

    if total_crops > 0:
        print(f"\nExporting YOLO labels for all processed images...")
        export_yolo_labels(meta, new_images)
    else:
        print(f"\nNo new crops. Re-exporting YOLO labels...")
        export_yolo_labels(meta, new_images)

    print(f"\nTotal time: {(time.time() - t0) / 60:.1f} min")
    print("Done.")


if __name__ == "__main__":
    main()
