"""Build Dataset/image_registry.csv by scanning all store image directories.

Scans Dataset/{store}/ for images, extracts EXIF dates + dimensions + sha256 hash,
and writes a unified registry CSV. Idempotent — re-running updates existing entries
by filename and appends new ones.

Usage:
    python experiments/shared/build_image_registry.py
    python experiments/shared/build_image_registry.py --force   # Re-hash all images
"""

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared import hardware_utils as hw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = PROJECT_ROOT / "Dataset" / "image_registry.csv"
TRACKING_PATH = PROJECT_ROOT / "Dataset" / "image_registry_meta.json"
# Directories to scan — everything under Dataset/ that contains images and is not a metadata dir
SKIP_DIRS = {
    "processed_retrieval", "processed_yolo", "processed_yolo_sku",
    "curated_subset", "ocr_cache", "sku_catalogue", "recognition",
    "raw", "__pycache__",
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser(description="Build image registry from store directories")
    parser.add_argument("--force", action="store_true",
                        help="Re-hash all images (default: skip unchanged files)")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def try_get_exif_date(path: Path) -> str | None:
    """Extract EXIF DateTimeOriginal as YYYY-MM-DD, or None."""
    try:
        from PIL import Image
        img = Image.open(path)
        exif = img.getexif()
        if not exif:
            return None
        # EXIF tag 36867 = DateTimeOriginal, 36868 = DateTimeDigitized, 306 = DateTime
        for tag in (36867, 36868, 306):
            val = exif.get(tag)
            if val:
                # Format: "2026:06:01 16:59:59"
                dt = datetime.strptime(val.strip(), "%Y:%m:%d %H:%M:%S")
                return dt.strftime("%Y-%m-%d")
    except Exception:
        pass
    return None


def get_image_dims(path: Path) -> tuple[int, int]:
    """Return (width, height) using PIL."""
    from PIL import Image
    with Image.open(path) as img:
        return img.size


def infer_capture_date(path: Path, store_dir: Path) -> str:
    """Infer capture date: from date-subdir name, EXIF, or file mtime."""
    # Check if parent dir looks like a date: YYYY-MM-DD
    parent = path.parent
    if parent != store_dir:
        try:
            datetime.strptime(parent.name, "%Y-%m-%d")
            return parent.name
        except ValueError:
            pass

    # Try EXIF
    exif_date = try_get_exif_date(path)
    if exif_date:
        return exif_date

    # Fallback to file mtime
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return mtime.strftime("%Y-%m-%d")


def find_store_dirs() -> list[Path]:
    """Return all Dataset subdirs that contain images (excluding metadata dirs)."""
    dataset = PROJECT_ROOT / "Dataset"
    stores = []
    for entry in sorted(dataset.iterdir()):
        if not entry.is_dir() or entry.name in SKIP_DIRS or entry.name.startswith("."):
            continue
        # Check if it contains image files (directly or in subdirs)
        has_images = any(
            p.suffix.lower() in IMAGE_EXTS
            for p in entry.rglob("*")
            if p.is_file()
        )
        if has_images:
            stores.append(entry)
    return stores


def load_previous_registry() -> dict[str, dict]:
    """Load existing registry as {filename: row}."""
    if not REGISTRY_PATH.exists():
        return {}
    rows = {}
    with open(REGISTRY_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["filename"]] = row
    return rows


def main():
    args = parse_args()
    t0 = time.time()

    print("=" * 60)
    print(hw.hw_summary())
    print("=" * 60)

    store_dirs = find_store_dirs()
    if not store_dirs:
        print("ERROR: No store directories with images found under Dataset/")
        sys.exit(1)

    print(f"Found {len(store_dirs)} store dirs: {[d.name for d in store_dirs]}")

    # Load previous registry for idempotency
    prev = {} if args.force else load_previous_registry()

    # Collect all image files
    all_images: list[Path] = []
    for store_dir in store_dirs:
        for p in sorted(store_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                all_images.append(p)

    print(f"Total image files: {len(all_images)}")

    # Build new rows
    rows = []
    new_count = 0
    updated_count = 0
    unchanged_count = 0

    for img_path in all_images:
        rel = img_path.relative_to(PROJECT_ROOT / "Dataset")
        store = img_path.parent.name

        # For date-subdir stores, the store name is the grandparent
        grandparent = img_path.parent.parent
        if grandparent.name != "Dataset" and (grandparent.parent.name == "Dataset" or str(grandparent.parent).endswith("Dataset")):
            store = grandparent.name

        filename = str(rel)

        # Check if unchanged (same mtime)
        if not args.force and filename in prev:
            prev_row = prev[filename]
            prev_mtime = prev_row.get("_mtime", "")
            current_mtime = str(int(img_path.stat().st_mtime))
            prev_hash = prev_row.get("hash", "")
            if prev_mtime == current_mtime and prev_hash:
                rows.append(prev_row)
                unchanged_count += 1
                continue

        # Extract metadata
        file_size = img_path.stat().st_size
        capture_date = infer_capture_date(img_path, img_path.parent)
        file_hash = sha256_file(img_path)
        width, height = get_image_dims(img_path)
        current_mtime = str(int(img_path.stat().st_mtime))

        row = {
            "filename": filename,
            "store": store,
            "capture_date": capture_date,
            "width": str(width),
            "height": str(height),
            "hash": file_hash,
            "status": "active",
            "notes": "",
            "_size": str(file_size),
            "_mtime": current_mtime,
        }
        rows.append(row)

        if filename in prev:
            updated_count += 1
        else:
            new_count += 1

    # Write CSV (excluding internal fields)
    fieldnames = ["filename", "store", "capture_date", "width", "height", "hash", "status", "notes"]
    with open(REGISTRY_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(out)

    # Save tracking metadata for next incremental run
    tracking = {
        "last_run": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_images": len(rows),
        "stores": [d.name for d in store_dirs],
    }
    with open(TRACKING_PATH, "w") as f:
        json.dump(tracking, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nImage registry built: {REGISTRY_PATH}")
    print(f"  Total entries: {len(rows)}")
    print(f"  New: {new_count}, Updated: {updated_count}, Unchanged: {unchanged_count}")
    print(f"  Stores: {tracking['stores']}")
    print(f"  Time: {elapsed / 60:.1f} min")
    print("Done.")


if __name__ == "__main__":
    main()
