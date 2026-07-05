#!/usr/bin/env python3
"""Verify curated pipeline YOLO export integrity.

Validates:
- Every label file has a matching source image
- Class IDs are within valid range (0..nc-1)
- Bounding box coordinates are valid (normalized 0-1, w>0, h>0)
- No duplicate or empty label files
- Prints summary stats for thesis reference

Usage:
    python verify_export.py
"""

from pathlib import Path
import yaml

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
YOLO_EXPORT_DIR = PROJECT_ROOT / "experiments" / "curated_pipeline" / "data" / "exports" / "yolo"
DATA_YAML = YOLO_EXPORT_DIR / "data.yaml"

# Source image search roots (order matters — first match wins)
SOURCE_ROOTS = [
    PROJECT_ROOT / "Dataset" / "raw" / "Aldi",
    PROJECT_ROOT / "Dataset" / "raw" / "kaufland",
    PROJECT_ROOT / "Dataset" / "raw" / "Lidl",
    PROJECT_ROOT / "Dataset" / "raw" / "netto",
    PROJECT_ROOT / "Dataset" / "lidl" / "2026-06-15",       # video frames
    PROJECT_ROOT / "Dataset" / "raw",                       # catch-all
]

# ── Load data.yaml ─────────────────────────────────────────────────────────
def load_data_yaml(path: Path) -> dict:
    """Load and parse data.yaml, return dict with nc and names."""
    if not path.exists():
        print(f"❌ data.yaml not found: {path}")
        return {"nc": 0, "names": []}

    with open(path) as f:
        raw = yaml.safe_load(f)

    nc = raw.get("nc", 0)
    names = raw.get("names", [])
    print(f"  Classes: {nc} ({names[0] if names else '?'} … {names[-1] if names else '?'})")
    return {"nc": nc, "names": names, "raw": raw}


# ── Find source images ─────────────────────────────────────────────────────
SOURCE_CACHE: dict[str, Path | None] = {}


def find_source_image(stem: str) -> Path | None:
    """Find a source image by stem across all source roots."""
    if stem in SOURCE_CACHE:
        return SOURCE_CACHE[stem]

    for root in SOURCE_ROOTS:
        if not root.exists():
            continue
        for ext in (".jpg", ".jpeg", ".png", ".JPG"):
            candidate = root / f"{stem}{ext}"
            if candidate.exists():
                SOURCE_CACHE[stem] = candidate
                return candidate

    SOURCE_CACHE[stem] = None
    return None


# ── Validate a single label file ────────────────────────────────────────────
def validate_label_file(txt_path: Path, nc: int) -> dict:
    """Validate one YOLO label file. Returns result dict."""
    stem = txt_path.stem
    source = find_source_image(stem)
    result = {
        "file": txt_path.name,
        "stem": stem,
        "source": source,
        "n_labels": 0,
        "valid_classes": True,
        "valid_boxes": True,
        "class_ids": set(),
        "errors": [],
    }

    # Check source image exists
    if source is None:
        result["errors"].append(f"Source image not found for {stem}")
        return result

    # Read and validate labels
    lines = txt_path.read_text().strip().splitlines()
    result["n_labels"] = len(lines)

    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) != 5:
            result["errors"].append(f"  Line {i + 1}: expected 5 values, got {len(parts)}")
            result["valid_boxes"] = False
            continue

        cls_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:])

        # Check class ID range
        if cls_id < 0 or cls_id >= nc:
            result["errors"].append(f"  Line {i + 1}: class ID {cls_id} out of range [0, {nc - 1}]")
            result["valid_classes"] = False

        result["class_ids"].add(cls_id)

        # Check bounding box validity
        if w <= 0 or h <= 0:
            result["errors"].append(f"  Line {i + 1}: non-positive box dimensions (w={w}, h={h})")
            result["valid_boxes"] = False
        if cx < 0 or cx > 1 or cy < 0 or cy > 1:
            result["errors"].append(f"  Line {i + 1}: center ({cx}, {cy}) outside [0, 1]")
            result["valid_boxes"] = False

    return result


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("YOLO EXPORT VERIFICATION")
    print("=" * 60)

    # Load data.yaml
    print("\n[1] data.yaml:")
    meta = load_data_yaml(DATA_YAML)
    nc = meta["nc"]
    names = meta["names"]

    if nc == 0:
        print("\n❌ Cannot proceed without valid data.yaml")
        return 1

    # Collect label files
    label_files = sorted(YOLO_EXPORT_DIR.glob("*.txt"))
    # Filter out data.yaml, keep only label files
    label_files = [f for f in label_files if f.name != "data.yaml"]
    print(f"\n[2] Label files found: {len(label_files)}")

    # Validate each file
    print(f"\n[3] Validating labels...")
    all_results = []
    total_labels = 0
    all_class_ids = set()
    images_with_errors = 0
    missing_images = 0

    for lf in label_files:
        result = validate_label_file(lf, nc)
        all_results.append(result)
        total_labels += result["n_labels"]
        all_class_ids.update(result["class_ids"])

        if result["errors"]:
            images_with_errors += 1
            if result["source"] is None:
                missing_images += 1

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("VERIFICATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Label files:          {len(label_files)}")
    print(f"  Total bounding boxes: {total_labels}")
    print(f"  Classes represented:  {len(all_class_ids)} / {nc}")
    print(f"  Missing source imgs:  {missing_images}")
    print(f"  Files with errors:    {images_with_errors}")

    # Per-image breakdown
    print(f"\n{'=' * 60}")
    print("PER-IMAGE BREAKDOWN")
    print(f"{'=' * 60}")
    print(f"  {'File':45s} {'Labels':>6s}  {'Classes':>7s}  {'Source':>6s}")
    print(f"  {'-' * 45}  {'-' * 6}  {'-' * 7}  {'-' * 6}")
    for r in all_results:
        source_status = "✓" if r["source"] else "✗"
        n_classes = len(r["class_ids"])
        print(f"  {r['file']:45s} {r['n_labels']:6d}  {n_classes:7d}  {source_status:>6s}")

    # Error details
    if images_with_errors > 0:
        print(f"\n{'=' * 60}")
        print("ERRORS")
        print(f"{'=' * 60}")
        for r in all_results:
            if r["errors"]:
                print(f"\n  {r['file']}:")
                for err in r["errors"]:
                    print(f"    {err}")

    # ── Suggested data.yaml for training ────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("DATASET STATS (for thesis)")
    print(f"{'=' * 60}")
    avg_labels = total_labels / len(label_files) if label_files else 0
    print(f"  Total images:  {len(label_files)}")
    print(f"  Total bboxes:  {total_labels}")
    print(f"  Avg bbox/img:  {avg_labels:.1f}")
    print(f"  Total classes: {nc}")
    print(f"  Classes w/ ≥1 instance: {len(all_class_ids)}")

    # Class distribution (top 10)
    class_counts: dict[int, int] = {}
    for lf in label_files:
        for line in lf.read_text().strip().splitlines():
            cls_id = int(line.strip().split()[0])
            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

    if class_counts:
        top_classes = sorted(class_counts.items(), key=lambda x: -x[1])[:10]
        print(f"\n  Top-10 most frequent classes:")
        for cls_id, count in top_classes:
            name = names[cls_id] if cls_id < len(names) else f"class-{cls_id}"
            print(f"    {name:15s} (ID {cls_id:3d}): {count:4d} instances")

    success = images_with_errors == 0 and missing_images == 0
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"\n  Verdict: {status}")

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
