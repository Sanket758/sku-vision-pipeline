"""
detr_crops_to_sku_yolo.py

Reads:
  - Dataset/detections.json      (crop_fname -> orig_img, box_2d in absolute coords)
  - experiments/annotation_tool/data/clusters.json              (592 OCR clusters)
  - experiments/annotation_tool/data/class_catalogue.json       (83 SKU classes)
  - Dataset/processed_yolo/  (existing train/val/test labels with class 0 only)

Output:
  - Dataset/processed_yolo_sku/  (YOLO labels with SKU class ids)
  - Dataset/processed_yolo_sku/data.yaml

Mapping pipeline:
  crop_fname (from detections) -> member of which cluster? -> cluster maps to which SKU?
  SKU -> numeric class_id (sorted by sku_code for deterministic ordering)
"""

import json
import shutil
from pathlib import Path
from collections import OrderedDict
import yaml  # pip install pyyaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/home/sanket758/Education/BSBI/Masters-Thesis")
DETECTIONS_JSON = PROJECT_ROOT / "Dataset" / "detections.json"
CLUSTERS_JSON = PROJECT_ROOT / "experiments" / "annotation_tool" / "data" / "clusters.json"
CATALOGUE_JSON = PROJECT_ROOT / "experiments" / "annotation_tool" / "data" / "class_catalogue.json"
YOLO_IMAGES_DIR = PROJECT_ROOT / "Dataset" / "processed_yolo"
OUTPUT_DIR = PROJECT_ROOT / "Dataset" / "processed_yolo_sku"

# ---------------------------------------------------------------------------
# 1. Load detections (crop_fname -> bbox)
# ---------------------------------------------------------------------------
print("[1/6] Loading detections.json ...")
with open(DETECTIONS_JSON) as f:
    detections = json.load(f)
print(f"       {len(detections)} crop entries loaded.")

# Build crop -> (orig_img, box_2d, orig_size) lookup
# box_2d = [x1, y1, x2, y2] in original image coords (absolute pixels)
crop_to_det = {}
for crop_fname, info in detections.items():
    crop_to_det[crop_fname] = info

# ---------------------------------------------------------------------------
# 2. Load clusters & build crop_fname -> cluster_key mapping
# ---------------------------------------------------------------------------
print("[2/6] Loading clusters.json ...")
with open(CLUSTERS_JSON) as f:
    cluster_data = json.load(f)

# crop_fname -> cluster_key
crop_to_cluster = {}
for cluster_entry in cluster_data["clusters"]:
    key = cluster_entry["key"]
    for fname in cluster_entry["member_fnames"]:
        crop_to_cluster[fname] = key

print(f"       {len(crop_to_cluster)} crops mapped to {len(cluster_data['clusters'])} clusters.")

# ---------------------------------------------------------------------------
# 3. Load class catalogue & build crop_fname -> SKU mapping
# ---------------------------------------------------------------------------
print("[3/6] Loading class_catalogue.json ...")
with open(CATALOGUE_JSON) as f:
    catalogue_data = json.load(f)

# crop_fname -> sku_code
crop_to_sku = {}
sku_info = OrderedDict()  # sku_code -> class_name, brand, super_category, n_crops
for cls in catalogue_data["classes"]:
    sku_code = cls["sku_code"]
    sku_info[sku_code] = {
        "class_name": cls["class_name"],
        "brand": cls["brand"],
        "super_category": cls["super_category"],
        "n_crops": cls["n_crops"],
    }
    for fname in cls["crop_fnames"]:
        crop_to_sku[fname] = sku_code

print(f"       {len(crop_to_sku)} crops mapped to {len(sku_info)} SKU classes.")

# ---------------------------------------------------------------------------
# 4. Assign numeric class ids (sorted for determinism)
# ---------------------------------------------------------------------------
print("[4/6] Assigning class IDs ...")
sku_codes_sorted = sorted(sku_info.keys())
sku_to_class_id = {code: idx for idx, code in enumerate(sku_codes_sorted)}
class_names = [f"{sku_info[code]['class_name']} ({code})" for code in sku_codes_sorted]

# Which crops have both a detection entry and a SKU label?
# This gives us the set we can convert.
mappable_crops = set(crop_to_det.keys()) & set(crop_to_sku.keys())
print(f"       {len(mappable_crops)} crops have both detection bbox + SKU label.")
print(f"       Total SKU classes: {len(sku_codes_sorted)}")

# ---------------------------------------------------------------------------
# 5. Process each image in processed_yolo and produce per-SKU labels
# ---------------------------------------------------------------------------
print("[5/6] Generating per-SKU YOLO labels ...")

# Clear/create output directory
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True)

splits = ["train", "val", "test"]

stats = {"train": 0, "val": 0, "test": 0, "unknown_class": 0, "no_sku": 0, "total_crops": 0}

for split in splits:
    label_dir = YOLO_IMAGES_DIR / "labels" / split
    src_image_dir = YOLO_IMAGES_DIR / "images" / split
    dst_label_dir = OUTPUT_DIR / "labels" / split
    dst_image_dir = OUTPUT_DIR / "images" / split

    if not label_dir.exists():
        print(f"       [SKIP] {label_dir} does not exist.")
        continue

    dst_label_dir.mkdir(parents=True)
    dst_image_dir.mkdir(parents=True)

    # Copy images over (they stay the same)
    for img_path in sorted(src_image_dir.iterdir()):
        if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            shutil.copy2(img_path, dst_image_dir / img_path.name)

    # Process each label file
    for label_path in sorted(label_dir.iterdir()):
        if label_path.suffix != ".txt":
            continue

        image_stem = label_path.stem
        out_lines = []

        with open(label_path) as f:
            raw_lines = f.readlines()

        for line in raw_lines:
            line = line.strip()
            if not line:
                continue
            # Parse: class_id x_center y_center width height (normalized 0-1)
            parts = line.split()
            if len(parts) != 5:
                continue

            # Keep the bbox coords, we only change class_id
            _, cx, cy, w, h = parts

            # The label file corresponds to a source image.
            # We need to find which DETR crops belong to this image.
            # detections.json maps crop_fname -> orig_img.
            # We look for crops whose orig_img ends with this image_stem.

            # Actually, simpler approach: collect all crop boxes that map to this orig_img.
            # We'll do that below using a pre-built index.

            # For now, we need a reverse lookup: orig_img -> list of (crop_fname, box, sku)
            out_lines.append(line)  # placeholder

        # Write new label file
        # (We'll rebuild properly below)
        pass

    stats[split] = len(list(label_dir.iterdir()))
    print(f"       {split}: {stats[split]} images processed")

# ---------------------------------------------------------------------------
# 6. Build reverse lookup: orig_img -> list of crop annotations with SKU
# ---------------------------------------------------------------------------
# The above approach is wrong because YOLO labels are per source image,
# not per crop. We need to group crops by their source image.

print("[5b/6] Built approach: generate label files by source image ...")

# Build orig_img -> list of (box_2d, class_id)
orig_img_to_labels = {}  # orig_img path -> [(class_id, x_center, y_center, width, height, score)]

for crop_fname, det_info in crop_to_det.items():
    if crop_fname not in crop_to_sku:
        stats["no_sku"] += 1
        continue

    sku_code = crop_to_sku[crop_fname]
    class_id = sku_to_class_id[sku_code]
    box = det_info["box_2d"]  # [x1, y1, x2, y2] absolute
    orig_img = det_info["orig_img"]
    orig_h, orig_w = det_info["orig_size"]
    score = det_info["score"]

    # Convert to YOLO normalized format
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0 / orig_w
    cy = (y1 + y2) / 2.0 / orig_h
    bw = (x2 - x1) / orig_w
    bh = (y2 - y1) / orig_h

    # Clamp to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    bw = max(0.0, min(1.0, bw))
    bh = max(0.0, min(1.0, bh))

    if orig_img not in orig_img_to_labels:
        orig_img_to_labels[orig_img] = []
    orig_img_to_labels[orig_img].append((class_id, cx, cy, bw, bh, score))
    stats["total_crops"] += 1

# Now determine which split each orig_img belongs to
# The processed_yolo split labels map by source image stem
#
# processed_yolo/images/train/ contains the SOURCE images (like IMG20260529095752.jpg)
# processed_yolo/labels/train/ has matching .txt files with all boxes for that source image
#
# So we use the same split info from processed_yolo to determine
# which orig_imgs go to train/val/test.

# Map image stem -> split
img_to_split = {}
for split in splits:
    img_dir = YOLO_IMAGES_DIR / "images" / split
    if not img_dir.exists():
        continue
    for img_path in img_dir.iterdir():
        if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            img_to_split[img_path.stem] = split

# For each orig_img path, find which split it belongs to
# orig_img is like "Dataset/kaufland/IMG20260601165959.jpg"
# We extract the stem and look it up in img_to_split
# If not found, the orig_img may not be in processed_yolo (different collection)

orig_img_split_counts = {s: 0 for s in splits}
orig_img_missing = 0

for orig_img_rel, labels in orig_img_to_labels.items():
    orig_img_path = Path(orig_img_rel)
    stem = orig_img_path.stem

    if stem in img_to_split:
        split = img_to_split[stem]
        # Sort labels by score descending (best detections first), optional dedup
        labels_sorted = sorted(labels, key=lambda x: x[5], reverse=True)

        # Write label file
        dst_label_file = OUTPUT_DIR / "labels" / split / f"{stem}.txt"
        dst_label_file.parent.mkdir(parents=True, exist_ok=True)

        with open(dst_label_file, "w") as f:
            for class_id, cx, cy, bw, bh, score in labels_sorted:
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        orig_img_split_counts[split] += 1
    else:
        orig_img_missing += 1
        # This orig_img wasn't in the processed_yolo split files
        # This can happen if crops from this image exist in detections
        # but the source image isn't in the YOLO split

# Copy images for splits that have labels
for split in splits:
    src_img_dir = YOLO_IMAGES_DIR / "images" / split
    dst_img_dir = OUTPUT_DIR / "images" / split
    dst_lbl_dir = OUTPUT_DIR / "labels" / split

    if not dst_lbl_dir.exists():
        dst_lbl_dir.mkdir(parents=True)
        dst_img_dir.mkdir(parents=True)
        continue

    # Copy images that have label files
    for label_path in dst_lbl_dir.iterdir():
        if label_path.suffix != ".txt":
            continue
        stem = label_path.stem
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            src_img = src_img_dir / f"{stem}{ext}"
            if src_img.exists():
                shutil.copy2(src_img, dst_img_dir / src_img.name)
                break

print(f"       Train: {orig_img_split_counts['train']} images")
print(f"       Val:   {orig_img_split_counts['val']} images")
print(f"       Test:  {orig_img_split_counts['test']} images")
print(f"       Missing from split: {orig_img_missing} images (crops exist but source not in YOLO split)")
print(f"       Total mapped crops: {stats['total_crops']}")
print(f"       Crops with no SKU: {stats['no_sku']} (these are unlabeled crops, not in catalogue)")

# ---------------------------------------------------------------------------
# 7. Write data.yaml
# ---------------------------------------------------------------------------
print("[6/6] Writing data.yaml ...")

data_yaml = {
    "train": str(OUTPUT_DIR / "images" / "train"),
    "val": str(OUTPUT_DIR / "images" / "val"),
    "test": str(OUTPUT_DIR / "images" / "test"),
    "nc": len(sku_codes_sorted),
    "names": class_names,
}

with open(OUTPUT_DIR / "data.yaml", "w") as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

# Also save a class mapping file for reference
class_map = {}
for code, idx in sku_to_class_id.items():
    info = sku_info[code]
    class_map[str(idx)] = {
        "sku_code": code,
        "class_name": info["class_name"],
        "brand": info["brand"],
        "super_category": info["super_category"],
        "n_crops": info["n_crops"],
    }

with open(OUTPUT_DIR / "class_map.json", "w") as f:
    json.dump(class_map, f, indent=2)

print(f"\n[DONE] Output at: {OUTPUT_DIR}")
print(f"       {OUTPUT_DIR / 'data.yaml'}")
print(f"       {OUTPUT_DIR / 'class_map.json'}")
print(f"       nc={len(sku_codes_sorted)} classes")
print(f"       names: {class_names[:5]}... ({len(class_names)} total)")
