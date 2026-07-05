#!/usr/bin/env python3
"""Build YOLO dataset from curated pipeline exports + Albumentations augmentation.

Usage:
    python prepare_data.py                     # Full run
    python prepare_data.py --dry-run           # Preview only
    python prepare_data.py --augment 2         # 2 augmented copies per image
    python prepare_data.py --force             # Rebuild from scratch
    python prepare_data.py --seed 42           # Custom seed
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import yaml

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
YOLO_EXPORT_DIR = PROJECT_ROOT / "experiments" / "curated_pipeline" / "data" / "exports" / "yolo"
DATA_YAML_SRC = YOLO_EXPORT_DIR / "data.yaml"
OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "curated_yolo"
MANIFEST_PATH = OUTPUT_DIR / ".manifest.json"

# Source image search roots (first match wins)
SOURCE_ROOTS = [
    PROJECT_ROOT / "Dataset" / "raw" / "Aldi",
    PROJECT_ROOT / "Dataset" / "raw" / "kaufland",
    PROJECT_ROOT / "Dataset" / "raw" / "Lidl",
    PROJECT_ROOT / "Dataset" / "raw" / "netto",
    PROJECT_ROOT / "Dataset" / "lidl" / "2026-06-15",
    PROJECT_ROOT / "Dataset" / "raw",
    PROJECT_ROOT / "Dataset" / "raw" / "mix",
]

SPLIT_NAMES = ("train", "val", "test")

# ── Augmentation pipeline ──────────────────────────────────────────────────
def build_augmentation() -> A.Compose:
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=0, p=0.5),
        A.Blur(blur_limit=3, p=0.2),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.85, 1.15), translate_percent=(-0.1, 0.1), p=0.5),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
    ], bbox_params=A.BboxParams(format="yolo", min_visibility=0.3, label_fields=["class_labels"]))


# ── Load metadata ──────────────────────────────────────────────────────────
def load_data_yaml(path: Path) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return {"nc": raw["nc"], "names": raw["names"], "raw": raw}


def find_source_image(stem: str) -> Path | None:
    for root in SOURCE_ROOTS:
        if not root.exists():
            continue
        for ext in (".jpg", ".jpeg", ".png", ".JPG"):
            c = root / f"{stem}{ext}"
            if c.exists():
                return c
    return None


def collect_label_files(exports_dir: Path) -> list[Path]:
    files = sorted(exports_dir.glob("*.txt"))
    return [f for f in files if f.name != "data.yaml"]


def parse_labels(txt_path: Path) -> list[list[float]]:
    """Return list of [class_id, cx, cy, w, h]."""
    lines = txt_path.read_text().strip().splitlines()
    labels = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            labels.append([float(x) for x in parts])
    return labels


# ── Determine store origin from filename ──────────────────────────────────
def infer_store(stem: str) -> str:
    if stem.startswith("VID"):
        return "lidl"
    if stem.startswith("IMG20260615"):
        return "aldi"
    if stem.startswith("IMG20260611") or stem.startswith("IMG20260620"):
        return "kaufland"
    return "unknown"


# ── Augment a single image + labels ────────────────────────────────────────
def augment_image(image_path: Path, labels: list[list[float]],
                  aug_pipeline: A.Compose, n_aug: int,
                  output_img_dir: Path, output_lbl_dir: Path,
                  stem_prefix: str):
    """Apply augmentation and write augmented copies."""
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # Build bbox array for Albumentations: [x, y, x2, y2] normalized → Alb uses [cx, cy, w, h] yolo format
    bboxes = [lbl[1:] for lbl in labels]
    class_labels = [int(lbl[0]) for lbl in labels]

    for aug_idx in range(n_aug):
        try:
            transformed = aug_pipeline(image=img, bboxes=bboxes, class_labels=class_labels)
            aug_img = transformed["image"]
            aug_bboxes = transformed["bboxes"]
            aug_classes = transformed["class_labels"]
        except Exception:
            continue

        if len(aug_bboxes) == 0:
            continue

        aug_stem = f"{stem_prefix}_aug{aug_idx}"
        out_img = output_img_dir / f"{aug_stem}.jpg"
        out_txt = output_lbl_dir / f"{aug_stem}.txt"

        # Save image
        aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_img), aug_img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Save labels (albumentations returns [cx, cy, w, h] in yolo format)
        lines = []
        for cls_id, bbox in zip(aug_classes, aug_bboxes):
            cx, cy, bw, bh = bbox
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        out_txt.write_text("\n".join(lines) + "\n")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset from curated pipeline exports")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--augment", type=int, default=2, help="Number of augmented copies per image (default: 2)")
    parser.add_argument("--force", action="store_true", help="Rebuild from scratch")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split (default: 42)")
    args = parser.parse_args()

    # ── Load data.yaml ────────────────────────────────────────────────────
    if not DATA_YAML_SRC.exists():
        print(f"❌ data.yaml not found: {DATA_YAML_SRC}")
        return 1

    meta = load_data_yaml(DATA_YAML_SRC)
    nc = meta["nc"]
    names = meta["names"]
    print(f"Loaded {DATA_YAML_SRC.name}: {nc} classes")

    # ── Collect labels ────────────────────────────────────────────────────
    label_files = collect_label_files(YOLO_EXPORT_DIR)
    if not label_files:
        print("❌ No label files found in export dir")
        return 1
    print(f"Label files: {len(label_files)}")

    # ── Map labels → source images ────────────────────────────────────────
    entries = []
    missing = 0
    for lf in label_files:
        stem = lf.stem
        src_img = find_source_image(stem)
        if src_img is None:
            print(f"  ⚠ Source image not found: {stem}")
            missing += 1
            continue
        labels = parse_labels(lf)
        store = infer_store(stem)
        entries.append({
            "stem": stem,
            "source_image": src_img,
            "labels": labels,
            "n_labels": len(labels),
            "store": store,
        })

    if missing:
        print(f"  ⚠ {missing} source images missing — they will be skipped")

    print(f"Valid entries: {len(entries)}")
    print(f"Total bboxes: {sum(e['n_labels'] for e in entries)}")

    if args.dry_run:
        print("\n── DRY RUN ──")
        print(f"Would process {len(entries)} images")
        print(f"Would generate {args.augment} augmented copies each → {len(entries) * (1 + args.augment)} total")
        stores = set(e["store"] for e in entries)
        print(f"Stores: {', '.join(sorted(stores))}")
        for e in entries[:5]:
            print(f"  {e['stem']:50s} {e['n_labels']:4d} bboxes  store={e['store']}")
        if len(entries) > 5:
            print(f"  ... and {len(entries) - 5} more")
        return 0

    # ── Split ─────────────────────────────────────────────────────────────
    random.seed(args.seed)

    # Group by store for stratified split
    by_store: dict[str, list[dict]] = {}
    for e in entries:
        by_store.setdefault(e["store"], []).append(e)

    train_entries: list[dict] = []
    val_entries: list[dict] = []
    test_entries: list[dict] = []

    for store, store_entries in by_store.items():
        random.shuffle(store_entries)
        n = len(store_entries)
        n_val = max(1, round(n * 0.15))
        n_test = max(1, round(n * 0.15))
        n_train = n - n_val - n_test
        if n_train <= 0:
            # Small store group: put at least 1 in train
            n_train = 1
            remaining = n - n_train
            n_val = remaining // 2
            n_test = remaining - n_val
        train_entries.extend(store_entries[:n_train])
        val_entries.extend(store_entries[n_train:n_train + n_val])
        test_entries.extend(store_entries[n_train + n_val:])

    splits = {
        "train": train_entries,
        "val": val_entries,
        "test": test_entries,
    }

    print(f"\nSplit:")
    for split_name, split_entries in splits.items():
        total_with_aug = len(split_entries) * (1 + args.augment)
        bboxes = sum(e["n_labels"] for e in split_entries)
        print(f"  {split_name}: {len(split_entries)} originals → {total_with_aug} total ({bboxes} bboxes)")

    # ── Build directories ─────────────────────────────────────────────────
    for split_name in SPLIT_NAMES:
        (OUTPUT_DIR / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    # ── Copy originals + augment ──────────────────────────────────────────
    aug_pipeline = build_augmentation()
    total_originals = 0
    total_augmented = 0
    total_train = 0
    total_val = 0
    total_test = 0

    for split_name in SPLIT_NAMES:
        split_entries = splits[split_name]
        img_dir = OUTPUT_DIR / "images" / split_name
        lbl_dir = OUTPUT_DIR / "labels" / split_name

        for e in split_entries:
            # Copy original
            dst_img = img_dir / f"{e['stem']}.jpg"
            dst_txt = lbl_dir / f"{e['stem']}.txt"
            shutil.copy2(str(e["source_image"]), str(dst_img))
            lines = [f"{int(l[0])} {l[1]:.6f} {l[2]:.6f} {l[3]:.6f} {l[4]:.6f}" for l in e["labels"]]
            dst_txt.write_text("\n".join(lines) + "\n")
            total_originals += 1
            if split_name == "train":
                total_train += 1
            elif split_name == "val":
                total_val += 1
            else:
                total_test += 1

            # Generate augmented versions (train only)
            if split_name == "train" and args.augment > 0:
                augment_image(
                    image_path=e["source_image"],
                    labels=e["labels"],
                    aug_pipeline=aug_pipeline,
                    n_aug=args.augment,
                    output_img_dir=img_dir,
                    output_lbl_dir=lbl_dir,
                    stem_prefix=e["stem"],
                )
                total_augmented += args.augment

    # ── Write data.yaml ───────────────────────────────────────────────────
    abs_output = OUTPUT_DIR.resolve()
    yaml_content = {
        "train": str(abs_output / "images" / "train"),
        "val": str(abs_output / "images" / "val"),
        "test": str(abs_output / "images" / "test"),
        "nc": nc,
        "names": names,
    }
    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    print(f"\nWritten: {yaml_path}")

    # ── Write manifest ────────────────────────────────────────────────────
    processed = [{"stem": e["stem"], "n_labels": e["n_labels"], "store": e["store"]} for e in entries]
    manifest = {
        "n_original": len(entries),
        "n_total_images": len(entries) + total_augmented,
        "augment_factor": args.augment,
        "seed": args.seed,
        "processed": processed,
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    # ── Count bboxes in output ────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("DATASET SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Classes:     {nc}")
    print(f"  Augment:     {args.augment}× (train only)")

    total_images = total_originals + total_augmented
    print(f"  Originals:   {total_originals}")
    print(f"  Augmented:   {total_augmented}")
    print(f"  Total:       {total_images}")
    print(f"  Train:       {total_train} originals + {total_augmented} aug = {total_train + total_augmented}")
    print(f"  Val:         {total_val}")
    print(f"  Test:        {total_test}")

    # Count bboxes per split
    for split_name in SPLIT_NAMES:
        lbl_dir = OUTPUT_DIR / "labels" / split_name
        total_boxes = 0
        if lbl_dir.exists():
            for tf in lbl_dir.glob("*.txt"):
                total_boxes += len(tf.read_text().strip().splitlines())
        print(f"  Bboxes in {split_name}: {total_boxes}")

    print(f"\n✅ Dataset ready at: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    exit(main())
