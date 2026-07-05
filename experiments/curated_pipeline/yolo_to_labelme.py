#!/usr/bin/env python3
"""Convert YOLO .txt exports to LabelMe JSON format for visual validation.

Reads data/exports/yolo/*.txt + data.yaml, converts to LabelMe JSON files
with the source images copied alongside, so you can open the output folder
directly in LabelMe.

Usage
-----
    python yolo_to_labelme.py
    python yolo_to_labelme.py --input-dir data/exports/yolo
    python yolo_to_labelme.py --output-dir data/exports/labelme
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from pathlib import Path

from PIL import Image

_PIPELINE_DIR = Path(__file__).resolve().parent

import config as cfg

logger = logging.getLogger("yolo_to_labelme")

DEFAULT_INPUT = cfg.YOLO_EXPORT_DIR
DEFAULT_OUTPUT = cfg.EXPORTS_DIR / "labelme"
DEFAULT_IMAGES_DIR = cfg.INPUT_DIR


def parse_data_yaml(yaml_path: Path) -> dict[int, str]:
    """Parse data.yaml to get {class_id: class_name} mapping.

    Handles both inline list format:
        names: [SKU-001, SKU-002, ...]
    and line-by-line format:
        names:
          0: SKU-001
          1: SKU-002
    """
    text = yaml_path.read_text()
    cls_map: dict[int, str] = {}

    # Inline list: names: [SKU-001, SKU-002, ...]
    m = re.search(r"names:\s*\[(.+?)\]", text)
    if m:
        items = [x.strip().strip("'\"") for x in m.group(1).split(",")]
        for idx, name in enumerate(items):
            if name:
                cls_map[idx] = name
        return cls_map

    # Line-by-line: names:\n  ... SKU-001\n ...
    lines = text.splitlines()
    in_names = False
    for line in lines:
        if line.strip().startswith("names:") and "[" not in line:
            in_names = True
            continue
        if in_names:
            m2 = re.match(r"\s*(\d+):\s*['\"]?(.+?)['\"]?\s*$", line)
            if m2:
                cls_map[int(m2.group(1))] = m2.group(2).strip()
            elif re.match(r"^\s*#", line) or line.strip() == "":
                continue
            else:
                in_names = False

    return cls_map


def yolo_to_pixel(
    cx: float, cy: float, w: float, h: float, img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    """Convert YOLO normalized (cx, cy, w, h) to absolute pixel (x1, y1, x2, y2)."""
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return x1, y1, x2, y2


def build_labelme_json(
    image_path: Path,
    shapes: list[dict],
    img_w: int,
    img_h: int,
    rel_image_path: str,
) -> dict:
    """Build a LabelMe-format JSON dict for one image."""
    return {
        "version": "5.3.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": rel_image_path,
        "imageData": None,
        "imageHeight": img_h,
        "imageWidth": img_w,
    }


def convert(
    input_dir: Path,
    output_dir: Path,
    images_dir: Path,
    copy_images: bool = True,
) -> int:
    """Convert all YOLO .txt files in input_dir to LabelMe JSON in output_dir."""
    yaml_path = input_dir / "data.yaml"
    if not yaml_path.exists():
        logger.error("No data.yaml found in %s — cannot map class IDs to names", input_dir)
        return 1

    cls_map = parse_data_yaml(yaml_path)
    if not cls_map:
        logger.error("Could not parse class names from %s", yaml_path)
        return 1

    logger.info("Class mapping (%d classes): %s", len(cls_map), cls_map)

    txt_files = sorted(input_dir.glob("*.txt"))
    # Exclude data.yaml from being mistaken as a label file
    txt_files = [f for f in txt_files if f.name != "data.yaml"]

    if not txt_files:
        logger.warning("No .txt files found in %s", input_dir)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    total_shapes = 0
    n_files = 0

    for txt_path in txt_files:
        stem = txt_path.stem  # e.g. IMG20260611210028

        # Find the source image
        src_img = None
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            candidate = images_dir / f"{stem}{ext}"
            if candidate.exists():
                src_img = candidate
                break
        if src_img is None:
            # Fallback: search recursively in input dir
            for ext in (".jpg", ".jpeg", ".png", ".webp"):
                for found in images_dir.rglob(f"{stem}{ext}"):
                    src_img = found
                    break
            if src_img is None:
                logger.warning("Source image not found for %s — skipping", stem)
                continue

        # Get image dimensions
        with Image.open(src_img) as img:
            img_w, img_h = img.size

        # Parse YOLO lines
        lines = txt_path.read_text().strip().splitlines()
        shapes: list[dict] = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                logger.warning("  Skipping malformed line: %s", line)
                continue

            cls_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            if cls_id not in cls_map:
                logger.warning("  Unknown class ID %d in %s — skipping", cls_id, txt_path.name)
                continue

            label = cls_map[cls_id]
            x1, y1, x2, y2 = yolo_to_pixel(cx, cy, w, h, img_w, img_h)

            shapes.append({
                "label": label,
                "points": [[x1, y1], [x2, y2]],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {},
            })

        if not shapes:
            logger.info("  %s: no valid shapes", txt_path.name)
            # Still write an empty annotation so LabelMe opens the image
        else:
            logger.info("  %s: %d shapes", txt_path.name, len(shapes))

        # Copy image to output
        dst_img = output_dir / src_img.name
        if copy_images:
            shutil.copy2(str(src_img), str(dst_img))
        rel_path = src_img.name  # relative to output dir

        # Write LabelMe JSON
        labelme_data = build_labelme_json(
            src_img, shapes, img_w, img_h, rel_path,
        )
        json_path = output_dir / f"{stem}.json"
        with open(json_path, "w") as f:
            json.dump(labelme_data, f, indent=2)

        total_shapes += len(shapes)
        n_files += 1

    logger.info("")
    logger.info("Done! %d images, %d shapes → %s", n_files, total_shapes, output_dir)
    logger.info("Open the folder in LabelMe:")
    logger.info("  labelme %s", output_dir)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert YOLO exports to LabelMe JSON format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input-dir", type=str, default=str(DEFAULT_INPUT),
        help="YOLO export directory (default: data/exports/yolo)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT),
        help="LabelMe output directory (default: data/exports/labelme)",
    )
    parser.add_argument(
        "--images-dir", type=str, default=str(DEFAULT_IMAGES_DIR),
        help="Source images directory (default: data/input)",
    )
    parser.add_argument(
        "--no-copy-images", action="store_true",
        help="Don't copy images to output (JSON will reference originals)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    return parser.parse_args(argv)


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    images_dir = Path(args.images_dir).resolve()

    return convert(
        input_dir, output_dir, images_dir,
        copy_images=not args.no_copy_images,
    )


if __name__ == "__main__":
    sys.exit(main())
