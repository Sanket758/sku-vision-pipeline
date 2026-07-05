"""Generate synthetic training images via copy-paste augmentation.

Extracts clean product "stickers" from existing DETR crops, then pastes
them onto shelf background images at random positions with random scales.
Outputs augmented images + YOLO labels into Dataset/processed_yolo/augmented/.

Usage:
    /home/sanket758/Education/BSBI/Masters-Thesis/.venv/bin/python experiments/shared/generate_synthetic_augmentations.py
"""

import json
import math
import random
import sys
from pathlib import Path

from PIL import Image

YOLO_DIR = Path("/home/sanket758/Education/BSBI/Masters-Thesis/Dataset/processed_yolo")
DET_FILE = Path("/home/sanket758/Education/BSBI/Masters-Thesis/Dataset/detections.json")
CROP_DIR = Path("/home/sanket758/Education/BSBI/Masters-Thesis/Dataset/processed_retrieval")
AUG_DIR = YOLO_DIR / "augmented"

SEED = 43
AUGS_PER_IMAGE = 3
MAX_STICKERS_PER_IMAGE = 25
MIN_STICKER_CONFIDENCE = 0.85
STICKER_AREA_PCT_RANGE = (0.005, 0.05)
N_SOURCE_IMAGES = 100
N_STICKER_CANDIDATES = 2000
OVERLAP_THRESHOLD = 0.3


def log(msg):
    print(msg, flush=True)


def filter_candidates(det):
    """Filter detection metadata (no image loading) for sticker-worthy candidates."""
    candidates = []
    for crop_id, info in det.items():
        score = info["score"]
        if score < MIN_STICKER_CONFIDENCE:
            continue
        box = info["box_2d"]
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        area = w * h
        orig_size = info["orig_size"]
        img_area = orig_size[0] * orig_size[1]
        area_pct = area / img_area if img_area > 0 else 0
        if area_pct < 0.001 or area_pct > 0.3:
            continue
        crop_stem = Path(crop_id).stem
        candidates.append({
            "crop_stem": crop_stem,
            "score": score,
            "area_pct": area_pct,
            "source_extent": (w, h),
        })
    return candidates


def load_sticker_images(candidates, n_sample):
    """Load crop images for a random sample of candidates."""
    selected = random.Random(SEED).sample(candidates, min(n_sample, len(candidates)))

    crop_paths = {}
    for p in CROP_DIR.glob("*.jpg"):
        crop_paths[p.stem] = p

    stickers = []
    loaded = 0
    failed = 0
    for c in selected:
        path = crop_paths.get(c["crop_stem"])
        if path is None:
            failed += 1
            continue
        try:
            img = Image.open(path).convert("RGBA")
            stickers.append({**c, "image": img, "width": img.width, "height": img.height})
            loaded += 1
        except Exception:
            failed += 1
        if loaded % 500 == 0:
            log(f"    Loaded {loaded}/{len(selected)} sticker images...")

    log(f"    Loaded {loaded} sticker images ({failed} failed)")
    return stickers


def random_scale_sticker(sticker, bg_w, bg_h, rng):
    target_area_pct = rng.uniform(*STICKER_AREA_PCT_RANGE)
    bg_area = bg_w * bg_h
    target_area = bg_area * target_area_pct
    cur_area = sticker["width"] * sticker["height"]
    if cur_area <= 0:
        return None
    scale = math.sqrt(target_area / cur_area)
    scale = max(0.3, min(scale, 2.0))
    new_w = max(10, min(int(sticker["width"] * scale), bg_w))
    new_h = max(10, min(int(sticker["height"] * scale), bg_h))
    scaled = sticker["image"].resize((new_w, new_h), Image.BILINEAR)
    rotation = rng.uniform(-15, 15)
    if abs(rotation) > 1:
        scaled = scaled.rotate(rotation, expand=True, fillcolor=(0, 0, 0, 0))
    return scaled


def boxes_overlap(b1, b2, threshold=OVERLAP_THRESHOLD):
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return False
    inter = (ix2 - ix1) * (iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    m = min(a1, a2)
    return inter / m > threshold if m > 0 else False


def generate_augmented(bg_path, stickers, aug_idx, out_dir):
    try:
        bg = Image.open(bg_path).convert("RGBA")
    except Exception:
        return None
    bg_w, bg_h = bg.size
    rng = random.Random(bg_path.stem + str(aug_idx))
    placed = []
    used = []
    n_stickers = min(MAX_STICKERS_PER_IMAGE, len(stickers))
    candidates = rng.sample(stickers, n_stickers)

    for sticker in candidates:
        scaled = random_scale_sticker(sticker, bg_w, bg_h, rng)
        if scaled is None:
            continue
        sw, sh = scaled.size
        for _ in range(10):
            x = rng.randint(0, max(0, bg_w - sw))
            y = rng.randint(0, max(0, bg_h - sh))
            box = (x, y, x + sw, y + sh)
            if any(boxes_overlap(box, p) for p in placed):
                continue
            placed.append(box)
            used.append((box, sticker))
            bg.paste(scaled, (x, y), scaled)
            break

    if not used:
        return None

    bg_rgb = bg.convert("RGB")
    stem = f"{bg_path.stem}_aug{aug_idx:03d}"
    img_path = out_dir / "images" / f"{stem}.jpg"
    lbl_path = out_dir / "labels" / f"{stem}.txt"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    lbl_path.parent.mkdir(parents=True, exist_ok=True)
    bg_rgb.save(img_path, "JPEG", quality=85)

    lines = []
    for (x1, y1, x2, y2), _ in used:
        x_c = (x1 + x2) / 2 / bg_w
        y_c = (y1 + y2) / 2 / bg_h
        w_n = (x2 - x1) / bg_w
        h_n = (y2 - y1) / bg_h
        lines.append(f"0 {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")
    lbl_path.write_text("\n".join(lines))
    return stem


def main():
    log("=== Loading detections ===")
    det = json.load(open(DET_FILE))
    log(f"  Loaded {len(det)} detections")

    log("\n=== Filtering candidates (metadata only) ===")
    candidates = filter_candidates(det)
    log(f"  {len(candidates)} candidates pass score + area filter")
    if not candidates:
        log("  ERROR: No candidates. Cannot generate augmentations.")
        return

    log(f"\n=== Loading {N_STICKER_CANDIDATES} sticker images ===")
    stickers = load_sticker_images(candidates, N_STICKER_CANDIDATES)
    if len(stickers) < 100:
        log(f"  ERROR: Only {len(stickers)} stickers loaded, need >= 100.")
        return
    log(f"  Using {len(stickers)} stickers for augmentation")

    log(f"\n=== Selecting training background images ===")
    train_imgs = sorted((YOLO_DIR / "images" / "train").glob("*.*"))
    log(f"  Found {len(train_imgs)} training images")
    rng = random.Random(SEED)
    selected = rng.sample(train_imgs, min(N_SOURCE_IMAGES, len(train_imgs)))
    log(f"  Selected {len(selected)} source images")

    log(f"\n=== Generating augmented images ===")
    AUG_DIR.mkdir(parents=True, exist_ok=True)
    augmented = []
    total = len(selected) * AUGS_PER_IMAGE
    done = 0

    for bg_path in selected:
        for aug_idx in range(AUGS_PER_IMAGE):
            result = generate_augmented(bg_path, stickers, aug_idx, AUG_DIR)
            if result:
                augmented.append(result)
            done += 1
            if done % 50 == 0:
                log(f"  Progress: {done}/{total} ({len(augmented)} generated)")

    n_imgs = len(list((AUG_DIR / "images").glob("*.*")))
    n_lbls = len(list((AUG_DIR / "labels").glob("*.txt")))
    log(f"\n=== Summary ===")
    log(f"  Generated {len(augmented)} augmented images")
    log(f"  Images: {n_imgs}")
    log(f"  Labels: {n_lbls}")
    log(f"  Location: {AUG_DIR}/")
    log("Done.")


if __name__ == "__main__":
    main()
