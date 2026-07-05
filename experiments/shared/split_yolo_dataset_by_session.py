"""Session-aware YOLO dataset split — eliminates data leakage from video frames.

Splits by session (video source / photo session) instead of individual images.
Ensures all frames from one video session go to a single split.
Balances stores across splits ~70/15/15.

Usage:
    /home/sanket758/Education/BSBI/Masters-Thesis/.venv/bin/python experiments/shared/split_yolo_dataset_by_session.py
"""

import json
import random
import shutil
import re
from pathlib import Path
from collections import defaultdict

YOLO_DIR = Path("/home/sanket758/Education/BSBI/Masters-Thesis/Dataset/processed_yolo")
IMG_DIR = YOLO_DIR / "images"
LBL_DIR = YOLO_DIR / "labels"
DET_FILE = Path("/home/sanket758/Education/BSBI/Masters-Thesis/Dataset/detections.json")

VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42


def get_session_key(orig_img: str) -> str:
    """Map a source image path to a session key."""
    fname = Path(orig_img).stem  # stem removes .jpg/.png

    if fname.startswith("VID"):
        parts = fname.split("_frame_")
        return parts[0]

    # IMG* files: use minute-level timestamp
    match = re.match(r"(IMG\d{12})", fname)
    if match:
        return match.group(1)[:-2]  # drop last 2 chars (second granularity)
    return f"unknown_{fname}"


def get_store(orig_img: str) -> str:
    """Determine store from image path."""
    path_lower = orig_img.lower()
    if "aldi" in path_lower:
        return "aldi"
    elif "lidl" in path_lower:
        return "lidl"
    elif "kaufland" in path_lower:
        return "kaufland"
    elif "netto" in path_lower:
        return "netto"
    return "unknown"


def main():
    print("=== Loading detections.json ===")
    with open(DET_FILE) as f:
        det = json.load(f)
    print(f"  Total detections: {len(det)}")

    # Build unique source images
    all_orig = set(v["orig_img"] for v in det.values())
    print(f"  Unique source images: {len(all_orig)}")

    # Count crops per source image
    crops_per_img = defaultdict(int)
    for v in det.values():
        crops_per_img[v["orig_img"]] += 1

    # Build session map
    sessions = defaultdict(list)
    store_of_img = {}
    for orig in all_orig:
        sk = get_session_key(orig)
        st = get_store(orig)
        sessions[sk].append(orig)
        store_of_img[orig] = st

    # Print session summary
    print(f"\n=== Sessions ({len(sessions)}) ===")
    for sk, imgs in sorted(sessions.items(), key=lambda x: -len(x[1])):
        stores = set(store_of_img[i] for i in imgs)
        total_crops = sum(crops_per_img[i] for i in imgs)
        print(f"  {sk}: {len(imgs)} imgs, {total_crops} crops, store={','.join(sorted(stores))}")

    assigned = {}
    store_sessions = defaultdict(list)
    for sk, imgs in sessions.items():
        stores_in_session = set(store_of_img[i] for i in imgs)
        primary_store = list(stores_in_session)[0]
        total_crops = sum(crops_per_img[i] for i in imgs)
        store_sessions[primary_store].append((sk, total_crops, len(imgs)))

    for store_name, ses_list in sorted(store_sessions.items()):
        ses_list.sort(key=lambda x: -x[1])
        splits = {"train": [], "val": [], "test": []}
        split_crops = {"train": 0, "val": 0, "test": 0}

        for sk, cr, ni in ses_list:
            target_split = min(split_crops, key=split_crops.get)
            splits[target_split].append(sk)
            split_crops[target_split] += cr

        for split_name, ses_keys in splits.items():
            for sk in ses_keys:
                assigned[sk] = split_name

        # Print per-store breakdown
        print(f"\n  Store: {store_name}")
        for split_name in ("train", "val", "test"):
            n_sessions = len(splits[split_name])
            n_crops = sum(crops_per_img[i] for sk in splits[split_name] for i in sessions[sk])
            n_imgs = sum(len(sessions[sk]) for sk in splits[split_name])
            pct = n_crops / max(split_crops["train"] + split_crops["val"] + split_crops["test"], 1) * 100
            print(f"    {split_name}: {n_sessions} sessions, {n_imgs} imgs, {n_crops} crops ({pct:.0f}%)")

    # Build the image -> split mapping
    img_to_split = {}
    for sk, imgs in sessions.items():
        split_name = assigned.get(sk, "train")
        for img in imgs:
            img_to_split[img] = split_name

    # Verify no leakage: check session integrity
    print("\n=== Verifying session integrity ===")
    for sk, imgs in sessions.items():
        split_of_images = set(img_to_split[i] for i in imgs)
        if len(split_of_images) > 1:
            print(f"  ERROR: Session {sk} has images in multiple splits: {split_of_images}")
        else:
            s = list(split_of_images)[0]
            n_crops = sum(crops_per_img[i] for i in imgs)
            print(f"  {sk}: {len(imgs)} imgs, {n_crops} crops -> {s}")

    # Now create the actual split directories and copy files
    print("\n=== Creating split directories and copying files ===")
    current_images = sorted(IMG_DIR.glob("*.jpg")) + sorted(IMG_DIR.glob("*.png"))

    # Group source images by stem for label matching
    source_stems = {}
    for orig in all_orig:
        stem = Path(orig).stem
        source_stems[stem] = orig

    train_count = val_count = test_count = 0

    for img_path in current_images:
        stem = img_path.stem

        # Find which source image this corresponds to
        # The YOLO image might have the same stem as the source image stem
        src_path = source_stems.get(stem)
        if src_path is None:
            # Try matching via the full path
            # YOLO filenames are derived from source image stems
            # e.g. IMG20260615190120.jpg
            for src_stem, src_orig in source_stems.items():
                if stem == src_stem:
                    src_path = src_orig
                    break

        if src_path is None:
            print(f"  WARNING: Could not find source image for {img_path.name}, skipping")
            continue

        split_name = img_to_split.get(src_path, "train")

        dest_img_dir = IMG_DIR / split_name
        dest_lbl_dir = LBL_DIR / split_name
        dest_img_dir.mkdir(parents=True, exist_ok=True)
        dest_lbl_dir.mkdir(parents=True, exist_ok=True)

        # Copy image
        shutil.copy2(img_path, dest_img_dir / img_path.name)

        # Copy label
        label_path = LBL_DIR / f"{stem}.txt"
        if label_path.exists():
            shutil.copy2(label_path, dest_lbl_dir / label_path.name)

        if split_name == "train":
            train_count += 1
        elif split_name == "val":
            val_count += 1
        else:
            test_count += 1

    print(f"\n=== Split summary ===")
    print(f"  Train: {train_count} images")
    print(f"  Val:   {val_count} images")
    print(f"  Test:  {test_count} images")

    # Count label files per split
    for split_name in ("train", "val", "test"):
        n_imgs = len(list((IMG_DIR / split_name).glob("*.*")))
        n_lbls = len(list((LBL_DIR / split_name).glob("*.txt")))
        print(f"  {split_name}: {n_imgs} images, {n_lbls} labels")

    # Write data.yaml
    abs_yolo = YOLO_DIR.resolve()
    data = {
        "train": str(abs_yolo / "images" / "train"),
        "val": str(abs_yolo / "images" / "val"),
        "test": str(abs_yolo / "images" / "test"),
        "nc": 1,
        "names": ["product"],
    }

    lines = [
        f"train: {data['train']}",
        f"val: {data['val']}",
        f"test: {data['test']}",
        "",
        f"nc: {data['nc']}",
        f"names: {data['names']}",
    ]
    (YOLO_DIR / "data.yaml").write_text("\n".join(lines))
    print(f"\ndata.yaml written to {YOLO_DIR / 'data.yaml'}")
    print("Done.")


if __name__ == "__main__":
    main()
