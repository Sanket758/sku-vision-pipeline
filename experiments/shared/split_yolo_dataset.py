"""Split flat processed_yolo into train/val/test splits + write data.yaml."""

import random
import shutil
from pathlib import Path

YOLO_DIR = Path("/home/sanket758/Education/BSBI/Masters-Thesis/Dataset/processed_yolo")
IMG_DIR = YOLO_DIR / "images"
LBL_DIR = YOLO_DIR / "labels"

VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42

# Get all image files
images = sorted(IMG_DIR.glob("*.jpg")) + sorted(IMG_DIR.glob("*.png"))
print(f"Total images: {len(images)}")

# Shuffle and split
random.seed(SEED)
random.shuffle(images)

n = len(images)
n_val = int(n * VAL_SPLIT)
n_test = int(n * TEST_SPLIT)
n_train = n - n_val - n_test

splits = {
    "train": images[:n_train],
    "val": images[n_train:n_train + n_val],
    "test": images[n_train + n_val:],
}

# Create split directories and move files
for split_name, split_images in splits.items():
    split_img_dir = IMG_DIR / split_name
    split_lbl_dir = LBL_DIR / split_name
    split_img_dir.mkdir(parents=True, exist_ok=True)
    split_lbl_dir.mkdir(parents=True, exist_ok=True)

    for img_path in split_images:
        # Copy image
        shutil.copy2(img_path, split_img_dir / img_path.name)
        # Copy label
        label_path = LBL_DIR / f"{img_path.stem}.txt"
        if label_path.exists():
            shutil.copy2(label_path, split_lbl_dir / label_path.name)

    print(f"  {split_name}: {len(split_images)} images")

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

# Count labels per split
for split_name in ("train", "val", "test"):
    n_labels = len(list((LBL_DIR / split_name).glob("*.txt")))
    print(f"  {split_name} labels: {n_labels}")

print(f"\ndata.yaml written to {YOLO_DIR / 'data.yaml'}")
print(f"\nDetection dataset summary:")
print(f"  {n_train} train, {n_val} val, {n_test} test")
print(f"  Classes: 1 (product)")
print(f"  Ready for YOLOv8/v10 training")
