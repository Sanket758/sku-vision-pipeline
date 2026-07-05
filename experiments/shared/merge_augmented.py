"""Copy augmented images into YOLO train set and update data.yaml."""
import shutil
import sys
from pathlib import Path

YOLO_DIR = Path("/home/sanket758/Education/BSBI/Masters-Thesis/Dataset/processed_yolo")
AUG_DIR = YOLO_DIR / "augmented"
DATA_YAML = YOLO_DIR / "data.yaml"

def log(m): print(m, flush=True)

log("=== Merging augmented images into train set ===")
train_img_dir = YOLO_DIR / "images" / "train"
train_lbl_dir = YOLO_DIR / "labels" / "train"

aug_imgs = sorted((AUG_DIR / "images").glob("*.*"))
aug_lbls = sorted((AUG_DIR / "labels").glob("*.txt"))
log(f"  Augmented images: {len(aug_imgs)}")
log(f"  Augmented labels: {len(aug_lbls)}")

n_copied = 0
for img in aug_imgs:
    dst = train_img_dir / img.name
    shutil.copy2(img, dst)
    lbl = AUG_DIR / "labels" / f"{img.stem}.txt"
    if lbl.exists():
        shutil.copy2(lbl, train_lbl_dir / lbl.name)
    n_copied += 1

n_train_orig = len(list(train_img_dir.glob("*.*"))) - n_copied
n_train_final = len(list(train_img_dir.glob("*.*")))
log(f"  Original train images: {n_train_orig}")
log(f"  Copied augmented: {n_copied}")
log(f"  Final train images: {n_train_final}")

log("\n=== Updating data.yaml ===")
if DATA_YAML.exists():
    text = DATA_YAML.read_text()
    if "augmented" not in text:
        log("  data.yaml exists; augmented images are in the train/ dir so no change needed")
    else:
        log(f"  data.yaml content:\n{text[:500]}")
    n_train = len(list((YOLO_DIR / "labels" / "train").glob("*.txt")))
    n_val = len(list((YOLO_DIR / "labels" / "val").glob("*.txt")))
    n_test = len(list((YOLO_DIR / "labels" / "test").glob("*.txt")))
    log(f"  Train labels: {n_train}, Val labels: {n_val}, Test labels: {n_test}")
else:
    log("  WARNING: data.yaml not found!")
    
log("\nDone. Training set ready with augmented images.")
