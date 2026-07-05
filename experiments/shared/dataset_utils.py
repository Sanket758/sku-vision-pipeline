import shutil
import random
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "Dataset" / "raw"
YOLO_DIR = PROJECT_ROOT / "Dataset" / "processed_yolo"
RETRIEVAL_DIR = PROJECT_ROOT / "Dataset" / "processed_retrieval"


def organize_yolo_dataset(
    raw_dir: Optional[Path] = None,
    yolo_dir: Optional[Path] = None,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
):
    """Prepare YOLO directory structure from raw shelf images.
    
    Expects raw images under raw_dir/{store_name}/*.jpg.
    Splits into train/val/test. Labels dir is left empty for annotation.
    """
    raw_dir = raw_dir or RAW_DIR
    yolo_dir = yolo_dir or YOLO_DIR

    raw_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    image_paths = sorted(raw_dir.rglob("*.jpg")) + sorted(raw_dir.rglob("*.png"))
    if not image_paths:
        image_paths = sorted(PROJECT_ROOT.glob("Dataset/*/*.jpg"))
    if not image_paths:
        print("No images found. Place images in Dataset/raw/ or Dataset/{store}/")
        return

    random.seed(seed)
    random.shuffle(image_paths)

    n = len(image_paths)
    n_val = int(n * val_split)
    n_test = int(n * test_split)
    n_train = n - n_val - n_test

    splits = {
        "train": image_paths[:n_train],
        "val": image_paths[n_train:n_train + n_val],
        "test": image_paths[n_train + n_val:],
    }

    for split_name, paths in splits.items():
        for src in paths:
            dst = yolo_dir / "images" / split_name / src.name
            shutil.copy2(src, dst)

    print(f"YOLO dataset prepared at {yolo_dir}:")
    print(f"  Train: {len(splits['train'])} images")
    print(f"  Val:   {len(splits['val'])} images")
    print(f"  Test:  {len(splits['test'])} images")
    print("NOTE: Labels directory created empty. Add YOLO-format .txt files before training.")


def write_data_yaml(yolo_dir: Optional[Path] = None, nc: int = 1, names: Optional[list[str]] = None):
    """Write YOLO data.yaml for the processed YOLO dataset."""
    yolo_dir = yolo_dir or YOLO_DIR
    names = names or ["product"]

    abs_yolo = yolo_dir.resolve()
    data = {
        "train": str(abs_yolo / "images" / "train"),
        "val": str(abs_yolo / "images" / "val"),
        "test": str(abs_yolo / "images" / "test"),
        "nc": nc,
        "names": names,
    }

    lines = [
        f"train: {data['train']}",
        f"val: {data['val']}",
        f"test: {data['test']}",
        "",
        f"nc: {nc}",
        f"names: {names}",
        "",
    ]
    (yolo_dir / "data.yaml").write_text("\n".join(lines))
    print(f"data.yaml written with nc={nc}, names={names}")
    return yolo_dir / "data.yaml"
