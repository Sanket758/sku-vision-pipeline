"""Generate multi-class YOLO labels from curated SKU subset.
Connects the human-renamed clusters back to original shelf images.

Usage:
    python generate_curated_labels.py --curated ../../Dataset/curated_subset --meta ../../Dataset/detections.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared import hardware_utils as hw
from shared.dataset_utils import YOLO_DIR, write_data_yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Curated Subset to YOLO Label Generator")
    parser.add_argument("--curated", type=str, default="../../Dataset/curated_subset",
                        help="Directory with renamed SKU cluster folders")
    parser.add_argument("--meta", type=str, default="../../Dataset/detections.json",
                        help="Path to detections.json metadata")
    parser.add_argument("--output", type=str, default="../../Dataset/processed_yolo",
                        help="YOLO dataset root")
    parser.add_argument("--val_split", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()

    curated_dir = Path(args.curated).resolve()
    meta_path = Path(args.meta).resolve()
    yolo_root = Path(args.output).resolve()

    if not curated_dir.exists():
        print(f"ERROR: Curated directory not found: {curated_dir}")
        sys.exit(1)
    if not meta_path.exists():
        print(f"ERROR: Metadata not found: {meta_path}")
        print("Run detect_crop_index.py first.")
        sys.exit(1)

    print("=" * 60)
    print(f"Source: {curated_dir}")
    print(f"Meta:   {meta_path}")
    print("=" * 60)

    # 1. Load Metadata
    with open(meta_path) as f:
        meta = json.load(f)

    # 2. Identify Classes from folder names
    # Skip folders that haven't been renamed (still start with 'cluster_')
    class_folders = sorted([d for d in curated_dir.iterdir() if d.is_dir()])
    
    sku_to_id = {}
    class_names = []
    
    for d in class_folders:
        if d.name.startswith("cluster_"):
            # Optional: Decide if we want to skip or include unnamed clusters as 'generic'
            # For this thesis, we only want the 30-60 target SKUs
            continue
        
        sku_to_id[d.name] = len(class_names)
        class_names.append(d.name)

    if not class_names:
        print("No renamed SKU folders found. Rename your clusters in Dataset/curated_subset/ first.")
        sys.exit(1)

    print(f"Found {len(class_names)} target classes.")

    # 3. Map Crops to Labels
    # image_labels[orig_img_path] = list of [class_id, x_center, y_center, width, height]
    image_labels = {}

    for sku_name, class_id in sku_to_id.items():
        sku_dir = curated_dir / sku_name
        for crop_path in sku_dir.glob("*.jpg"):
            if crop_path.name in meta:
                m = meta[crop_path.name]
                orig_img = m["orig_img"]
                box = m["box_2d"] # [x1, y1, x2, y2]
                w, h = m["orig_size"]

                # Convert to YOLO format (normalized xywh)
                x_center = ((box[0] + box[2]) / 2) / w
                y_center = ((box[1] + box[3]) / 2) / h
                bw = (box[2] - box[0]) / w
                bh = (box[3] - box[1]) / h

                if orig_img not in image_labels:
                    image_labels[orig_img] = []
                
                image_labels[orig_img].append([class_id, x_center, y_center, bw, bh])

    # 4. Prepare YOLO structure
    import random
    import shutil

    for split in ("train", "val"):
        (yolo_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parents[2]
    
    img_list = list(image_labels.keys())
    random.seed(42)
    random.shuffle(img_list)
    
    val_count = int(len(img_list) * args.val_split)
    splits = {
        "val": img_list[:val_count],
        "train": img_list[val_count:]
    }

    print(f"Processing {len(img_list)} images into YOLO format...")
    
    for split_name, imgs in splits.items():
        for img_rel_path in imgs:
            src_img = project_root / img_rel_path
            if not src_img.exists():
                continue
            
            dst_img = yolo_root / "images" / split_name / src_img.name
            shutil.copy2(src_img, dst_img)
            
            label_txt = yolo_root / "labels" / split_name / (src_img.stem + ".txt")
            with open(label_txt, "w") as f:
                for label in image_labels[img_rel_path]:
                    f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

    # 5. Write data.yaml
    write_data_yaml(yolo_root, nc=len(class_names), names=class_names)

    print(f"\nSuccess. YOLO dataset ready at {yolo_root}")
    print(f"  Classes: {len(class_names)}")
    print(f"  Images:  {len(img_list)} ({len(splits['train'])} train, {len(splits['val'])} val)")


if __name__ == "__main__":
    main()
