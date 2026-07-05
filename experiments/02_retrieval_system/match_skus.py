"""Match retrieval product crops to SKU reference images using CLIP embeddings.

This script finds the closest SKU in the `pre-master` catalog for each of the
79k cropped images in `Dataset/processed_retrieval/`. It outputs the SKU data
distribution to a CSV file, copies matching crops of the top-N SKUs to folders
for visual verification, and generates a YOLO format dataset on full shelf images.

Usage:
    # 1. Run analysis and save SKU data distribution:
    python match_skus.py

    # 2. Select top 50 SKUs, copy crops for verification, and generate YOLO dataset:
    python match_skus.py --copy_crops --gen_yolo --top_n 50
"""

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

# Setup path to import shared modules
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared import hardware_utils as hw
from shared.dataset_utils import write_data_yaml


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for SKU matching and curation.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="CLIP-based SKU matching and dataset generator")
    parser.add_argument(
        "--sku_dir",
        type=str,
        default="pre-master",
        help="Directory with reference SKU images",
    )
    parser.add_argument(
        "--retrieval_dir",
        type=str,
        default="Dataset/processed_retrieval",
        help="Directory with cropped product images",
    )
    parser.add_argument(
        "--meta",
        type=str,
        default="Dataset/detections.json",
        help="Path to detections metadata JSON",
    )
    parser.add_argument(
        "--output_dist",
        type=str,
        default="Dataset/sku_data_distribution.csv",
        help="Output CSV file path for SKU data distribution",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="Dataset",
        help="Directory to cache retrieval embeddings",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.82,
        help="Cosine similarity threshold for a valid SKU match",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=50,
        help="Number of top SKUs to select for dataset generation (30 to 60)",
    )
    parser.add_argument(
        "--copy_crops",
        action="store_true",
        help="Copy matching crops of top SKUs to validation folders",
    )
    parser.add_argument(
        "--gen_yolo",
        action="store_true",
        help="Generate YOLO format shelf image dataset for top SKUs",
    )
    parser.add_argument(
        "--yolo_dir",
        type=str,
        default="Dataset/processed_yolo",
        help="Output directory for YOLO dataset",
    )
    parser.add_argument(
        "--recog_dir",
        type=str,
        default="Dataset/recognition",
        help="Output directory for visual verification folders",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.20,
        help="Fraction of images to allocate for validation",
    )
    return parser.parse_args()


def load_model(device: str) -> "SentenceTransformer":
    """Load the SentenceTransformer CLIP model.

    Args:
        device (str): Device to load the model onto.

    Returns:
        SentenceTransformer: Loaded CLIP model.
    """
    from sentence_transformers import SentenceTransformer

    print(f"Loading CLIP model 'clip-ViT-B-32' on {device}...")
    return SentenceTransformer("clip-ViT-B-32", device=device)


def embed_images(
    image_paths: List[Path],
    model: "SentenceTransformer",
    batch_size: int = 128,
) -> np.ndarray:
    """Compute normalized CLIP embeddings for a list of image paths in batches.

    Args:
        image_paths (List[Path]): List of image paths.
        model (SentenceTransformer): CLIP model instance.
        batch_size (int): Batch size for inference.

    Returns:
        np.ndarray: Normalized embeddings matrix of shape (N, 512).
    """
    embeddings_list = []
    total = len(image_paths)

    for i in range(0, total, batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_images = []
        for p in batch_paths:
            try:
                batch_images.append(Image.open(p).convert("RGB"))
            except Exception as e:
                print(f"  Warning: Cannot open image {p.name}: {e}")

        if not batch_images:
            continue

        # Compute embeddings
        embs = model.encode(batch_images, convert_to_numpy=True, show_progress_bar=False)
        # Normalize L2
        embs = embs / np.linalg.norm(embs, axis=-1, keepdims=True)
        embeddings_list.append(embs.astype(np.float32))

        if (i // batch_size) % 20 == 0 or (i + batch_size) >= total:
            print(f"  Processed {min(i + batch_size, total)}/{total} images...")

    return np.vstack(embeddings_list) if embeddings_list else np.empty((0, 512), dtype=np.float32)


def main() -> None:
    """Main execution function for SKU matching, statistical analysis, and dataset curation."""
    args = parse_args()

    sku_dir = Path(args.sku_dir).resolve()
    retrieval_dir = Path(args.retrieval_dir).resolve()
    meta_path = Path(args.meta).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    output_dist = Path(args.output_dist).resolve()

    if not sku_dir.exists():
        print(f"ERROR: SKU directory not found at {sku_dir}")
        sys.exit(1)
    if not retrieval_dir.exists():
        print(f"ERROR: Retrieval directory not found at {retrieval_dir}")
        sys.exit(1)
    if not meta_path.exists():
        print(f"ERROR: Metadata not found at {meta_path}")
        sys.exit(1)

    print("=" * 60)
    print(hw.hw_summary())
    print(f"SKUs Source:     {sku_dir}")
    print(f"Retrieval Crops: {retrieval_dir}")
    print(f"Metadata Path:   {meta_path}")
    print(f"Output Dist CSV: {output_dist}")
    print("=" * 60)

    # 1. Discover files
    sku_paths = sorted(
        [p for p in sku_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    )
    sku_names = [p.stem for p in sku_paths]
    print(f"Found {len(sku_paths)} SKUs in reference catalog.")

    crop_paths = sorted(
        [
            p
            for p in retrieval_dir.glob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]
    )
    print(f"Found {len(crop_paths)} cropped product images in retrieval folder.")

    if not sku_paths or not crop_paths:
        print("ERROR: Missing SKU reference images or crop images.")
        sys.exit(1)

    # 2. Get device
    device = "cuda" if hw.get_vram_gb() > 0 else "cpu"

    # 3. Load or compute retrieval crop embeddings (cached)
    emb_cache_file = cache_dir / "retrieval_embeddings.npy"
    names_cache_file = cache_dir / "retrieval_filenames.txt"

    if emb_cache_file.exists() and names_cache_file.exists():
        print(f"\nLoading cached crop embeddings from {emb_cache_file}...")
        crop_embs = np.load(str(emb_cache_file))
        cached_names = names_cache_file.read_text().strip().split("\n")

        # Sanity check
        if len(cached_names) == len(crop_paths) and crop_embs.shape[0] == len(crop_paths):
            print("Cached embeddings verify successfully.")
            crop_names = cached_names
        else:
            print("Warning: Cache mismatch with current files. Re-computing embeddings...")
            model = load_model(device)
            crop_embs = embed_images(crop_paths, model, batch_size=256)
            crop_names = [p.name for p in crop_paths]
            np.save(str(emb_cache_file), crop_embs)
            names_cache_file.write_text("\n".join(crop_names))
            print(f"Saved crop embeddings cache to {emb_cache_file}")
    else:
        print("\nComputing embeddings for retrieval crops (this will run once and cache)...")
        model = load_model(device)
        crop_embs = embed_images(crop_paths, model, batch_size=256)
        crop_names = [p.name for p in crop_paths]
        np.save(str(emb_cache_file), crop_embs)
        names_cache_file.write_text("\n".join(crop_names))
        print(f"Saved crop embeddings cache to {emb_cache_file}")

    # 4. Compute SKU reference embeddings
    print("\nComputing embeddings for reference SKUs...")
    model = load_model(device)
    sku_embs = embed_images(sku_paths, model, batch_size=128)

    # 5. Compute Similarity Matrix on GPU if possible
    print("\nCalculating matches using cosine similarity matrix...")
    t_start = time.time()
    
    # Convert matrices to PyTorch tensors
    crop_t = torch.from_numpy(crop_embs).to(device)
    sku_t = torch.from_numpy(sku_embs).to(device)

    # Compute cosine similarity matrix (both are normalized, so matmul is cosine similarity)
    # shape: (num_crops, num_skus)
    sim_matrix = torch.matmul(crop_t, sku_t.T)

    # For each crop, find the highest score and the matching SKU index
    scores, best_idx = torch.max(sim_matrix, dim=1)

    # Move back to CPU
    scores = scores.cpu().numpy()
    best_idx = best_idx.cpu().numpy()
    print(f"Matrix multiplication and matching took {time.time() - t_start:.2f} seconds.")

    # 6. Analyze SKU Data Distribution
    print("\nAnalyzing matches at different thresholds...")
    thresholds = [0.75, 0.80, 0.82, 0.85, 0.90]
    
    # sku_stats[sku_name] = dictionary of threshold -> match_count
    sku_stats: Dict[str, Dict[float, int]] = {
        name: {t: 0 for t in thresholds} for name in sku_names
    }

    # Record matched details for each crop
    crop_matches: List[Tuple[str, str, float]] = []
    for c_name, score, idx in zip(crop_names, scores, best_idx):
        matched_sku = sku_names[idx]
        crop_matches.append((c_name, matched_sku, float(score)))

        for t in thresholds:
            if score >= t:
                sku_stats[matched_sku][t] += 1

    # 7. Write SKU Data Distribution to CSV
    output_dist.parent.mkdir(parents=True, exist_ok=True)
    with open(output_dist, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sku_name"] + [f"matches_at_{t:.2f}" for t in thresholds])
        for name in sku_names:
            writer.writerow([name] + [sku_stats[name][t] for t in thresholds])

    print(f"Saved SKU data distribution to {output_dist}")

    # 8. Sort and Display Top SKUs
    target_t = args.threshold
    sorted_skus = sorted(sku_names, key=lambda name: sku_stats[name][target_t], reverse=True)

    print(f"\nTop 100 SKUs by Crop Count (at threshold >= {target_t}):")
    print("-" * 60)
    print(f"{'Rank':<6}{'SKU Name':<40}{f'Matches (>= {target_t:.2f})':<15}")
    print("-" * 60)
    for rank, name in enumerate(sorted_skus[:100], 1):
        print(f"{rank:<6}{name:<40}{sku_stats[name][target_t]:<15}")
    print("-" * 60)

    # Count how many SKUs have > 0 matches
    skus_with_data = sum(1 for name in sku_names if sku_stats[name][target_t] > 0)
    print(f"Total SKUs with at least 1 match (>= {target_t}): {skus_with_data} / {len(sku_names)}")

    # 9. Perform Curation Tasks
    selected_skus = sorted_skus[: args.top_n]
    print(f"\nAutomatically selected top {args.top_n} SKUs for dataset operations.")

    if args.copy_crops:
        recog_dir = Path(args.recog_dir).resolve()
        if recog_dir.exists():
            print(f"\nClearing existing recognition directory: {recog_dir}")
            shutil.rmtree(recog_dir)
        recog_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nCopying matching crops for top {len(selected_skus)} SKUs to {recog_dir}...")
        copied_count = 0
        for c_name, matched_sku, score in crop_matches:
            if matched_sku in selected_skus and score >= target_t:
                sku_folder = recog_dir / matched_sku
                sku_folder.mkdir(parents=True, exist_ok=True)
                shutil.copy2(retrieval_dir / c_name, sku_folder / c_name)
                copied_count += 1

        print(f"Curation complete: Copied {copied_count} crop images into class-organized folders.")

    if args.gen_yolo:
        yolo_dir = Path(args.yolo_dir).resolve()
        print(f"\nGenerating YOLO full shelf image dataset at {yolo_dir}...")

        # Load detections metadata
        with open(meta_path) as f:
            meta = json.load(f)

        # Map selected SKU name -> YOLO class ID
        sku_to_id = {name: i for i, name in enumerate(selected_skus)}

        # Clear existing YOLO directories
        if yolo_dir.exists():
            print(f"Clearing existing YOLO directory: {yolo_dir}")
            shutil.rmtree(yolo_dir)

        for split in ("train", "val"):
            (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        # Map: original_shelf_image -> list of [class_id, x_center, y_center, width, height]
        shelf_labels: Dict[str, List[List[float]]] = {}

        matched_yolo_crops = 0
        for c_name, matched_sku, score in crop_matches:
            if matched_sku in selected_skus and score >= target_t:
                if c_name in meta:
                    m = meta[c_name]
                    orig_img = m["orig_img"]
                    box = m["box_2d"]  # [x1, y1, x2, y2]
                    w, h = m["orig_size"]

                    # Convert to YOLO format (normalized center x, center y, width, height)
                    x_center = ((box[0] + box[2]) / 2.0) / w
                    y_center = ((box[1] + box[3]) / 2.0) / h
                    bw = (box[2] - box[0]) / w
                    bh = (box[3] - box[1]) / h

                    class_id = sku_to_id[matched_sku]

                    if orig_img not in shelf_labels:
                        shelf_labels[orig_img] = []
                    
                    shelf_labels[orig_img].append([class_id, x_center, y_center, bw, bh])
                    matched_yolo_crops += 1

        # Train/Val Split on unique shelf images
        import random
        random.seed(42)

        shelf_images = list(shelf_labels.keys())
        random.shuffle(shelf_images)

        val_count = int(len(shelf_images) * args.val_split)
        splits = {
            "val": shelf_images[:val_count],
            "train": shelf_images[val_count:]
        }

        project_root = Path(__file__).resolve().parents[2]
        processed_images = 0

        for split_name, imgs in splits.items():
            for img_rel_path in imgs:
                src_img = project_root / img_rel_path
                if not src_img.exists():
                    # Fallback to Dataset/raw/<store>/<filename> if moved
                    parts = Path(img_rel_path).parts
                    if len(parts) >= 3 and parts[0] == "Dataset":
                        raw_attempt = project_root / "Dataset" / "raw" / parts[1] / parts[2]
                        if raw_attempt.exists():
                            src_img = raw_attempt
                        else:
                            # Fallback to absolute path check
                            src_img = Path(img_rel_path)
                            if not src_img.exists():
                                continue
                    else:
                        # Fallback to absolute path check
                        src_img = Path(img_rel_path)
                        if not src_img.exists():
                            continue

                dst_img = yolo_dir / "images" / split_name / src_img.name
                shutil.copy2(src_img, dst_img)

                label_txt = yolo_dir / "labels" / split_name / (src_img.stem + ".txt")
                with open(label_txt, "w") as lf:
                    for label in shelf_labels[img_rel_path]:
                        lf.write(
                            f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n"
                        )
                processed_images += 1

        # Write data.yaml
        write_data_yaml(yolo_dir, nc=len(selected_skus), names=selected_skus)

        print(f"YOLO Generation complete:")
        print(f"  Processed {processed_images} shelf images with annotations.")
        print(f"  Mapped {matched_yolo_crops} bounding boxes matching selected SKUs.")
        print(f"  Train: {len(splits['train'])} images, Val: {len(splits['val'])} images")
        print(f"  Dataset YAML saved to: {yolo_dir}/data.yaml")

    print("\nProcessing finished successfully.")
    print("Next steps:")
    print("  1. Inspect the matching CSV at: Dataset/sku_data_distribution.csv")
    if args.copy_crops:
        print(f"  2. Visually verify matches in the folders under: Dataset/recognition/")
    if args.gen_yolo:
        print(f"  3. Train a YOLO model using: Dataset/processed_yolo/data.yaml")


if __name__ == "__main__":
    main()
