"""End-to-end: detect products with YOLOv5 SKU-110K → crop → re-index with CLIP.

Usage:
    python detect_crop_index.py                    # Full pipeline (detect + index)
    python detect_crop_index.py --index_only        # Index existing crops only
    python detect_crop_index.py --detect_only       # Detect and crop only
    python detect_crop_index.py --conf 0.25         # Custom confidence threshold
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared import hardware_utils as hw
from shared.metrics_logger import MetricsLogger
from shared.yolo_detector import YOLODetector


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIRS = [PROJECT_ROOT / "Dataset" / "kaufland", PROJECT_ROOT / "Dataset" / "netto"]
CROP_DIR = PROJECT_ROOT / "Dataset" / "processed_retrieval"
INDEX_PATH = PROJECT_ROOT / ".faiss_index" / "products.index"


def parse_args():
    parser = argparse.ArgumentParser(description="Detect → Crop → Re-index pipeline")
    parser.add_argument("--conf", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    parser.add_argument("--detect_only", action="store_true", help="Skip indexing step")
    parser.add_argument("--index_only", action="store_true", help="Skip detection, index existing crops only")
    return parser.parse_args()


def detect_and_crop(args):
    import json
    from PIL import Image

    print(f"\nLoading YOLOv5 SKU-110K model...")
    detector = YOLODetector.get_instance()
    if args.device:
        detector.model.to(args.device)

    CROP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Metadata for multi-class mapping
    detection_metadata = {}

    image_extensions = {".jpg", ".jpeg", ".png"}
    image_paths = sorted(
        p for raw_dir in RAW_DIRS for p in raw_dir.glob("*") if p.suffix.lower() in image_extensions
    )

    total_crops = 0
    skipped = 0

    for img_idx, img_path in enumerate(image_paths):
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [{img_idx + 1}/{len(image_paths)}] Skipping {img_path.name}: {e}")
            skipped += 1
            continue

        detections = detector.detect(image, conf=args.conf)
        n_detections = len(detections)

        for det_idx, det in enumerate(detections):
            x1, y1, x2, y2 = det["box"]

            crop = image.crop((x1, y1, x2, y2))
            crop_name = f"{img_path.stem}_det{det_idx:03d}.jpg"
            crop.save(CROP_DIR / crop_name, quality=92)
            
            # Save metadata
            detection_metadata[crop_name] = {
                "orig_img": str(img_path.relative_to(PROJECT_ROOT)),
                "box_2d": det["box"],
                "score": det["score"],
                "orig_size": [image.width, image.height]
            }
            total_crops += 1

        if (img_idx + 1) % 10 == 0:
            print(f"  [{img_idx + 1}/{len(image_paths)}] {n_detections} detections, total crops: {total_crops}")
    
    # Save detection metadata
    meta_path = PROJECT_ROOT / "Dataset" / "detections.json"
    with open(meta_path, "w") as f:
        json.dump(detection_metadata, f, indent=2)

    print(f"\nDetection complete.")
    print(f"  Total images: {len(image_paths)}")
    print(f"  Skipped (corrupt): {skipped}")
    print(f"  Total crops saved: {total_crops} to {CROP_DIR}")
    print(f"  Metadata saved to: {meta_path}")
    return total_crops


def build_index():
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from shared import hardware_utils as hw
    from shared.metrics_logger import MetricsLogger
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    from PIL import Image

    device = "cuda" if hw.get_vram_gb() > 0 else "cpu"

    logger = MetricsLogger("faiss_index_crops", config={"source": str(CROP_DIR)})
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading CLIP model (clip-ViT-B-32) on {device}...")
    model = SentenceTransformer("clip-ViT-B-32", device=device)

    image_paths = sorted(CROP_DIR.glob("*.jpg"))
    print(f"Encoding {len(image_paths)} product crops...")

    all_embeddings = []
    all_paths = []

    batch_size = 64
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = [Image.open(p).convert("RGB") for p in batch_paths]

        embeddings = model.encode(batch_images, convert_to_numpy=True, show_progress_bar=False)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)

        all_embeddings.append(embeddings.astype(np.float32))
        all_paths.extend(batch_paths)

        if (i // batch_size) % 10 == 0:
            print(f"  {min(i + batch_size, len(image_paths))}/{len(image_paths)}")

    if not all_embeddings:
        print("No crops to index.")
        return

    embeddings_matrix = np.vstack(all_embeddings)
    dim = embeddings_matrix.shape[1]

    print(f"Building FAISS index (dim={dim}, vectors={len(embeddings_matrix)})...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_matrix)
    faiss.write_index(index, str(INDEX_PATH))

    id_map_path = INDEX_PATH.with_suffix(".paths.txt")
    id_map_path.write_text("\n".join(str(p.relative_to(CROP_DIR)) for p in all_paths))

    logger.log_hyperparams({
        "model": "clip-ViT-B-32",
        "dimension": dim,
        "num_vectors": len(embeddings_matrix),
        "device": device,
    })
    logger.log_metrics({
        "num_vectors": len(embeddings_matrix),
        "dimension": dim,
        "index_size_mb": round(INDEX_PATH.stat().st_size / 1e6, 2),
    })
    logger.flush()

    print(f"\nIndex rebuilt successfully.")
    print(f"  Vectors: {len(embeddings_matrix)} product crops")
    print(f"  Index: {INDEX_PATH}")
    print(f"  Size: {INDEX_PATH.stat().st_size / 1e6:.2f} MB")
    print(f"  Log: {logger.get_run_path()}")


def main():
    args = parse_args()
    print("=" * 60)
    print(hw.hw_summary())
    print("=" * 60)

    t0 = time.time()

    if args.index_only:
        print("\nIndex-only mode: using existing crops...")
        build_index()
    else:
        if args.detect_only:
            print("\nDetection-only mode...")
        else:
            print("\nFull pipeline: detect → crop → index...")

        old_crops = list(CROP_DIR.glob("*.jpg"))
        if old_crops:
            print(f"Clearing {len(old_crops)} existing crops...")
            for f in old_crops:
                f.unlink()

        total_crops = detect_and_crop(args)

        if total_crops == 0:
            print("No crops generated. Exiting.")
            return

        if not args.detect_only:
            t1 = time.time()
            build_index()
            print(f"Indexing time: {time.time() - t1:.1f}s")

    print(f"\nTotal time: {(time.time() - t0) / 60:.1f} min")
    print("Done.")


if __name__ == "__main__":
    main()
