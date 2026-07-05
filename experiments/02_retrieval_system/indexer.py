"""Build a FAISS index from product images using CLIP embeddings (via sentence-transformers).

Usage:
    python indexer.py --source ../../Dataset/processed_retrieval --index ../../.faiss_index/products.index
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared import hardware_utils as hw
from shared.metrics_logger import MetricsLogger


def parse_args():
    parser = argparse.ArgumentParser(description="FAISS Index Builder from Product Images")
    parser.add_argument("--source", type=str, default="../../Dataset/processed_retrieval",
                        help="Directory with product images")
    parser.add_argument("--index", type=str, default="../../.faiss_index/products.index",
                        help="Output FAISS index path")
    parser.add_argument("--model_name", type=str, default="clip-ViT-B-32",
                        help="SentenceTransformer model name")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for encoding")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    return parser.parse_args()


def main():
    args = parse_args()

    source_dir = Path(args.source).resolve()
    index_path = Path(args.index).resolve()

    if not source_dir.exists():
        print(f"ERROR: Source directory not found: {source_dir}")
        print("Place product images in Dataset/processed_retrieval/")
        sys.exit(1)

    index_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(hw.hw_summary())
    print(f"Source:  {source_dir}")
    print(f"Index:   {index_path}")
    print(f"Model:   {args.model_name}")
    print("=" * 60)

    logger = MetricsLogger("faiss_index", config=vars(args))

    try:
        from sentence_transformers import SentenceTransformer
        import faiss
        from PIL import Image
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Install with: pip install sentence-transformers faiss-cpu pillow")
        sys.exit(1)

    device = args.device or ("cuda" if hw.get_vram_gb() > 0 else "cpu")
    print(f"\nLoading model '{args.model_name}' on {device}...")
    model = SentenceTransformer(args.model_name, device=device)

    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    image_paths = sorted([p for p in source_dir.rglob("*") if p.suffix.lower() in image_extensions])

    if not image_paths:
        print(f"No images found in {source_dir}.")
        sys.exit(1)

    print(f"Found {len(image_paths)} images. Generating embeddings...")

    all_embeddings = []
    all_paths = []

    for i in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[i:i + args.batch_size]
        batch_images = []
        valid_paths = []

        for img_path in batch_paths:
            try:
                batch_images.append(Image.open(img_path).convert("RGB"))
                valid_paths.append(img_path)
            except Exception as e:
                print(f"  Skipping {img_path.name}: {e}")

        if not batch_images:
            continue

        embeddings = model.encode(batch_images, convert_to_numpy=True, show_progress_bar=False)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)

        all_embeddings.append(embeddings.astype(np.float32))
        all_paths.extend(valid_paths)

        if (i // args.batch_size) % 10 == 0:
            print(f"  Processed {min(i + args.batch_size, len(image_paths))}/{len(image_paths)}")

    if not all_embeddings:
        print("No embeddings generated.")
        sys.exit(1)

    embeddings_matrix = np.vstack(all_embeddings).astype(np.float32)
    dim = embeddings_matrix.shape[1]

    print(f"\nBuilding FAISS index (dim={dim}, vectors={len(embeddings_matrix)})...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_matrix)
    faiss.write_index(index, str(index_path))

    id_map_path = index_path.with_suffix(".paths.txt")
    id_map_path.write_text("\n".join(str(p.relative_to(source_dir)) for p in all_paths))

    logger.log_hyperparams({
        "model": args.model_name,
        "dimension": dim,
        "num_vectors": len(embeddings_matrix),
        "device": device,
        "source_size": len(image_paths),
    })
    logger.log_metrics({
        "num_vectors": len(embeddings_matrix),
        "dimension": dim,
        "index_size_mb": round(index_path.stat().st_size / 1e6, 2),
    })
    logger.flush()

    print(f"\nIndex built successfully.")
    print(f"  Vectors: {len(embeddings_matrix)}")
    print(f"  Dimension: {dim}")
    print(f"  Index: {index_path}")
    print(f"  Path mapping: {id_map_path}")
    print(f"  Size: {index_path.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
