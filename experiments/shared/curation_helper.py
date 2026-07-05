"""Helper script to curate the 30-60 SKU subset using CLIP embeddings and K-Means.
Helps group visually similar products from the 27,000+ crops.

Usage:
    python curation_helper.py --clusters 100 --top_n 60
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared import hardware_utils as hw


def parse_args():
    parser = argparse.ArgumentParser(description="SKU Curation Helper via Clustering")
    parser.add_argument("--index", type=str, default="../../.faiss_index/products.index",
                        help="Path to FAISS index")
    parser.add_argument("--paths", type=str, default=None,
                        help="Path to mapping file")
    parser.add_argument("--source", type=str, default="../../Dataset/processed_retrieval",
                        help="Source directory of crops")
    parser.add_argument("--output", type=str, default="../../Dataset/curated_subset",
                        help="Where to save organized clusters")
    parser.add_argument("--clusters", type=int, default=100, help="Number of clusters (K)")
    parser.add_argument("--top_n", type=int, default=60, help="Number of largest clusters to export")
    parser.add_argument("--min_size", type=int, default=10, help="Minimum images per class")
    return parser.parse_args()


def main():
    args = parse_args()

    index_path = Path(args.index).resolve()
    paths_file = Path(args.paths or index_path.with_suffix(".paths.txt"))
    source_dir = Path(args.source).resolve()
    output_dir = Path(args.output).resolve()

    if not index_path.exists():
        print(f"ERROR: Index not found at {index_path}")
        sys.exit(1)

    print("=" * 60)
    print(hw.hw_summary())
    print(f"Clustering {args.clusters} groups from {index_path.name}")
    print("=" * 60)

    try:
        import faiss
    except ImportError:
        print("ERROR: faiss-cpu not installed.")
        sys.exit(1)

    # 1. Load Index
    print("Loading vectors from index...")
    index = faiss.read_index(str(index_path))
    num_vectors = index.ntotal
    dim = index.d
    
    # Extract all vectors (reconstruct)
    # Note: IndexFlatIP supports reconstruction
    vectors = np.zeros((num_vectors, dim), dtype=np.float32)
    for i in range(num_vectors):
        vectors[i] = index.reconstruct(i)

    # 2. Run K-Means
    print(f"Running K-Means (K={args.clusters})...")
    kmeans = KMeans(n_clusters=args.clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vectors)

    # 3. Load Paths
    paths = paths_file.read_text().strip().split("\n")
    if len(paths) != num_vectors:
        print(f"WARNING: Path mismatch ({len(paths)} vs {num_vectors} vectors)")

    # 4. Group by cluster
    cluster_groups = {}
    for i, label in enumerate(labels):
        if label not in cluster_groups:
            cluster_groups[label] = []
        cluster_groups[label].append(paths[i])

    # 5. Sort by size and export top N
    sorted_clusters = sorted(cluster_groups.items(), key=lambda x: len(x[1]), reverse=True)
    
    print(f"\nExporting top {args.top_n} clusters to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_count = 0
    for cluster_id, cluster_paths in sorted_clusters:
        if len(cluster_paths) < args.min_size:
            continue
        
        if exported_count >= args.top_n:
            break

        class_dir = output_dir / f"cluster_{cluster_id:03d}_size_{len(cluster_paths)}"
        class_dir.mkdir(exist_ok=True)
        
        # Copy first 100 images to prevent bloat, but enough for review
        for p_str in cluster_paths[:100]:
            src = source_dir / p_str
            if src.exists():
                shutil.copy2(src, class_dir / src.name)
        
        print(f"  Class {exported_count+1:02d}: {class_dir.name}")
        exported_count += 1

    print(f"\nSuccess. {exported_count} clusters exported.")
    print("Next Steps:")
    print("1. Review clusters in Dataset/curated_subset/")
    print("2. Delete clusters that are low-quality (e.g. background, blurry).")
    print("3. Rename folders to SKU names (e.g. 'Coca_Cola_0.5L').")
    print("4. Use organized folders as the ground truth for retrieval and YOLO.")


if __name__ == "__main__":
    main()
