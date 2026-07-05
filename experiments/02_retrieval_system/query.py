"""Query the FAISS index and evaluate retrieval accuracy.

Usage:
    python query.py --query_image test.jpg --index ../../.faiss_index/products.index
    python query.py --eval --topk 5
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared import hardware_utils as hw
from shared.metrics_logger import MetricsLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Retrieval Evaluation for Product Search")
    parser.add_argument("--index", type=str, default="../../.faiss_index/products.index",
                        help="Path to FAISS index")
    parser.add_argument("--paths", type=str, default=None,
                        help="Path mapping file (auto: index path + .paths.txt)")
    parser.add_argument("--source", type=str, default="../../Dataset/processed_retrieval",
                        help="Source directory of product images")
    parser.add_argument("--query_image", type=str, default=None,
                        help="Single query image path")
    parser.add_argument("--eval", action="store_true", default=False,
                        help="Run batch evaluation on held-out queries")
    parser.add_argument("--topk", type=int, default=5, help="Top-K results for evaluation")
    parser.add_argument("--model_name", type=str, default="clip-ViT-B-32",
                        help="SentenceTransformer model name")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_index(index_path: Path, paths_file: Path):
    import faiss
    index = faiss.read_index(str(index_path))
    id_to_path = {}
    if paths_file.exists():
        lines = paths_file.read_text().strip().split("\n")
        for i, line in enumerate(lines):
            id_to_path[i] = line
    return index, id_to_path


def embed_image(image_path, model):
    from PIL import Image
    embedding = model.encode([Image.open(image_path).convert("RGB")], convert_to_numpy=True)
    embedding = embedding / np.linalg.norm(embedding, axis=-1, keepdims=True)
    return embedding.astype(np.float32)


def main():
    args = parse_args()

    index_path = Path(args.index).resolve()
    paths_file = Path(args.paths or index_path.with_suffix(".paths.txt"))
    source_dir = Path(args.source).resolve()

    if not index_path.exists():
        print(f"ERROR: Index not found at {index_path}")
        print("Run indexer.py first.")
        sys.exit(1)

    print("=" * 60)
    print(hw.hw_summary())
    print(f"Index: {index_path}")
    print(f"Paths: {paths_file}")
    print("=" * 60)

    logger = MetricsLogger("retrieval_query", config=vars(args))

    try:
        from sentence_transformers import SentenceTransformer
        import faiss
    except ImportError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    device = args.device or ("cuda" if hw.get_vram_gb() > 0 else "cpu")
    print(f"\nLoading model '{args.model_name}' on {device}...")
    model = SentenceTransformer(args.model_name, device=device)

    index, id_to_path = load_index(index_path, paths_file)
    print(f"Loaded index with {index.ntotal} vectors")

    if args.query_image:
        query_path = Path(args.query_image)
        if not query_path.exists():
            print(f"Query image not found: {query_path}")
            sys.exit(1)

        print(f"\n--- Query: {query_path.name} ---")
        query_emb = embed_image(query_path, model)
        scores, indices = index.search(query_emb, args.topk)

        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            path_str = id_to_path.get(int(idx), f"id_{idx}")
            print(f"  {rank + 1}. [{score:.4f}] {path_str}")

        logger.log_metrics({
            "top1_score": float(scores[0][0]),
            "topk": args.topk,
        })
        logger.save_artifact(str(query_path), f"query_{query_path.name}")

    if args.eval:
        print(f"\n--- Batch Evaluation (Top-{args.topk}) ---")
        # Support both flat and class-organized sources
        image_extensions = {".jpg", ".jpeg", ".png"}
        held_out = sorted([p for p in source_dir.rglob("*") if p.suffix.lower() in image_extensions])
        
        if not held_out:
            print(f"No held-out query images found in {source_dir}.")
            return

        top1_hits = 0
        topk_hits = 0
        total = 0

        for qpath in held_out:
            # Determine class from folder name if possible
            gt_class = qpath.parent.name if qpath.parent != source_dir else None
            
            query_emb = embed_image(qpath, model)
            scores, indices = index.search(query_emb, args.topk)

            for rank, idx in enumerate(indices[0]):
                retrieved_rel_path = id_to_path.get(int(idx), "")
                if not retrieved_rel_path:
                    continue
                
                # Success if:
                # 1. Exact path match (instance retrieval)
                # 2. Folder name match (class retrieval)
                is_hit = False
                if gt_class:
                    retrieved_class = Path(retrieved_rel_path).parent.name
                    if retrieved_class == gt_class:
                        is_hit = True
                else:
                    # Fallback to instance match
                    if retrieved_rel_path in str(qpath.relative_to(source_dir.parent)):
                        is_hit = True

                if is_hit:
                    if rank == 0:
                        top1_hits += 1
                    topk_hits += 1
                    break

            total += 1
            if total % 100 == 0:
                print(f"  Evaluated {total}/{len(held_out)}...")

        top1_acc = top1_hits / total if total else 0
        topk_acc = topk_hits / total if total else 0

        print(f"  Total queries: {total}")
        print(f"  Top-1 Accuracy: {top1_acc:.4f} ({top1_hits}/{total})")
        print(f"  Top-{args.topk} Accuracy: {topk_acc:.4f} ({topk_hits}/{total})")

        logger.log_metrics({
            "top1_accuracy": top1_acc,
            f"top{args.topk}_accuracy": topk_acc,
            "num_queries": total,
        })

    logger.flush()
    print(f"\nResults saved to: {logger.get_run_path()}")


if __name__ == "__main__":
    main()
