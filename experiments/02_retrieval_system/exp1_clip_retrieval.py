"""Experiment 1: CLIP ViT-B/32 Retrieval Accuracy.

Evaluates retrieval accuracy by querying each val image against the full
FAISS index (32,670 DETR crops). A retrieval is "correct" if any of the
top-K results belong to the same class (matching via filename→class map
built from the labeled recognition dataset).

The FAISS index contains embeddings from sentence-transformers/clip-ViT-B-32.

Usage:
    python exp1_clip_retrieval.py
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

# ── Paths ────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[2]
INDEX_PATH = BASE / ".faiss_index" / "products.index"
PATHS_FILE = BASE / ".faiss_index" / "products.paths.txt"
TRAIN_DIR = BASE / "Dataset" / "recognition" / "train"
VAL_DIR = BASE / "Dataset" / "recognition" / "val"
RESULTS_DIR = BASE / "experiments" / "02_retrieval_system" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "clip-ViT-B-32"
TOP_K_VALUES = [1, 5, 10]
MAX_K = max(TOP_K_VALUES)


def build_filename_to_class(
    train_dir: Path, val_dir: Path
) -> dict[str, str]:
    """Build a mapping from image filename to class name."""
    fname_to_class: dict[str, str] = {}
    for split_dir in [train_dir, val_dir]:
        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            for img in cls_dir.iterdir():
                if img.suffix.lower() in {".jpg", ".png", ".jpeg"}:
                    fname_to_class[img.name] = cls_dir.name
    return fname_to_class


def load_val_images(val_dir: Path) -> list[tuple[Path, str]]:
    """Load val image paths with their class labels."""
    images: list[tuple[Path, str]] = []
    for cls_dir in sorted(val_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        for img in sorted(cls_dir.iterdir()):
            if img.suffix.lower() in {".jpg", ".png", ".jpeg"}:
                images.append((img, cls_dir.name))
    return images


def main() -> None:
    """Run CLIP retrieval evaluation."""
    print("=" * 60)
    print("Experiment 1: CLIP ViT-B/32 Retrieval Accuracy")
    print("=" * 60)

    # ── Load FAISS index and path mapping ────────────────────────────
    print(f"\nLoading FAISS index from {INDEX_PATH}...")
    index = faiss.read_index(str(INDEX_PATH))
    print(f"  Index vectors: {index.ntotal}")

    faiss_paths = PATHS_FILE.read_text().strip().split("\n")
    print(f"  Path mappings: {len(faiss_paths)}")

    # ── Build filename → class mapping ───────────────────────────────
    fname_to_class = build_filename_to_class(TRAIN_DIR, VAL_DIR)
    print(f"  Labeled filenames: {len(fname_to_class)}")
    print(f"  Unique classes: {len(set(fname_to_class.values()))}")

    labeled_in_index = sum(1 for p in faiss_paths if p in fname_to_class)
    print(f"  Labeled entries in index: {labeled_in_index}/{len(faiss_paths)}")

    # ── Load model ───────────────────────────────────────────────────
    print(f"\nLoading model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME, device="cuda")
    print("  Model loaded on CUDA")

    # ── Load val images ──────────────────────────────────────────────
    val_images = load_val_images(VAL_DIR)
    print(f"\nVal images: {len(val_images)}")

    # ── Evaluate retrieval ───────────────────────────────────────────
    print(f"\nRunning retrieval evaluation (Top-K = {TOP_K_VALUES})...")
    t0 = time.time()

    # Track per-class hits
    per_class_hits: dict[str, dict[int, int]] = defaultdict(
        lambda: {k: 0 for k in TOP_K_VALUES}
    )
    per_class_total: dict[str, int] = defaultdict(int)
    overall_hits = {k: 0 for k in TOP_K_VALUES}
    total = 0

    # Track self-retrieval (query image appearing in results)
    self_retrieval_count = 0

    for i, (img_path, gt_class) in enumerate(val_images):
        # Encode query image
        img = Image.open(img_path).convert("RGB")
        query_emb = model.encode(
            [img], convert_to_numpy=True, show_progress_bar=False
        )
        query_emb = query_emb / np.linalg.norm(
            query_emb, axis=-1, keepdims=True
        )
        query_emb = query_emb.astype(np.float32)

        # Search FAISS index
        scores, indices = index.search(query_emb, MAX_K + 1)

        # Filter results, excluding self-retrieval
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(faiss_paths):
                continue
            retrieved_fname = faiss_paths[idx]
            if retrieved_fname == img_path.name:
                self_retrieval_count += 1
                continue
            results.append((score, retrieved_fname))
            if len(results) >= MAX_K:
                break

        # Check hits at each K
        per_class_total[gt_class] += 1
        total += 1

        for k in TOP_K_VALUES:
            found = False
            for _, retrieved_fname in results[:k]:
                retrieved_class = fname_to_class.get(retrieved_fname)
                if retrieved_class == gt_class:
                    found = True
                    break
            if found:
                overall_hits[k] += 1
                per_class_hits[gt_class][k] += 1

        if (i + 1) % 25 == 0 or (i + 1) == len(val_images):
            elapsed = time.time() - t0
            print(f"  [{i + 1}/{len(val_images)}] {elapsed:.1f}s")

    elapsed_total = time.time() - t0

    # ── Compute metrics ──────────────────────────────────────────────
    overall_acc = {k: overall_hits[k] / total for k in TOP_K_VALUES}

    per_class_acc: dict[str, dict[str, float]] = {}
    for cls_name in sorted(per_class_total.keys()):
        n = per_class_total[cls_name]
        per_class_acc[cls_name] = {
            f"top{k}_acc": per_class_hits[cls_name][k] / n
            for k in TOP_K_VALUES
        }
        per_class_acc[cls_name]["num_val_images"] = n

    # ── Print results ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Total queries: {total}")
    print(f"  Self-retrievals excluded: {self_retrieval_count}")
    print(f"  Time: {elapsed_total:.1f}s")
    print()
    for k in TOP_K_VALUES:
        print(
            f"  Top-{k:>2} Accuracy: {overall_acc[k]:.4f} "
            f"({overall_hits[k]}/{total})"
        )

    print(f"\nPer-class Top-1 Accuracy:")
    for cls_name in sorted(per_class_acc.keys()):
        info = per_class_acc[cls_name]
        n = int(info["num_val_images"])
        t1 = info["top1_acc"]
        print(f"  {cls_name[:50]:<52} {t1:.2f} (n={n})")

    # ── Save results ─────────────────────────────────────────────────
    results = {
        "experiment": "clip_retrieval_accuracy",
        "model": MODEL_NAME,
        "index_path": str(INDEX_PATH),
        "index_vectors": index.ntotal,
        "labeled_in_index": labeled_in_index,
        "num_val_queries": total,
        "num_classes": len(set(per_class_total.keys())),
        "self_retrievals_excluded": self_retrieval_count,
        "top_k_values": TOP_K_VALUES,
        "overall_accuracy": {
            f"top{k}": round(overall_acc[k], 4) for k in TOP_K_VALUES
        },
        "per_class_accuracy": per_class_acc,
        "elapsed_seconds": round(elapsed_total, 1),
    }

    out_path = RESULTS_DIR / "exp1_clip_retrieval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")

    # Clean up GPU memory
    del model
    import torch
    torch.cuda.empty_cache()
    print("GPU memory freed.")


if __name__ == "__main__":
    main()
