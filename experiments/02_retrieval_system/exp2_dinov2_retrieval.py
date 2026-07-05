"""Experiment 2: DINOv2 ViT-B/14 Retrieval Accuracy.

Uses labeled-only approach (300 train as database, 76 val as queries)
since rebuilding the full 32K FAISS index with DINOv2 would be expensive.

This is noted as a limitation: the DINOv2 evaluation uses a smaller,
labeled-only database, while CLIP is evaluated against the full 32K index.

Usage:
    python exp2_dinov2_retrieval.py
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import torch
from PIL import Image

# ── Paths ────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[2]
TRAIN_DIR = BASE / "Dataset" / "recognition" / "train"
VAL_DIR = BASE / "Dataset" / "recognition" / "val"
RESULTS_DIR = BASE / "experiments" / "02_retrieval_system" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "facebook/dinov2-base"
TOP_K_VALUES = [1, 5, 10]
MAX_K = max(TOP_K_VALUES)
BATCH_SIZE = 16


def load_class_images(
    split_dir: Path,
) -> list[tuple[Path, str]]:
    """Load image paths with their class labels."""
    images: list[tuple[Path, str]] = []
    for cls_dir in sorted(split_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        for img in sorted(cls_dir.iterdir()):
            if img.suffix.lower() in {".jpg", ".png", ".jpeg"}:
                images.append((img, cls_dir.name))
    return images


def encode_images_dinov2(
    model: torch.nn.Module,
    processor: "AutoImageProcessor",
    image_paths: list[Path],
    device: str,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Encode images with DINOv2, return L2-normalized embeddings."""
    all_embs = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        images = [
            Image.open(p).convert("RGB") for p in batch_paths
        ]
        inputs = processor(images=images, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Use CLS token embedding
            embs = outputs.last_hidden_state[:, 0, :]

        embs = embs.cpu().numpy()
        all_embs.append(embs)

    all_embs_np = np.concatenate(all_embs, axis=0)
    # L2 normalize
    norms = np.linalg.norm(all_embs_np, axis=-1, keepdims=True)
    all_embs_np = all_embs_np / norms
    return all_embs_np.astype(np.float32)


def main() -> None:
    """Run DINOv2 retrieval evaluation using labeled-only database."""
    print("=" * 60)
    print("Experiment 2: DINOv2 ViT-B/14 Retrieval Accuracy")
    print("  (Labeled-only: train as database, val as queries)")
    print("=" * 60)

    # ── Load dataset ─────────────────────────────────────────────────
    train_images = load_class_images(TRAIN_DIR)
    val_images = load_class_images(VAL_DIR)
    print(f"Train (database): {len(train_images)} images")
    print(f"Val (queries): {len(val_images)} images")
    print(f"Classes: {len(set(c for _, c in train_images))}")

    # ── Load DINOv2 model ────────────────────────────────────────────
    from transformers import AutoImageProcessor, AutoModel

    print(f"\nLoading DINOv2 model '{MODEL_NAME}'...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    print(f"  Model loaded on {device}")

    # ── Encode train (database) images ───────────────────────────────
    print("\nEncoding train images (database)...")
    t0 = time.time()
    train_paths = [p for p, _ in train_images]
    train_labels = [c for _, c in train_images]
    train_embs = encode_images_dinov2(
        model, processor, train_paths, device
    )
    print(f"  Train embeddings: {train_embs.shape} ({time.time() - t0:.1f}s)")

    # ── Build FAISS index from train embeddings ──────────────────────
    dim = train_embs.shape[1]
    print(f"\nBuilding FAISS index (dim={dim})...")
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine on L2-normed)
    index.add(train_embs)
    print(f"  Index: {index.ntotal} vectors")

    # ── Encode val (query) images ────────────────────────────────────
    print("\nEncoding val images (queries)...")
    t0 = time.time()
    val_paths = [p for p, _ in val_images]
    val_labels = [c for _, c in val_images]
    val_embs = encode_images_dinov2(
        model, processor, val_paths, device
    )
    print(f"  Val embeddings: {val_embs.shape} ({time.time() - t0:.1f}s)")

    # ── Evaluate retrieval ───────────────────────────────────────────
    print(f"\nRunning retrieval evaluation (Top-K = {TOP_K_VALUES})...")
    t0 = time.time()

    scores, indices = index.search(val_embs, MAX_K)

    overall_hits = {k: 0 for k in TOP_K_VALUES}
    per_class_hits: dict[str, dict[int, int]] = defaultdict(
        lambda: {k: 0 for k in TOP_K_VALUES}
    )
    per_class_total: dict[str, int] = defaultdict(int)
    total = 0

    for i, gt_class in enumerate(val_labels):
        per_class_total[gt_class] += 1
        total += 1

        for k in TOP_K_VALUES:
            found = False
            for j in range(min(k, len(indices[i]))):
                idx = indices[i][j]
                if idx < 0 or idx >= len(train_labels):
                    continue
                if train_labels[idx] == gt_class:
                    found = True
                    break
            if found:
                overall_hits[k] += 1
                per_class_hits[gt_class][k] += 1

    elapsed = time.time() - t0
    overall_acc = {k: overall_hits[k] / total for k in TOP_K_VALUES}

    # Per-class accuracy
    per_class_acc: dict[str, dict[str, float]] = {}
    for cls_name in sorted(per_class_total.keys()):
        n = per_class_total[cls_name]
        per_class_acc[cls_name] = {
            f"top{k}_acc": per_class_hits[cls_name][k] / n
            for k in TOP_K_VALUES
        }
        per_class_acc[cls_name]["num_val_images"] = n

    # ── Print results ────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print("=" * 60)
    print(f"  Total queries: {total}")
    print(f"  Database size: {index.ntotal}")
    print(f"  Embedding dim: {dim}")
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

    # ── Also run CLIP on same labeled-only setup for fair comparison ──
    print(f"\n{'=' * 60}")
    print("CLIP Labeled-Only Baseline (for fair comparison)")
    print("=" * 60)

    # Free DINOv2 from GPU
    del model
    torch.cuda.empty_cache()

    from sentence_transformers import SentenceTransformer
    clip_model = SentenceTransformer("clip-ViT-B-32", device="cuda")

    # Encode train with CLIP
    print("Encoding train images with CLIP...")
    train_imgs_pil = [Image.open(p).convert("RGB") for p in train_paths]
    clip_train_embs = clip_model.encode(
        train_imgs_pil, convert_to_numpy=True,
        show_progress_bar=True, batch_size=32
    )
    clip_train_embs = clip_train_embs / np.linalg.norm(
        clip_train_embs, axis=-1, keepdims=True
    )
    clip_train_embs = clip_train_embs.astype(np.float32)
    del train_imgs_pil

    # Encode val with CLIP
    print("Encoding val images with CLIP...")
    val_imgs_pil = [Image.open(p).convert("RGB") for p in val_paths]
    clip_val_embs = clip_model.encode(
        val_imgs_pil, convert_to_numpy=True,
        show_progress_bar=True, batch_size=32
    )
    clip_val_embs = clip_val_embs / np.linalg.norm(
        clip_val_embs, axis=-1, keepdims=True
    )
    clip_val_embs = clip_val_embs.astype(np.float32)
    del val_imgs_pil

    # Build CLIP index
    clip_dim = clip_train_embs.shape[1]
    clip_index = faiss.IndexFlatIP(clip_dim)
    clip_index.add(clip_train_embs)

    # Evaluate CLIP
    clip_scores, clip_indices = clip_index.search(clip_val_embs, MAX_K)

    clip_overall_hits = {k: 0 for k in TOP_K_VALUES}
    for i, gt_class in enumerate(val_labels):
        for k in TOP_K_VALUES:
            found = False
            for j in range(min(k, len(clip_indices[i]))):
                idx = clip_indices[i][j]
                if idx >= 0 and idx < len(train_labels):
                    if train_labels[idx] == gt_class:
                        found = True
                        break
            if found:
                clip_overall_hits[k] += 1

    clip_overall_acc = {
        k: clip_overall_hits[k] / total for k in TOP_K_VALUES
    }

    print("CLIP Labeled-Only Results:")
    for k in TOP_K_VALUES:
        print(
            f"  Top-{k:>2}: {clip_overall_acc[k]:.4f} "
            f"({clip_overall_hits[k]}/{total})"
        )

    # ── Save results ─────────────────────────────────────────────────
    output = {
        "experiment": "dinov2_retrieval_accuracy",
        "model": MODEL_NAME,
        "evaluation_mode": "labeled_only",
        "limitation": (
            "DINOv2 evaluated on labeled-only database (304 train images) "
            "while CLIP Experiment 1 uses the full 32K FAISS index. "
            "This section also includes a CLIP labeled-only baseline "
            "for fair apples-to-apples comparison."
        ),
        "database_size": len(train_images),
        "num_val_queries": len(val_images),
        "num_classes": len(set(train_labels)),
        "embedding_dim": dim,
        "top_k_values": TOP_K_VALUES,
        "dinov2_overall_accuracy": {
            f"top{k}": round(overall_acc[k], 4) for k in TOP_K_VALUES
        },
        "dinov2_per_class_accuracy": per_class_acc,
        "clip_labeled_only_accuracy": {
            f"top{k}": round(clip_overall_acc[k], 4) for k in TOP_K_VALUES
        },
    }

    out_path = RESULTS_DIR / "exp2_dinov2_retrieval.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")

    # Clean up
    del clip_model
    torch.cuda.empty_cache()
    print("GPU memory freed.")


if __name__ == "__main__":
    main()
