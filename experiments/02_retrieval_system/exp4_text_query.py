"""Experiment 4: VLM Text-Query Evaluation with CLIP.

For each of the 52 SKU classes, creates a text prompt and encodes it
with CLIP's text encoder. Then for each val image, computes cosine
similarity with all 52 text embeddings to check if the correct class
ranks in the top-K.

Tests multiple prompt templates to find the best-performing one.

Usage:
    python exp4_text_query.py
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

# ── Paths ────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[2]
VAL_DIR = BASE / "Dataset" / "recognition" / "val"
CLASS_NAMES_FILE = BASE / "Dataset" / "recognition" / "class_names.txt"
RESULTS_DIR = BASE / "experiments" / "02_retrieval_system" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "clip-ViT-B-32"
TOP_K_VALUES = [1, 5, 10]

# Multiple prompt templates to evaluate
PROMPT_TEMPLATES = {
    "simple": "a photo of {class_name}",
    "shelf": "a photo of {class_name} on a supermarket shelf",
    "product": "a product photo of {class_name}",
    "packaging": "the packaging of {class_name}",
    "closeup": "a close-up photo of {class_name} product",
}


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
    """Run text-query evaluation."""
    print("=" * 60)
    print("Experiment 4: VLM Text-Query Evaluation")
    print("=" * 60)

    # ── Load class names ─────────────────────────────────────────────
    class_names = [
        line.strip() for line in CLASS_NAMES_FILE.read_text().strip().split("\n")
        if line.strip()
    ]
    print(f"Class names: {len(class_names)}")

    # ── Load val images ──────────────────────────────────────────────
    val_images = load_val_images(VAL_DIR)
    print(f"Val images: {len(val_images)}")

    # Verify all val classes are in class_names
    val_classes = set(cls for _, cls in val_images)
    missing = val_classes - set(class_names)
    if missing:
        print(f"WARNING: {len(missing)} val classes not in class_names.txt")
        # Add missing classes
        class_names = sorted(set(class_names) | val_classes)
        print(f"  Extended to {len(class_names)} classes")

    # ── Load model ───────────────────────────────────────────────────
    print(f"\nLoading model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME, device="cuda")
    print("  Model loaded on CUDA")

    # ── Pre-encode all val images ────────────────────────────────────
    print("\nEncoding val images...")
    val_imgs_pil = [
        Image.open(p).convert("RGB") for p, _ in val_images
    ]
    val_embs = model.encode(
        val_imgs_pil, convert_to_numpy=True,
        show_progress_bar=True, batch_size=32
    )
    val_embs = val_embs / np.linalg.norm(
        val_embs, axis=-1, keepdims=True
    )
    val_embs = val_embs.astype(np.float32)
    print(f"  Val embeddings shape: {val_embs.shape}")

    # Free PIL images
    del val_imgs_pil

    # ── Evaluate each prompt template ────────────────────────────────
    all_template_results: dict[str, dict] = {}

    for template_name, template in PROMPT_TEMPLATES.items():
        print(f"\n--- Template: '{template_name}' ---")
        print(f"    Example: '{template.format(class_name=class_names[0])}'")
        t0 = time.time()

        # Encode text prompts for all classes
        text_prompts = [
            template.format(class_name=cn) for cn in class_names
        ]
        text_embs = model.encode(
            text_prompts, convert_to_numpy=True,
            show_progress_bar=False
        )
        text_embs = text_embs / np.linalg.norm(
            text_embs, axis=-1, keepdims=True
        )
        text_embs = text_embs.astype(np.float32)

        # Create class name to index mapping
        class_to_idx = {cn: i for i, cn in enumerate(class_names)}

        # Compute similarity matrix: (num_val, num_classes)
        sim_matrix = val_embs @ text_embs.T

        # Evaluate
        overall_hits = {k: 0 for k in TOP_K_VALUES}
        per_class_hits: dict[str, dict[int, int]] = defaultdict(
            lambda: {k: 0 for k in TOP_K_VALUES}
        )
        per_class_total: dict[str, int] = defaultdict(int)
        total = 0

        for i, (_, gt_class) in enumerate(val_images):
            gt_idx = class_to_idx.get(gt_class)
            if gt_idx is None:
                continue

            sims = sim_matrix[i]
            ranked_indices = np.argsort(-sims)

            per_class_total[gt_class] += 1
            total += 1

            for k in TOP_K_VALUES:
                if gt_idx in ranked_indices[:k]:
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

        # Print summary
        for k in TOP_K_VALUES:
            print(
                f"  Top-{k:>2}: {overall_acc[k]:.4f} "
                f"({overall_hits[k]}/{total})"
            )
        print(f"  Time: {elapsed:.1f}s")

        all_template_results[template_name] = {
            "template": template,
            "num_queries": total,
            "overall_accuracy": {
                f"top{k}": round(overall_acc[k], 4) for k in TOP_K_VALUES
            },
            "per_class_accuracy": per_class_acc,
            "elapsed_seconds": round(elapsed, 1),
        }

    # ── Find best template ───────────────────────────────────────────
    best_template = max(
        all_template_results.items(),
        key=lambda x: x[1]["overall_accuracy"]["top1"]
    )
    print(f"\n{'=' * 60}")
    print(f"Best template: '{best_template[0]}' "
          f"(Top-1: {best_template[1]['overall_accuracy']['top1']:.4f})")

    # ── Save results ─────────────────────────────────────────────────
    output = {
        "experiment": "text_query_evaluation",
        "model": MODEL_NAME,
        "num_classes": len(class_names),
        "num_val_queries": len(val_images),
        "top_k_values": TOP_K_VALUES,
        "prompt_templates": PROMPT_TEMPLATES,
        "best_template": best_template[0],
        "results_by_template": all_template_results,
    }

    out_path = RESULTS_DIR / "exp4_text_query.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")

    # Clean up
    del model
    import torch
    torch.cuda.empty_cache()
    print("GPU memory freed.")


if __name__ == "__main__":
    main()
