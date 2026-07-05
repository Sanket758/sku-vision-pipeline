"""Experiment 3: Few-Shot N-Way K-Shot Evaluation with CLIP.

Evaluates CLIP ViT-B/32 in a few-shot episodic setting:
- Sample N classes, K support images per class from train set
- 1 query image per class from val set
- Classify by nearest support image (cosine similarity)
- Test N=5,10,20 and K=1,3,5 with 100 episodes each

Usage:
    python exp3_fewshot.py
"""

import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

# ── Paths ────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[2]
TRAIN_DIR = BASE / "Dataset" / "recognition" / "train"
VAL_DIR = BASE / "Dataset" / "recognition" / "val"
RESULTS_DIR = BASE / "experiments" / "02_retrieval_system" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "clip-ViT-B-32"
N_VALUES = [5, 10, 20]
K_VALUES = [1, 3, 5]
NUM_EPISODES = 100
SEED = 42


def load_class_images(
    split_dir: Path,
) -> dict[str, list[Path]]:
    """Load images grouped by class."""
    class_images: dict[str, list[Path]] = {}
    for cls_dir in sorted(split_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        imgs = sorted(
            p for p in cls_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".png", ".jpeg"}
        )
        if imgs:
            class_images[cls_dir.name] = imgs
    return class_images


def encode_images(
    model: SentenceTransformer, paths: list[Path]
) -> np.ndarray:
    """Encode a batch of images, return L2-normalized embeddings."""
    imgs = [Image.open(p).convert("RGB") for p in paths]
    embs = model.encode(imgs, convert_to_numpy=True, show_progress_bar=False)
    embs = embs / np.linalg.norm(embs, axis=-1, keepdims=True)
    return embs.astype(np.float32)


def main() -> None:
    """Run few-shot N-way K-shot evaluation."""
    print("=" * 60)
    print("Experiment 3: Few-Shot N-Way K-Shot Evaluation")
    print("=" * 60)

    # ── Load dataset ─────────────────────────────────────────────────
    train_classes = load_class_images(TRAIN_DIR)
    val_classes = load_class_images(VAL_DIR)
    print(f"Train: {len(train_classes)} classes, "
          f"{sum(len(v) for v in train_classes.values())} images")
    print(f"Val: {len(val_classes)} classes, "
          f"{sum(len(v) for v in val_classes.values())} images")

    # Filter to classes that have both train and val images,
    # and have enough train images for max K
    eligible_classes = []
    for cls_name in sorted(train_classes.keys()):
        if cls_name not in val_classes:
            continue
        if len(train_classes[cls_name]) >= max(K_VALUES):
            eligible_classes.append(cls_name)

    print(f"Eligible classes (≥{max(K_VALUES)} train, has val): "
          f"{len(eligible_classes)}")

    if len(eligible_classes) < max(N_VALUES):
        print(f"WARNING: Only {len(eligible_classes)} eligible classes, "
              f"but max N={max(N_VALUES)}. Adjusting N_VALUES.")
        N_VALUES_ACTUAL = [n for n in N_VALUES if n <= len(eligible_classes)]
    else:
        N_VALUES_ACTUAL = N_VALUES

    # ── Load model ───────────────────────────────────────────────────
    print(f"\nLoading model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME, device="cuda")
    print("  Model loaded on CUDA")

    # ── Pre-encode all eligible images ───────────────────────────────
    # This is much more efficient than encoding per-episode
    print("\nPre-encoding all eligible images...")
    train_embeddings: dict[str, np.ndarray] = {}
    val_embeddings: dict[str, np.ndarray] = {}

    for i, cls_name in enumerate(eligible_classes):
        train_embs = encode_images(model, train_classes[cls_name])
        train_embeddings[cls_name] = train_embs

        val_embs = encode_images(model, val_classes[cls_name])
        val_embeddings[cls_name] = val_embs

        if (i + 1) % 10 == 0 or (i + 1) == len(eligible_classes):
            print(f"  [{i + 1}/{len(eligible_classes)}] classes encoded")

    # ── Run episodes ─────────────────────────────────────────────────
    rng = random.Random(SEED)
    np_rng = np.random.RandomState(SEED)

    all_results: dict[str, dict] = {}

    for n_way in N_VALUES_ACTUAL:
        for k_shot in K_VALUES:
            config_key = f"N{n_way}_K{k_shot}"
            print(f"\n--- {config_key}: {NUM_EPISODES} episodes ---")
            t0 = time.time()

            top1_accs = []
            top5_accs = []

            for ep in range(NUM_EPISODES):
                # Sample N classes
                sampled_classes = rng.sample(eligible_classes, n_way)

                # Build support set: K images per class
                support_embs_list = []
                support_labels = []
                for cls_name in sampled_classes:
                    n_train = len(train_embeddings[cls_name])
                    indices = np_rng.choice(
                        n_train, size=min(k_shot, n_train), replace=False
                    )
                    for idx in indices:
                        support_embs_list.append(
                            train_embeddings[cls_name][idx]
                        )
                        support_labels.append(cls_name)

                support_embs = np.stack(support_embs_list)  # (N*K, D)

                # Build query set: 1 image per class from val
                correct_top1 = 0
                correct_top5 = 0
                n_queries = 0

                for cls_name in sampled_classes:
                    n_val = len(val_embeddings[cls_name])
                    q_idx = np_rng.randint(0, n_val)
                    query_emb = val_embeddings[cls_name][q_idx]  # (D,)

                    # Cosine similarity (embeddings are L2-normalized)
                    sims = query_emb @ support_embs.T  # (N*K,)

                    # Rank by similarity
                    ranked_indices = np.argsort(-sims)
                    ranked_labels = [
                        support_labels[j] for j in ranked_indices
                    ]

                    # Top-1 accuracy
                    if ranked_labels[0] == cls_name:
                        correct_top1 += 1

                    # Top-5 accuracy (check if correct class
                    # appears in top-5 unique predictions)
                    seen_classes = []
                    for lbl in ranked_labels:
                        if lbl not in seen_classes:
                            seen_classes.append(lbl)
                        if len(seen_classes) >= 5:
                            break
                    if cls_name in seen_classes:
                        correct_top5 += 1

                    n_queries += 1

                top1_accs.append(correct_top1 / n_queries)
                top5_accs.append(correct_top5 / n_queries)

            elapsed = time.time() - t0
            mean_top1 = np.mean(top1_accs)
            std_top1 = np.std(top1_accs)
            mean_top5 = np.mean(top5_accs)
            std_top5 = np.std(top5_accs)

            print(f"  Top-1: {mean_top1:.4f} ± {std_top1:.4f}")
            print(f"  Top-5: {mean_top5:.4f} ± {std_top5:.4f}")
            print(f"  Time: {elapsed:.1f}s")

            all_results[config_key] = {
                "n_way": n_way,
                "k_shot": k_shot,
                "num_episodes": NUM_EPISODES,
                "top1_mean": round(float(mean_top1), 4),
                "top1_std": round(float(std_top1), 4),
                "top5_mean": round(float(mean_top5), 4),
                "top5_std": round(float(std_top5), 4),
                "elapsed_seconds": round(elapsed, 1),
            }

    # ── Save results ─────────────────────────────────────────────────
    output = {
        "experiment": "fewshot_nway_kshot",
        "model": MODEL_NAME,
        "n_values": N_VALUES_ACTUAL,
        "k_values": K_VALUES,
        "num_episodes": NUM_EPISODES,
        "seed": SEED,
        "num_eligible_classes": len(eligible_classes),
        "eligible_classes": eligible_classes,
        "results": all_results,
    }

    out_path = RESULTS_DIR / "exp3_fewshot.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")

    # Clean up GPU memory
    del model
    import torch
    torch.cuda.empty_cache()
    print("GPU memory freed.")


if __name__ == "__main__":
    main()
