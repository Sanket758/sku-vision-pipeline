#!/usr/bin/env python3
"""
Pre-compute retrieval demo results from the SKU registry.

Loads the 148-SKU registry, computes centroid embeddings per SKU,
selects 20 diverse query exemplars, computes top-5 retrieval results
for three methods (DINOv3-only, MobileNetV2-only, DINOv3+MobileNetV2 Hybrid),
and writes the results as a JS module for the demo webapp.

Usage: python scripts/run_retrieval.py
Output: ../src/retrieval_results.js
"""

import json
import os
import shutil
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DEMO_UI_DIR = SCRIPT_DIR.parent
SRC_DIR = DEMO_UI_DIR / "src"
PUBLIC_DIR = DEMO_UI_DIR / "public"

REGISTRY_PATH = Path(
    "/home/sanket758/Education/BSBI/Masters-Thesis"
    "/experiments/curated_pipeline/sku_registry.json"
)
EXEMPLAR_PUBLIC_DIR = PUBLIC_DIR / "demo" / "exemplars"
OUTPUT_JS_PATH = SRC_DIR / "retrieval_results.js"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_QUERIES = 20
TOP_K = 5
HYBRID_ALPHA = 0.55  # DINOv3 weight in hybrid score
HYBRID_BETA = 0.45   # MobileNetV2 weight in hybrid score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def l2_normalise(v: np.ndarray) -> np.ndarray:
    """L2-normalise a vector (no-op if zero-norm)."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def centroid_from_embeddings(
    embeddings: list | np.ndarray,
) -> np.ndarray:
    """Mean of embedding list → L2-normalised centroid."""
    mean_emb = np.mean(embeddings, axis=0)
    return l2_normalise(mean_emb)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two unit vectors (equivalent to dot product)."""
    return float(np.dot(a, b))


def make_result_entry(top_k: list[tuple]) -> list[dict]:
    """Build the per-method result list (rank, sku, similarity, thumb)."""
    return [
        {
            "rank": rank,
            "sku": sku,
            "similarity": round(float(sim), 3),
            "thumb": f"demo/skus/{sku}/thumb.jpg",
        }
        for rank, (sku, sim) in enumerate(top_k, 1)
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("Retrieval Demo Data Generator")
    print("=" * 60)

    # ---- [1/5] Load SKU registry -----------------------------------------
    print(f"\n[1/5] Loading SKU registry from: {REGISTRY_PATH}")
    with open(REGISTRY_PATH) as f:
        registry: dict = json.load(f)
    print(f"  Loaded {len(registry)} SKUs")

    # ---- [2/5] Compute centroids -----------------------------------------
    print("\n[2/5] Computing centroid embeddings for each SKU…")
    centroids: dict[str, dict[str, np.ndarray]] = {}
    exemplar_counts: dict[str, int] = {}

    for sku, entry in registry.items():
        dino_embs = np.array(
            entry["exemplar_embeddings"]["dino"], dtype=np.float32
        )
        mnet_embs = np.array(
            entry["exemplar_embeddings"]["mnet"], dtype=np.float32
        )

        centroids[sku] = {
            "dino": centroid_from_embeddings(dino_embs),
            "mnet": centroid_from_embeddings(mnet_embs),
        }
        exemplar_counts[sku] = len(entry["exemplars"])

    assert len(centroids) == len(registry)
    print(f"  Computed centroids for {len(centroids)} SKUs")

    # ---- [3/5] Select 20 diverse query SKUs ------------------------------
    print(f"\n[3/5] Selecting {NUM_QUERIES} diverse query SKUs…")

    # Sort by exemplar count (ascending) so we span the range
    sorted_skus = sorted(
        exemplar_counts.keys(), key=lambda s: exemplar_counts[s]
    )
    n = len(sorted_skus)

    # Evenly-spaced indices across the sorted list
    query_skus = [
        sorted_skus[int(i * (n - 1) / (NUM_QUERIES - 1))]
        for i in range(NUM_QUERIES)
    ]

    print(f"  Selected {len(query_skus)} SKUs (exemplar count range: "
          f"{min(exemplar_counts[s] for s in query_skus)}–"
          f"{max(exemplar_counts[s] for s in query_skus)}):")
    for sku in query_skus:
        print(f"    {sku}: {exemplar_counts[sku]} exemplars")

    # ---- [4/5] Compute top-K results for each query ----------------------
    print(f"\n[4/5] Computing top-{TOP_K} retrieval results…")

    os.makedirs(EXEMPLAR_PUBLIC_DIR, exist_ok=True)
    queries: list[dict] = []

    for query_sku in query_skus:
        entry = registry[query_sku]

        # First exemplar → query embedding
        query_dino = np.array(
            entry["exemplar_embeddings"]["dino"][0], dtype=np.float32
        )
        query_mnet = np.array(
            entry["exemplar_embeddings"]["mnet"][0], dtype=np.float32
        )

        # Score against all 148 SKU centroids
        dino_scores: dict[str, float] = {}
        mnet_scores: dict[str, float] = {}
        hybrid_scores: dict[str, float] = {}

        for sku in registry:
            d = cosine_similarity(query_dino, centroids[sku]["dino"])
            m = cosine_similarity(query_mnet, centroids[sku]["mnet"])
            h = HYBRID_ALPHA * d + HYBRID_BETA * m
            dino_scores[sku] = d
            mnet_scores[sku] = m
            hybrid_scores[sku] = h

        # Top-K per method
        dino_topk = sorted(
            dino_scores.items(), key=lambda x: x[1], reverse=True
        )[:TOP_K]
        mnet_topk = sorted(
            mnet_scores.items(), key=lambda x: x[1], reverse=True
        )[:TOP_K]
        hybrid_topk = sorted(
            hybrid_scores.items(), key=lambda x: x[1], reverse=True
        )[:TOP_K]

        # Copy query exemplar image to public dir
        first_exemplar_path = entry["exemplars"][0]
        sku_exemplar_dest = EXEMPLAR_PUBLIC_DIR / query_sku
        os.makedirs(sku_exemplar_dest, exist_ok=True)
        dest_path = sku_exemplar_dest / "crop_000.jpg"

        if os.path.isfile(first_exemplar_path):
            shutil.copy2(first_exemplar_path, dest_path)
        else:
            # Fallback: master reference thumbnail
            thumb_fallback = (
                PUBLIC_DIR / "demo" / "skus" / query_sku / "thumb.jpg"
            )
            if thumb_fallback.is_file():
                shutil.copy2(str(thumb_fallback), dest_path)
                print(f"  ⚠ Exemplar not found for {query_sku}, "
                      f"using thumb fallback")
            else:
                print(f"  ✗ Neither exemplar nor thumb found for {query_sku}")

        query_entry = {
            "query_sku": query_sku,
            "query_image": f"demo/exemplars/{query_sku}/crop_000.jpg",
            "query_thumb": f"demo/skus/{query_sku}/thumb.jpg",
            "results": {
                "dinov3": make_result_entry(dino_topk),
                "mobilenetv2": make_result_entry(mnet_topk),
                "hybrid": make_result_entry(hybrid_topk),
            },
        }
        queries.append(query_entry)

        # Progress line
        print(
            f"  {query_sku} ({exemplar_counts[query_sku]} ex.)  |  "
            f"DINO: {dino_topk[0][0]} {dino_topk[0][1]:.3f}  |  "
            f"MNet: {mnet_topk[0][0]} {mnet_topk[0][1]:.3f}  |  "
            f"Hybrid: {hybrid_topk[0][0]} {hybrid_topk[0][1]:.3f}"
        )

    # ---- [5/5] Write output JS module ------------------------------------
    print(f"\n[5/5] Writing {OUTPUT_JS_PATH.name}…")

    retrieval_data = {"queries": queries}
    js_content = (
        "const retrievalData = "
        + json.dumps(retrieval_data, indent=2)
        + ";\nexport default retrievalData;\n"
    )

    with open(OUTPUT_JS_PATH, "w") as f:
        f.write(js_content)

    print(f"  Written to: {OUTPUT_JS_PATH}")

    # Summary
    n_exemplar_copied = len(
        list(EXEMPLAR_PUBLIC_DIR.glob("*/crop_000.jpg"))
    )
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Queries generated:   {len(queries)}")
    print(f"  SKUs searched:       {len(registry)}")
    print(f"  Exemplar images:     {n_exemplar_copied} (in {EXEMPLAR_PUBLIC_DIR})")
    print(f"  JS output:           {OUTPUT_JS_PATH}")


if __name__ == "__main__":
    main()
