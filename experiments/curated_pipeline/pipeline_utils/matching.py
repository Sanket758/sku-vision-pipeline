"""SKU registry and cross-image matching for the curated pipeline.

Maintains a persistent SKU database (sku_registry.json) that grows across
images.  Each new crop is matched against all known SKUs using a hybrid
DINOv3 + MobileNetV2 similarity score, with optional positional bonus.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from experiments.curated_pipeline import config

logger = logging.getLogger(__name__)

# Minimum exemplars before centroid is reliable
_MIN_EXEMPLARS_FOR_CENTROID = 2


class SKURegistry:
    """Persistent SKU registry with cross-image memory.

    The registry is stored as a JSON file at ``config.REGISTRY_FILE``.
    Each SKU entry stores:
    - label, category
    - list of exemplar crop paths
    - DINOv3 + MobileNetV2 centroids (rolling average)
    - all individual exemplar embeddings (for max-over-exemplars matching)
    - source image names
    """

    _next_inst_id = 0

    def __init__(self, registry_path: str | Path | None = None) -> None:
        self._inst_id = SKURegistry._next_inst_id
        SKURegistry._next_inst_id += 1
        self._path = Path(registry_path or config.REGISTRY_FILE)
        self._data: dict[str, dict] = {}
        self._np_cache: dict[str, dict] = {}
        self._np_cache_dirty: bool = True
        self._load()
        logger.info(
            "SKURegistry[%d] created — path=%s  loaded=%d SKUs",
            self._inst_id,
            self._path,
            len(self._data),
        )

    # ── Persistence ─────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._path.exists():
            raw = self._path.read_text().strip()
            if not raw:
                logger.warning(
                    "SKURegistry[%d] _load() — Registry file %s is empty — starting fresh",
                    self._inst_id,
                    self._path,
                )
                self._data = {}
                return
            try:
                self._data = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning(
                    "SKURegistry[%d] _load() — Registry file %s is corrupt — starting fresh",
                    self._inst_id,
                    self._path,
                )
                self._data = {}
                return
            logger.info(
                "SKURegistry[%d] _load() — loaded %d SKUs from %s",
                self._inst_id,
                len(self._data),
                self._path,
            )
        else:
            logger.info(
                "SKURegistry[%d] _load() — No existing registry at %s — starting fresh",
                self._inst_id,
                self._path,
            )
            self._data = {}

        self._build_np_cache()

    def _save(self) -> None:
        """Atomically write registry to disk (write to temp, rename).

        Uses a single atomic write via temp+move to avoid the corruption
        caused by a previous double-write pattern (temp+move then direct
        dump) where an interrupted second write would truncate the file.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            suffix=".json", prefix="sku_registry_", dir=self._path.parent
        )
        try:
            with open(fd, "w") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, str(self._path))
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise

        logger.info(
            "SKURegistry[%d] _save() — data has %d keys",
            self._inst_id,
            len(self._data),
        )

    def _build_np_cache(self) -> None:
        """Pre-convert exemplar embeddings to numpy arrays for fast matching."""
        self._np_cache = {}
        for sku_id, entry in self._data.items():
            dino_raw = entry.get("exemplar_embeddings", {}).get("dino", [])
            mnet_raw = entry.get("exemplar_embeddings", {}).get("mnet", [])
            self._np_cache[sku_id] = {
                "dino_np": [np.array(e, dtype=np.float32) for e in dino_raw],
                "mnet_np": [np.array(e, dtype=np.float32) for e in mnet_raw],
                "dino_centroid_np": np.array(entry.get("dino_centroid", [0.0] * 384), dtype=np.float32),
                "mnet_centroid_np": np.array(entry.get("mnet_centroid", [0.0] * 1280), dtype=np.float32),
            }
        self._np_cache_dirty = False

    # ── SKU ID management ──────────────────────────────────────────────────

    def _next_sku_id(self) -> str:
        existing = {int(k.split("-")[1]) for k in self._data if k.startswith("SKU-")}
        n = 1
        while n in existing:
            n += 1
        return f"SKU-{n:03d}"

    # ── Match logic ────────────────────────────────────────────────────────

    def match_new_crop(
        self,
        dino_emb: np.ndarray,
        mnet_emb: np.ndarray,
        box_norm: np.ndarray | None = None,
    ) -> list[dict]:
        """Return ranked candidate SKUs for a new crop embedding.

        Parameters
        ----------
        dino_emb:
            L2-normalized DINOv3 embedding of shape ``(384,)``.
        mnet_emb:
            L2-normalized MobileNetV2 embedding of shape ``(1280,)``.
        box_norm:
            Optional normalized bounding box ``[x_c, y_c, w, h]`` for the
            positional bonus.

        Returns
        -------
        list[dict]
            Each dict has keys ``sku_id``, ``label``, ``score``, ``zone``
            (one of ``"high"``, ``"ambiguous"``, ``"none"``), sorted by
            descending score.  Empty list if the registry is empty.
        """
        if not self._data:
            return []

        candidates: list[tuple[str, float]] = []

        if self._np_cache_dirty:
            self._build_np_cache()

        for sku_id, entry in self._data.items():
            cache = self._np_cache.get(sku_id, {})
            dino_centroid = cache.get("dino_centroid_np", np.array(entry.get("dino_centroid", [0.0] * 384), dtype=np.float32))
            mnet_centroid = cache.get("mnet_centroid_np", np.array(entry.get("mnet_centroid", [0.0] * 1280), dtype=np.float32))

            # Centroid similarity
            dino_sim = float(np.dot(dino_emb, dino_centroid))
            mnet_sim = float(np.dot(mnet_emb, mnet_centroid))
            centroid_score = (
                config.DINOV3_WEIGHT * dino_sim + config.MOBILENET_WEIGHT * mnet_sim
            )

            # Max-over-exemplars similarity (using cached numpy arrays)
            dino_exemplars = cache.get("dino_np", [])
            mnet_exemplars = cache.get("mnet_np", [])

            exemplar_scores: list[float] = []
            for de, me in zip(dino_exemplars, mnet_exemplars):
                ds = float(np.dot(dino_emb, de))
                ms = float(np.dot(mnet_emb, me))
                exemplar_scores.append(config.DINOV3_WEIGHT * ds + config.MOBILENET_WEIGHT * ms)

            final_score = centroid_score
            if exemplar_scores:
                max_ex_score = max(exemplar_scores)
                final_score = max(centroid_score, max_ex_score)

            # Positional bonus
            if box_norm is not None and entry.get("exemplar_embeddings"):
                # Compare against exemplar boxes — approximate with centroid position
                pass  # positional bonus applied via row_similarity externally

            candidates.append((sku_id, final_score))

        # Sort descending by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        top = candidates[:3]

        results = []
        for sku_id, score in top:
            entry = self._data[sku_id]
            if score >= config.HIGH_CONFIDENCE:
                zone = "high"
            elif score >= config.LOW_CONFIDENCE:
                zone = "ambiguous"
            else:
                zone = "none"

            anchor = entry["exemplars"][0] if entry["exemplars"] else ""
            results.append({
                "sku_id": sku_id,
                "label": entry["label"],
                "score": round(float(score), 4),
                "dino_score": round(float(dino_sim), 4),
                "mnet_score": round(float(mnet_sim), 4),
                "n_exemplars": len(entry["exemplars"]),
                "zone": zone,
                "anchor_exemplar": anchor,
            })

        return results

    # ── Registry mutation ──────────────────────────────────────────────────

    def add_exemplar(
        self,
        sku_id: str,
        crop_path: str,
        dino_emb: np.ndarray,
        mnet_emb: np.ndarray,
        source_image: str,
    ) -> None:
        """Add a confirmed exemplar to an existing SKU.

        Updates the rolling-average centroids and appends the exemplar.
        """
        if sku_id not in self._data:
            raise KeyError(f"SKU {sku_id} not found in registry")

        entry = self._data[sku_id]
        n = len(entry["exemplars"])

        # Rolling-average centroid update
        dino_centroid = np.array(entry["dino_centroid"], dtype=np.float32)
        mnet_centroid = np.array(entry["mnet_centroid"], dtype=np.float32)

        new_dino_centroid = (dino_centroid * n + dino_emb) / (n + 1)
        new_mnet_centroid = (mnet_centroid * n + mnet_emb) / (n + 1)

        # Re-normalize
        nd = np.linalg.norm(new_dino_centroid)
        nm = np.linalg.norm(new_mnet_centroid)
        entry["dino_centroid"] = (new_dino_centroid / nd).tolist() if nd > 0 else new_dino_centroid.tolist()
        entry["mnet_centroid"] = (new_mnet_centroid / nm).tolist() if nm > 0 else new_mnet_centroid.tolist()

        entry["exemplars"].append(str(crop_path))
        entry["exemplar_embeddings"]["dino"].append(dino_emb.tolist())
        entry["exemplar_embeddings"]["mnet"].append(mnet_emb.tolist())

        if source_image not in entry["source_images"]:
            entry["source_images"].append(source_image)

        entry["updated"] = datetime.now(timezone.utc).isoformat()
        self._np_cache_dirty = True
        self._save()
        logger.info(
            "SKURegistry[%d] add_exemplar(%s) — data now has %d keys",
            self._inst_id,
            sku_id,
            len(self._data),
        )

    def create_sku(
        self,
        label: str,
        category: str,
        crop_path: str,
        dino_emb: np.ndarray,
        mnet_emb: np.ndarray,
        source_image: str,
    ) -> str:
        """Create a new SKU entry and return its ID."""
        sku_id = self._next_sku_id()

        dino_list = dino_emb.tolist()
        mnet_list = mnet_emb.tolist()
        now = datetime.now(timezone.utc).isoformat()

        self._data[sku_id] = {
            "label": label,
            "category": category,
            "exemplars": [str(crop_path)],
            "dino_centroid": dino_list,
            "mnet_centroid": mnet_list,
            "exemplar_embeddings": {
                "dino": [dino_list],
                "mnet": [mnet_list],
            },
            "source_images": [source_image],
            "created": now,
            "updated": now,
        }
        self._np_cache_dirty = True
        self._save()
        logger.info(
            "SKURegistry[%d] create_sku(%s) — data now has %d keys",
            self._inst_id,
            sku_id,
            len(self._data),
        )
        return sku_id

    def sync_exemplar_files(
        self, sku_id: str | None = None, ex_dir: str | Path | None = None,
    ) -> int:
        """Copy exemplar crop files to ``exemplars/{SKU_ID}/crop_NNN.ext``.

        Idempotent — re-running overwrites old files.  Missing crop files are
        skipped with a warning.

        Parameters
        ----------
        sku_id:
            If provided, only sync this one SKU (fast path for live review).
            If ``None``, sync all SKUs.
        ex_dir:
            Target directory (defaults to ``config.PIPELINE_DIR / "exemplars"``).

        Returns
        -------
        int
            Number of files copied.
        """
        if ex_dir is None:
            ex_dir = config.PIPELINE_DIR / "exemplars"
        ex_dir = Path(ex_dir)
        ex_dir.mkdir(parents=True, exist_ok=True)

        total_copied = 0

        entries = (
            [(sku_id, self._data[sku_id])] if sku_id
            else list(self._data.items())
        )

        for sku_id, entry in entries:
            sku_dir = ex_dir / sku_id
            sku_dir.mkdir(parents=True, exist_ok=True)

            for idx, src_path_str in enumerate(entry.get("exemplars", [])):
                src = Path(src_path_str)
                if not src.exists():
                    logger.warning(
                        "  Exemplar not found, skipping: %s", src,
                    )
                    continue

                dst = sku_dir / f"crop_{idx:03d}{src.suffix}"
                try:
                    shutil.copy2(str(src), str(dst))
                    total_copied += 1
                except shutil.SameFileError:
                    pass  # already in place
                except Exception as e:
                    logger.error("  Failed to copy %s: %s", src, e)

        logger.info(
            "sync_exemplar_files: %d files synced to %s", total_copied, ex_dir,
        )
        return total_copied

    def delete_sku(self, sku_id: str) -> None:
        """Remove a SKU from the registry."""
        if sku_id in self._data:
            del self._data[sku_id]
            self._np_cache_dirty = True
            self._save()
            logger.info("Deleted SKU %s", sku_id)

    def remove_exemplar(self, sku_id: str, crop_path: str) -> None:
        """Remove a single exemplar from a SKU, recalculate centroid, save."""
        if sku_id not in self._data:
            raise KeyError(f"SKU {sku_id} not found in registry")

        entry = self._data[sku_id]
        normalized_crop = os.path.normpath(crop_path)

        idx = None
        for i, stored_path in enumerate(entry["exemplars"]):
            if os.path.normpath(stored_path) == normalized_crop:
                idx = i
                break

        if idx is None:
            raise ValueError(f"Exemplar {crop_path} not found in SKU {sku_id}")

        entry["exemplars"].pop(idx)
        entry["exemplar_embeddings"]["dino"].pop(idx)
        entry["exemplar_embeddings"]["mnet"].pop(idx)

        remaining = len(entry["exemplars"])
        if remaining > 0:
            dino_embs = [np.array(e, dtype=np.float32) for e in entry["exemplar_embeddings"]["dino"]]
            mnet_embs = [np.array(e, dtype=np.float32) for e in entry["exemplar_embeddings"]["mnet"]]
            dino_centroid = np.mean(dino_embs, axis=0)
            mnet_centroid = np.mean(mnet_embs, axis=0)
            nd = np.linalg.norm(dino_centroid)
            nm = np.linalg.norm(mnet_centroid)
            entry["dino_centroid"] = (dino_centroid / nd).tolist() if nd > 0 else dino_centroid.tolist()
            entry["mnet_centroid"] = (mnet_centroid / nm).tolist() if nm > 0 else mnet_centroid.tolist()
        else:
            entry["dino_centroid"] = [0.0] * 384
            entry["mnet_centroid"] = [0.0] * 1280

        self._np_cache_dirty = True
        self._save()
        logger.info("  Removed exemplar %s from %s", crop_path, sku_id)

    # ── Queries ────────────────────────────────────────────────────────────

    def get_sku_info(self, sku_id: str) -> dict | None:
        """Return SKU metadata, or None if not found."""
        entry = self._data.get(sku_id)
        if entry is None:
            return None
        return {
            "sku_id": sku_id,
            "label": entry["label"],
            "category": entry["category"],
            "n_exemplars": len(entry["exemplars"]),
            "anchor_exemplar": entry["exemplars"][0] if entry["exemplars"] else None,
            "source_images": entry["source_images"],
            "created": entry["created"],
            "updated": entry["updated"],
        }

    def list_skus(self) -> list[str]:
        """Return sorted list of all SKU IDs."""
        return sorted(self._data.keys())

    def get_all_entries(self) -> dict[str, dict]:
        """Return the full registry data (read-only view)."""
        return dict(self._data)

    def __bool__(self) -> bool:
        """Always truthy regardless of _data size (prevents 'registry or SKURegistry()'
        from silently replacing an empty registry with a new empty one)."""
        return True

    def __len__(self) -> int:
        logger.info(
            "SKURegistry[%d] __len__() returning %d",
            self._inst_id,
            len(self._data),
        )
        return len(self._data)
