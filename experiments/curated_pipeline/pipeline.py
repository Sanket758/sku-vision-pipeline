#!/usr/bin/env python3
"""CLI pipeline: one shelf image → detect → embed → match → review → export.

Usage
-----
    # Process one image
    python pipeline.py --image data/input/aldi_001.jpg

    # Process + export after review
    python pipeline.py --image data/input/aldi_001.jpg --export yolo,retrieval

    # Dry-run (detect + embed only, skip review)
    python pipeline.py --image data/input/aldi_001.jpg --dry-run

    # Resume most recent session (skip steps 1-5, go to review)
    python pipeline.py --resume

    # List all SKUs in registry
    python pipeline.py --list-skus

    # Get info about one SKU
    python pipeline.py --sku-info SKU-001
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from PIL import Image

# Ensure project root + pipeline dir are on sys.path for all imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PIPELINE_DIR = Path(__file__).resolve().parent
for p in [str(_PROJECT_ROOT), str(_PIPELINE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import config
from pipeline_utils.detection import Detector
from pipeline_utils.embeddings import DINOv3Extractor, MobileNetV2Extractor
from pipeline_utils.positional import normalize_boxes, group_shelf_rows, row_similarity
from pipeline_utils.matching import SKURegistry
from pipeline_utils.export import build_class_mapping, export_yolo, export_retrieval_folders
import yolo_to_labelme as y2l
from review_server import ReviewSession, launch_server

logger = logging.getLogger("pipeline")

# Sentinel to detect whether --resume was explicitly passed
_RESUME_UNSET = "___RESUME_UNSET___"


# ── Session persistence ─────────────────────────────────────────────────────


def _save_session(
    results: list[dict],
    dino_embs: list,
    mnet_embs: list,
    crops_dir: Path,
    image_name: str,
    image_path: str = "",
) -> Path:
    """Persist pipeline state after step_match so --resume can skip steps 1-5.

    Returns the path to the saved session file.
    """
    config.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    session_path = config.SESSIONS_DIR / f"{image_name}.json"

    payload = {
        "image_name": image_name,
        "image_path": image_path,
        "crops_dir": str(crops_dir),
        "items": results,
        "dino_embs": [e.tolist() for e in dino_embs],
        "mnet_embs": [e.tolist() for e in mnet_embs],
    }

    with open(session_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    logger.info("Session saved → %s", session_path)
    return session_path


def _load_latest_session() -> tuple[list[dict], list, list, str, str]:
    """Load the most recent session file from config.SESSIONS_DIR.

    Returns (results, dino_embs, mnet_embs, image_name, crops_dir).
    Embedding lists are converted back to numpy arrays.
    """
    session_dir = config.SESSIONS_DIR
    if not session_dir.is_dir():
        raise FileNotFoundError(f"No sessions directory: {session_dir}")

    json_files = sorted(session_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not json_files:
        done_dir = session_dir / "done"
        if done_dir.is_dir():
            json_files = sorted(done_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not json_files:
        raise FileNotFoundError(f"No session files found in {session_dir}")

    latest = json_files[-1]
    logger.info("Resuming from session → %s", latest)

    with open(latest, encoding="utf-8") as f:
        payload = json.load(f)

    results = payload["items"]
    dino_embs = [np.array(e) for e in payload["dino_embs"]]
    mnet_embs = [np.array(e) for e in payload["mnet_embs"]]
    image_name = payload["image_name"]
    crops_dir = payload["crops_dir"]

    if not image_name:
        image_name = latest.stem

    return results, dino_embs, mnet_embs, image_name, crops_dir


def _load_session_by_name(session_name: str) -> tuple[list[dict], list, list, str, str]:
    """Load a specific session file by image stem name.

    Returns (results, dino_embs, mnet_embs, image_name, crops_dir).
    Embedding lists are converted back to numpy arrays.
    """
    session_file = config.SESSIONS_DIR / f"{session_name}.json"
    if not session_file.exists():
        session_file = config.SESSIONS_DIR / "done" / f"{session_name}.json"
    if not session_file.exists():
        raise FileNotFoundError(f"Session file not found: {session_file}")

    logger.info("Loading session → %s", session_file)
    with open(session_file, encoding="utf-8") as f:
        payload = json.load(f)

    results = payload["items"]
    dino_embs = [np.array(e) for e in payload["dino_embs"]]
    mnet_embs = [np.array(e) for e in payload["mnet_embs"]]
    image_name = payload.get("image_name", session_name)
    crops_dir = payload.get("crops_dir", str(config.CROPS_DIR / image_name))

    return results, dino_embs, mnet_embs, image_name, crops_dir, payload


def _check_session_review_status(
    session_data: dict,
    registry: SKURegistry,
) -> tuple[dict[int, str], bool]:
    """Check which session crops already exist as exemplars in the registry.

    Returns
    -------
    (pre_confirmed_map, is_complete)
    * pre_confirmed_map:     ``{item_index: sku_id}`` for crops already
                             registered as exemplars.
    * is_complete:           ``True`` when **every** crop in the session is
                             already registered.
    """
    # Build reverse map: crop_path -> SKU ID
    # NOTE: Full path match only. Filename-only matching is intentionally
    # excluded — crops across different images share generic names like
    # ``crop_003.jpg`` and must NOT be treated as pre-confirmed.
    crop_to_sku: dict[str, str] = {}
    for sku_id, entry in registry.get_all_entries().items():
        for ex_path in entry.get("exemplars", []):
            crop_to_sku[ex_path] = sku_id

    items = session_data.get("items", [])
    pre_confirmed_map: dict[int, str] = {}

    for idx, item in enumerate(items):
        cp = item.get("crop_path", "")
        if not cp:
            continue
        sku = crop_to_sku.get(cp)
        if sku is not None:
            pre_confirmed_map[idx] = sku

    is_complete = bool(items) and len(pre_confirmed_map) == len(items)
    return pre_confirmed_map, is_complete


def _move_session_to_done(session_file: Path) -> None:
    """Move a completed session file to ``data/sessions/done/``.

    Creates the ``done/`` subdirectory if it does not exist.
    Silently skips if the file is already under ``done/``.
    """
    if "done" in session_file.parts:
        return
    done_dir = session_file.parent / "done"
    done_dir.mkdir(parents=True, exist_ok=True)
    dst = done_dir / session_file.name
    if dst.exists():
        dst.unlink()
    session_file.rename(dst)
    logger.info("Moved completed session → %s", dst)


def _list_sessions() -> None:
    """Print a table of all session files with review status.

    Columns: Session | Crops | Reviewed | Status
    """
    session_dir = config.SESSIONS_DIR
    if not session_dir.is_dir():
        print(f"No sessions directory: {session_dir}")
        return

    registry = SKURegistry()
    json_files = sorted(
        session_dir.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    done_dir = session_dir / "done"
    done_files: list[Path] = []
    if done_dir.is_dir():
        done_files = sorted(
            done_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    if not json_files and not done_files:
        print("No session files found.")
        return

    header = f"{'Session':30s}  {'Crops':6s}  {'Reviewed':8s}  {'Status':15s}"
    sep = "-" * 62
    print(f"\n{header}")
    print(sep)

    for sf in json_files:
        try:
            with open(sf, encoding="utf-8") as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            print(f"{sf.stem:30s}  {'?':6s}  {'?':8s}  {'corrupt':15s}")
            continue

        items = payload.get("items", [])
        n_crops = len(items)
        pre_confirmed, is_complete = _check_session_review_status(payload, registry)
        n_reviewed = len(pre_confirmed)

        if is_complete:
            status = "complete"
        elif n_reviewed > 0:
            status = "partial"
        else:
            status = "pending"

        reviewed_str = f"{n_reviewed}/{n_crops}" if n_crops else "—"

        print(f"{sf.stem:30s}  {n_crops:6d}  {reviewed_str:>8s}  {status:15s}")

    for sf in done_files:
        try:
            with open(sf, encoding="utf-8") as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            print(f"{sf.stem:30s}  {'?':6s}  {'?':8s}  {'corrupt':15s}")
            continue
        items = payload.get("items", [])
        n_crops = len(items)
        reviewed_str = f"{n_crops}/{n_crops}" if n_crops else "—"
        print(f"{sf.stem:30s}  {n_crops:6d}  {reviewed_str:>8s}  {'done':15s}")

    print()


def _batch_process(
    batch_dir: str,
    conf_threshold: float | None = None,
    tta: bool = False,
    verbose: bool = False,
) -> int:
    """Process all images in a directory through detect → embed → match (no review/export).

    Parameters
    ----------
    batch_dir : str
        Path to directory containing shelf images.
    conf_threshold : float or None
        Detection confidence threshold.  None = use config default.
    tta : bool
        Enable Test Time Augmentation for detection.
    verbose : bool
        Enable verbose logging (passed through to step functions).

    Returns
    -------
    int
        0 on success, 1 on error.
    """
    batch_path = Path(batch_dir).resolve()

    if not batch_path.is_dir():
        print(f"Batch directory not found: {batch_path}")
        return 1

    # Scan for image files sorted alphabetically
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    image_paths: list[str] = []
    for p in sorted(batch_path.iterdir()):
        if p.is_file() and p.suffix.lower() in image_extensions:
            image_paths.append(str(p.resolve()))

    if not image_paths:
        logger.warning("No image files found in %s — nothing to do", batch_path)
        return 0

    # Override confidence threshold if provided
    if conf_threshold is not None:
        config.CONF_THRESHOLD = conf_threshold
        Detector._instance = None

    registry = SKURegistry()
    t_start = time.time()
    total = len(image_paths)

    logger.info("=" * 55)
    logger.info("Batch Processing — %d image(s) in %s", total, batch_path)
    logger.info("=" * 55)

    for i, image_path in enumerate(image_paths, 1):
        filename = Path(image_path).name
        stem = Path(image_path).stem

        logger.info("")
        logger.info("[%d/%d] %s", i, total, filename)

        # Step 1-2: Detect + crop
        img, detections, crops_dir = step_detect(
            image_path,
            conf_threshold=config.CONF_THRESHOLD,
            augment=tta,
        )

        if len(detections) == 0:
            logger.warning("  → No detections — skipping %s", filename)
            continue

        # Step 3: Embed
        dino_embs, mnet_embs, valid_dets = step_embed(detections, crops_dir)

        if len(valid_dets) == 0:
            logger.warning("  → No valid embeddings — skipping %s", filename)
            continue

        # Step 4: Positional
        boxes_norm = step_positional(img, valid_dets)

        # Step 5: Match against registry
        results = step_match(registry, dino_embs, mnet_embs, valid_dets, boxes_norm)

        # Save session
        _save_session(results, dino_embs, mnet_embs, crops_dir, stem, image_path)

        n_detections = len(results)
        logger.info(
            "  ✓ [%d/%d] %s — %d detections",
            i, total, filename, n_detections,
        )

    elapsed = time.time() - t_start
    logger.info("")
    logger.info("=" * 55)
    logger.info("  Batch complete! %d image(s) processed in %.1fs", total, elapsed)
    logger.info("=" * 55)

    return 0


def _batch_review(registry: SKURegistry) -> int:
    """Review all pending/partial sessions sequentially.

    Scans ``config.SESSIONS_DIR`` for session files, filters to sessions
    that are not fully reviewed, and runs ``step_review`` + ``step_update_registry``
    for each.  Handles ``KeyboardInterrupt`` gracefully — progress from completed
    sessions is preserved.

    Returns
    -------
    int
        0 on success, 1 on error.
    """
    session_dir = config.SESSIONS_DIR
    if not session_dir.is_dir():
        print(f"No sessions directory: {session_dir}")
        return 1

    json_files = sorted(
        session_dir.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
    )

    if not json_files:
        print("No session files found.")
        return 0

    # Identify pending / partially-reviewed sessions
    pending: list[tuple[Path, dict, dict[int, str]]] = []
    for sf in json_files:
        try:
            with open(sf, encoding="utf-8") as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("Skipping corrupt session: %s", sf)
            continue

        pre_confirmed_map, is_complete = _check_session_review_status(payload, registry)
        if not is_complete:
            pending.append((sf, payload, pre_confirmed_map))

    if not pending:
        print("No pending sessions found.")
        return 0

    total = len(pending)
    n_reviewed_total = 0
    n_sessions_done = 0
    i = 0  # keep last index for summary after break

    for i, (session_file, payload, pre_confirmed_map) in enumerate(pending, 1):
        image_name = payload.get("image_name", session_file.stem)
        image_path = payload.get("image_path", "")
        crops_dir_str = payload.get("crops_dir", str(config.CROPS_DIR / image_name))
        results = payload["items"]
        dino_embs = [np.array(e) for e in payload.get("dino_embs", [])]
        mnet_embs = [np.array(e) for e in payload.get("mnet_embs", [])]

        # Resolve image path for review server
        img_for_review = image_path or str(config.INPUT_DIR / f"{image_name}.jpg")
        if not Path(img_for_review).exists():
            img_for_review = str(config.INPUT_DIR / f"{image_name}.jpg")
        if not Path(img_for_review).exists():
            logger.warning("Image not found for %s: %s — skipping", image_name, img_for_review)
            continue

        n_pending = len(results) - len(pre_confirmed_map)
        print(f"\n{'='*50}")
        print(f"  Session [{i}/{total}]: {image_name}")
        print(f"  {n_pending}/{len(results)} crops to review")
        if pre_confirmed_map:
            print(f"  ({len(pre_confirmed_map)} pre-confirmed, skipping)")
        print(f"{'='*50}\n")

        try:
            confirmed = step_review(
                img_for_review, results, Path(crops_dir_str), Path(crops_dir_str),
                dino_embs=dino_embs, mnet_embs=mnet_embs, registry=registry,
                pre_confirmed_map=pre_confirmed_map or None,
            )
        except KeyboardInterrupt:
            print("\n\nInterrupted — stopping after current session. Progress saved.")
            break

        if not confirmed:
            logger.warning("No confirmations received for %s — skipping", image_name)
            continue

        source_label = image_name
        idx_to_sku = step_update_registry(registry, results, confirmed, [], [], source_label)
        n_sessions_done += 1
        n_reviewed_total += len(idx_to_sku)

        # Move fully-reviewed session to done/ so it doesn't appear in future scans
        _move_session_to_done(session_file)

        print(f"  ✓ Session {i}/{total} complete. {len(confirmed)} new confirmations.")

    # Cleanup pass: move any already-complete sessions out of the main dir
    remaining_json = sorted(session_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    for sf in remaining_json:
        try:
            with open(sf, encoding="utf-8") as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        _, is_complete = _check_session_review_status(payload, registry)
        if is_complete:
            _move_session_to_done(sf)

    print(f"\n{'='*50}")
    print(f"  Batch review complete! {n_reviewed_total} crops confirmed "
          f"across {n_sessions_done} sessions.")
    print(f"  Run --batch-export to export.")
    print(f"{'='*50}\n")
    return 0


def _batch_export(registry: SKURegistry, export_flags: list[str] | None = None) -> int:
    """Export all reviewed sessions.

    Scans ``config.SESSIONS_DIR``, checks each session against the registry,
    and calls ``step_export`` for every session that has at least one
    registered crop.

    Parameters
    ----------
    export_flags:
        Export formats (``"yolo"``, ``"retrieval"``, ``"labelme"``).
        Defaults to ``["yolo"]`` when ``None`` or empty.

    Returns
    -------
    int
        0 on success, 1 on error.
    """
    if not export_flags:
        export_flags = ["yolo"]

    session_dir = config.SESSIONS_DIR
    if not session_dir.is_dir():
        print(f"No sessions directory: {session_dir}")
        return 1

    json_files = sorted(
        session_dir.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
    )

    done_dir = session_dir / "done"
    if done_dir.is_dir():
        done_files = sorted(
            done_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
        )
        json_files.extend(done_files)
        json_files.sort(key=lambda p: p.stat().st_mtime)

    if not json_files:
        print("No session files found.")
        return 0

    n_exported = 0
    n_sessions = 0

    for sf in json_files:
        try:
            with open(sf, encoding="utf-8") as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("Skipping corrupt session: %s", sf)
            continue

        results = payload.get("items", [])
        if not results:
            continue

        # Reconstruct idx→sku map from registry exemplars
        pre_confirmed_map, _ = _check_session_review_status(payload, registry)

        if not pre_confirmed_map:
            continue  # No reviewed crops in this session

        image_path = payload.get("image_path", "")
        image_name = payload.get("image_name", sf.stem)
        if not image_path or not Path(image_path).exists():
            image_path = str(config.INPUT_DIR / f"{image_name}.jpg")
        if not Path(image_path).exists():
            logger.warning("Image not found for session %s: %s — skipping", image_name, image_path)
            continue

        n_sessions += 1
        n_exported += len(pre_confirmed_map)
        step_export(image_path, results, pre_confirmed_map, export_flags, registry=registry)
        print(f"  ✓ Exported {image_name}: {len(pre_confirmed_map)} crops")

    if n_sessions == 0:
        print("No reviewed sessions found.")
        return 0

    print(f"\n{'='*50}")
    print(f"  Batch export complete! {n_exported} crops exported across {n_sessions} sessions.")
    print(f"{'='*50}\n")
    return 0


# ── Step helpers ────────────────────────────────────────────────────────────


def step_detect(image_path: str, conf_threshold: float | None = None, augment: bool = False) -> tuple[Image.Image, list[dict], Path]:
    """Step 1-2: Detect products with YOLO, save crops, return metadata.

    Returns
    -------
    (pil_image, detections, crops_subdir)
    """
    logger.info("─" * 50)
    logger.info("STEP 1: Detecting products with SKU110K-v3%s", " + TTA" if augment else "")
    image_path = str(Path(image_path).resolve())

    detector = Detector.get_instance(conf_threshold=conf_threshold)
    crops_dir = str(config.CROPS_DIR)
    detections = detector.detect(image_path, crops_dir, augment=augment)

    img = Image.open(image_path).convert("RGB")
    stem = Path(image_path).stem
    crops_subdir = config.CROPS_DIR / stem

    logger.info("  → %d detections (conf >= %.2f)", len(detections), config.CONF_THRESHOLD)
    logger.info("  → Crops saved to %s", crops_subdir)
    return img, detections, crops_subdir


def step_embed(detections: list[dict], crops_dir: Path) -> tuple[list, list, list]:
    """Step 3: Compute DINOv3 + MobileNetV2 embeddings for all crops.

    Returns
    -------
    (dino_embs, mnet_embs, valid_detections)
    where dino_embs[i] is 384-dim ndarray, mnet_embs[i] is 1280-dim ndarray.
    Only detections that successfully embedded both models are returned.
    """
    logger.info("STEP 2: Extracting DINOv3 + MobileNetV2 embeddings")

    crop_paths = [d["crop_path"] for d in detections]
    dino_ext = DINOv3Extractor()
    mnet_ext = MobileNetV2Extractor()

    dino_embs = dino_ext.extract_batch(crop_paths)
    mnet_embs = mnet_ext.extract_batch(crop_paths)

    # Filter out failed embeddings
    valid_dets: list[dict] = []
    valid_dino: list = []
    valid_mnet: list = []
    for det, de, me in zip(detections, dino_embs, mnet_embs):
        if de is not None and me is not None:
            valid_dets.append(det)
            valid_dino.append(de)
            valid_mnet.append(me)

    logger.info("  → %d/%d crops embedded successfully", len(valid_dets), len(detections))
    return valid_dino, valid_mnet, valid_dets


def step_positional(img: Image.Image, detections: list[dict]) -> list:
    """Step 4: Compute normalized boxes and shelf row IDs."""
    logger.info("STEP 3: Computing positional features")
    boxes = [d["box"] for d in detections]
    boxes_norm = normalize_boxes(boxes, img.size)
    row_ids = group_shelf_rows(boxes_norm, eps=config.POSITIONAL_ROW_EPS)
    logger.info("  → %d shelf rows detected", len(set(row_ids) - {i for i in row_ids if i < 0}))
    return boxes_norm


def step_match(
    registry: SKURegistry,
    dino_embs: list,
    mnet_embs: list,
    detections: list[dict],
    boxes_norm: list | None = None,
) -> list[dict]:
    """Step 5: Match each crop against the SKU registry.

    Returns enriched detection list with ``sku_id``, ``label``, ``score``,
    ``zone``, ``candidates``.
    """
    logger.info("STEP 4: Matching against SKU registry")

    if len(registry) == 0:
        logger.info("  → Registry is empty — all crops will be NEW")
        results = []
        for idx, det in enumerate(detections):
            box_norm = boxes_norm[idx] if boxes_norm is not None else None
            results.append({
                **det,
                "sku_id": None,
                "label": "New SKU",
                "score": 0.0,
                "zone": "none",
                "candidates": [],
            })
        return results

    results = []
    for i, (de, me) in enumerate(zip(dino_embs, mnet_embs)):
        box_norm = boxes_norm[i] if boxes_norm is not None else None
        candidates = registry.match_new_crop(de, me, box_norm)

        if not candidates:
            results.append({
                **detections[i],
                "sku_id": None,
                "label": "New SKU",
                "score": 0.0,
                "zone": "none",
                "candidates": [],
            })
        else:
            top = candidates[0]
            results.append({
                **detections[i],
                "sku_id": top["sku_id"],
                "label": top["label"],
                "score": top["score"],
                "dino_score": top.get("dino_score"),
                "mnet_score": top.get("mnet_score"),
                "n_exemplars": top.get("n_exemplars", 0),
                "zone": top["zone"],
                "anchor_exemplar": top.get("anchor_exemplar", ""),
                "candidates": candidates,
            })

    high = sum(1 for r in results if r["zone"] == "high")
    ambig = sum(1 for r in results if r["zone"] == "ambiguous")
    none_z = sum(1 for r in results if r["zone"] == "none")
    logger.info("  → %d high-confidence, %d ambiguous, %d new", high, ambig, none_z)
    for i, r in enumerate(results):
        top_score = r.get("score", 0.0)
        top_sku = r.get("sku_id") or "—"
        dino = r.get("dino_score", "—")
        mnet = r.get("mnet_score", "—")
        label = (r.get("label") or "new")[:24]
        logger.info(
            "    crop #%03d  zone=%-9s  sku=%-8s  score=%.4f  dino= %s  mnet= %s  label=%s",
            i, r["zone"], top_sku, top_score,
            f"{dino:.4f}" if isinstance(dino, float) else "—",
            f"{mnet:.4f}" if isinstance(mnet, float) else "—",
            label,
        )
    return results


def step_review(
    image_path: str,
    results: list[dict],
    crops_dir: Path,
    anchors_dir: Path,
    dino_embs: list | None = None,
    mnet_embs: list | None = None,
    registry: SKURegistry | None = None,
    pre_confirmed_map: dict[int, str] | None = None,
) -> dict[int, dict]:
    """Step 6: Launch visual review server, return confirmed actions.

    If there are no detections, returns empty dict immediately.

    Parameters
    ----------
    pre_confirmed_map:
        Optional mapping ``{item_index: sku_id}`` for crops already registered
        as exemplars.  These are pre-set as confirmed so the review UI skips
        them.
    """
    logger.info("STEP 5: Launching visual review server")

    if len(results) == 0:
        logger.info("  → No detections to review")
        return {}

    _reg = registry or SKURegistry()
    session = ReviewSession(
        title=Path(image_path).name,
        items=results,
        crops_dir=crops_dir,
        dino_embs=dino_embs or [],
        mnet_embs=mnet_embs or [],
        registry=_reg,
    )

    # Pre-set status for already-registered crops so the review server
    # treats them as confirmed from the start.
    if pre_confirmed_map:
        for idx, sku_id in pre_confirmed_map.items():
            if idx < len(results):
                session.crop_status[idx] = "confirmed"
                session.crop_action[idx] = {"action": "accept", "sku_id": sku_id}

    logger.info("  step_review: registry = SKURegistry[%d] (session.registry id=%s)", _reg._inst_id, id(session.registry))
    logger.info("  → Review page: http://%s:%d", config.REVIEW_HOST, config.REVIEW_PORT)
    confirmed = launch_server(session)
    return confirmed


def _sync_exemplars(registry: SKURegistry) -> int:
    """Copy all registry exemplar paths to ``exemplars/{SKU_ID}/crop_NNN.ext``.

    Idempotent — re-running overwrites old files.  Missing crop files are
    skipped with a warning.

    Returns the number of files copied.
    """
    ex_dir = config.PIPELINE_DIR / "exemplars"
    ex_dir.mkdir(parents=True, exist_ok=True)

    total_copied = 0

    for sku_id, entry in registry.get_all_entries().items():
        sku_dir = ex_dir / sku_id
        sku_dir.mkdir(parents=True, exist_ok=True)

        exemplars = entry.get("exemplars", [])
        if not exemplars:
            continue

        for idx, src_path_str in enumerate(exemplars):
            src = Path(src_path_str)
            if not src.exists():
                logger.warning("  Exemplar not found, skipping: %s", src)
                continue

            dst = sku_dir / f"crop_{idx:03d}{src.suffix}"
            try:
                shutil.copy2(str(src), str(dst))
                total_copied += 1
            except shutil.SameFileError:
                pass  # already in place
            except Exception as e:
                logger.error("  Failed to copy %s: %s", src, e)

    logger.info("Sync complete: %d exemplars synced to %s", total_copied, ex_dir)
    return total_copied


def step_update_registry(
    registry: SKURegistry,
    results: list[dict],
    confirmed: dict[int, dict],
    dino_embs: list,
    mnet_embs: list,
    source_image: str,
) -> dict[int, str]:
    """Step 7: Build ``{idx: sku_id}`` mapping from review actions.

    Registry is already mutated during live review — this only reads
    the action payloads to produce the mapping for export.
    """
    logger.info("STEP 6: Building SKU mapping from review")

    idx_to_sku: dict[int, str] = {}

    for idx_str, action in confirmed.items():
        idx = int(idx_str)
        act = action.get("action")
        sku_id = action.get("sku_id")
        if sku_id and act in ("accept", "create", "new", "rename", "skip"):
            idx_to_sku[idx] = sku_id
            if act != "skip":
                logger.info("  → %s assigned to %s", Path(results[idx]["crop_path"]).name, sku_id)

    logger.info("  step_update_registry: registry SKURegistry[%d] — _data keys: %s", registry._inst_id, list(registry._data.keys()))
    in_memory = len(registry)
    logger.info("  → Registry now has %d SKU(s) in memory", in_memory)
    registry._save()
    on_disk = len(json.loads(Path(registry._path).read_text())) if Path(registry._path).exists() else 0
    logger.info("  → Registry now has %d SKU(s) on disk", on_disk)
    logger.info("  step_update_registry: on-disk=%d — confirming %d actions (first 5: %s)", on_disk, len(confirmed), list(confirmed.keys())[:5])

    _sync_exemplars(registry)

    return idx_to_sku


def step_export(
    image_path: str,
    results: list[dict],
    idx_to_sku: dict[int, str],
    export_flags: list[str],
    registry: SKURegistry | None = None,
) -> None:
    """Step 8: Export YOLO labels and/or retrieval folders.

    Parameters
    ----------
    registry:
        Live registry instance from the current pipeline run.  When ``None``
        (e.g. re-export path), a fresh ``SKURegistry`` is loaded from disk.
    """
    if not export_flags:
        logger.info("STEP 7: Export skipped (no --export flag)")
        return

    logger.info("STEP 7: Exporting")

    # Use the live registry when available (avoids depending on
    # disk correctness), otherwise fall back to loading from disk.
    reg = registry or SKURegistry()

    # Enrich detections with SKU IDs
    enriched = []
    for idx, det in enumerate(results):
        if idx in idx_to_sku:
            enriched.append({**det, "sku_id": idx_to_sku[idx]})

    if not enriched:
        logger.info("  → No confirmed detections to export")
        return

    if "yolo" in export_flags:
        class_map = build_class_mapping(reg.get_all_entries())
        result = export_yolo(image_path, enriched, class_map)
        logger.info("  → YOLO: %s (%d labels)", result["label_file"], result["n_labels"])

    if "labelme" in export_flags:
        if "yolo" not in export_flags:
            logger.warning("  → LabelMe export requested but YOLO not in flags; running YOLO first")
            class_map = build_class_mapping(reg.get_all_entries())
            export_yolo(image_path, enriched, class_map)
        y2l.main([
            "--input-dir", str(config.YOLO_EXPORT_DIR),
            "--output-dir", str(config.EXPORTS_DIR / "labelme"),
            "--images-dir", str(config.INPUT_DIR),
        ])
        logger.info("  → LabelMe: converted to %s", config.EXPORTS_DIR / "labelme")

    if "retrieval" in export_flags:
        result = export_retrieval_folders(reg.get_all_entries())
        logger.info("  → Retrieval: %d SKUs, %d files → %s", result["n_skus"], result["n_files"], result["output_dir"])


# ── CLI ─────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Curated SKU labelling pipeline — one shelf image at a time.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mutual exclusive: process image OR query registry
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--image", type=str, help="Path to a shelf image to process")
    group.add_argument("--list-skus", action="store_true", help="List all SKUs in registry")
    group.add_argument("--sku-info", type=str, metavar="SKU-ID", help="Show info for one SKU")
    group.add_argument("--delete-sku", type=str, metavar="SKU-ID", help="Delete a SKU from registry")
    group.add_argument("--resume", nargs="?", const=None, default=_RESUME_UNSET,
                       metavar="SESSION_NAME",
                       help="Resume a session (skip steps 1-5). Optionally specify the session image stem, "
                            "e.g. --resume IMG20260626184606. Omit to resume most recent session. "
                            "Use --resume --list to list all sessions.")
    group.add_argument("--re-export", type=str, metavar="IMAGE_PATH",
                       help="Re-export from a previously reviewed image (skips detect→review)")
    group.add_argument("--batch-dir", type=str,
                       help="Process all images in a directory (detect→embed→match, no review/export)")
    group.add_argument("--batch-review", action="store_true",
                       help="Review all pending/partial sessions sequentially")

    parser.add_argument("--export", type=str, nargs="+",
                        choices=["yolo", "retrieval", "labelme"],
                        help="Export format(s) after confirmation (default: yolo)")
    parser.add_argument("--batch-export", type=str, nargs="*",
                        choices=["yolo", "retrieval", "labelme"],
                        help="Export all reviewed sessions in specified format(s) (default: yolo)")
    parser.add_argument("--no-export", action="store_true",
                        help="Skip all exports even if defaults are set")
    parser.add_argument("--dry-run", action="store_true",
                        help="Detect + embed only, skip review and export")
    parser.add_argument("--conf", type=float, default=None,
                        help=f"Detection confidence threshold (default: {config.CONF_THRESHOLD})")
    parser.add_argument("--tta", action="store_true",
                        help="Enable YOLOv5 Test Time Augmentation (TTA) for better detection")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--list", action="store_true",
                        help="List all sessions with review status (used with --resume)")

    return parser.parse_args(argv)


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)

    # ── Registry queries ───────────────────────────────────────────────────
    registry = SKURegistry()
    logger.info("main: registry instance = SKURegistry[%d]", registry._inst_id)

    if args.list_skus:
        skus = registry.list_skus()
        print(f"\nSKU Registry ({len(skus)} entries):\n")
        for sid in skus:
            info = registry.get_sku_info(sid)
            print(f"  {sid}  |  {info['label']:40s}  |  {info['category']:15s}  |  {info['n_exemplars']} exemplars")
        return 0

    if args.sku_info:
        info = registry.get_sku_info(args.sku_info)
        if info is None:
            print(f"SKU {args.sku_info} not found in registry")
            return 1
        print(f"\nSKU: {info['sku_id']}")
        print(f"  Label:        {info['label']}")
        print(f"  Category:     {info['category']}")
        print(f"  Exemplars:    {info['n_exemplars']}")
        print(f"  Anchor:       {info['anchor_exemplar']}")
        print(f"  Source imgs:  {', '.join(info['source_images'])}")
        print(f"  Created:      {info['created']}")
        print(f"  Updated:      {info['updated']}")
        return 0

    if args.delete_sku:
        info = registry.get_sku_info(args.delete_sku)
        if info is None:
            print(f"SKU {args.delete_sku} not found")
            return 1
        print(f"Deleting {info['sku_id']}: {info['label']}")
        registry.delete_sku(args.delete_sku)
        print("Done.")
        return 0

    # ── Batch review (sequential) ────────────────────────────────────────
    if args.batch_review:
        return _batch_review(registry)

    # ── Batch processing ──────────────────────────────────────────────────
    if args.batch_dir:
        return _batch_process(
            batch_dir=args.batch_dir,
            conf_threshold=args.conf,
            tta=args.tta,
            verbose=args.verbose,
        )

    # ── Re-export from completed session ──────────────────────────────────
    if args.re_export:
        image_path = str(Path(args.re_export).resolve())
        if not Path(image_path).exists():
            print(f"Image not found: {image_path}")
            return 1

        image_name = Path(image_path).stem
        session_file = config.SESSIONS_DIR / f"{image_name}.json"
        if not session_file.exists():
            print(f"No session file for {image_name} — run --image first")
            return 1

        with open(session_file, encoding="utf-8") as f:
            payload = json.load(f)
        results = payload["items"]
        crops_dir = payload.get("crops_dir", str(config.CROPS_DIR / image_name))

        registry = SKURegistry()
        logger.info("main: registry instance = SKURegistry[%d]", registry._inst_id)
        # Reconstruct idx→sku from registry exemplars matching this image's crops
        idx_to_sku: dict[int, str] = {}
        crop_to_sku: dict[str, str] = {}
        for sku_id, entry in registry.get_all_entries().items():
            for ex_path in entry.get("exemplars", []):
                crop_to_sku[ex_path] = sku_id
        for idx, det in enumerate(results):
            cp = det.get("crop_path", "")
            if cp in crop_to_sku:
                idx_to_sku[idx] = crop_to_sku[cp]

        if not idx_to_sku:
            print("No confirmed crops found for this image in registry")
            return 1

        if args.no_export:
            export_flags = []
        else:
            export_flags = args.export or ["yolo"]
        step_export(image_path, results, idx_to_sku, export_flags)

        print(f"\n  Re-export complete: {len(idx_to_sku)} crops exported")
        return 0

    # ── Session listing (--resume --list) ──────────────────────────────────
    if args.list:
        _list_sessions()
        return 0

    # ── Resume session ────────────────────────────────────────────────────
    if args.resume is not _RESUME_UNSET:
        resume_session_name = args.resume

        try:
            if resume_session_name:
                results, dino_embs, mnet_embs, image_name, crops_dir, payload = (
                    _load_session_by_name(resume_session_name)
                )
                session_file = config.SESSIONS_DIR / f"{resume_session_name}.json"
                if not session_file.exists():
                    session_file = config.SESSIONS_DIR / "done" / f"{resume_session_name}.json"
            else:
                results, dino_embs, mnet_embs, image_name, crops_dir = _load_latest_session()
                # Find matching session file (may be in main dir or done/)
                session_file = next(
                    sorted(config.SESSIONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
                    + sorted(
                        (config.SESSIONS_DIR / "done").glob("*.json"),
                        key=lambda p: p.stat().st_mtime,
                    ),
                    None,
                )
                if session_file is None:
                    raise FileNotFoundError("No session files found")
                with open(session_file, encoding="utf-8") as f:
                    payload = json.load(f)
        except FileNotFoundError as exc:
            print(f"Error: {exc}")
            return 1

        pre_confirmed_map, is_complete = _check_session_review_status(payload, registry)

        if is_complete:
            n = len(payload.get("items", []))
            print(f"\n  Session '{payload.get('image_name', '?')}' — all {n} crops already registered.")
            print("  Nothing to review.\n")
            return 0

        img_for_review = payload.get("image_path", "") or str(
            config.INPUT_DIR / f"{image_name}.jpg"
        )
        if not Path(img_for_review).exists():
            img_for_review = str(config.INPUT_DIR / f"{image_name}.jpg")
        source_label = payload.get("image_name", image_name)

        print(f"\n{'='*50}")
        print(f"  Resumed session — {source_label}")
        if pre_confirmed_map:
            n_total = len(results)
            n_pre = len(pre_confirmed_map)
            print(f"  {n_pre}/{n_total} crops pre-confirmed (already in registry) — skipping")
        print(f"{'='*50}\n")

        confirmed = step_review(
            img_for_review, results, Path(crops_dir), Path(crops_dir),
            dino_embs=dino_embs, mnet_embs=mnet_embs, registry=registry,
            pre_confirmed_map=pre_confirmed_map or None,
        )

        if not confirmed:
            logger.warning("No confirmations received — aborting")
            return 1

        idx_to_sku = step_update_registry(registry, results, confirmed, [], [], source_label)

        _move_session_to_done(session_file)

        # Step 8: Export
        if args.no_export:
            export_flags = []
        else:
            export_flags = args.export or ["yolo"]
        step_export(img_for_review, results, idx_to_sku, export_flags, registry=registry)

        n_user_confirmed = len(confirmed)
        n_total = len(idx_to_sku)
        print(f"\n{'='*50}")
        print(f"  Done! {n_total} crops confirmed ({n_user_confirmed} via review, "
              f"{len(pre_confirmed_map)} pre-confirmed)")
        print(f"{'='*50}")
        return 0

    # ── Batch export (scan all reviewed sessions) ─────────────────────────
    if args.batch_export is not None:
        return _batch_export(registry, args.batch_export)

    # ── Image processing ───────────────────────────────────────────────────
    if not args.image:
        print("No command specified. Use --image or --resume or --list-skus or --help")
        return 1

    image_path = str(Path(args.image).resolve())
    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        return 1

    # Override confidence threshold if provided
    if args.conf is not None:
        import config as cfg_mod
        cfg_mod.CONF_THRESHOLD = args.conf
        # Force new detector instance with updated threshold
        Detector._instance = None

    print(f"\n{'='*50}")
    print(f"  Curated Pipeline — {Path(image_path).name}")
    print(f"{'='*50}\n")

    t_start = time.time()

    # Step 1-2: Detect + crop
    img, detections, crops_dir = step_detect(image_path, conf_threshold=config.CONF_THRESHOLD, augment=args.tta)

    if len(detections) == 0:
        logger.warning("No detections found in image — nothing to do")
        return 0

    # Step 3: Embed
    dino_embs, mnet_embs, valid_dets = step_embed(detections, crops_dir)

    if len(valid_dets) == 0:
        logger.warning("No valid embeddings — skipping review")
        return 0

    # Step 4: Positional (optional)
    boxes_norm = step_positional(img, valid_dets)

    # Step 5: Match against registry
    results = step_match(registry, dino_embs, mnet_embs, valid_dets, boxes_norm)

    # Persist session for potential --resume
    _save_session(results, dino_embs, mnet_embs, crops_dir, Path(image_path).stem, image_path)

    if args.dry_run:
        print("\n── DRY RUN — no review or export ──")
        print(f"  Detections: {len(results)}")
        print(f"  Registry:   {len(registry)} SKUs")
        print(f"  Elapsed:    {time.time() - t_start:.1f}s")
        return 0

    confirmed = step_review(
        image_path, results, crops_dir, crops_dir,
        dino_embs=dino_embs, mnet_embs=mnet_embs, registry=registry,
    )

    if not confirmed:
        logger.warning("No confirmations received — aborting")
        return 1

    idx_to_sku = step_update_registry(registry, results, confirmed, [], [], Path(image_path).name)

    # Step 8: Export
    if args.no_export:
        export_flags = []
    else:
        export_flags = args.export or ["yolo"]
    step_export(image_path, results, idx_to_sku, export_flags, registry=registry)

    elapsed = time.time() - t_start
    print(f"\n{'='*50}")
    print(f"  Done! {len(confirmed)} crops confirmed in {elapsed:.1f}s")
    print(f"{'='*50}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
