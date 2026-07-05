"""YOLO SKU110K-v3 detection wrapper for the curated pipeline.

Singleton-pattern detector that loads the SKU110K_V3.pt weights once per
process, runs inference, extracts bounding-box crops, saves them to disk,
and returns structured detection metadata.

Usage
-----
    from pipeline_utils.detection import Detector

    det = Detector.get_instance()
    results = det.detect("data/input/shelf_001.jpg", "data/crops")
    # results -> [{"box": [x1,y1,x2,y2], "score": 0.87, "idx": 0, "crop_path": "..."}, ...]
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from PIL import Image

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


class Detector:
    """Singleton YOLOv5 detector for the curated pipeline.

    Wraps ``torch.hub.load('ultralytics/yolov5', 'custom', ...)`` with the
    SKU110K_V3 weights.  The model is loaded exactly once and reused for
    every subsequent call in the same process.

    Parameters
    ----------
    model_path : str
        Path to the YOLOv5 weights file.
    device : str
        Torch device string (e.g. ``"cuda:0"`` or ``"cpu"``).
    conf_threshold : float
        Minimum confidence score to keep a detection.
    """

    _instance: Detector | None = None

    def __init__(
        self,
        model_path: str,
        device: str = "",
        conf_threshold: float = config.CONF_THRESHOLD,
        augment: bool = False,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.device = device
        self.augment = augment
        try:
            logger.info("Loading YOLOv5 model from %s", model_path)
            self.model = torch.hub.load(
                "ultralytics/yolov5",
                "custom",
                path=model_path,
                force_reload=False,
                trust_repo=True,
            )
            if device:
                self.model.to(device)
            if augment:
                self.model.augment = True  # Enable TTA (Test Time Augmentation)
            logger.info("Model loaded successfully on device=%s (augment=%s)", device or "default", augment)
        except Exception:
            logger.exception("Failed to load YOLOv5 model from %s", model_path)
            raise

    # ── Singleton ─────────────────────────────────────────────────────────

    @classmethod
    def get_instance(
        cls,
        model_path: str | None = None,
        device: str | None = None,
        conf_threshold: float | None = None,
        augment: bool = False,
    ) -> Detector:
        """Return or create the singleton detector.

        Parameters
        ----------
        model_path : str or None
            Path to YOLOv5 weights.  Defaults to
            ``config.SKU110K_MODEL_PATH``.
        device : str or None
            Torch device string.  Defaults to ``""`` (auto-select).
        conf_threshold : float or None
            Confidence threshold.  Defaults to ``config.CONF_THRESHOLD``.
        augment : bool
            Enable Test Time Augmentation (TTA) for better detection.
        """
        if cls._instance is None:
            _model_path = model_path or config.SKU110K_MODEL_PATH
            _device = device or ""
            _conf = conf_threshold if conf_threshold is not None else config.CONF_THRESHOLD
            cls._instance = cls(_model_path, _device, _conf, augment)
        return cls._instance

    # ── Detection ─────────────────────────────────────────────────────────

    def detect(self, image_path: str, output_dir: str, augment: bool = False) -> list[dict]:
        """Run detection, save crops, and return metadata.

        Parameters
        ----------
        image_path : str
            Path to the source shelf image.
        output_dir : str
            Root directory where crop images are saved.  Crops are
            written to ``{output_dir}/{image_stem}/{idx}_{score:.4f}.jpg``.
        augment : bool
            Enable Test Time Augmentation (TTA) for this inference.

        Returns
        -------
        list[dict]
            Each dict contains:

            - ``box`` (list[int]): ``[x1, y1, x2, y2]`` clamped to image
              bounds, as integers.
            - ``score`` (float): Detection confidence, rounded to 4 decimals.
            - ``idx`` (int): Zero-based detection index (after filtering).
            - ``crop_path`` (str): Absolute path to the saved crop image.
        """
        src = Path(image_path)
        stem = src.stem
        img = Image.open(src).convert("RGB")
        img_w, img_h = img.size

        if augment:
            self.model.augment = True

        results = self.model(img)
        detections: list[dict] = []

        for *box, score, _cls_id in results.xyxy[0].cpu().numpy():
            x1, y1, x2, y2 = (int(b) for b in box)
            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)

            # Skip tiny detections
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue

            if float(score) < self.conf_threshold:
                continue

            detections.append(
                {"box": [x1, y1, x2, y2], "score": round(float(score), 4)}
            )

        # Save crops and enrich metadata
        crop_root = Path(output_dir) / stem
        crop_root.mkdir(parents=True, exist_ok=True)

        enriched: list[dict] = []
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det["box"]
            crop = img.crop((x1, y1, x2, y2))
            crop_filename = f"{idx}_{det['score']:.4f}.jpg"
            crop_path = crop_root / crop_filename
            crop.save(crop_path, quality=95)

            enriched.append(
                {
                    "box": det["box"],
                    "score": det["score"],
                    "idx": idx,
                    "crop_path": str(crop_path.resolve()),
                }
            )

        logger.info(
            "Detected %d objects in %s (conf>=%.2f)",
            len(enriched),
            src.name,
            self.conf_threshold,
        )
        return enriched
