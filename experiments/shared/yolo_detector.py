"""Singleton YOLOv5 detector for SKU-110K product detection.

Usage:
    from shared.yolo_detector import YOLODetector

    detector = YOLODetector.get_instance()
    detections = detector.detect(pil_image, conf=0.3)
    # Returns list of {"box": [x1, y1, x2, y2], "score": float}
"""

from pathlib import Path

import torch
from PIL import Image

from . import hardware_utils as hw


class YOLODetector:
    """Singleton YOLOv5 detector.

    The model is loaded exactly once via torch.hub and shared across all
    callers. Default model: models/SKU110K_V3.pt (YOLOv5x trained on SKU-110K).
    """

    _instance = None

    def __init__(self, model_path: str, device: str = ""):
        self.model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=model_path,
            force_reload=False,
            trust_repo=True,
        )
        if device:
            self.model.to(device)

    @classmethod
    def get_instance(cls, model_path=None, device=None):
        """Return or create the singleton detector.

        Parameters
        ----------
        model_path : str or None
            Path to the YOLOv5 weights file. Defaults to
            ``models/SKU110K_V3.pt`` relative to the project root.
        device : torch.device or str or None
            Target device. Defaults to ``hardware_utils.get_device()``.
        """
        if cls._instance is None:
            project_root = Path(__file__).resolve().parents[2]
            if model_path is None:
                model_path = str(project_root / "models" / "SKU110K_V3.pt")
            if device is None:
                device = hw.get_device()
            cls._instance = cls(str(model_path), str(device))
        return cls._instance

    def detect(self, image: Image.Image, conf: float = 0.3, augment: bool = False) -> list[dict]:
        """Run detection on a PIL image.

        Parameters
        ----------
        image : PIL.Image.Image
            RGB image to detect products in.
        conf : float
            Confidence threshold (0-1).
        augment : bool
            Enable Test-Time Augmentation (TTA) via YOLOv5's built-in flips
            and multi-scale inference. Defaults to False.

        Returns
        -------
        list[dict]
            Each dict: ``{"box": [x1, y1, x2, y2], "score": float}``.
            Box coordinates are absolute pixel values clamped to image bounds.
        """
        results = self.model(image, augment=augment)
        dets = []
        for *box, score, _cls_id in results.xyxy[0].cpu().numpy():
            x1, y1, x2, y2 = [int(b) for b in box]
            w, h = image.size
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            dets.append({"box": [x1, y1, x2, y2], "score": round(float(score), 4)})
        return dets
