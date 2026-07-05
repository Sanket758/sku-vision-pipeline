"""Dual embedding extractors for the curated SKU labeling pipeline.

Provides DINOv3 (384-dim) and MobileNetV2 (1280-dim) L2-normalized
embedding extractors used in the retrieval and matching stages.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ── Project root for logic/ imports ────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from logic.dinov3_wrapper import DINOv3Wrapper  # noqa: E402

from experiments.curated_pipeline import config as _cfg  # noqa: E402
DINOV3_MODEL_PATH = _cfg.DINOV3_MODEL_PATH
MOBILENET_DIM = _cfg.MOBILENET_DIM
MOBILENET_INPUT_SIZE = _cfg.MOBILENET_INPUT_SIZE

# ── Device ─────────────────────────────────────────────────────────────────
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── ImageNet normalization constants ───────────────────────────────────────
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _mobilenet_transform() -> transforms.Compose:
    """Standard ImageNet transform for MobileNetV2: resize 256, center-crop 224."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(MOBILENET_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


class DINOv3Extractor:
    """Wraps logic.dinov3_wrapper.DINOv3Wrapper for pipeline use.

    Returns L2-normalized 384-dim float32 embeddings.
    """

    def __init__(self, model_path: str = DINOV3_MODEL_PATH) -> None:
        self._wrapper = DINOv3Wrapper(model_path=model_path)

    def extract(self, image_path: str) -> Optional[np.ndarray]:
        """Extract a single L2-normalized 384-dim embedding.

        Args:
            image_path: Path to an image file.

        Returns:
            np.ndarray of shape [384] (float32, L2-normalized),
            or None on failure.
        """
        return self._wrapper.extract_features(image_path)

    def extract_batch(
        self, paths: list[str], batch_size: int = 32
    ) -> list[Optional[np.ndarray]]:
        """Extract embeddings for multiple images.

        Args:
            paths: List of image file paths.
            batch_size: Number of images per forward pass.

        Returns:
            List of np.ndarray [384] or None, one per input path.
        """
        return self._wrapper.extract_features_batch(paths, batch_size=batch_size)


class MobileNetV2Extractor:
    """MobileNetV2 embedding extractor (features + avgpool, no classifier).

    Returns L2-normalized 1280-dim float32 embeddings.
    """

    def __init__(self) -> None:
        import torchvision.models as models

        backbone = models.mobilenet_v2(weights="DEFAULT")
        # backbone.children: features (Sequential) + classifier (Dropout+Linear)
        # features output: [B, 1280, 7, 7] → need AdaptiveAvgPool2d → [B, 1280, 1, 1]
        self._backbone = torch.nn.Sequential(
            backbone.features,
            torch.nn.AdaptiveAvgPool2d(1),
        )
        self._backbone.to(_DEVICE)
        self._backbone.eval()
        self._transform = _mobilenet_transform()

    @torch.no_grad()
    def extract(self, image_path: str) -> Optional[np.ndarray]:
        """Extract a single L2-normalized 1280-dim embedding.

        Args:
            image_path: Path to an image file.

        Returns:
            np.ndarray of shape [1280] (float32, L2-normalized),
            or None on failure.
        """
        try:
            image = Image.open(image_path).convert("RGB")
            tensor = self._transform(image).unsqueeze(0).to(_DEVICE)
            features = self._backbone(tensor)  # [1, 1280, 1, 1]
            features = features.flatten(1)     # [1, 1280]
            features = F.normalize(features, p=2, dim=1)
            return features.squeeze(0).cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"MobileNetV2: failed to extract {image_path}: {e}")
            return None

    @torch.no_grad()
    def extract_batch(
        self, paths: list[str], batch_size: int = 32
    ) -> list[Optional[np.ndarray]]:
        """Extract embeddings for multiple images.

        Args:
            paths: List of image file paths.
            batch_size: Number of images per forward pass.

        Returns:
            List of np.ndarray [1280] or None, one per input path.
        """
        results: list[Optional[np.ndarray]] = [None] * len(paths)

        for start in range(0, len(paths), batch_size):
            end = start + batch_size
            batch_paths = paths[start:end]
            tensors: list[torch.Tensor] = []
            valid_indices: list[int] = []

            for j, path in enumerate(batch_paths):
                try:
                    img = Image.open(path).convert("RGB")
                    tensors.append(self._transform(img))
                    valid_indices.append(start + j)
                except Exception as e:
                    print(f"MobileNetV2: skipping {path}: {e}")

            if not tensors:
                continue

            batch_tensor = torch.stack(tensors).to(_DEVICE)
            features = self._backbone(batch_tensor)  # [B, 1280, 1, 1]
            features = features.flatten(1)            # [B, 1280]
            features = F.normalize(features, p=2, dim=1)
            features_np = features.cpu().numpy().astype(np.float32)

            for local_i, global_i in enumerate(valid_indices):
                results[global_i] = features_np[local_i]

        return results
