"""DINOv3 feature extraction wrapper for few-shot SKU labeling.

Loads local DINOv3 ViT-S/16+ checkpoint and extracts L2-normalized
CLS token embeddings for product image retrieval.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import Optional, List, Tuple


class DINOv3Wrapper:
    """DINOv3 embedding extractor with local checkpoint support.

    Usage:
        wrapper = DINOv3Wrapper()
        success, msg = wrapper.load_model()
        emb = wrapper.extract_features("crop.jpg")  # np.ndarray [384]
    """

    def __init__(self, model_path: str = "models/dinov3/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"):
        self.model_path = model_path
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load_model(self) -> Tuple[bool, str]:
        """Load DINOv3 ViT-S/16+ from github source + local checkpoint weights."""
        try:
            print(f"Loading DINOv3 architecture from facebookresearch/dinov3...")
            self.model = torch.hub.load(
                'facebookresearch/dinov3',
                'dinov3_vits16plus',
                pretrained=False,
                source='github',
            )

            print(f"Loading local weights from {self.model_path}...")
            state_dict = torch.load(self.model_path, map_location=self.device)

            # Handle wrapped checkpoint formats
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            # Remove 'module.' prefix if present (from DDP training)
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"  Missing keys: {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")

            self.model.to(self.device)
            self.model.eval()
            return True, f"DINOv3 ViT-S/16+ loaded on {self.device}"

        except Exception as e:
            return False, f"Failed to load DINOv3: {e}"

    @torch.no_grad()
    def extract_features(self, image_path: str) -> Optional[np.ndarray]:
        """Extract L2-normalized CLS token embedding for one image.

        Returns:
            np.ndarray of shape [384] (L2-normalized), or None on failure.
        """
        if self.model is None:
            ok, msg = self.load_model()
            if not ok:
                print(msg)
                return None

        try:
            image = Image.open(image_path).convert("RGB")
            tensor = self.transform(image).unsqueeze(0).to(self.device)

            # DINOv3 returns a dict; CLS token is under 'x_norm_clstoken'
            output = self.model(tensor)

            if isinstance(output, dict):
                if 'x_norm_clstoken' in output:
                    features = output['x_norm_clstoken']
                elif 'x' in output:
                    features = output['x'][:, 0]
                else:
                    # Fallback: first key, first token
                    features = output[list(output.keys())[0]][:, 0]
            else:
                # Raw tensor: [B, N, D] or [B, D]
                if output.dim() == 3:
                    features = output[:, 0]  # CLS token
                else:
                    features = output

            features = F.normalize(features, p=2, dim=1)
            return features.squeeze(0).cpu().numpy().astype(np.float32)

        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None

    @torch.no_grad()
    def extract_features_batch(
        self, image_paths: List[str], batch_size: int = 32
    ) -> List[Optional[np.ndarray]]:
        """Extract features for multiple images in batches."""
        if self.model is None:
            ok, msg = self.load_model()
            if not ok:
                print(msg)
                return [None] * len(image_paths)

        results: List[Optional[np.ndarray]] = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            tensors = []
            valid_idx = []

            for j, path in enumerate(batch_paths):
                try:
                    img = Image.open(path).convert("RGB")
                    tensors.append(self.transform(img))
                    valid_idx.append(j)
                except Exception as e:
                    print(f"  Skipping {path}: {e}")
                    results.append(None)

            if not tensors:
                continue

            batch_t = torch.stack(tensors).to(self.device)
            output = self.model(batch_t)

            if isinstance(output, dict):
                if 'x_norm_clstoken' in output:
                    feats = output['x_norm_clstoken']
                elif 'x' in output:
                    feats = output['x'][:, 0]
                else:
                    feats = output[list(output.keys())[0]][:, 0]
            else:
                if output.dim() == 3:
                    feats = output[:, 0]
                else:
                    feats = output

            feats = F.normalize(feats, p=2, dim=1)
            feats_np = feats.cpu().numpy().astype(np.float32)

            for vi in valid_idx:
                results.append(feats_np[vi])

        return results


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized vectors."""
    return float(np.dot(a, b))


def find_nearest(
    query_emb: np.ndarray,
    gallery: List[Tuple[str, np.ndarray]],
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """Find top-k nearest neighbors by cosine similarity.

    Args:
        query_emb: Query embedding of shape [D].
        gallery: List of (label, embedding) tuples.
        top_k: Number of results.

    Returns:
        List of (label, similarity) sorted descending.
    """
    if not gallery:
        return []

    labels = [g[0] for g in gallery]
    embeddings = np.stack([g[1] for g in gallery])  # [N, D]
    similarities = embeddings @ query_emb  # dot = cosine (both L2-normed)

    top_idx = np.argsort(similarities)[::-1][:top_k]
    return [(labels[i], float(similarities[i])) for i in top_idx]
