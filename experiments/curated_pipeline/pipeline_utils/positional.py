"""Spatial encoding and shelf-row grouping heuristic."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN


def normalize_boxes(boxes: np.ndarray, img_size: tuple[int, int]) -> np.ndarray:
    """Convert absolute [x1, y1, x2, y2] boxes to normalized [x_c, y_c, w, h].

    Parameters
    ----------
    boxes:
        Array of shape (N, 4) with absolute pixel coordinates [x1, y1, x2, y2].
    img_size:
        (width, height) of the image in pixels.

    Returns
    -------
    Normalized array of shape (N, 4) with [x_center, y_center, width, height]
    in [0, 1] range.
    """
    img_w, img_h = img_size
    boxes = np.asarray(boxes, dtype=np.float64)

    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"Expected boxes of shape (N, 4), got {boxes.shape}")

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    x_center = ((x1 + x2) / 2.0) / img_w
    y_center = ((y1 + y2) / 2.0) / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h

    # Clamp to [0, 1] to guard against rounding / annotation errors
    x_center = np.clip(x_center, 0.0, 1.0)
    y_center = np.clip(y_center, 0.0, 1.0)
    width = np.clip(width, 0.0, 1.0)
    height = np.clip(height, 0.0, 1.0)

    return np.column_stack([x_center, y_center, width, height])


def group_shelf_rows(boxes_norm: np.ndarray, eps: float = 0.05) -> list[int]:
    """Cluster boxes into shelf rows via DBSCAN on normalized y-center.

    Parameters
    ----------
    boxes_norm:
        Array of shape (N, 4) from :func:`normalize_boxes`.
    eps:
        DBSCAN neighbourhood radius on the y-center axis.

    Returns
    -------
    List of length N with row IDs.  Noise points (singletons) are assigned
    their own unique negative IDs so they never match any real row.
    """
    boxes_norm = np.asarray(boxes_norm, dtype=np.float64)
    if boxes_norm.ndim != 2 or boxes_norm.shape[1] != 4:
        raise ValueError(f"Expected boxes_norm of shape (N, 4), got {boxes_norm.shape}")

    if len(boxes_norm) == 0:
        return []

    y_centers = boxes_norm[:, 1].reshape(-1, 1)

    db = DBSCAN(eps=eps, min_samples=1, metric="euclidean")
    labels = db.fit_predict(y_centers)

    # Noise points get unique negative IDs so they never collide with real rows
    row_ids: list[int] = []
    next_noise = -1
    for label in labels:
        if label == -1:
            row_ids.append(next_noise)
            next_noise -= 1
        else:
            row_ids.append(int(label))

    return row_ids


def row_similarity(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Return a positional bonus if two boxes are on the same shelf row and close in x.

    Parameters
    ----------
    box_a, box_b:
        Normalized boxes of shape (4,) – [x_center, y_center, width, height].

    Returns
    -------
    0.10 if same row (|y_center_a - y_center_b| < 0.05) **and**
    x-proximate (|x_center_a - x_center_b| < 0.3), otherwise 0.0.
    """
    box_a = np.asarray(box_a, dtype=np.float64)
    box_b = np.asarray(box_b, dtype=np.float64)

    if box_a.shape != (4,) or box_b.shape != (4,):
        raise ValueError("box_a and box_b must each be 1-D arrays of length 4")

    y_diff = abs(box_a[1] - box_b[1])
    x_diff = abs(box_a[0] - box_b[0])

    if y_diff < 0.05 and x_diff < 0.3:
        return 0.10
    return 0.0


def positional_embedding(box_norm: np.ndarray) -> np.ndarray:
    """Return a 4-dimensional positional feature vector.

    Parameters
    ----------
    box_norm:
        Normalized box of shape (4,) – [x_center, y_center, width, height].

    Returns
    -------
    Array of shape (4,) with the normalized positional features.
    """
    box_norm = np.asarray(box_norm, dtype=np.float64)
    if box_norm.shape != (4,):
        raise ValueError(f"Expected box_norm of shape (4,), got {box_norm.shape}")

    return box_norm.copy()
