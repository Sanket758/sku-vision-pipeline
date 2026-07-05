"""Curated-pipeline utility modules."""

from .detection import Detector
from .positional import (
    group_shelf_rows,
    normalize_boxes,
    positional_embedding,
    row_similarity,
)

__all__ = [
    "Detector",
    "group_shelf_rows",
    "normalize_boxes",
    "positional_embedding",
    "row_similarity",
]

