"""Batch collation helpers for variable-size detection targets."""

from __future__ import annotations

from typing import Any

from .yolo_detection_dataset import DetectionTarget


def detection_collate_fn(
    batch: list[tuple[Any, DetectionTarget]],
) -> tuple[tuple[Any, ...], tuple[DetectionTarget, ...]]:
    """Keep images and targets as sequences instead of stacking target tensors."""
    if not batch:
        return (), ()
    images, targets = zip(*batch)
    return tuple(images), tuple(targets)
