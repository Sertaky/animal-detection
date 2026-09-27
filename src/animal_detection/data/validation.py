"""Reusable validation for dataset-facing and model-facing detection targets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import torch
from torch import Tensor

LabelSpace = Literal["dataset", "model"]
BOUNDARY_TOLERANCE = 1e-4


class TargetValidationError(ValueError):
    """Raised when a detection target violates shape, dtype, or geometry rules."""


def validate_detection_target(
    target: Mapping[str, Tensor | str],
    image_size: tuple[int, int],
    *,
    label_space: LabelSpace = "dataset",
) -> None:
    """Validate a target against an image size expressed as ``(height, width)``."""
    if label_space not in {"dataset", "model"}:
        raise ValueError(f"Unsupported label space: {label_space!r}")
    height, width = image_size
    if height <= 0 or width <= 0:
        raise ValueError(f"Image dimensions must be positive, got {image_size}")

    boxes = target.get("boxes")
    labels = target.get("labels")
    if not isinstance(boxes, Tensor):
        raise TargetValidationError("target['boxes'] must be a torch.Tensor")
    if not isinstance(labels, Tensor):
        raise TargetValidationError("target['labels'] must be a torch.Tensor")
    if boxes.ndim != 2 or boxes.shape[1:] != (4,):
        raise TargetValidationError(f"boxes must have shape (N, 4), got {tuple(boxes.shape)}")
    if labels.ndim != 1 or labels.shape[0] != boxes.shape[0]:
        raise TargetValidationError(
            f"labels must have shape ({boxes.shape[0]},), got {tuple(labels.shape)}"
        )
    if boxes.dtype != torch.float32:
        raise TargetValidationError(f"boxes must have dtype torch.float32, got {boxes.dtype}")
    if labels.dtype != torch.int64:
        raise TargetValidationError(f"labels must have dtype torch.int64, got {labels.dtype}")
    if not bool(torch.all(torch.isfinite(boxes))):
        raise TargetValidationError("boxes contain NaN or infinite values")

    if boxes.numel():
        if not bool(torch.all(boxes[:, 2] > boxes[:, 0])):
            raise TargetValidationError("every box must satisfy x2 > x1")
        if not bool(torch.all(boxes[:, 3] > boxes[:, 1])):
            raise TargetValidationError("every box must satisfy y2 > y1")
        if bool(
            torch.any(
                (boxes[:, 0] < -BOUNDARY_TOLERANCE)
                | (boxes[:, 1] < -BOUNDARY_TOLERANCE)
                | (boxes[:, 2] > width + BOUNDARY_TOLERANCE)
                | (boxes[:, 3] > height + BOUNDARY_TOLERANCE)
            )
        ):
            raise TargetValidationError(
                f"box coordinates must stay within image bounds width={width}, height={height}"
            )

    label_min, label_max = (0, 19) if label_space == "dataset" else (1, 20)
    if labels.numel() and bool(torch.any((labels < label_min) | (labels > label_max))):
        raise TargetValidationError(
            f"{label_space} labels must be in [{label_min}, {label_max}]"
        )

    for key, dtype in (("area", torch.float32), ("iscrowd", torch.int64)):
        value = target.get(key)
        if value is not None:
            if not isinstance(value, Tensor):
                raise TargetValidationError(f"target[{key!r}] must be a torch.Tensor")
            if value.shape != (boxes.shape[0],):
                raise TargetValidationError(
                    f"target[{key!r}] must have shape ({boxes.shape[0]},), got {tuple(value.shape)}"
                )
            if value.dtype != dtype:
                raise TargetValidationError(
                    f"target[{key!r}] must have dtype {dtype}, got {value.dtype}"
                )


def validate_dataset_target(
    target: Mapping[str, Tensor | str], image_size: tuple[int, int]
) -> None:
    """Validate a target whose animal labels use the raw dataset space 0..19."""
    validate_detection_target(target, image_size, label_space="dataset")


def validate_model_target(
    target: Mapping[str, Tensor | str], image_size: tuple[int, int]
) -> None:
    """Validate a target whose animal labels use torchvision space 1..20."""
    validate_detection_target(target, image_size, label_space="model")
