"""Explicit label mapping between dataset and torchvision detector spaces.

The immutable YOLO dataset uses animal class IDs 0..19. Torchvision detectors
reserve 0 for background, so their foreground animal labels are 1..20.
"""

from __future__ import annotations

import torch
from torch import Tensor

DATASET_LABEL_MIN = 0
DATASET_LABEL_MAX = 19
MODEL_BACKGROUND_LABEL = 0
MODEL_LABEL_MIN = 1
MODEL_LABEL_MAX = 20

ANIMAL_CLASS_NAMES = (
    "Buffalo",
    "Camel",
    "Cat",
    "Cheetah",
    "Cow",
    "Deer",
    "Dog",
    "Elephant",
    "Goat",
    "Gorilla",
    "Hippo",
    "Horse",
    "Lion",
    "Monkeys",
    "Panda",
    "Rat",
    "Rhino",
    "Tiger",
    "Wolf",
    "Zebra",
)

_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _validate_scalar(value: int, minimum: int, maximum: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}], got {value}")


def _validate_tensor(labels: Tensor, minimum: int, maximum: int, name: str) -> None:
    if labels.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"{name} tensor must have an integer dtype, got {labels.dtype}")
    if labels.numel() and bool(torch.any((labels < minimum) | (labels > maximum))):
        invalid = labels[(labels < minimum) | (labels > maximum)][0].item()
        raise ValueError(
            f"{name} values must be in [{minimum}, {maximum}], found {invalid}"
        )


def dataset_label_to_model_label(label: int | Tensor) -> int | Tensor:
    """Map dataset animal labels 0..19 to model foreground labels 1..20.

    Tensor inputs retain their dtype and device. The input tensor is not mutated.
    """
    if isinstance(label, Tensor):
        _validate_tensor(label, DATASET_LABEL_MIN, DATASET_LABEL_MAX, "dataset label")
        return label + 1
    _validate_scalar(label, DATASET_LABEL_MIN, DATASET_LABEL_MAX, "dataset label")
    return label + 1


def model_label_to_dataset_label(label: int | Tensor) -> int | Tensor:
    """Map model foreground labels 1..20 back to dataset animal labels 0..19.

    Model label 0 is background and therefore has no dataset animal equivalent.
    Tensor inputs retain their dtype and device. The input tensor is not mutated.
    """
    if isinstance(label, Tensor):
        _validate_tensor(label, MODEL_LABEL_MIN, MODEL_LABEL_MAX, "model label")
        return label - 1
    _validate_scalar(label, MODEL_LABEL_MIN, MODEL_LABEL_MAX, "model label")
    return label - 1
