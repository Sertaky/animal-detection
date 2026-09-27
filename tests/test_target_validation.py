from __future__ import annotations

import pytest
import torch

from animal_detection.data import (
    TargetValidationError,
    adapt_target_for_torchvision,
    validate_dataset_target,
    validate_model_target,
)


def valid_target() -> dict:
    return {
        "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]], dtype=torch.float32),
        "labels": torch.tensor([0], dtype=torch.int64),
    }


def test_dataset_and_model_targets_validate() -> None:
    target = valid_target()
    validate_dataset_target(target, (100, 100))
    validate_model_target(adapt_target_for_torchvision(target), (100, 100))


def test_zero_object_targets_validate() -> None:
    target = {
        "boxes": torch.empty((0, 4), dtype=torch.float32),
        "labels": torch.empty((0,), dtype=torch.int64),
    }
    validate_dataset_target(target, (100, 100))
    validate_model_target(adapt_target_for_torchvision(target), (100, 100))


@pytest.mark.parametrize(
    ("boxes", "message"),
    [
        ([[30.0, 20.0, 10.0, 40.0]], "x2 > x1"),
        ([[10.0, 40.0, 30.0, 20.0]], "y2 > y1"),
        ([[-2.0, 10.0, 20.0, 30.0]], "within image bounds"),
        ([[10.0, 20.0, float("nan"), 40.0]], "NaN or infinite"),
    ],
)
def test_validation_catches_malformed_geometry(boxes: list[list[float]], message: str) -> None:
    target = {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor([0], dtype=torch.int64),
    }
    with pytest.raises(TargetValidationError, match=message):
        validate_dataset_target(target, (100, 100))


def test_validation_catches_wrong_label_space() -> None:
    target = valid_target()
    with pytest.raises(TargetValidationError, match="model labels"):
        validate_model_target(target, (100, 100))
