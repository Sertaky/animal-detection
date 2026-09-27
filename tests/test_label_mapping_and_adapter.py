from __future__ import annotations

import pytest
import torch

from animal_detection.data import (
    adapt_target_for_torchvision,
    dataset_label_to_model_label,
    model_label_to_dataset_label,
)


def test_scalar_label_mapping_boundaries() -> None:
    assert dataset_label_to_model_label(0) == 1
    assert dataset_label_to_model_label(19) == 20
    assert model_label_to_dataset_label(1) == 0
    assert model_label_to_dataset_label(20) == 19


def test_tensor_label_mapping_preserves_dtype_and_input() -> None:
    dataset_labels = torch.tensor([0, 7, 19], dtype=torch.int64)
    model_labels = dataset_label_to_model_label(dataset_labels)
    assert model_labels.dtype == dataset_labels.dtype
    assert torch.equal(model_labels, torch.tensor([1, 8, 20]))
    assert torch.equal(dataset_labels, torch.tensor([0, 7, 19]))
    assert torch.equal(model_label_to_dataset_label(model_labels), dataset_labels)


@pytest.mark.parametrize("label", [-1, 20])
def test_invalid_dataset_scalar_labels_fail(label: int) -> None:
    with pytest.raises(ValueError, match="dataset label"):
        dataset_label_to_model_label(label)


@pytest.mark.parametrize("label", [0, 21])
def test_background_or_invalid_model_scalar_labels_fail(label: int) -> None:
    with pytest.raises(ValueError, match="model label"):
        model_label_to_dataset_label(label)


def test_non_integer_mapping_inputs_fail() -> None:
    with pytest.raises(TypeError, match="integer"):
        dataset_label_to_model_label(torch.tensor([0.0]))
    with pytest.raises(TypeError, match="integer"):
        model_label_to_dataset_label(1.0)  # type: ignore[arg-type]


def test_target_adapter_does_not_mutate_and_adds_standard_fields() -> None:
    target = {
        "boxes": torch.tensor([[10.0, 20.0, 30.0, 50.0], [0.0, 0.0, 4.0, 5.0]]),
        "labels": torch.tensor([0, 19], dtype=torch.int64),
        "image_id": torch.tensor(4, dtype=torch.int64),
        "image_path": "example.jpg",
        "original_size": torch.tensor([100, 200], dtype=torch.int64),
        "size": torch.tensor([100, 200], dtype=torch.int64),
    }
    original_boxes = target["boxes"].clone()
    original_labels = target["labels"].clone()

    adapted = adapt_target_for_torchvision(target)

    assert torch.equal(target["boxes"], original_boxes)
    assert torch.equal(target["labels"], original_labels)
    assert adapted["boxes"].data_ptr() != target["boxes"].data_ptr()
    assert torch.equal(adapted["boxes"], original_boxes)
    assert torch.equal(adapted["labels"], torch.tensor([1, 20]))
    assert torch.equal(adapted["area"], torch.tensor([600.0, 20.0]))
    assert adapted["area"].dtype == torch.float32
    assert torch.equal(adapted["iscrowd"], torch.zeros(2, dtype=torch.int64))
    assert adapted["image_path"] == target["image_path"]
    assert torch.equal(adapted["image_id"], target["image_id"])


def test_target_adapter_handles_empty_target() -> None:
    target = {
        "boxes": torch.empty((0, 4), dtype=torch.float32),
        "labels": torch.empty((0,), dtype=torch.int64),
    }
    adapted = adapt_target_for_torchvision(target)
    assert adapted["boxes"].shape == (0, 4)
    assert adapted["labels"].shape == (0,)
    assert adapted["area"].shape == (0,)
    assert adapted["area"].dtype == torch.float32
    assert adapted["iscrowd"].shape == (0,)
    assert adapted["iscrowd"].dtype == torch.int64
