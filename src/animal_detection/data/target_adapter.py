"""Adapters from dataset-facing targets to torchvision detector targets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .label_mapping import dataset_label_to_model_label
from .yolo_detection_dataset import DetectionTarget


def adapt_target_for_torchvision(
    target: Mapping[str, Tensor | str],
) -> DetectionTarget:
    """Return a copied target with foreground labels shifted from 0..19 to 1..20.

    Boxes and existing tensor metadata are cloned. ``area`` and ``iscrowd`` are
    calculated in torchvision's expected dtypes. The input target is not mutated.
    """
    boxes = target.get("boxes")
    labels = target.get("labels")
    if not isinstance(boxes, Tensor):
        raise TypeError("target['boxes'] must be a torch.Tensor")
    if not isinstance(labels, Tensor):
        raise TypeError("target['labels'] must be a torch.Tensor")
    if boxes.ndim != 2 or boxes.shape[1:] != (4,):
        raise ValueError(f"target boxes must have shape (N, 4), got {tuple(boxes.shape)}")
    if labels.ndim != 1 or labels.shape[0] != boxes.shape[0]:
        raise ValueError(
            "target labels must have shape (N,) matching boxes; "
            f"got boxes={tuple(boxes.shape)} labels={tuple(labels.shape)}"
        )

    adapted: DetectionTarget = {
        key: value.clone() if isinstance(value, Tensor) else value
        for key, value in target.items()
    }
    adapted_boxes = boxes.clone()
    adapted["boxes"] = adapted_boxes
    adapted["labels"] = dataset_label_to_model_label(labels)
    adapted["area"] = (
        (adapted_boxes[:, 2] - adapted_boxes[:, 0])
        * (adapted_boxes[:, 3] - adapted_boxes[:, 1])
    ).to(dtype=torch.float32)
    adapted["iscrowd"] = torch.zeros(
        (adapted_boxes.shape[0],), dtype=torch.int64, device=adapted_boxes.device
    )
    return adapted


class TorchvisionDetectionDataset(Dataset):
    """Wrap a detection dataset and explicitly adapt each target for torchvision."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> tuple[Any, DetectionTarget]:
        image, target = self.dataset[index]
        return image, adapt_target_for_torchvision(target)
