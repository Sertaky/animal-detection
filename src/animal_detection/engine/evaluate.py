"""COCO-style validation metrics for torchvision detection models."""

from __future__ import annotations

import time
from typing import Any

import torch
from torch import Tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from animal_detection.data.label_mapping import (
    ANIMAL_CLASS_NAMES,
    model_label_to_dataset_label,
)


def _cpu_prediction(output: dict[str, Tensor]) -> dict[str, Tensor]:
    return {
        key: output[key].detach().cpu()
        for key in ("boxes", "scores", "labels")
    }


def _cpu_ground_truth(target: dict[str, Any]) -> dict[str, Tensor]:
    boxes, labels = target["boxes"], target["labels"]
    if not isinstance(boxes, Tensor) or not isinstance(labels, Tensor):
        raise TypeError("Ground-truth boxes and labels must be tensors")
    return {"boxes": boxes.detach().cpu(), "labels": labels.detach().cpu()}


def _scalar(result: dict[str, Tensor], key: str) -> float | None:
    value = result.get(key)
    return float(value.item()) if isinstance(value, Tensor) and value.numel() == 1 else None


def evaluate_map(
    model,
    dataloader,
    device: torch.device | str,
    class_metrics: bool = True,
) -> dict[str, Any]:
    """Evaluate validation predictions with TorchMetrics MeanAveragePrecision."""
    device = torch.device(device)
    model.eval()
    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=class_metrics,
        backend="pycocotools",
    )
    started = time.perf_counter()
    image_count = 0
    batch_count = 0

    with torch.no_grad():
        for images, targets in dataloader:
            device_images = [image.to(device) for image in images]
            outputs = model(device_images)
            if not isinstance(outputs, list) or len(outputs) != len(device_images):
                raise TypeError("Inference must return one prediction dictionary per image")
            metric.update(
                [_cpu_prediction(output) for output in outputs],
                [_cpu_ground_truth(target) for target in targets],
            )
            image_count += len(device_images)
            batch_count += 1

    if image_count == 0:
        raise ValueError("Validation dataloader produced no samples")
    result = metric.compute()
    summary: dict[str, Any] = {
        "map": _scalar(result, "map"),
        "map_50": _scalar(result, "map_50"),
        "map_75": _scalar(result, "map_75"),
        "mar_100": _scalar(result, "mar_100"),
        "num_images": image_count,
        "num_batches": batch_count,
        "runtime_seconds": time.perf_counter() - started,
        "per_class": [],
    }

    if class_metrics:
        classes = result.get("classes", torch.empty(0, dtype=torch.int64)).reshape(-1)
        per_class_ap = result.get("map_per_class", torch.empty(0)).reshape(-1)
        per_class_ar = result.get("mar_100_per_class", torch.empty(0)).reshape(-1)
        for index, model_label_tensor in enumerate(classes):
            model_label = int(model_label_tensor.item())
            if not 1 <= model_label <= 20:
                continue
            dataset_label = int(model_label_to_dataset_label(model_label))
            summary["per_class"].append(
                {
                    "model_label": model_label,
                    "dataset_label": dataset_label,
                    "class_name": ANIMAL_CLASS_NAMES[dataset_label],
                    "ap": float(per_class_ap[index].item()),
                    "ar_100": float(per_class_ar[index].item()),
                }
            )
    return summary
