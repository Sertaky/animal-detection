#!/usr/bin/env python3
"""Run one Faster R-CNN train step and one inference pass as a smoke test."""

from __future__ import annotations

import argparse
import gc
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.optim import SGD
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from animal_detection.data import (
    YoloDetectionDataset,
    adapt_target_for_torchvision,
    detection_collate_fn,
    get_eval_transforms,
    get_train_transforms,
    validate_model_target,
)
from animal_detection.models import DEFAULT_FASTER_RCNN_WEIGHTS, build_faster_rcnn

EXPECTED_LOSS_KEYS = {
    "loss_classifier",
    "loss_box_reg",
    "loss_objectness",
    "loss_rpn_box_reg",
}
MIB = 1024**2


def move_target_to_device(target: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in target.items()
    }


def memory_mib(value: int) -> float:
    return value / MIB


def run_smoke(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    size = (args.image_size, args.image_size)

    train_dataset = YoloDetectionDataset(
        args.dataset_root,
        "train",
        transforms=get_train_transforms(
            size=size,
            horizontal_flip_probability=0.0,
        ),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=detection_collate_fn,
    )
    images, dataset_targets = next(iter(train_loader))
    images = [image.to(device) for image in images]
    targets = [
        move_target_to_device(adapt_target_for_torchvision(target), device)
        for target in dataset_targets
    ]
    for image, target in zip(images, targets):
        validate_model_target(target, (int(image.shape[-2]), int(image.shape[-1])))

    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")
    print(f"architecture=FasterRCNN ResNet50-FPN weights={DEFAULT_FASTER_RCNN_WEIGHTS}")
    print(f"num_classes=21 batch_size={len(images)} input_size={size}")

    model = build_faster_rcnn(
        num_classes=21,
        pretrained=True,
        min_size=args.image_size,
        max_size=args.image_size,
    ).to(device)
    optimizer = SGD(
        model.parameters(),
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005,
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_dict = model(images, targets)
    if not isinstance(loss_dict, dict):
        raise TypeError(f"Training forward must return a loss dict, got {type(loss_dict).__name__}")
    missing = EXPECTED_LOSS_KEYS - loss_dict.keys()
    if missing:
        raise RuntimeError(f"Missing expected loss keys: {sorted(missing)}")
    for name, loss in loss_dict.items():
        if loss.ndim != 0 or not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Loss {name} must be a finite scalar, got {loss}")
        print(f"loss.{name}={float(loss.detach().cpu()):.6f}")
    total_loss = sum(loss_dict.values())
    total_value = float(total_loss.detach().cpu())
    if not math.isfinite(total_value) or total_value <= 0:
        raise RuntimeError(f"Total loss must be finite and positive, got {total_value}")
    print(f"loss.total={total_value:.6f}")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        print(
            f"memory.after_forward_allocated_mib="
            f"{memory_mib(torch.cuda.memory_allocated(device)):.1f}"
        )
        print(
            f"memory.after_forward_reserved_mib="
            f"{memory_mib(torch.cuda.memory_reserved(device)):.1f}"
        )

    total_loss.backward()
    gradient_parameter_count = sum(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if gradient_parameter_count == 0:
        raise RuntimeError("Backward completed but no trainable parameter received a gradient")
    if any(
        not bool(torch.all(torch.isfinite(parameter.grad)))
        for parameter in model.parameters()
        if parameter.grad is not None
    ):
        raise RuntimeError("A trainable parameter has a non-finite gradient")
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        print(
            f"memory.peak_allocated_mib="
            f"{memory_mib(torch.cuda.max_memory_allocated(device)):.1f}"
        )
        print(
            f"memory.peak_reserved_mib="
            f"{memory_mib(torch.cuda.max_memory_reserved(device)):.1f}"
        )
    print(f"backward_succeeded=True gradient_parameter_count={gradient_parameter_count}")
    print("optimizer_step_succeeded=True steps=1")

    optimizer.zero_grad(set_to_none=True)
    del optimizer, total_loss, loss_dict, targets, images, dataset_targets
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    eval_dataset = YoloDetectionDataset(
        args.dataset_root,
        "valid",
        transforms=get_eval_transforms(size=size),
    )
    eval_image, _ = eval_dataset[0]
    eval_images = [eval_image.to(device)]
    model.eval()
    with torch.no_grad():
        outputs = model(eval_images)
    if not isinstance(outputs, list) or len(outputs) != 1 or not isinstance(outputs[0], dict):
        raise TypeError("Inference must return a list containing one output dictionary")
    output = outputs[0]
    required_keys = {"boxes", "labels", "scores"}
    if not required_keys <= output.keys():
        raise RuntimeError(f"Inference output lacks keys: {sorted(required_keys - output.keys())}")
    boxes, labels, scores = output["boxes"], output["labels"], output["scores"]
    count = int(labels.numel())
    if boxes.shape != (count, 4) or labels.shape != (count,) or scores.shape != (count,):
        raise RuntimeError(
            f"Unexpected inference shapes: boxes={tuple(boxes.shape)} "
            f"labels={tuple(labels.shape)} scores={tuple(scores.shape)}"
        )
    if not bool(torch.all(torch.isfinite(boxes))) or not bool(torch.all(torch.isfinite(scores))):
        raise RuntimeError("Inference boxes or scores contain NaN/inf")
    if count and bool(torch.any((labels < 1) | (labels > 20))):
        raise RuntimeError("Predicted foreground labels must be in model range 1..20")

    print(f"inference.detections={count}")
    print(
        f"inference.boxes_shape={tuple(boxes.shape)} dtype={boxes.dtype} "
        f"labels_shape={tuple(labels.shape)} dtype={labels.dtype} "
        f"scores_shape={tuple(scores.shape)} dtype={scores.dtype}"
    )
    if count:
        print(f"inference.label_range={int(labels.min())}..{int(labels.max())}")
        print(
            f"inference.score_range={float(scores.min()):.6f}.."
            f"{float(scores.max()):.6f}"
        )
    else:
        print("inference.label_range=empty")
        print("inference.score_range=empty")
    print("inference_succeeded=True")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Square smoke-test size; 512 is chosen for the 4 GB GPU baseline.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.image_size < 64:
        parser.error("--image-size must be at least 64")

    try:
        return run_smoke(args)
    except RuntimeError as exc:
        if args.device.startswith("cuda") and "out of memory" in str(exc).casefold():
            print(f"CUDA OOM during smoke test: {exc}", file=sys.stderr)
            if args.batch_size > 1:
                print("Retry with --batch-size 1.", file=sys.stderr)
            torch.cuda.empty_cache()
            return 2
        raise


if __name__ == "__main__":
    raise SystemExit(main())
