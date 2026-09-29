"""One-epoch training loop for torchvision detection models."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any

import torch
from torch import Tensor

EXPECTED_LOSS_KEYS = {
    "loss_classifier",
    "loss_box_reg",
    "loss_objectness",
    "loss_rpn_box_reg",
}
MIB = 1024**2


def _move_target(target: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in target.items()
    }


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device: torch.device | str,
    epoch: int | None = None,
    print_freq: int | None = None,
) -> dict[str, float | int | None]:
    """Train for exactly one pass over ``dataloader`` and return weighted averages."""
    device = torch.device(device)
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    weighted_sums: defaultdict[str, float] = defaultdict(float)
    sample_count = 0
    batch_count = 0

    for batch_index, (images, targets) in enumerate(dataloader):
        images = [image.to(device) for image in images]
        targets = [_move_target(target, device) for target in targets]
        batch_size = len(images)

        optimizer.zero_grad(set_to_none=True)
        loss_dict = model(images, targets)
        if not isinstance(loss_dict, dict):
            raise TypeError(f"Model must return a loss dictionary, got {type(loss_dict).__name__}")
        missing = EXPECTED_LOSS_KEYS - loss_dict.keys()
        if missing:
            raise RuntimeError(f"Missing Faster R-CNN loss keys: {sorted(missing)}")
        total_loss = sum(loss_dict.values())
        loss_values = {name: float(loss.detach().cpu()) for name, loss in loss_dict.items()}
        total_value = float(total_loss.detach().cpu())
        if not math.isfinite(total_value) or any(
            not math.isfinite(value) for value in loss_values.values()
        ):
            raise FloatingPointError(
                f"Non-finite loss at batch {batch_index}: "
                f"losses={loss_values}, total_loss={total_value}"
            )

        total_loss.backward()
        optimizer.step()

        for name, value in loss_values.items():
            weighted_sums[name] += value * batch_size
        weighted_sums["total_loss"] += total_value * batch_size
        sample_count += batch_size
        batch_count += 1
        if print_freq and (batch_index + 1) % print_freq == 0:
            print(
                f"epoch={epoch} batch={batch_index + 1} "
                f"samples={sample_count} total_loss={total_value:.6f}"
            )

    if sample_count == 0:
        raise ValueError("Training dataloader produced no samples")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result: dict[str, float | int | None] = {
        name: weighted_sums[name] / sample_count
        for name in (*sorted(EXPECTED_LOSS_KEYS), "total_loss")
    }
    result.update(
        {
            "epoch": epoch,
            "num_batches": batch_count,
            "num_samples": sample_count,
            "runtime_seconds": time.perf_counter() - started,
            "gpu_peak_allocated_mib": (
                torch.cuda.max_memory_allocated(device) / MIB if device.type == "cuda" else 0.0
            ),
            "gpu_peak_reserved_mib": (
                torch.cuda.max_memory_reserved(device) / MIB if device.type == "cuda" else 0.0
            ),
        }
    )
    return result
