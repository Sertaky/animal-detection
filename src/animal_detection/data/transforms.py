"""Paired detection transforms that update images and xyxy boxes together."""

from __future__ import annotations

import random
import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image, ImageOps
from torch import Tensor

from .yolo_detection_dataset import DetectionTarget

PairedTransform = Callable[[Any, DetectionTarget], tuple[Any, DetectionTarget]]


def _image_size(image: Image.Image | Tensor) -> tuple[int, int]:
    """Return image dimensions as ``(height, width)``."""
    if isinstance(image, Image.Image):
        width, height = image.size
        return height, width
    if isinstance(image, Tensor) and image.ndim == 3:
        return int(image.shape[-2]), int(image.shape[-1])
    raise TypeError("Expected a Pillow image or a CHW torch.Tensor")


class Compose:
    """Apply paired image/target transforms in the declared order."""

    def __init__(self, transforms: Sequence[PairedTransform]) -> None:
        self.transforms = tuple(transforms)

    def __call__(self, image: Any, target: DetectionTarget) -> tuple[Any, DetectionTarget]:
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


class ToTensor:
    """Convert an RGB Pillow image from HWC uint8 to CHW float32 in [0, 1]."""

    def __call__(
        self, image: Image.Image, target: DetectionTarget
    ) -> tuple[Tensor, DetectionTarget]:
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected a Pillow image, got {type(image).__name__}")
        array = np.array(image, dtype=np.float32, copy=True)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"Expected an RGB HWC image, got shape {array.shape}")
        image_tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous().div_(255.0)
        return image_tensor, target


class RandomHorizontalFlip:
    """Randomly mirror an image and its xyxy boxes with probability ``p``."""

    def __init__(self, p: float = 0.5) -> None:
        if not 0 <= p <= 1:
            raise ValueError(f"Flip probability must be in [0, 1], got {p}")
        self.p = float(p)

    def __call__(
        self, image: Image.Image | Tensor, target: DetectionTarget
    ) -> tuple[Image.Image | Tensor, DetectionTarget]:
        if random.random() >= self.p:
            return image, target
        _, width = _image_size(image)
        if isinstance(image, Image.Image):
            flipped_image: Image.Image | Tensor = ImageOps.mirror(image)
        else:
            flipped_image = torch.flip(image, dims=(-1,))

        updated = dict(target)
        boxes = target["boxes"]
        if not isinstance(boxes, Tensor):
            raise TypeError("target['boxes'] must be a torch.Tensor")
        flipped_boxes = boxes.clone()
        if flipped_boxes.numel():
            old_x1 = boxes[:, 0].clone()
            old_x2 = boxes[:, 2].clone()
            flipped_boxes[:, 0] = width - old_x2
            flipped_boxes[:, 2] = width - old_x1
        updated["boxes"] = flipped_boxes
        return flipped_image, updated


class Resize:
    """Resize to ``(height, width)`` and scale xyxy boxes by the same factors."""

    def __init__(self, size: tuple[int, int]) -> None:
        if len(size) != 2 or any(isinstance(value, bool) or not isinstance(value, int) for value in size):
            raise TypeError("Resize size must be a pair of integers: (height, width)")
        if size[0] <= 0 or size[1] <= 0:
            raise ValueError(f"Resize dimensions must be positive, got {size}")
        self.size = size

    def __call__(
        self, image: Image.Image | Tensor, target: DetectionTarget
    ) -> tuple[Image.Image | Tensor, DetectionTarget]:
        old_height, old_width = _image_size(image)
        new_height, new_width = self.size
        if isinstance(image, Image.Image):
            resized_image: Image.Image | Tensor = image.resize(
                (new_width, new_height), resample=Image.Resampling.BILINEAR
            )
        else:
            if not image.is_floating_point():
                raise TypeError("Tensor resize expects a floating-point CHW tensor")
            resized_image = functional.interpolate(
                image.unsqueeze(0),
                size=(new_height, new_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        updated = dict(target)
        boxes = target["boxes"]
        if not isinstance(boxes, Tensor):
            raise TypeError("target['boxes'] must be a torch.Tensor")
        resized_boxes = boxes.clone()
        if resized_boxes.numel():
            resized_boxes[:, [0, 2]] *= new_width / old_width
            resized_boxes[:, [1, 3]] *= new_height / old_height
        updated["boxes"] = resized_boxes
        updated["size"] = torch.tensor(
            [new_height, new_width], dtype=torch.int64, device=boxes.device
        )
        return resized_image, updated


class RandomScaleJitter512:
    """Resize to a sampled scale, then randomly pad or crop to a fixed canvas.

    Scaled dimensions use deterministic half-up rounding:
    ``floor(output_dimension * scale + 0.5)``. Random choices use Python's
    process-wide ``random`` module so the project's existing seed controls scale,
    crop, and padding decisions.
    """

    def __init__(
        self,
        scale_factors: Sequence[float] = (0.8, 1.0, 1.2),
        output_size: tuple[int, int] = (512, 512),
    ) -> None:
        if not scale_factors or any(factor <= 0 for factor in scale_factors):
            raise ValueError("scale_factors must contain positive values")
        if len(output_size) != 2 or any(value <= 0 for value in output_size):
            raise ValueError("output_size must contain two positive dimensions")
        self.scale_factors = tuple(float(factor) for factor in scale_factors)
        self.output_size = tuple(int(value) for value in output_size)

    def scaled_size(self, scale_factor: float) -> tuple[int, int]:
        height, width = self.output_size
        return (
            math.floor(height * scale_factor + 0.5),
            math.floor(width * scale_factor + 0.5),
        )

    def __call__(
        self, image: Image.Image, target: DetectionTarget
    ) -> tuple[Image.Image, DetectionTarget]:
        scale_factor = random.choice(self.scale_factors)
        scaled_height, scaled_width = self.scaled_size(scale_factor)
        output_height, output_width = self.output_size
        offset_x = random.randint(0, abs(output_width - scaled_width))
        offset_y = random.randint(0, abs(output_height - scaled_height))
        transformed_image, transformed_target, _ = self.apply_with_parameters(
            image, target, scale_factor=scale_factor,
            offset_x=offset_x, offset_y=offset_y,
        )
        return transformed_image, transformed_target

    def apply_with_parameters(
        self,
        image: Image.Image,
        target: DetectionTarget,
        *,
        scale_factor: float,
        offset_x: int,
        offset_y: int,
    ) -> tuple[Image.Image, DetectionTarget, dict[str, Any]]:
        """Apply explicit scale/offset parameters and return diagnostic metadata."""
        if not isinstance(image, Image.Image):
            raise TypeError("RandomScaleJitter512 expects a Pillow image before ToTensor")
        if scale_factor not in self.scale_factors:
            raise ValueError(f"scale_factor must be one of {self.scale_factors}")
        boxes_value = target.get("boxes")
        labels_value = target.get("labels")
        if not isinstance(boxes_value, Tensor) or not isinstance(labels_value, Tensor):
            raise TypeError("target boxes and labels must be tensors")
        boxes = boxes_value.clone().to(dtype=torch.float32)
        labels = labels_value.clone()
        if boxes.shape != (labels.shape[0], 4):
            raise ValueError("boxes must have shape (N,4) aligned with labels")

        old_height, old_width = _image_size(image)
        scaled_height, scaled_width = self.scaled_size(scale_factor)
        output_height, output_width = self.output_size
        max_x = abs(output_width - scaled_width)
        max_y = abs(output_height - scaled_height)
        if not 0 <= offset_x <= max_x or not 0 <= offset_y <= max_y:
            raise ValueError("offset is outside the valid padding/crop range")

        resized = image.resize((scaled_width, scaled_height), Image.Resampling.BILINEAR)
        if boxes.numel():
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * (scaled_width / old_width)
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * (scaled_height / old_height)

        clipped_mask = torch.zeros((boxes.shape[0],), dtype=torch.bool)
        if scaled_width < output_width or scaled_height < output_height:
            canvas = Image.new(image.mode, (output_width, output_height), color=0)
            canvas.paste(resized, (offset_x, offset_y))
            boxes[:, [0, 2]] = boxes[:, [0, 2]] + offset_x
            boxes[:, [1, 3]] = boxes[:, [1, 3]] + offset_y
            operation = "padding"
            output_image = canvas
        elif scaled_width > output_width or scaled_height > output_height:
            before_clip = boxes.clone()
            boxes[:, [0, 2]] = boxes[:, [0, 2]] - offset_x
            boxes[:, [1, 3]] = boxes[:, [1, 3]] - offset_y
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, output_width)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, output_height)
            shifted_unclipped = before_clip.clone()
            shifted_unclipped[:, [0, 2]] -= offset_x
            shifted_unclipped[:, [1, 3]] -= offset_y
            clipped_mask = torch.any(boxes != shifted_unclipped, dim=1)
            output_image = resized.crop(
                (offset_x, offset_y, offset_x + output_width, offset_y + output_height)
            )
            operation = "crop"
        else:
            if offset_x or offset_y:
                raise ValueError("scale 1.0 requires zero offsets")
            output_image = resized
            operation = "resize"

        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        updated = dict(target)
        updated["boxes"] = boxes[keep].reshape(-1, 4).to(dtype=torch.float32)
        updated["labels"] = labels[keep].reshape(-1).to(dtype=torch.int64)
        for key in ("area", "iscrowd"):
            value = target.get(key)
            if isinstance(value, Tensor) and value.ndim >= 1 and value.shape[0] == keep.shape[0]:
                updated[key] = value.clone()[keep]
        if "area" in updated:
            kept_boxes = updated["boxes"]
            updated["area"] = (
                (kept_boxes[:, 2] - kept_boxes[:, 0])
                * (kept_boxes[:, 3] - kept_boxes[:, 1])
            ).to(dtype=torch.float32)
        updated["size"] = torch.tensor(
            [output_height, output_width], dtype=torch.int64, device=boxes.device
        )
        metadata = {
            "scale_factor": scale_factor,
            "rounding_rule": "floor(output_dimension * scale + 0.5)",
            "size_before": [old_height, old_width],
            "size_after_scaling": [scaled_height, scaled_width],
            "final_canvas_size": [output_height, output_width],
            "operation": operation,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "boxes_before": boxes_value.tolist(),
            "boxes_after": updated["boxes"].tolist(),
            "objects_before": int(boxes_value.shape[0]),
            "objects_after": int(updated["boxes"].shape[0]),
            "boxes_clipped": int(clipped_mask.sum()),
            "boxes_removed": int((~keep).sum()),
        }
        return output_image, updated, metadata
