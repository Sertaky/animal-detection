"""Paired detection transforms that update images and xyxy boxes together."""

from __future__ import annotations

import random
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
