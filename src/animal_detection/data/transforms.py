"""Minimal transforms that update an image/target pair."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from .yolo_detection_dataset import DetectionTarget


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
