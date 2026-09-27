"""Minimal transform presets for pre-training data verification."""

from __future__ import annotations

from .transforms import Compose, RandomHorizontalFlip, Resize, ToTensor

DEFAULT_IMAGE_SIZE = (640, 640)


def get_train_transforms(
    size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    *,
    horizontal_flip_probability: float = 0.5,
) -> Compose:
    """Resize, lightly augment, and tensorize a training sample.

    No ImageNet normalization is applied because torchvision detection models
    generally perform their own input normalization internally.
    """
    return Compose(
        [Resize(size), RandomHorizontalFlip(horizontal_flip_probability), ToTensor()]
    )


def get_eval_transforms(size: tuple[int, int] = DEFAULT_IMAGE_SIZE) -> Compose:
    """Deterministically resize and tensorize an evaluation sample."""
    return Compose([Resize(size), ToTensor()])
