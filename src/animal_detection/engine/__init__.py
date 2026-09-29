"""Reusable training and validation loops for object detection."""

from .evaluate import evaluate_map
from .train import train_one_epoch

__all__ = ["evaluate_map", "train_one_epoch"]
