"""Faster R-CNN model construction for the animal detector."""

from __future__ import annotations

from typing import Any

from torchvision.models.detection import (
    FasterRCNN,
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

DEFAULT_FASTER_RCNN_WEIGHTS = FasterRCNN_ResNet50_FPN_Weights.DEFAULT


def build_faster_rcnn(
    num_classes: int = 21,
    pretrained: bool = True,
    **model_kwargs: Any,
) -> FasterRCNN:
    """Build Faster R-CNN ResNet50-FPN with a task-specific predictor.

    ``num_classes`` includes background: 20 animal foreground classes plus
    background means 21 outputs. When ``pretrained`` is false, both detector and
    backbone weights are disabled so construction never triggers a download.
    Extra keyword arguments are forwarded to torchvision's FasterRCNN builder.
    """
    if isinstance(num_classes, bool) or not isinstance(num_classes, int):
        raise TypeError("num_classes must be an integer including background")
    if num_classes < 2:
        raise ValueError("num_classes must include background and at least one foreground class")
    if not isinstance(pretrained, bool):
        raise TypeError("pretrained must be a boolean")
    if "weights" in model_kwargs or "weights_backbone" in model_kwargs:
        raise ValueError(
            "Use the pretrained argument; do not pass weights or weights_backbone directly"
        )

    weights = DEFAULT_FASTER_RCNN_WEIGHTS if pretrained else None
    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=None,
        **model_kwargs,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
