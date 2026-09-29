from __future__ import annotations

import pytest
import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from animal_detection.models import build_faster_rcnn


@pytest.fixture(scope="module")
def tiny_model() -> FasterRCNN:
    torch.manual_seed(0)
    return build_faster_rcnn(
        num_classes=21,
        pretrained=False,
        min_size=64,
        max_size=64,
        rpn_pre_nms_top_n_train=40,
        rpn_post_nms_top_n_train=20,
        rpn_pre_nms_top_n_test=40,
        rpn_post_nms_top_n_test=20,
        box_detections_per_img=10,
    )


def test_builder_returns_faster_rcnn_with_21_class_predictor(
    tiny_model: FasterRCNN,
) -> None:
    assert isinstance(tiny_model, FasterRCNN)
    predictor = tiny_model.roi_heads.box_predictor
    assert isinstance(predictor, FastRCNNPredictor)
    assert predictor.cls_score.out_features == 21
    assert predictor.bbox_pred.out_features == 21 * 4


@pytest.mark.parametrize("num_classes", [0, 1, -2])
def test_invalid_num_classes_fails(num_classes: int) -> None:
    with pytest.raises(ValueError, match="background"):
        build_faster_rcnn(num_classes=num_classes, pretrained=False)


def test_training_forward_returns_expected_losses(tiny_model: FasterRCNN) -> None:
    tiny_model.train()
    image = torch.rand((3, 64, 64), dtype=torch.float32)
    target = {
        "boxes": torch.tensor([[8.0, 8.0, 48.0, 48.0]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
    }
    losses = tiny_model([image], [target])
    assert {
        "loss_classifier",
        "loss_box_reg",
        "loss_objectness",
        "loss_rpn_box_reg",
    } <= losses.keys()
    assert all(loss.ndim == 0 and torch.isfinite(loss) for loss in losses.values())


def test_inference_output_structure(tiny_model: FasterRCNN) -> None:
    tiny_model.eval()
    image = torch.rand((3, 64, 64), dtype=torch.float32)
    with torch.no_grad():
        outputs = tiny_model([image])
    assert isinstance(outputs, list) and len(outputs) == 1
    output = outputs[0]
    assert {"boxes", "labels", "scores"} <= output.keys()
    count = output["labels"].numel()
    assert output["boxes"].shape == (count, 4)
    assert output["labels"].shape == (count,)
    assert output["scores"].shape == (count,)
    assert output["boxes"].dtype == torch.float32
    assert output["labels"].dtype == torch.int64
    assert output["scores"].dtype == torch.float32
