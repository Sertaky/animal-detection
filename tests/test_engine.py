from __future__ import annotations

import torch
from torch import nn
from torch.optim import SGD

from animal_detection.engine import evaluate_map, train_one_epoch


class SyntheticLossModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, images, targets):
        base = self.weight.square()
        return {
            "loss_classifier": base,
            "loss_box_reg": base * 0.5,
            "loss_objectness": base * 0.25,
            "loss_rpn_box_reg": base * 0.125,
        }


class PerfectPredictionModel(nn.Module):
    def eval(self):
        super().eval()
        return self

    def forward(self, images):
        outputs = []
        for index, _ in enumerate(images):
            if index == 0:
                outputs.append(
                    {
                        "boxes": torch.tensor([[1.0, 1.0, 9.0, 9.0]]),
                        "labels": torch.tensor([1], dtype=torch.int64),
                        "scores": torch.tensor([0.99]),
                    }
                )
            else:
                outputs.append(
                    {
                        "boxes": torch.empty((0, 4), dtype=torch.float32),
                        "labels": torch.empty((0,), dtype=torch.int64),
                        "scores": torch.empty((0,), dtype=torch.float32),
                    }
                )
        return outputs


def test_train_one_epoch_returns_expected_loss_metrics() -> None:
    model = SyntheticLossModel()
    optimizer = SGD(model.parameters(), lr=0.01)
    target = {
        "boxes": torch.tensor([[1.0, 1.0, 9.0, 9.0]]),
        "labels": torch.tensor([1], dtype=torch.int64),
    }
    dataloader = [([torch.rand(3, 10, 10)], [target])]
    result = train_one_epoch(model, dataloader, optimizer, "cpu", epoch=1)
    assert {
        "loss_classifier",
        "loss_box_reg",
        "loss_objectness",
        "loss_rpn_box_reg",
        "total_loss",
    } <= result.keys()
    assert result["num_batches"] == 1
    assert result["num_samples"] == 1
    assert result["total_loss"] > 0


def test_evaluator_returns_map_and_accepts_zero_object_target() -> None:
    images = [torch.rand(3, 10, 10), torch.rand(3, 10, 10)]
    targets = [
        {
            "boxes": torch.tensor([[1.0, 1.0, 9.0, 9.0]]),
            "labels": torch.tensor([1], dtype=torch.int64),
        },
        {
            "boxes": torch.empty((0, 4), dtype=torch.float32),
            "labels": torch.empty((0,), dtype=torch.int64),
        },
    ]
    result = evaluate_map(
        PerfectPredictionModel(),
        [(images, targets)],
        "cpu",
        class_metrics=True,
    )
    assert result["map"] == 1.0
    assert result["map_50"] == 1.0
    assert result["num_images"] == 2
    assert result["per_class"][0]["class_name"] == "Buffalo"
