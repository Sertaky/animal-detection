from __future__ import annotations

import json

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import Dataset

from animal_detection.data import TorchvisionDetectionDataset
from animal_detection.utils import (
    append_training_history,
    load_checkpoint,
    save_checkpoint,
    write_training_history,
)


class SyntheticDataset(Dataset):
    def __init__(self, empty: bool = False) -> None:
        self.empty = empty

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        return torch.zeros((3, 10, 10)), {
            "boxes": (
                torch.empty((0, 4), dtype=torch.float32)
                if self.empty
                else torch.tensor([[1.0, 1.0, 4.0, 4.0], [5.0, 5.0, 9.0, 9.0]])
            ),
            "labels": (
                torch.empty((0,), dtype=torch.int64)
                if self.empty
                else torch.tensor([0, 19], dtype=torch.int64)
            ),
            "image_id": torch.tensor(index),
        }


def test_wrapper_maps_labels_and_preserves_empty_target() -> None:
    _, target = TorchvisionDetectionDataset(SyntheticDataset())[0]
    assert torch.equal(target["labels"], torch.tensor([1, 20]))
    assert target["area"].shape == (2,)
    _, empty = TorchvisionDetectionDataset(SyntheticDataset(empty=True))[0]
    assert empty["boxes"].shape == (0, 4)
    assert empty["labels"].shape == (0,)
    assert empty["area"].shape == (0,)


def test_checkpoint_round_trip(tmp_path) -> None:
    model = nn.Linear(2, 1)
    optimizer = SGD(model.parameters(), lr=0.1)
    destination = tmp_path / "checkpoint.pt"
    save_checkpoint(
        destination,
        epoch=3,
        model=model,
        optimizer=optimizer,
        validation_metrics={"map": 0.25},
        training_metrics={"total_loss": 1.5},
        configuration={"batch_size": 1},
        class_mapping={"1": {"class_name": "Buffalo"}},
    )
    loaded = load_checkpoint(destination)
    assert loaded["epoch"] == 3
    assert loaded["validation_metrics"]["map"] == 0.25
    assert loaded["training_metrics"]["total_loss"] == 1.5
    assert "model_state_dict" in loaded
    assert "optimizer_state_dict" in loaded
    assert "saved_at_utc" in loaded


def test_training_history_write_and_append(tmp_path) -> None:
    path = tmp_path / "history.json"
    write_training_history(path, [{"epoch": 1, "loss": 2.0}])
    history = append_training_history(path, {"epoch": 2, "loss": 1.0})
    assert history == [{"epoch": 1, "loss": 2.0}, {"epoch": 2, "loss": 1.0}]
    assert json.loads(path.read_text(encoding="utf-8")) == history
