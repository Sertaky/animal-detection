from __future__ import annotations

import torch

from animal_detection.data import detection_collate_fn


def test_collate_keeps_variable_length_targets_separate() -> None:
    image_a = torch.zeros((3, 8, 8), dtype=torch.float32)
    image_b = torch.ones((3, 8, 8), dtype=torch.float32)
    target_a = {
        "boxes": torch.zeros((1, 4), dtype=torch.float32),
        "labels": torch.zeros((1,), dtype=torch.int64),
    }
    target_b = {
        "boxes": torch.zeros((3, 4), dtype=torch.float32),
        "labels": torch.tensor([1, 2, 3], dtype=torch.int64),
    }

    images, targets = detection_collate_fn(
        [(image_a, target_a), (image_b, target_b)]
    )

    assert isinstance(images, tuple)
    assert isinstance(targets, tuple)
    assert len(images) == len(targets) == 2
    assert targets[0]["boxes"].shape == (1, 4)
    assert targets[1]["boxes"].shape == (3, 4)


def test_collate_accepts_empty_batch() -> None:
    assert detection_collate_fn([]) == ((), ())
