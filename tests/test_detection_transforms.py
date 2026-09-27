from __future__ import annotations

import torch
from PIL import Image

from animal_detection.data import Compose, RandomHorizontalFlip, Resize, ToTensor


def target_with_boxes(boxes: list[list[float]], labels: list[int]) -> dict:
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "original_size": torch.tensor([300, 400], dtype=torch.int64),
        "size": torch.tensor([300, 400], dtype=torch.int64),
    }


def test_horizontal_flip_math_and_no_input_mutation() -> None:
    image = Image.new("RGB", (500, 200), color="white")
    target = target_with_boxes([[100, 40, 200, 140]], [3])
    original = target["boxes"].clone()

    flipped_image, flipped_target = RandomHorizontalFlip(p=1.0)(image, target)

    assert flipped_image.size == image.size
    assert torch.equal(target["boxes"], original)
    torch.testing.assert_close(
        flipped_target["boxes"], torch.tensor([[300.0, 40.0, 400.0, 140.0]])
    )


def test_resize_math_and_metadata() -> None:
    image = Image.new("RGB", (400, 300), color="white")
    target = target_with_boxes([[50, 60, 170, 180]], [4])

    resized_image, resized_target = Resize((600, 800))(image, target)

    assert resized_image.size == (800, 600)
    torch.testing.assert_close(
        resized_target["boxes"], torch.tensor([[100.0, 120.0, 340.0, 360.0]])
    )
    assert torch.equal(resized_target["size"], torch.tensor([600, 800]))
    assert torch.equal(resized_target["original_size"], torch.tensor([300, 400]))


def test_zero_object_resize_and_flip() -> None:
    image = Image.new("RGB", (400, 300), color="white")
    target = target_with_boxes([], [])
    transformed_image, transformed_target = Compose(
        [Resize((150, 200)), RandomHorizontalFlip(p=1.0), ToTensor()]
    )(image, target)

    assert transformed_image.shape == (3, 150, 200)
    assert transformed_target["boxes"].shape == (0, 4)
    assert transformed_target["labels"].shape == (0,)


def test_multi_object_resize_and_flip() -> None:
    image = Image.new("RGB", (100, 100), color="white")
    target = target_with_boxes([[0, 10, 20, 30], [40, 50, 90, 100]], [1, 2])
    _, transformed = Compose([Resize((200, 300)), RandomHorizontalFlip(p=1.0)])(
        image, target
    )
    torch.testing.assert_close(
        transformed["boxes"],
        torch.tensor([[240.0, 20.0, 300.0, 60.0], [30.0, 100.0, 180.0, 200.0]]),
    )


def test_compose_applies_transforms_in_declared_order() -> None:
    calls: list[str] = []

    class Recorder:
        def __init__(self, name: str) -> None:
            self.name = name

        def __call__(self, image, target):
            calls.append(self.name)
            return image, target

    image = Image.new("RGB", (10, 10))
    Compose([Recorder("resize"), Recorder("flip"), Recorder("tensor")])(image, {})
    assert calls == ["resize", "flip", "tensor"]
