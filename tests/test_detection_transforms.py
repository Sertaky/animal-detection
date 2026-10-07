from __future__ import annotations

import torch
from PIL import Image
from unittest.mock import patch
import random

from animal_detection.data import (
    Compose, RandomHorizontalFlip, RandomScaleJitter512, Resize, ToTensor,
)


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


def jitter_target(boxes: list[list[float]], labels: list[int]) -> dict:
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "original_size": torch.tensor([100, 100], dtype=torch.int64),
        "size": torch.tensor([100, 100], dtype=torch.int64),
    }


def test_scale_one_matches_fixed_resize_512() -> None:
    image = Image.new("RGB", (100, 100), color="white")
    target = jitter_target([[10, 20, 30, 40]], [2])
    jitter_image, jittered, metadata = RandomScaleJitter512().apply_with_parameters(
        image, target, scale_factor=1.0, offset_x=0, offset_y=0
    )
    resize_image, resized = Resize((512, 512))(image, target)
    assert jitter_image.size == resize_image.size == (512, 512)
    torch.testing.assert_close(jittered["boxes"], resized["boxes"])
    assert metadata["operation"] == "resize"


def test_scale_point_eight_pads_and_shifts_boxes() -> None:
    image = Image.new("RGB", (100, 100), color="white")
    output, target, metadata = RandomScaleJitter512().apply_with_parameters(
        image, jitter_target([[10, 20, 30, 40]], [2]),
        scale_factor=0.8, offset_x=7, offset_y=11,
    )
    assert metadata["size_after_scaling"] == [410, 410]
    assert output.size == (512, 512)
    torch.testing.assert_close(
        target["boxes"], torch.tensor([[48.0, 93.0, 130.0, 175.0]])
    )


def test_scale_one_point_two_crops_shifts_and_clips() -> None:
    image = Image.new("RGB", (100, 100), color="white")
    output, target, metadata = RandomScaleJitter512().apply_with_parameters(
        image, jitter_target([[10, 20, 90, 100]], [4]),
        scale_factor=1.2, offset_x=50, offset_y=80,
    )
    assert metadata["size_after_scaling"] == [614, 614]
    assert output.size == (512, 512)
    torch.testing.assert_close(
        target["boxes"], torch.tensor([[11.4, 42.8, 502.6, 512.0]]), atol=1e-4, rtol=1e-5
    )
    assert metadata["boxes_clipped"] == 1


def test_crop_removes_only_fully_invisible_boxes_and_keeps_labels_aligned() -> None:
    image = Image.new("RGB", (100, 100), color="white")
    _, target, metadata = RandomScaleJitter512().apply_with_parameters(
        image,
        jitter_target([[0, 0, 10, 10], [10, 0, 30, 30]], [3, 8]),
        scale_factor=1.2, offset_x=102, offset_y=0,
    )
    assert metadata["boxes_removed"] == 1
    assert target["boxes"].shape == (1, 4)
    assert target["labels"].tolist() == [8]
    assert target["boxes"][0, 0] == 0
    assert target["boxes"][0, 2] > 0


def test_crop_can_return_correct_empty_target_shapes() -> None:
    image = Image.new("RGB", (100, 100), color="white")
    _, target, metadata = RandomScaleJitter512().apply_with_parameters(
        image, jitter_target([[0, 0, 10, 10]], [3]),
        scale_factor=1.2, offset_x=102, offset_y=102,
    )
    assert metadata["boxes_removed"] == 1
    assert target["boxes"].shape == (0, 4)
    assert target["boxes"].dtype == torch.float32
    assert target["labels"].shape == (0,)
    assert target["labels"].dtype == torch.int64


def test_flip_still_aligns_after_scale_jitter() -> None:
    image = Image.new("RGB", (100, 100), color="white")
    jitter = RandomScaleJitter512()
    image, target, _ = jitter.apply_with_parameters(
        image, jitter_target([[10, 20, 30, 40]], [2]),
        scale_factor=0.8, offset_x=7, offset_y=11,
    )
    _, target = RandomHorizontalFlip(p=1.0)(image, target)
    torch.testing.assert_close(
        target["boxes"], torch.tensor([[382.0, 93.0, 464.0, 175.0]])
    )


def test_scale_selection_uses_only_declared_factors() -> None:
    transform = RandomScaleJitter512()
    image = Image.new("RGB", (100, 100), color="white")
    target = jitter_target([], [])
    with patch("animal_detection.data.transforms.random.choice", return_value=1.0) as choice:
        transform(image, target)
    assert choice.call_args.args[0] == (0.8, 1.0, 1.2)


def test_scale_jitter_is_deterministic_with_fixed_seed_and_non_mutating() -> None:
    transform = RandomScaleJitter512()
    image = Image.new("RGB", (100, 100), color="white")
    target = jitter_target([[10, 20, 30, 40]], [2])
    original_boxes = target["boxes"].clone()
    random.seed(42)
    first_image, first = transform(image, target)
    random.seed(42)
    second_image, second = transform(image, target)
    assert first_image.tobytes() == second_image.tobytes()
    torch.testing.assert_close(first["boxes"], second["boxes"])
    torch.testing.assert_close(target["boxes"], original_boxes)
