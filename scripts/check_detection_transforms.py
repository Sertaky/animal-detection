#!/usr/bin/env python3
"""Numerically verify resize, horizontal flip, empty, and dense targets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from animal_detection.data import (
    Compose,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
    YoloDetectionDataset,
    adapt_target_for_torchvision,
    validate_dataset_target,
    validate_model_target,
)

EMPTY_STEM = "cow-19-_jpg.rf.79ab2f17459bf2b455ba932f730a35d2"
MULTI_STEM = "rat-38-_jpg.rf.445d47cfa6520071db240711501f1cf4"
RESIZED_SIZE = (320, 480)


def index_for_stem(dataset: YoloDetectionDataset, stem: str) -> int:
    return next(index for index, path in enumerate(dataset.image_paths) if path.stem == stem)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    args = parser.parse_args()

    dataset = YoloDetectionDataset(args.dataset_root, "train")
    normal_index = next(
        index
        for index, label_path in enumerate(dataset.label_paths)
        if label_path.read_text(encoding="utf-8-sig").strip()
    )
    image, target = dataset[normal_index]
    original_boxes = target["boxes"].clone()
    resized_image, resized_target = Resize(RESIZED_SIZE)(image, target)
    expected_resized = original_boxes * torch.tensor([0.75, 0.5, 0.75, 0.5])
    torch.testing.assert_close(resized_target["boxes"], expected_resized)

    flipped_image, flipped_target = RandomHorizontalFlip(p=1.0)(
        resized_image, resized_target
    )
    expected_flipped = expected_resized.clone()
    expected_flipped[:, 0] = RESIZED_SIZE[1] - expected_resized[:, 2]
    expected_flipped[:, 2] = RESIZED_SIZE[1] - expected_resized[:, 0]
    torch.testing.assert_close(flipped_target["boxes"], expected_flipped)
    validate_dataset_target(flipped_target, RESIZED_SIZE)

    empty_image, empty_target = dataset[index_for_stem(dataset, EMPTY_STEM)]
    empty_tensor, transformed_empty = Compose(
        [Resize(RESIZED_SIZE), RandomHorizontalFlip(p=1.0), ToTensor()]
    )(empty_image, empty_target)
    validate_dataset_target(transformed_empty, RESIZED_SIZE)
    assert transformed_empty["boxes"].shape == (0, 4)

    multi_image, multi_target = dataset[index_for_stem(dataset, MULTI_STEM)]
    multi_tensor, transformed_multi = Compose(
        [Resize(RESIZED_SIZE), RandomHorizontalFlip(p=1.0), ToTensor()]
    )(multi_image, multi_target)
    validate_dataset_target(transformed_multi, RESIZED_SIZE)
    model_target = adapt_target_for_torchvision(transformed_multi)
    validate_model_target(model_target, RESIZED_SIZE)
    assert torch.equal(multi_target["labels"], torch.full_like(multi_target["labels"], 15))
    assert torch.equal(model_target["labels"], torch.full_like(model_target["labels"], 16))

    print(f"normal_image: before={image.size[::-1]} after={flipped_image.size[::-1]}")
    print(f"normal_first_box_before={original_boxes[0].tolist()}")
    print(f"normal_first_box_resized={resized_target['boxes'][0].tolist()}")
    print(f"normal_first_box_flipped={flipped_target['boxes'][0].tolist()}")
    print(
        f"empty_sample: image_shape={tuple(empty_tensor.shape)} "
        f"boxes={tuple(transformed_empty['boxes'].shape)} "
        f"labels={tuple(transformed_empty['labels'].shape)}"
    )
    print(
        f"multi_sample: image_shape={tuple(multi_tensor.shape)} "
        f"objects={transformed_multi['labels'].numel()} "
        f"dataset_label_range={int(transformed_multi['labels'].min())}.."
        f"{int(transformed_multi['labels'].max())} "
        f"model_label_range={int(model_target['labels'].min())}.."
        f"{int(model_target['labels'].max())}"
    )
    print("all_transform_checks_passed=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
