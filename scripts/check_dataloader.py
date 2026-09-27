#!/usr/bin/env python3
"""Load a few detection batches and report shapes, counts, and dtypes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from animal_detection.data import ToTensor, YoloDetectionDataset, detection_collate_fn


def representative_indices(dataset: YoloDetectionDataset, limit: int) -> list[int]:
    """Choose deterministic zero-, single-, and multi-object samples when present."""
    by_category: dict[str, int] = {}
    for index, label_path in enumerate(dataset.label_paths):
        count = sum(
            bool(row.strip())
            for row in label_path.read_text(encoding="utf-8-sig").splitlines()
        )
        category = "zero" if count == 0 else "single" if count == 1 else "multi"
        by_category.setdefault(category, index)

    selected = [
        by_category[category]
        for category in ("zero", "single", "multi")
        if category in by_category
    ]
    for index in range(len(dataset)):
        if len(selected) >= limit:
            break
        if index not in selected:
            selected.append(index)
    return selected[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="train")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-batches", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.num_batches < 1:
        parser.error("--num-batches must be positive")

    dataset = YoloDetectionDataset(args.dataset_root, args.split, transforms=ToTensor())
    sample_indices = representative_indices(dataset, args.batch_size * args.num_batches)
    loader = DataLoader(
        Subset(dataset, sample_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=detection_collate_fn,
    )
    print(
        f"split={args.split} dataset_length={len(dataset)} "
        f"sample_indices={sample_indices}"
    )

    observed_counts: set[int] = set()
    for batch_index, (images, targets) in enumerate(loader):
        if batch_index >= args.num_batches:
            break
        print(f"batch={batch_index} images={len(images)}")
        for item_index, (image, target) in enumerate(zip(images, targets)):
            boxes = target["boxes"]
            labels = target["labels"]
            if not isinstance(boxes, torch.Tensor) or not isinstance(labels, torch.Tensor):
                raise TypeError("Target boxes and labels must be tensors")
            object_count = int(labels.numel())
            observed_counts.add(object_count)
            label_range = (
                f"{int(labels.min().item())}..{int(labels.max().item())}"
                if object_count
                else "empty"
            )
            print(
                f"  item={item_index} image_shape={tuple(image.shape)} "
                f"objects={object_count} boxes_shape={tuple(boxes.shape)} "
                f"labels_shape={tuple(labels.shape)} label_range={label_range} "
                f"image_dtype={image.dtype} boxes_dtype={boxes.dtype} "
                f"labels_dtype={labels.dtype}"
            )
    if len(observed_counts) < 2:
        print("warning: checked batches happened to contain only one object-count value")
    else:
        print(f"variable_object_counts_verified={sorted(observed_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
