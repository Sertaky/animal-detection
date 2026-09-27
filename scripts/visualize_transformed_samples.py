#!/usr/bin/env python3
"""Render deterministic samples after paired detection transforms."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from animal_detection.data import (
    YoloDetectionDataset,
    adapt_target_for_torchvision,
    get_eval_transforms,
    get_train_transforms,
    model_label_to_dataset_label,
    validate_dataset_target,
    validate_model_target,
)

COLORS = ["#ef4444", "#22c55e", "#3b82f6", "#eab308", "#a855f7", "#06b6d4"]


def load_names(path: Path) -> dict[int, str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    names = config.get("names")
    if isinstance(names, list):
        return dict(enumerate(map(str, names)))
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    raise ValueError("data.yaml does not contain a valid names list or mapping")


def select_indices(dataset: YoloDetectionDataset, count: int, seed: int) -> list[int]:
    if count > len(dataset):
        raise ValueError(f"Requested {count} samples from a dataset of {len(dataset)}")
    empty = [
        index
        for index, path in enumerate(dataset.label_paths)
        if not path.read_text(encoding="utf-8-sig").strip()
    ]
    selected = empty[:1]
    pool = [index for index in range(len(dataset)) if index not in selected]
    selected.extend(random.Random(seed).sample(pool, count - len(selected)))
    return selected


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = (
        image.detach().cpu().clamp(0, 1).mul(255).round().to(torch.uint8)
        .permute(1, 2, 0).numpy()
    )
    return Image.fromarray(array, mode="RGB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="train")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/transformed_samples"))
    parser.add_argument("--preset", choices=("auto", "train", "eval"), default="auto")
    parser.add_argument("--model-facing-labels", action="store_true")
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("--num-samples must be positive")

    root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.resolve()
    if output_dir == root or root in output_dir.parents:
        parser.error("Output directory must be outside the immutable raw dataset")
    preset = args.preset if args.preset != "auto" else ("train" if args.split == "train" else "eval")
    transforms = get_train_transforms() if preset == "train" else get_eval_transforms()
    dataset = YoloDetectionDataset(root, args.split, transforms=transforms)
    indices = select_indices(dataset, args.num_samples, args.seed)
    names = load_names(root / "data.yaml")
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    for output_index, dataset_index in enumerate(indices, 1):
        image_tensor, target = dataset[dataset_index]
        image_size = (int(image_tensor.shape[-2]), int(image_tensor.shape[-1]))
        validate_dataset_target(target, image_size)
        draw_target = adapt_target_for_torchvision(target) if args.model_facing_labels else target
        if args.model_facing_labels:
            validate_model_target(draw_target, image_size)

        image = tensor_to_pil(image_tensor)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=16)
        boxes = draw_target["boxes"]
        labels = draw_target["labels"]
        for box, label_value in zip(boxes.tolist(), labels.tolist()):
            raw_label = (
                int(model_label_to_dataset_label(int(label_value)))
                if args.model_facing_labels
                else int(label_value)
            )
            color = COLORS[raw_label % len(COLORS)]
            draw.rectangle(box, outline=color, width=3)
            suffix = f" model={label_value}" if args.model_facing_labels else ""
            text = f"{raw_label}: {names[raw_label]}{suffix}"
            left, top, right, bottom = draw.textbbox((box[0], box[1]), text, font=font)
            label_height = bottom - top + 6
            label_top = max(0, box[1] - label_height)
            draw.rectangle((box[0], label_top, box[0] + right - left + 8, label_top + label_height), fill=color)
            draw.text((box[0] + 4, label_top + 2), text, fill="white", font=font, stroke_width=1, stroke_fill="black")
        if not labels.numel():
            draw.rectangle((8, 8, 265, 38), fill="#b91c1c")
            draw.text((14, 13), "NO ANNOTATED OBJECTS", fill="white", font=font)

        label_space = "model" if args.model_facing_labels else "dataset"
        destination = output_dir / (
            f"{args.split}_{preset}_{label_space}_{output_index:02d}_"
            f"{dataset.image_paths[dataset_index].stem}.jpg"
        )
        image.save(destination, format="JPEG", quality=94)
        print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
