#!/usr/bin/env python3
"""Simulate and visualize the fixed Experiment 05 training scale jitter."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from animal_detection.data import RandomScaleJitter512, YoloDetectionDataset
from animal_detection.data.label_mapping import ANIMAL_CLASS_NAMES


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def size_bin(box: torch.Tensor, height: int, width: int) -> str:
    area = float((box[2] - box[0]) * (box[3] - box[1])) / (height * width)
    return "small" if area < 0.10 else "medium" if area < 0.40 else "large"


def boundary_count(boxes: torch.Tensor, height: int, width: int) -> int:
    if not boxes.numel():
        return 0
    epsilon = 1e-5
    touches = (
        (boxes[:, 0] <= epsilon) | (boxes[:, 1] <= epsilon)
        | (boxes[:, 2] >= width - epsilon) | (boxes[:, 3] >= height - epsilon)
    )
    return int(touches.sum())


def draw_sample(
    image: Image.Image, target: dict[str, Any], metadata: dict[str, Any], path: Path,
) -> None:
    rendered = image.copy()
    draw = ImageDraw.Draw(rendered)
    labels = target["labels"].tolist()
    for box, label in zip(target["boxes"].tolist(), labels):
        draw.rectangle(box, outline=(0, 255, 0), width=3)
        draw.text((box[0] + 2, box[1] + 2), ANIMAL_CLASS_NAMES[label], fill=(255, 255, 0))
    caption = (
        f"scale={metadata['scale_factor']} {metadata['operation']} "
        f"offset=({metadata['offset_x']},{metadata['offset_y']}) "
        f"objects={metadata['objects_before']}->{metadata['objects_after']}"
    )
    draw.rectangle((0, 0, 512, 20), fill=(0, 0, 0))
    draw.text((4, 4), caption, fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(path, quality=92)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclusions-manifest", type=Path,
        default=PROJECT_ROOT / "data" / "manifests" / "excluded_samples.json",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    visual_dir = output / "augmentation_visualizations"
    dataset = YoloDetectionDataset(args.dataset_root.resolve(), "train")
    manifest = json.loads(args.exclusions_manifest.read_text(encoding="utf-8"))
    excluded = {item["image_filename"] for item in manifest["samples"] if item["split"] == "train"}
    indices = [index for index, path in enumerate(dataset.image_paths) if path.name not in excluded]
    if len(dataset) != 1400 or len(indices) != 1399:
        raise RuntimeError("Unexpected usable training count")

    random.seed(args.seed)
    transform = RandomScaleJitter512((0.8, 1.0, 1.2), (512, 512))
    scales: Counter[str] = Counter()
    before_bins: Counter[str] = Counter()
    after_bins: Counter[str] = Counter()
    total_before = total_after = clipped = removed = zero_images = 0
    boundary_before = boundary_after = 0
    invalid_coordinates = mismatches = nonfinite = 0
    visual_counts: Counter[str] = Counter()
    visualization_paths: list[str] = []

    for index in indices:
        image, target = dataset[index]
        before_boxes = target["boxes"]
        before_labels = target["labels"]
        height, width = map(int, target["original_size"].tolist())
        factor = random.choice(transform.scale_factors)
        scaled_height, scaled_width = transform.scaled_size(factor)
        offset_x = random.randint(0, abs(512 - scaled_width))
        offset_y = random.randint(0, abs(512 - scaled_height))
        transformed_image, transformed, metadata = transform.apply_with_parameters(
            image, target, scale_factor=factor, offset_x=offset_x, offset_y=offset_y
        )
        after_boxes = transformed["boxes"]
        after_labels = transformed["labels"]
        scales[str(factor)] += 1
        total_before += len(before_boxes)
        total_after += len(after_boxes)
        clipped += metadata["boxes_clipped"]
        removed += metadata["boxes_removed"]
        zero_images += int(len(after_boxes) == 0)
        boundary_before += boundary_count(before_boxes, height, width)
        boundary_after += boundary_count(after_boxes, 512, 512)
        before_bins.update(size_bin(box, height, width) for box in before_boxes)
        after_bins.update(size_bin(box, 512, 512) for box in after_boxes)
        mismatches += int(len(after_boxes) != len(after_labels))
        nonfinite += int(not bool(torch.isfinite(after_boxes).all()))
        invalid_coordinates += int(bool(after_boxes.numel()) and bool(
            torch.any(after_boxes < 0) or torch.any(after_boxes > 512)
            or torch.any(after_boxes[:, 2] <= after_boxes[:, 0])
            or torch.any(after_boxes[:, 3] <= after_boxes[:, 1])
        ))

        categories: list[str] = []
        scale_key = f"scale_{str(factor).replace('.', 'p')}"
        if visual_counts[scale_key] < 3:
            categories.append(scale_key)
        if metadata["boxes_clipped"] and metadata["objects_after"] and visual_counts["cropped_partial"] < 1:
            categories.append("cropped_partial")
        if any(size_bin(box, height, width) == "small" for box in before_boxes) and visual_counts["small_object"] < 3:
            categories.append("small_object")
        if len(before_boxes) >= 4 and visual_counts["crowded"] < 3:
            categories.append("crowded")
        for category in categories:
            visual_counts[category] += 1
            filename = f"{category}_{visual_counts[category]:02d}_{dataset.image_paths[index].stem}.jpg"
            visual_path = visual_dir / filename
            draw_sample(transformed_image, transformed, metadata, visual_path)
            visualization_paths.append(str(visual_path.relative_to(output)))

    removed_fraction = removed / total_before if total_before else 0.0
    zero_fraction = zero_images / len(indices)
    safety = {
        "maximum_removed_object_fraction": 0.05,
        "maximum_zero_object_image_fraction": 0.01,
        "invalid_coordinate_samples": invalid_coordinates,
        "label_box_mismatch_samples": mismatches,
        "nonfinite_coordinate_samples": nonfinite,
        "removed_object_fraction": removed_fraction,
        "zero_object_image_fraction": zero_fraction,
    }
    safety["passed"] = (
        invalid_coordinates == 0 and mismatches == 0 and nonfinite == 0
        and removed_fraction <= safety["maximum_removed_object_fraction"]
        and zero_fraction <= safety["maximum_zero_object_image_fraction"]
        and set(scales) == {"0.8", "1.0", "1.2"}
    )
    result = {
        "split": "train",
        "test_split_constructed": False,
        "seed": args.seed,
        "usable_training_images": len(indices),
        "policy": {
            "scale_factors": [0.8, 1.0, 1.2],
            "output_size": [512, 512],
            "scaled_sizes": {"0.8": [410, 410], "1.0": [512, 512], "1.2": [614, 614]},
            "rounding_rule": "floor(output_dimension * scale + 0.5)",
            "padding": "random placement, zero fill",
            "crop": "random 512x512 crop",
        },
        "scale_assignments": dict(scales),
        "objects": {"before": total_before, "after": total_after},
        "size_bins": {"before": dict(before_bins), "after": dict(after_bins)},
        "boxes_clipped": clipped,
        "boxes_completely_removed": removed,
        "images_becoming_zero_object": zero_images,
        "mean_objects_per_image": {
            "before": total_before / len(indices), "after": total_after / len(indices),
        },
        "boundary_touch": {
            "before_count": boundary_before, "after_count": boundary_after,
            "before_fraction": boundary_before / total_before if total_before else 0.0,
            "after_fraction": boundary_after / total_after if total_after else 0.0,
        },
        "safety": safety,
        "visualizations": visualization_paths,
    }
    write_json(output / "augmentation_analysis.json", result)
    lines = [
        "SCALE-JITTER AUGMENTATION ANALYSIS", "",
        "Weights/classes are not used; this is a deterministic train-only transform simulation.",
        f"Scale assignments: 0.8={scales['0.8']}, 1.0={scales['1.0']}, 1.2={scales['1.2']}",
        f"Objects before/after: {total_before}/{total_after}",
        f"Size bins before: small={before_bins['small']} medium={before_bins['medium']} large={before_bins['large']}",
        f"Size bins after: small={after_bins['small']} medium={after_bins['medium']} large={after_bins['large']}",
        f"Clipped={clipped} removed={removed} zero-object images={zero_images}",
        f"Mean objects/image: {total_before / len(indices):.6f} -> {total_after / len(indices):.6f}",
        f"Boundary touch: {boundary_before}/{total_before} ({100*boundary_before/total_before:.2f}%) -> {boundary_after}/{total_after} ({100*boundary_after/total_after:.2f}%)",
        f"Safety passed: {safety['passed']}",
        "Test split constructed: false",
    ]
    (output / "augmentation_analysis.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not safety["passed"]:
        print("SAFETY GATE FAILED: do not start training", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
