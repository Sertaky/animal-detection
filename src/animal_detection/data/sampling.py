"""Reusable, explicit sampling helpers for detection datasets."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch.utils.data import WeightedRandomSampler


def compute_image_difficulty_flags(
    image_boxes: Sequence[Sequence[Sequence[float]] | torch.Tensor],
    image_sizes: Sequence[Sequence[int | float] | torch.Tensor],
    *,
    small_area_threshold: float,
    crowded_object_count: int,
) -> list[dict[str, bool]]:
    """Compute class-independent small-object and crowded-scene flags."""
    if len(image_boxes) != len(image_sizes):
        raise ValueError("image_boxes and image_sizes must have equal length")
    if not 0.0 < small_area_threshold <= 1.0:
        raise ValueError("small_area_threshold must be in (0, 1]")
    if crowded_object_count < 1:
        raise ValueError("crowded_object_count must be positive")
    flags: list[dict[str, bool]] = []
    for boxes_value, size_value in zip(image_boxes, image_sizes):
        boxes = torch.as_tensor(boxes_value, dtype=torch.float64).reshape(-1, 4)
        size = torch.as_tensor(size_value, dtype=torch.float64).reshape(-1)
        if size.numel() != 2 or bool(torch.any(size <= 0)):
            raise ValueError("each image size must be positive [height, width]")
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        if bool(torch.any(areas <= 0)):
            raise ValueError("boxes must have positive area")
        normalized_areas = areas / (size[0] * size[1])
        flags.append({
            "contains_small_object": bool(torch.any(normalized_areas < small_area_threshold)),
            "is_crowded": len(boxes) >= crowded_object_count,
        })
    return flags


def build_difficulty_aware_image_weights(
    difficulty_flags: Sequence[dict[str, bool]], *, difficulty_multiplier: float
) -> list[float]:
    """Weight difficult images once; small and crowded conditions do not stack."""
    if difficulty_multiplier < 1.0:
        raise ValueError("difficulty_multiplier must be at least 1.0")
    return [
        float(difficulty_multiplier)
        if flag["contains_small_object"] or flag["is_crowded"]
        else 1.0
        for flag in difficulty_flags
    ]


def build_difficulty_aware_sampler(
    weights: Sequence[float], *, num_samples: int, seed: int
) -> WeightedRandomSampler:
    """Build a deterministic replacement sampler for difficulty-aware weights."""
    return build_class_aware_sampler(weights, num_samples=num_samples, seed=seed)


def difficulty_sampling_diagnostics(
    difficulty_flags: Sequence[dict[str, bool]], sampled_indices: Sequence[int]
) -> dict[str, Any]:
    """Summarize raw and simulated exposure without consulting semantic labels."""
    draw_counts = Counter(map(int, sampled_indices))
    definitions = {
        "ordinary": lambda flag: not flag["contains_small_object"] and not flag["is_crowded"],
        "small_object_containing": lambda flag: flag["contains_small_object"],
        "crowded": lambda flag: flag["is_crowded"],
        "small_and_crowded": lambda flag: flag["contains_small_object"] and flag["is_crowded"],
    }
    groups: dict[str, dict[str, float | int | None]] = {}
    for name, predicate in definitions.items():
        raw = sum(int(predicate(flag)) for flag in difficulty_flags)
        sampled = sum(int(predicate(difficulty_flags[index])) for index in sampled_indices)
        groups[name] = {
            "raw_image_count": raw,
            "sampled_occurrences": sampled,
            "exposure_multiplier": sampled / raw if raw else None,
        }
    draws = len(sampled_indices)
    unique = len(draw_counts)
    return {
        "total_usable_images": len(difficulty_flags),
        "images_containing_small_objects": groups["small_object_containing"]["raw_image_count"],
        "crowded_images": groups["crowded"]["raw_image_count"],
        "images_both_small_and_crowded": groups["small_and_crowded"]["raw_image_count"],
        "ordinary_images": groups["ordinary"]["raw_image_count"],
        "epoch_draws": draws,
        "unique_images_sampled": unique,
        "repeated_image_draws": draws - unique,
        "unique_image_percent": 100.0 * unique / draws if draws else None,
        "maximum_draws_for_one_image": max(draw_counts.values(), default=0),
        "sampled_draws_containing_small_object_percent": (
            100.0 * groups["small_object_containing"]["sampled_occurrences"] / draws if draws else None
        ),
        "sampled_draws_crowded_percent": (
            100.0 * groups["crowded"]["sampled_occurrences"] / draws if draws else None
        ),
        "groups": groups,
    }


def build_class_aware_image_weights(
    image_class_ids: Sequence[Iterable[int]],
    *,
    weak_class_ids: Iterable[int],
    weak_multiplier: float,
) -> list[float]:
    """Assign one presence-based image weight; multiple weak classes do not stack."""
    if weak_multiplier < 1.0:
        raise ValueError("weak_multiplier must be at least 1.0")
    weak = {int(class_id) for class_id in weak_class_ids}
    if not weak:
        raise ValueError("weak_class_ids cannot be empty")
    return [
        float(weak_multiplier if weak.intersection(map(int, labels)) else 1.0)
        for labels in image_class_ids
    ]


def build_class_aware_sampler(
    weights: Sequence[float], *, num_samples: int, seed: int
) -> WeightedRandomSampler:
    """Build a deterministic replacement sampler with its own seeded generator."""
    if len(weights) == 0:
        raise ValueError("weights cannot be empty")
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=True,
        generator=generator,
    )


def sampling_diagnostics(
    image_labels: Sequence[Sequence[int]],
    sampled_indices: Sequence[int],
    *,
    weak_class_ids: Iterable[int],
    class_names: Sequence[str],
) -> dict[str, Any]:
    """Summarize raw annotation counts and image-level exposure for simulated draws."""
    weak = {int(class_id) for class_id in weak_class_ids}
    raw_objects: Counter[int] = Counter()
    raw_images: Counter[int] = Counter()
    for labels in image_labels:
        raw_objects.update(map(int, labels))
        raw_images.update(set(map(int, labels)))
    sampled_images: Counter[int] = Counter()
    weak_draws = 0
    draw_counts = Counter(map(int, sampled_indices))
    for index in sampled_indices:
        present = set(map(int, image_labels[index]))
        sampled_images.update(present)
        weak_draws += int(bool(present & weak))
    classes = []
    for class_id, class_name in enumerate(class_names):
        baseline = raw_images[class_id]
        sampled = sampled_images[class_id]
        classes.append({
            "class_id": class_id,
            "class_name": class_name,
            "raw_train_object_count": raw_objects[class_id],
            "raw_train_image_count": baseline,
            "sampled_image_occurrences": sampled,
            "exposure_multiplier": sampled / baseline if baseline else None,
        })
    unique = len(draw_counts)
    draws = len(sampled_indices)
    return {
        "epoch_draws": draws,
        "unique_images_sampled": unique,
        "repeated_image_draws": draws - unique,
        "maximum_draws_for_one_image": max(draw_counts.values(), default=0),
        "weak_class_image_draws": weak_draws,
        "weak_class_draw_percent": 100.0 * weak_draws / draws if draws else None,
        "classes": classes,
    }
