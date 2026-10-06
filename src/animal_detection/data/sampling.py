"""Reusable, explicit class-aware sampling helpers for detection datasets."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch.utils.data import WeightedRandomSampler


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
