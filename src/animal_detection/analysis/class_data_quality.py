"""Pure helpers for class-focused annotation distribution audits."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import numpy as np


SMALL_AREA_MAX = 0.10
MEDIUM_AREA_MAX = 0.40


def size_bucket(area: float) -> str:
    """Return the fixed size bin used by the validation error analysis."""
    if area < SMALL_AREA_MAX:
        return "small"
    if area < MEDIUM_AREA_MAX:
        return "medium"
    return "large"


def crowdedness_bucket(object_count: int) -> str:
    """Bucket an image by its total annotated object count."""
    if object_count < 1:
        raise ValueError("object_count must be positive for an annotated object")
    if object_count == 1:
        return "1"
    if object_count <= 3:
        return "2-3"
    return "4+"


def touches_boundary(
    box: Iterable[float], image_width: int, image_height: int, tolerance_px: float = 1.0
) -> bool:
    """Return whether pixel ``xyxy`` box has an edge within tolerance of an image edge."""
    x1, y1, x2, y2 = (float(value) for value in box)
    return (
        x1 <= tolerance_px
        or y1 <= tolerance_px
        or x2 >= image_width - tolerance_px
        or y2 >= image_height - tolerance_px
    )


def describe(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {"count": 0, "mean": None, "median": None, "q1": None, "q3": None, "min": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q1": float(np.quantile(array, 0.25)),
        "q3": float(np.quantile(array, 0.75)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def percentage(numerator: int, denominator: int) -> float | None:
    return 100.0 * numerator / denominator if denominator else None


def aggregate_objects(objects: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate normalized geometry and image-context fields for one class/split."""
    image_ids = {item["image_id"] for item in objects}
    multi_images = {
        item["image_id"] for item in objects if item["image_object_count"] > 1
    }
    sizes = Counter(size_bucket(float(item["area"])) for item in objects)
    crowd = Counter(crowdedness_bucket(int(item["image_object_count"])) for item in objects)
    object_count = len(objects)
    image_count = len(image_ids)
    return {
        "image_count": image_count,
        "object_count": object_count,
        "objects_per_image": object_count / image_count if image_count else None,
        "normalized_area": describe(item["area"] for item in objects),
        "normalized_width": describe(item["width"] for item in objects),
        "normalized_height": describe(item["height"] for item in objects),
        "aspect_ratio": describe(item["aspect_ratio"] for item in objects),
        "size_counts": {name: sizes[name] for name in ("small", "medium", "large")},
        "size_percent": {name: percentage(sizes[name], object_count) for name in ("small", "medium", "large")},
        "boundary_touch_count": sum(bool(item["touches_boundary"]) for item in objects),
        "boundary_touch_percent": percentage(
            sum(bool(item["touches_boundary"]) for item in objects), object_count
        ),
        "multi_object_image_count": len(multi_images),
        "multi_object_image_percent": percentage(len(multi_images), image_count),
        "object_crowdedness_counts": {name: crowd[name] for name in ("1", "2-3", "4+")},
        "object_crowdedness_percent": {
            name: percentage(crowd[name], object_count) for name in ("1", "2-3", "4+")
        },
    }


def split_shift(train: dict[str, Any], valid: dict[str, Any]) -> dict[str, Any]:
    """Report transparent train/validation differences and threshold-based flags."""
    def difference(path: tuple[str, ...]) -> float | None:
        left: Any = train
        right: Any = valid
        for key in path:
            left, right = left[key], right[key]
        return None if left is None or right is None else float(right) - float(left)

    area_delta = difference(("normalized_area", "median"))
    small_delta = difference(("size_percent", "small"))
    boundary_delta = difference(("boundary_touch_percent",))
    crowded_delta = difference(("multi_object_image_percent",))
    aspect_delta = difference(("aspect_ratio", "median"))
    flags = []
    if area_delta is not None and abs(area_delta) >= 0.10:
        flags.append("median_area_abs_delta_ge_0.10")
    if small_delta is not None and abs(small_delta) >= 15.0:
        flags.append("small_object_share_delta_ge_15pp")
    if boundary_delta is not None and abs(boundary_delta) >= 10.0:
        flags.append("boundary_touch_delta_ge_10pp")
    if crowded_delta is not None and abs(crowded_delta) >= 20.0:
        flags.append("multi_object_image_delta_ge_20pp")
    if aspect_delta is not None:
        baseline = train["aspect_ratio"]["median"]
        if baseline and abs(aspect_delta) / baseline >= 0.35:
            flags.append("median_aspect_ratio_relative_delta_ge_35pct")
    return {
        "train_object_count": train["object_count"],
        "valid_object_count": valid["object_count"],
        "train_image_count": train["image_count"],
        "valid_image_count": valid["image_count"],
        "valid_object_share_percent": percentage(
            valid["object_count"], train["object_count"] + valid["object_count"]
        ),
        "median_area_delta_valid_minus_train": area_delta,
        "small_object_share_delta_pp": small_delta,
        "boundary_touch_delta_pp": boundary_delta,
        "multi_object_image_delta_pp": crowded_delta,
        "median_aspect_ratio_delta": aspect_delta,
        "flags": flags,
    }
