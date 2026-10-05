from __future__ import annotations

import pytest

from animal_detection.analysis.class_data_quality import (
    aggregate_objects,
    crowdedness_bucket,
    size_bucket,
    split_shift,
    touches_boundary,
)


def test_fixed_size_and_crowdedness_boundaries() -> None:
    assert [size_bucket(value) for value in (0.01, 0.099, 0.10, 0.399, 0.40)] == [
        "small", "small", "medium", "medium", "large"
    ]
    assert [crowdedness_bucket(value) for value in (1, 2, 3, 4, 9)] == [
        "1", "2-3", "2-3", "4+", "4+"
    ]
    with pytest.raises(ValueError):
        crowdedness_bucket(0)


def test_boundary_touch_uses_one_pixel_tolerance() -> None:
    assert touches_boundary((1.0, 20.0, 80.0, 90.0), 100, 100)
    assert touches_boundary((20.0, 20.0, 99.0, 90.0), 100, 100)
    assert not touches_boundary((1.01, 20.0, 98.98, 90.0), 100, 100)


def test_aggregate_and_split_shift() -> None:
    train_objects = [
        {"image_id": "a", "area": 0.05, "width": 0.25, "height": 0.20, "aspect_ratio": 1.25, "touches_boundary": True, "image_object_count": 1},
        {"image_id": "b", "area": 0.20, "width": 0.50, "height": 0.40, "aspect_ratio": 1.25, "touches_boundary": False, "image_object_count": 4},
    ]
    valid_objects = [
        {"image_id": "c", "area": 0.50, "width": 0.50, "height": 1.00, "aspect_ratio": 0.50, "touches_boundary": False, "image_object_count": 1}
    ]
    train = aggregate_objects(train_objects)
    valid = aggregate_objects(valid_objects)
    assert train["image_count"] == 2
    assert train["size_counts"] == {"small": 1, "medium": 1, "large": 0}
    assert train["object_crowdedness_counts"]["4+"] == 1
    shift = split_shift(train, valid)
    assert shift["median_area_delta_valid_minus_train"] == pytest.approx(0.375)
    assert "median_area_abs_delta_ge_0.10" in shift["flags"]
