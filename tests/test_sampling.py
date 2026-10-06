from __future__ import annotations

import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset, WeightedRandomSampler

from animal_detection.data.sampling import (
    build_class_aware_image_weights,
    build_class_aware_sampler,
    build_difficulty_aware_image_weights,
    build_difficulty_aware_sampler,
    compute_image_difficulty_flags,
)


def test_presence_based_weights_do_not_stack_and_define_empty_behavior() -> None:
    weights = build_class_aware_image_weights(
        [[6], [14], [14, 13], []], weak_class_ids={14, 13}, weak_multiplier=2.0
    )
    assert weights == [1.0, 2.0, 2.0, 1.0]
    assert len(weights) == 4


def test_sampler_is_exact_length_and_deterministic() -> None:
    weights = [1.0, 2.0, 1.0, 2.0]
    first = list(build_class_aware_sampler(weights, num_samples=17, seed=42))
    second = list(build_class_aware_sampler(weights, num_samples=17, seed=42))
    assert len(first) == 17
    assert first == second


def test_dataloader_with_sampler_is_not_shuffled() -> None:
    dataset = TensorDataset(torch.arange(4))
    sampler = build_class_aware_sampler([1.0, 2.0, 1.0, 2.0], num_samples=4, seed=42)
    loader = DataLoader(dataset, sampler=sampler, shuffle=False)
    assert isinstance(loader.sampler, WeightedRandomSampler)
    assert not isinstance(loader.sampler, SequentialSampler)


def _flags(boxes: list[list[list[float]]]) -> list[dict[str, bool]]:
    return compute_image_difficulty_flags(
        boxes, [[100, 100]] * len(boxes),
        small_area_threshold=0.10, crowded_object_count=4,
    )


def test_difficulty_weights_cover_ordinary_small_crowded_and_overlap() -> None:
    flags = _flags([
        [[0, 0, 50, 50]],
        [[0, 0, 20, 20]],
        [[0, 0, 50, 50]] * 4,
        [[0, 0, 20, 20]] + [[0, 0, 50, 50]] * 3,
        [[0, 0, 50, 50]] * 3,
    ])
    assert flags == [
        {"contains_small_object": False, "is_crowded": False},
        {"contains_small_object": True, "is_crowded": False},
        {"contains_small_object": False, "is_crowded": True},
        {"contains_small_object": True, "is_crowded": True},
        {"contains_small_object": False, "is_crowded": False},
    ]
    assert build_difficulty_aware_image_weights(flags, difficulty_multiplier=2.0) == [1, 2, 2, 2, 1]


def test_small_area_threshold_is_strictly_less_than_point_one() -> None:
    flags = _flags([[[0, 0, 10, 100]], [[0, 0, 9.999, 100]]])
    assert flags[0]["contains_small_object"] is False
    assert flags[1]["contains_small_object"] is True


def test_difficulty_policy_ignores_semantic_labels() -> None:
    geometry = [[[0, 0, 20, 20]], [[0, 0, 50, 50]] * 4]
    weights_before = build_difficulty_aware_image_weights(_flags(geometry), difficulty_multiplier=2.0)
    arbitrary_class_ids = [[14], [0, 5, 13, 19]]
    arbitrary_class_ids[:] = [[6], [7, 8, 9, 10]]
    weights_after = build_difficulty_aware_image_weights(_flags(geometry), difficulty_multiplier=2.0)
    assert weights_before == weights_after == [2.0, 2.0]


def test_difficulty_sampler_length_determinism_and_loader_configuration() -> None:
    weights = build_difficulty_aware_image_weights(
        _flags([[[0, 0, 50, 50]], [[0, 0, 20, 20]], []]),
        difficulty_multiplier=2.0,
    )
    assert len(weights) == 3
    first = list(build_difficulty_aware_sampler(weights, num_samples=len(weights), seed=42))
    second = list(build_difficulty_aware_sampler(weights, num_samples=len(weights), seed=42))
    assert len(first) == len(weights)
    assert first == second
    loader = DataLoader(TensorDataset(torch.arange(3)), sampler=build_difficulty_aware_sampler(
        weights, num_samples=3, seed=42), shuffle=False)
    assert isinstance(loader.sampler, WeightedRandomSampler)
    assert not isinstance(loader.sampler, SequentialSampler)
