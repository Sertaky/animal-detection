from __future__ import annotations

import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset, WeightedRandomSampler

from animal_detection.data.sampling import (
    build_class_aware_image_weights,
    build_class_aware_sampler,
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
