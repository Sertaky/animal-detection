"""Explicit checkpoint save/load helpers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    *,
    epoch: int,
    model,
    optimizer,
    validation_metrics: Mapping[str, Any],
    training_metrics: Mapping[str, Any],
    configuration: Mapping[str, Any],
    class_mapping: Mapping[str, Any],
) -> Path:
    """Atomically save one requested training checkpoint."""
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_metrics": dict(validation_metrics),
        "training_metrics": dict(training_metrics),
        "configuration": dict(configuration),
        "class_mapping": dict(class_mapping),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint payload for explicit restoration or inspection."""
    return torch.load(Path(path), map_location=map_location, weights_only=False)
