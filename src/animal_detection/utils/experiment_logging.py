"""Small JSON experiment-history helpers."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def write_training_history(path: str | Path, records: Sequence[Mapping[str, Any]]) -> Path:
    """Atomically write training history as a JSON list."""
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(list(records), indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def append_training_history(path: str | Path, record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Append one record, rewrite atomically, and return the complete history."""
    destination = Path(path).resolve()
    history: list[dict[str, Any]] = []
    if destination.is_file():
        loaded = json.loads(destination.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError(f"Training history must contain a JSON list: {destination}")
        history = loaded
    history.append(dict(record))
    write_training_history(destination, history)
    return history
