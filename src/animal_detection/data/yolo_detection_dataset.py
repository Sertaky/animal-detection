"""PyTorch dataset for the project's immutable YOLO-format animal data.

The known empty-label cow and wolf samples deliberately remain empty targets.
Their missing annotations are a dataset-quality issue and are not repaired here.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

SUPPORTED_SPLITS = frozenset({"train", "valid", "test"})
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})
NUM_CLASSES = 20
COORDINATE_TOLERANCE = 1e-5

KNOWN_MISSING_ANNOTATION_STEMS = frozenset(
    {
        "cow-19-_jpg.rf.79ab2f17459bf2b455ba932f730a35d2",
        "wolf-109-_jpg.rf.a384d38588191d796589048d6cb38e97",
    }
)

DetectionTarget = dict[str, Tensor | str]
DetectionTransform = Callable[
    [Image.Image, DetectionTarget],
    tuple[Any, DetectionTarget],
]


class YoloAnnotationError(ValueError):
    """Raised when a YOLO annotation row is malformed or invalid."""


def _annotation_error(path: Path, line_number: int, row: str, reason: str) -> YoloAnnotationError:
    return YoloAnnotationError(f"{path}:{line_number}: {reason}; row={row!r}")


def _parse_yolo_annotation_file(
    label_path: Path,
    image_width: int,
    image_height: int,
) -> tuple[Tensor, Tensor]:
    """Parse one label file and convert normalized YOLO boxes to pixel xyxy."""
    boxes: list[list[float]] = []
    labels: list[int] = []

    for line_number, original_row in enumerate(
        label_path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        row = original_row.strip()
        if not row:
            continue
        fields = row.split()
        if len(fields) != 5:
            raise _annotation_error(
                label_path,
                line_number,
                original_row,
                f"expected 5 fields, found {len(fields)}",
            )

        try:
            class_id = int(fields[0])
        except ValueError as exc:
            raise _annotation_error(
                label_path, line_number, original_row, "class_id must be an integer"
            ) from exc
        try:
            center_x, center_y, box_width, box_height = map(float, fields[1:])
        except ValueError as exc:
            raise _annotation_error(
                label_path,
                line_number,
                original_row,
                "box coordinates must be numeric",
            ) from exc

        values = (center_x, center_y, box_width, box_height)
        if not all(math.isfinite(value) for value in values):
            raise _annotation_error(
                label_path, line_number, original_row, "box coordinates must be finite"
            )
        if not 0 <= class_id < NUM_CLASSES:
            raise _annotation_error(
                label_path,
                line_number,
                original_row,
                f"class_id must be in [0, {NUM_CLASSES - 1}]",
            )
        if not 0 <= center_x <= 1 or not 0 <= center_y <= 1:
            raise _annotation_error(
                label_path,
                line_number,
                original_row,
                "center_x and center_y must be in [0, 1]",
            )
        if not 0 < box_width <= 1 or not 0 < box_height <= 1:
            raise _annotation_error(
                label_path,
                line_number,
                original_row,
                "width and height must be in (0, 1]",
            )

        center_x_px = center_x * image_width
        center_y_px = center_y * image_height
        box_width_px = box_width * image_width
        box_height_px = box_height * image_height
        x1 = center_x_px - box_width_px / 2
        y1 = center_y_px - box_height_px / 2
        x2 = center_x_px + box_width_px / 2
        y2 = center_y_px + box_height_px / 2

        if x2 <= x1 or y2 <= y1:
            raise _annotation_error(
                label_path, line_number, original_row, "box must satisfy x2 > x1 and y2 > y1"
            )
        if (
            x1 < -COORDINATE_TOLERANCE
            or y1 < -COORDINATE_TOLERANCE
            or x2 > image_width + COORDINATE_TOLERANCE
            or y2 > image_height + COORDINATE_TOLERANCE
        ):
            raise _annotation_error(
                label_path,
                line_number,
                original_row,
                "box extends outside image bounds",
            )

        # Clamp only coordinates within the tiny accepted floating-point tolerance.
        boxes.append(
            [
                max(0.0, x1),
                max(0.0, y1),
                min(float(image_width), x2),
                min(float(image_height), y2),
            ]
        )
        labels.append(class_id)

    if boxes:
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    else:
        boxes_tensor = torch.empty((0, 4), dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.int64)
    return boxes_tensor, labels_tensor


class YoloDetectionDataset(Dataset[tuple[Any, DetectionTarget]]):
    """Load one dataset split with strict, read-only YOLO annotation parsing.

    Args:
        dataset_root: Root containing ``train``, ``valid``, and ``test`` folders.
        split: Exactly one of ``train``, ``valid``, or ``test``.
        transforms: Optional callable receiving and returning ``(image, target)``.

    Labels remain faithful to the raw YOLO IDs 0 through 19. Torchvision models
    that reserve label 0 for background will need an explicit mapping layer later.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        transforms: DetectionTransform | None = None,
    ) -> None:
        if split not in SUPPORTED_SPLITS:
            supported = ", ".join(sorted(SUPPORTED_SPLITS))
            raise ValueError(f"Unsupported split {split!r}; expected one of: {supported}")

        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.split = split
        self.transforms = transforms
        self.images_dir = self.dataset_root / split / "images"
        self.labels_dir = self.dataset_root / split / "labels"

        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {self.images_dir}")
        if not self.labels_dir.is_dir():
            raise FileNotFoundError(f"Label directory does not exist: {self.labels_dir}")

        image_paths = sorted(
            (
                path
                for path in self.images_dir.iterdir()
                if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
            ),
            key=lambda path: (path.name.casefold(), path.name),
        )
        if not image_paths:
            raise ValueError(f"No supported images found in: {self.images_dir}")

        stems: dict[str, Path] = {}
        for image_path in image_paths:
            stem_key = image_path.stem.casefold()
            if stem_key in stems:
                raise ValueError(
                    "Duplicate image stems cannot be paired unambiguously: "
                    f"{stems[stem_key]} and {image_path}"
                )
            stems[stem_key] = image_path

        label_paths = tuple(self.labels_dir / f"{path.stem}.txt" for path in image_paths)
        missing_labels = [path for path in label_paths if not path.is_file()]
        if missing_labels:
            preview = ", ".join(str(path) for path in missing_labels[:3])
            suffix = "" if len(missing_labels) <= 3 else f" (+{len(missing_labels) - 3} more)"
            raise FileNotFoundError(
                f"Missing {len(missing_labels)} label file(s): {preview}{suffix}"
            )

        self.image_paths = tuple(image_paths)
        self.label_paths = label_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[Any, DetectionTarget]:
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Dataset index out of range: {index}")

        image_path = self.image_paths[index]
        label_path = self.label_paths[index]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        width, height = image.size
        boxes, labels = _parse_yolo_annotation_file(label_path, width, height)

        target: DetectionTarget = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(index, dtype=torch.int64),
            "image_path": str(image_path),
            "original_size": torch.tensor([height, width], dtype=torch.int64),
        }
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target
