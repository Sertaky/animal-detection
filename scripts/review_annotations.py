#!/usr/bin/env python3
"""Generate targeted, read-only YOLO annotation review visualizations."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml
from PIL import Image, ImageDraw, ImageFont

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "valid", "test")
MUTED = "#64748b"
CATEGORY_COLORS = {
    "large_box": "#f97316",
    "boundary_box": "#06b6d4",
    "zebra": "#22c55e",
    "multi_object": "#e879f9",
}


@dataclass(frozen=True)
class Annotation:
    class_id: int
    cx: float
    cy: float
    width: float
    height: float

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def xyxy_normalized(self) -> tuple[float, float, float, float]:
        return (
            self.cx - self.width / 2,
            self.cy - self.height / 2,
            self.cx + self.width / 2,
            self.cy + self.height / 2,
        )


@dataclass(frozen=True)
class ImageRecord:
    split: str
    image_path: Path
    label_path: Path
    width: int
    height: int
    annotations: tuple[Annotation, ...]


def load_names(path: Path) -> dict[int, str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    names = config.get("names")
    if isinstance(names, list):
        return dict(enumerate(map(str, names)))
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    raise ValueError("data.yaml does not contain a valid names list or mapping")


def parse_label(path: Path, names: dict[int, str]) -> tuple[Annotation, ...]:
    annotations: list[Annotation] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not raw.strip():
            continue
        fields = raw.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 fields")
        try:
            class_value = float(fields[0])
            cx, cy, width, height = map(float, fields[1:])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: non-numeric field") from exc
        if not class_value.is_integer():
            raise ValueError(f"{path}:{line_number}: class ID is not an integer")
        class_id = int(class_value)
        if class_id not in names:
            raise ValueError(f"{path}:{line_number}: undeclared class ID {class_id}")
        annotation = Annotation(class_id, cx, cy, width, height)
        x1, y1, x2, y2 = annotation.xyxy_normalized
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"{path}:{line_number}: invalid normalized values")
        if x1 < -1e-9 or y1 < -1e-9 or x2 > 1 + 1e-9 or y2 > 1 + 1e-9:
            raise ValueError(f"{path}:{line_number}: box extends outside image")
        annotations.append(annotation)
    return tuple(annotations)


def scan_dataset(root: Path, names: dict[int, str]) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for split in SPLITS:
        images_dir = root / split / "images"
        labels_dir = root / split / "labels"
        for image_path in sorted(images_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.casefold() not in IMAGE_EXTENSIONS:
                continue
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise ValueError(f"Missing label for {image_path}")
            with Image.open(image_path) as image:
                width, height = image.size
            records.append(
                ImageRecord(
                    split=split,
                    image_path=image_path,
                    label_path=label_path,
                    width=width,
                    height=height,
                    annotations=parse_label(label_path, names),
                )
            )
    return records


def touches_boundary(annotation: Annotation, record: ImageRecord) -> bool:
    x1, y1, x2, y2 = annotation.xyxy_normalized
    return (
        x1 * record.width <= 1
        or y1 * record.height <= 1
        or x2 * record.width >= record.width - 1
        or y2 * record.height >= record.height - 1
    )


def select_records(
    candidates: list[ImageRecord], count: int, seed: int
) -> list[ImageRecord]:
    ordered = sorted(candidates, key=lambda record: (record.split, record.image_path.name.casefold()))
    if len(ordered) < count:
        raise ValueError(f"Requested {count} samples but only {len(ordered)} candidates exist")
    return random.Random(seed).sample(ordered, count)


def draw_label(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    text: str,
    color: str,
    font: ImageFont.ImageFont,
) -> None:
    left, top, right, bottom = draw.textbbox((box[0], box[1]), text, font=font, stroke_width=1)
    label_height = bottom - top + 6
    label_width = right - left + 8
    label_top = max(0, box[1] - label_height)
    draw.rectangle((box[0], label_top, box[0] + label_width, label_top + label_height), fill=color)
    draw.text((box[0] + 4, label_top + 2), text, fill="white", font=font, stroke_width=1, stroke_fill="black")


def render_record(
    record: ImageRecord,
    names: dict[int, str],
    destination: Path,
    category: str,
    highlight: Callable[[Annotation], bool],
) -> int:
    with Image.open(record.image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=16)
    highlighted = 0
    for annotation in record.annotations:
        selected = highlight(annotation)
        highlighted += int(selected)
        x1, y1, x2, y2 = annotation.xyxy_normalized
        box = (x1 * record.width, y1 * record.height, x2 * record.width, y2 * record.height)
        color = CATEGORY_COLORS[category] if selected else MUTED
        line_width = 4 if selected else 2
        draw.rectangle(box, outline=color, width=line_width)
        label = f"{annotation.class_id}: {names[annotation.class_id]}"
        if category == "large_box" and selected:
            label += f" area={annotation.area:.3f}"
        elif category == "boundary_box" and selected:
            label += " boundary"
        draw_label(draw, box, label, color, font)
    image.save(destination, format="JPEG", quality=94)
    return highlighted


def render_empty(record: ImageRecord, destination: Path) -> None:
    with Image.open(record.image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=24)
    message = "NO ANNOTATED OBJECTS"
    left, top, right, bottom = draw.textbbox((0, 0), message, font=font, stroke_width=1)
    text_width, text_height = right - left, bottom - top
    x = max(8, (image.width - text_width) / 2)
    draw.rectangle((x - 10, 8, x + text_width + 10, 20 + text_height), fill="#b91c1c")
    draw.text((x, 12), message, fill="white", font=font, stroke_width=1, stroke_fill="black")
    image.save(destination, format="JPEG", quality=94)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/review_samples"))
    parser.add_argument("--samples-per-category", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.samples_per_category < 3:
        parser.error("--samples-per-category must be at least 3")

    root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.resolve()
    if not root.is_dir() or not (root / "data.yaml").is_file():
        parser.error(f"Invalid dataset root: {root}")
    if output_dir == root or root in output_dir.parents:
        parser.error("Output directory must be outside the immutable raw dataset")
    output_dir.mkdir(parents=True, exist_ok=True)

    names = load_names(root / "data.yaml")
    zebra_ids = [class_id for class_id, name in names.items() if name.casefold() == "zebra"]
    if len(zebra_ids) != 1:
        raise ValueError(f"Expected one Zebra class, found {zebra_ids}")
    zebra_id = zebra_ids[0]
    records = scan_dataset(root, names)

    large_candidates = [record for record in records if any(a.area > 0.5 for a in record.annotations)]
    boundary_candidates = [record for record in records if any(touches_boundary(a, record) for a in record.annotations)]
    zebra_candidates = [record for record in records if any(a.class_id == zebra_id for a in record.annotations)]
    empty_records = [record for record in records if not record.annotations]
    if len(empty_records) != 2:
        raise ValueError(f"Expected exactly two empty-label images, found {len(empty_records)}")
    multi_record = max(records, key=lambda record: (len(record.annotations), record.image_path.name.casefold()))

    selections = {
        "large_box": select_records(large_candidates, args.samples_per_category, args.seed),
        "boundary_box": select_records(boundary_candidates, args.samples_per_category, args.seed + 1),
        "zebra": select_records(zebra_candidates, args.samples_per_category, args.seed + 2),
    }
    manifest: dict[str, object] = {
        "dataset_root": str(root),
        "definitions": {
            "large_box": "normalized area > 0.5",
            "boundary_box": "box edge within 1 pixel of image boundary",
        },
        "images": [],
    }
    manifest_images: list[dict[str, object]] = manifest["images"]  # type: ignore[assignment]

    for category, selected_records in selections.items():
        for index, record in enumerate(selected_records, 1):
            destination = output_dir / f"{category}_{index:02d}_{record.split}_{record.image_path.stem}.jpg"
            if category == "large_box":
                predicate = lambda annotation: annotation.area > 0.5
            elif category == "boundary_box":
                predicate = lambda annotation, current=record: touches_boundary(annotation, current)
            else:
                predicate = lambda annotation: annotation.class_id == zebra_id
            highlighted = render_record(record, names, destination, category, predicate)
            manifest_images.append({
                "category": category,
                "output": str(destination),
                "source_image": str(record.image_path),
                "source_label": str(record.label_path),
                "objects": len(record.annotations),
                "highlighted_objects": highlighted,
            })
            print(destination)

    for index, record in enumerate(sorted(empty_records, key=lambda item: item.image_path.name.casefold()), 1):
        destination = output_dir / f"empty_label_{index:02d}_{record.split}_{record.image_path.stem}.jpg"
        render_empty(record, destination)
        manifest_images.append({
            "category": "empty_label",
            "output": str(destination),
            "source_image": str(record.image_path),
            "source_label": str(record.label_path),
            "objects": 0,
            "highlighted_objects": 0,
        })
        print(destination)

    destination = output_dir / f"multi_object_01_{multi_record.split}_{multi_record.image_path.stem}.jpg"
    highlighted = render_record(multi_record, names, destination, "multi_object", lambda annotation: True)
    manifest_images.append({
        "category": "multi_object",
        "output": str(destination),
        "source_image": str(multi_record.image_path),
        "source_label": str(multi_record.label_path),
        "objects": len(multi_record.annotations),
        "highlighted_objects": highlighted,
    })
    print(destination)

    manifest_path = output_dir / "review_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
