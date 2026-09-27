#!/usr/bin/env python3
"""Render deterministic YOLO annotation previews without modifying source data."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
COLORS = ["#ef4444", "#22c55e", "#3b82f6", "#eab308", "#a855f7", "#06b6d4", "#f97316", "#ec4899"]


def load_names(path: Path) -> dict[int, str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    names = config.get("names")
    if isinstance(names, list):
        return dict(enumerate(map(str, names)))
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    raise ValueError("data.yaml does not contain a valid names list or mapping")


def parse_label(path: Path, class_names: dict[int, str]) -> list[tuple[int, float, float, float, float]]:
    annotations = []
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
        if class_id not in class_names:
            raise ValueError(f"{path}:{line_number}: undeclared class ID {class_id}")
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"{path}:{line_number}: invalid normalized values")
        x1, y1, x2, y2 = cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2
        if x1 < -1e-9 or y1 < -1e-9 or x2 > 1 + 1e-9 or y2 > 1 + 1e-9:
            raise ValueError(f"{path}:{line_number}: box extends outside image")
        annotations.append((class_id, x1, y1, x2, y2))
    return annotations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="train")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/annotation_samples"))
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("--num-samples must be positive")

    root = args.dataset_root.expanduser().resolve()
    images_dir, labels_dir = root / args.split / "images", root / args.split / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir() or not (root / "data.yaml").is_file():
        parser.error("Expected data.yaml and split images/labels directories")
    class_names = load_names(root / "data.yaml")
    candidates = sorted(
        image for image in images_dir.iterdir()
        if image.is_file() and image.suffix.casefold() in IMAGE_EXTENSIONS and (labels_dir / f"{image.stem}.txt").is_file()
    )
    if len(candidates) < args.num_samples:
        parser.error(f"Requested {args.num_samples} samples but only {len(candidates)} paired files exist")
    chosen = random.Random(args.seed).sample(candidates, args.num_samples)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=16)

    for image_path in chosen:
        annotations = parse_label(labels_dir / f"{image_path.stem}.txt", class_names)
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        for class_id, x1n, y1n, x2n, y2n in annotations:
            width, height = image.size
            box = (x1n * width, y1n * height, x2n * width, y2n * height)
            color = COLORS[class_id % len(COLORS)]
            draw.rectangle(box, outline=color, width=max(2, round(min(width, height) / 250)))
            label = f"{class_id}: {class_names[class_id]}"
            left, top, right, bottom = draw.textbbox((box[0], box[1]), label, font=font, stroke_width=1)
            label_height = bottom - top + 6
            label_top = max(0, box[1] - label_height)
            label_width = right - left + 8
            draw.rectangle((box[0], label_top, box[0] + label_width, label_top + label_height), fill=color)
            draw.text((box[0] + 4, label_top + 2), label, fill="white", font=font, stroke_width=1, stroke_fill="black")
        destination = output_dir / f"{args.split}_{image_path.stem}_annotated.jpg"
        image.save(destination, format="JPEG", quality=92)
        print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
