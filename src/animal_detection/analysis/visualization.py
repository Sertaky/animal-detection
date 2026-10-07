"""Deterministic validation examples drawn from cached detections."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from animal_detection.data.label_mapping import ANIMAL_CLASS_NAMES


def _name(label: int) -> str:
    return ANIMAL_CLASS_NAMES[label - 1]


def _best_score(row: dict[str, Any], category: str) -> float | None:
    records = row["match"]["predictions"]
    if category == "strongest_true_positives":
        values = [record["score"] for record in records if record["status"] == "true_positive"]
    elif category == "high_confidence_false_positives":
        values = [record["score"] for record in records if record["status"] == "false_positive"]
    elif category == "missed_objects":
        return float(len(row["match"]["false_negative_gt_indices"])) or None
    elif category == "crowded_scenes":
        count = len(row["image"]["ground_truth"])
        return float(count) if count >= 4 else None
    elif category == "small_object_cases":
        height, width = row["image"]["transformed_size"]
        small_indices = []
        for index, gt in enumerate(row["image"]["ground_truth"]):
            x1, y1, x2, y2 = gt["box"]
            if (x2 - x1) * (y2 - y1) / (height * width) < 0.10:
                small_indices.append(index)
        if not small_indices:
            return None
        missed = len(set(small_indices) & set(row["match"]["false_negative_gt_indices"]))
        return float(missed * 100 + len(small_indices))
    elif category.startswith("low_ap_"):
        label = ANIMAL_CLASS_NAMES.index(category.removeprefix("low_ap_").capitalize()) + 1
        gt_indices = [
            index for index, gt in enumerate(row["image"]["ground_truth"])
            if gt["label"] == label
        ]
        if not gt_indices:
            return None
        missed = len(set(gt_indices) & set(row["match"]["false_negative_gt_indices"]))
        false_positives = sum(
            record["status"] == "false_positive" and record["label"] == label
            for record in records
        )
        return float(missed * 100 + false_positives * 10 + len(gt_indices))
    else:
        values = [record["score"] for record in records if record["category"] == category]
    return max(values, default=None)


def _text(
    draw: ImageDraw.ImageDraw, xy: tuple[int, int], message: str,
    fill: str, font: ImageFont.ImageFont,
) -> None:
    bounds = draw.textbbox(xy, message, font=font)
    draw.rectangle(bounds, fill=fill)
    draw.text(xy, message, fill="white", font=font)


def _render(row: dict[str, Any], source: Path, destination: Path, category: str) -> None:
    with Image.open(source) as original:
        resized = original.convert("RGB").resize((1024, 1024), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (1024, 1130), "white")
    canvas.paste(resized, (0, 65))
    draw = ImageDraw.Draw(canvas)
    transformed_height, transformed_width = row["image"]["transformed_size"]
    scale_x = 1024 / transformed_width
    scale_y = 1024 / transformed_height
    try:
        font = ImageFont.truetype("arial.ttf", size=19)
    except OSError:
        font = ImageFont.load_default(size=19)
    _text(draw, (12, 10), category.replace("_", " ").title(), "#202020", font)
    _text(draw, (12, 35), "Ground truth: green    True positive: cyan    False positive: red", "#202020", font)
    for index, gt in enumerate(row["image"]["ground_truth"]):
        x1, y1, x2, y2 = gt["box"]
        draw.rectangle((scale_x * x1, scale_y * y1 + 65, scale_x * x2, scale_y * y2 + 65), outline="#00bb00", width=4)
        gt_label = f"G{index} {_name(gt['label'])}" if len(row["image"]["ground_truth"]) <= 5 else f"G{index}"
        _text(draw, (int(scale_x * x1), max(65, int(scale_y * y1) + 65)), gt_label, "#006000", font)
    shown = row["match"]["predictions"][:20]
    for record in shown:
        prediction = row["image"]["predictions"][record["prediction_index"]]
        x1, y1, x2, y2 = prediction["box"]
        color = "#007f9e" if record["status"] == "true_positive" else "#cf2828"
        draw.rectangle((scale_x * x1, scale_y * y1 + 65, scale_x * x2, scale_y * y2 + 65), outline=color, width=3)
        prefix = "TP" if record["status"] == "true_positive" else record["category"]
        suffix = f" IoU {record['iou']:.2f}" if record["iou"] is not None else ""
        label = f"{prefix} {_name(record['label'])} {record['score']:.2f}{suffix}"
        _text(draw, (int(scale_x * x1), min(1060, max(65, int(scale_y * y2) + 42))), label, color, font)
    gt_classes = {}
    for gt in row["image"]["ground_truth"]:
        name = _name(gt["label"])
        gt_classes[name] = gt_classes.get(name, 0) + 1
    classes = ", ".join(f"{name} {count}" for name, count in sorted(gt_classes.items()))
    footer = f"GT: {classes} | detections >=0.50: {len(row['match']['predictions'])} (shown {len(shown)})"
    _text(draw, (12, 1095), footer, "#202020", font)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=90)


def render_examples(
    rows: list[dict[str, Any]], root: Path, destination: Path
) -> dict[str, list[str]]:
    """Select at most three distinct validation images per requested bucket."""
    categories = (
        "strongest_true_positives",
        "high_confidence_false_positives",
        "missed_objects",
        "localization",
        "class_confusion",
        "duplicate",
        "small_object_cases",
        "crowded_scenes",
        "low_ap_panda",
        "low_ap_camel",
        "low_ap_monkeys",
        "low_ap_goat",
    )
    manifest: dict[str, list[str]] = {}
    for category in categories:
        eligible = [
            (score, row["image"]["image_id"], row)
            for row in rows
            if (score := _best_score(row, category)) is not None
        ]
        eligible.sort(key=lambda item: (-item[0], item[1]))
        paths = []
        for rank, (_, image_id, row) in enumerate(eligible[:3], start=1):
            if not image_id.startswith(("valid/images/", "test/images/")):
                raise ValueError(f"Unsupported image identifier in cache: {image_id}")
            tag = hashlib.sha256(image_id.encode("utf-8")).hexdigest()[:8]
            filename = f"{category}_{rank:02d}_{Path(image_id).stem[:42]}_{tag}.jpg"
            path = destination / filename
            _render(row, root / image_id, path, category)
            paths.append(str(path))
        manifest[category] = paths
    return manifest
