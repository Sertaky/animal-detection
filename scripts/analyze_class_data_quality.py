#!/usr/bin/env python3
"""Audit class-specific train/validation geometry and generate review-only visuals."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from animal_detection.analysis.class_data_quality import (
    aggregate_objects,
    split_shift,
    touches_boundary,
)
from animal_detection.data import YoloDetectionDataset
from animal_detection.data.label_mapping import ANIMAL_CLASS_NAMES

WEAK_CLASSES = ("Panda", "Monkeys", "Gorilla", "Goat", "Camel")
REFERENCE_CLASSES = ("Dog", "Wolf", "Rhino", "Lion")
CONTEXT_CLASSES = ("Deer",)
AUDIT_CLASSES = WEAK_CLASSES + REFERENCE_CLASSES + CONTEXT_CLASSES
PAIR_CLASSES = (("Monkeys", "Gorilla"), ("Goat", "Camel"), ("Goat", "Deer"))
SPLITS = ("train", "valid")
COLORS = ("#ef4444", "#22c55e", "#3b82f6", "#eab308", "#a855f7", "#06b6d4")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def train_valid_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for split in SPLITS:
        for folder in ("images", "labels"):
            for path in sorted((root / split / folder).iterdir()):
                if not path.is_file():
                    continue
                digest.update(path.relative_to(root).as_posix().encode("utf-8"))
                digest.update(b"\0")
                with path.open("rb") as source:
                    while block := source.read(1024 * 1024):
                        digest.update(block)
    return digest.hexdigest()


def scan(root: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    records: list[dict[str, Any]] = []
    objects_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for split in SPLITS:
        dataset = YoloDetectionDataset(root, split)
        for index in range(len(dataset)):
            image, target = dataset[index]
            width, height = image.size
            boxes = target["boxes"].tolist()
            labels = target["labels"].tolist()
            image_id = f"{split}/images/{Path(target['image_path']).name}"
            record = {
                "image_id": image_id,
                "split": split,
                "image_path": str(target["image_path"]),
                "label_path": str(dataset.label_paths[index]),
                "width": width,
                "height": height,
                "objects": [],
            }
            for object_index, (box, class_id) in enumerate(zip(boxes, labels)):
                x1, y1, x2, y2 = (float(value) for value in box)
                normalized_width = (x2 - x1) / width
                normalized_height = (y2 - y1) / height
                item = {
                    "image_id": image_id,
                    "split": split,
                    "image_path": str(target["image_path"]),
                    "label_path": str(dataset.label_paths[index]),
                    "object_index": object_index,
                    "class_id": int(class_id),
                    "class_name": ANIMAL_CLASS_NAMES[class_id],
                    "box": [x1, y1, x2, y2],
                    "width": normalized_width,
                    "height": normalized_height,
                    "area": normalized_width * normalized_height,
                    "aspect_ratio": normalized_width / normalized_height,
                    "touches_boundary": touches_boundary(box, width, height),
                    "image_object_count": len(boxes),
                    "image_width": width,
                    "image_height": height,
                }
                record["objects"].append(item)
                if item["class_name"] in AUDIT_CLASSES:
                    objects_by_class[item["class_name"]].append(item)
            records.append(record)
    return records, objects_by_class


def class_statistics(objects_by_class: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "definitions": {
            "size_bins": {"small": "area < 0.10", "medium": "0.10 <= area < 0.40", "large": "area >= 0.40"},
            "boundary_touch": "box edge within 1 pixel of image boundary",
            "crowdedness": {"1": "1 total annotated object", "2-3": "2-3 total annotated objects", "4+": "4+ total annotated objects"},
            "shift_flags": {
                "median_area": "absolute train/valid delta >= 0.10",
                "small_share": "absolute delta >= 15 percentage points",
                "boundary": "absolute delta >= 10 percentage points",
                "multi_object_images": "absolute delta >= 20 percentage points",
                "aspect_ratio": "median relative delta >= 35%",
            },
        },
        "groups": {"weak_confused": list(WEAK_CLASSES), "strong_reference": list(REFERENCE_CLASSES), "confusion_context": list(CONTEXT_CLASSES)},
        "classes": {},
    }
    for class_name in AUDIT_CLASSES:
        objects = objects_by_class[class_name]
        train = aggregate_objects([item for item in objects if item["split"] == "train"])
        valid = aggregate_objects([item for item in objects if item["split"] == "valid"])
        combined = aggregate_objects(objects)
        result["classes"][class_name] = {
            "train": train,
            "valid": valid,
            "combined": combined,
            "train_vs_valid": split_shift(train, valid),
        }
    return result


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, color: str, font: ImageFont.ImageFont) -> None:
    bounds = draw.textbbox(xy, text, font=font, stroke_width=1)
    draw.rectangle(bounds, fill=color)
    draw.text(xy, text, fill="white", font=font, stroke_width=1, stroke_fill="black")


def annotated_image(record: dict[str, Any], target_class: str, title: str, max_size: tuple[int, int] = (1100, 760)) -> Image.Image:
    with Image.open(record["image_path"]) as source:
        original = source.convert("RGB")
    scale = min(max_size[0] / original.width, max_size[1] / original.height, 1.0)
    resized = original.resize((round(original.width * scale), round(original.height * scale)), Image.Resampling.LANCZOS)
    banner = 64
    canvas = Image.new("RGB", (resized.width, resized.height + banner), "white")
    canvas.paste(resized, (0, banner))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default(size=18)
    draw.text((8, 6), title, fill="black", font=font)
    draw.text((8, 32), f"{record['split']} | total objects={len(record['objects'])} | {Path(record['image_path']).name}", fill="black", font=font)
    for item in record["objects"]:
        x1, y1, x2, y2 = item["box"]
        box = (x1 * scale, y1 * scale + banner, x2 * scale, y2 * scale + banner)
        selected = item["class_name"] == target_class
        color = "#ef4444" if selected else "#64748b"
        draw.rectangle(box, outline=color, width=4 if selected else 2)
        text = item["class_name"]
        if selected:
            text += f" area={item['area']:.3f}" + (" boundary" if item["touches_boundary"] else "")
        draw_label(draw, (max(0, int(box[0])), max(banner, int(box[1]))), text, color, font)
    return canvas


def choose_records(
    class_name: str, records: list[dict[str, Any]], objects: list[dict[str, Any]], seed: int
) -> dict[str, list[dict[str, Any]]]:
    by_id = {record["image_id"]: record for record in records}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in objects:
        grouped[item["image_id"]].append(item)
    candidates = [by_id[image_id] for image_id in grouped]

    def unique_sorted(key: Any, reverse: bool = False, eligible: Any = None) -> list[dict[str, Any]]:
        pool = [record for record in candidates if eligible is None or eligible(record)]
        return sorted(pool, key=lambda record: (key(record), record["image_id"]), reverse=reverse)[:3]

    smallest = unique_sorted(lambda record: min(item["area"] for item in grouped[record["image_id"]]))
    largest = unique_sorted(lambda record: max(item["area"] for item in grouped[record["image_id"]]), reverse=True)
    boundary = unique_sorted(
        lambda record: sum(item["touches_boundary"] for item in grouped[record["image_id"]]),
        reverse=True,
        eligible=lambda record: any(item["touches_boundary"] for item in grouped[record["image_id"]]),
    )
    crowded = unique_sorted(lambda record: len(record["objects"]), reverse=True, eligible=lambda record: len(record["objects"]) >= 4)
    areas = np.asarray([item["area"] for item in objects])
    q1, q3 = np.quantile(areas, [0.25, 0.75])
    typical_pool = [
        record for record in candidates
        if any(q1 <= item["area"] <= q3 for item in grouped[record["image_id"]])
    ]
    typical_pool.sort(key=lambda record: record["image_id"])
    typical = random.Random(seed).sample(typical_pool, min(3, len(typical_pool)))
    return {"smallest": smallest, "largest": largest, "boundary": boundary, "crowded": crowded, "typical": typical}


def save_class_visuals(
    records: list[dict[str, Any]], objects_by_class: dict[str, list[dict[str, Any]]], output: Path, seed: int
) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    overview_dir = output / "overviews"
    overview_dir.mkdir(parents=True, exist_ok=True)
    for class_index, class_name in enumerate(WEAK_CLASSES):
        selections = choose_records(class_name, records, objects_by_class[class_name], seed + class_index)
        manifest[class_name] = {}
        thumbnails: list[tuple[str, Image.Image]] = []
        for category, chosen in selections.items():
            paths = []
            category_dir = output / class_name.lower() / category
            category_dir.mkdir(parents=True, exist_ok=True)
            for rank, record in enumerate(chosen, 1):
                image = annotated_image(record, class_name, f"{class_name}: {category} #{rank}")
                destination = category_dir / f"{rank:02d}_{record['split']}_{Path(record['image_path']).stem}.jpg"
                image.save(destination, quality=92)
                paths.append({"output": str(destination), "source": record["image_path"], "split": record["split"], "object_count": len(record["objects"])})
                thumbnail = image.copy()
                thumbnail.thumbnail((360, 260), Image.Resampling.LANCZOS)
                thumbnails.append((f"{category} #{rank}", thumbnail))
            manifest[class_name][category] = paths
        sheet = Image.new("RGB", (3 * 370, 5 * 300), "white")
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default(size=16)
        for index, (label, thumb) in enumerate(thumbnails):
            x, y = (index % 3) * 370, (index // 3) * 300
            sheet.paste(thumb, (x, y + 25))
            draw.text((x + 4, y + 4), label, fill="black", font=font)
        overview = overview_dir / f"{class_name.lower()}_review_overview.jpg"
        sheet.save(overview, quality=92)
        manifest[class_name]["overview"] = str(overview)
    return manifest


def representative(record: dict[str, Any], class_name: str) -> dict[str, Any]:
    items = [item for item in record["objects"] if item["class_name"] == class_name]
    return min(items, key=lambda item: abs(item["area"] - np.median([entry["area"] for entry in items])))


def save_confusion_pairs(
    records: list[dict[str, Any]], objects_by_class: dict[str, list[dict[str, Any]]], output: Path
) -> dict[str, Any]:
    by_id = {record["image_id"]: record for record in records}
    manifest: dict[str, Any] = {}
    output.mkdir(parents=True, exist_ok=True)
    for left_name, right_name in PAIR_CLASSES:
        left_records = list({item["image_id"]: by_id[item["image_id"]] for item in objects_by_class[left_name]}.values())
        right_records = list({item["image_id"]: by_id[item["image_id"]] for item in objects_by_class[right_name]}.values())
        left_areas = np.asarray([item["area"] for item in objects_by_class[left_name]])
        median = float(np.median(left_areas))
        left_records.sort(key=lambda record: (abs(representative(record, left_name)["area"] - median), record["image_id"]))
        used_right: set[str] = set()
        paths = []
        for rank, left in enumerate(left_records[:3], 1):
            left_item = representative(left, left_name)
            available = [record for record in right_records if record["image_id"] not in used_right]
            right = min(
                available,
                key=lambda record: (
                    abs(math.log(max(representative(record, right_name)["area"], 1e-9) / max(left_item["area"], 1e-9)))
                    + 0.35 * abs(len(record["objects"]) - len(left["objects"])),
                    record["image_id"],
                ),
            )
            used_right.add(right["image_id"])
            left_image = annotated_image(left, left_name, f"{left_name} representative")
            right_image = annotated_image(right, right_name, f"{right_name} matched scale/crowding")
            left_image.thumbnail((620, 650), Image.Resampling.LANCZOS)
            right_image.thumbnail((620, 650), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (left_image.width + right_image.width + 10, max(left_image.height, right_image.height)), "white")
            canvas.paste(left_image, (0, 0))
            canvas.paste(right_image, (left_image.width + 10, 0))
            destination = output / f"{left_name.lower()}_vs_{right_name.lower()}_{rank:02d}.jpg"
            canvas.save(destination, quality=92)
            paths.append({
                "output": str(destination),
                "left_source": left["image_path"], "right_source": right["image_path"],
                "left_area": left_item["area"], "right_area": representative(right, right_name)["area"],
                "left_object_count": len(left["objects"]), "right_object_count": len(right["objects"]),
            })
        manifest[f"{left_name}_vs_{right_name}"] = paths
    return manifest


def write_csv(path: Path, statistics: dict[str, Any]) -> None:
    columns = (
        "class_name", "group", "split", "image_count", "object_count", "objects_per_image",
        "area_mean", "area_median", "area_q1", "area_q3", "width_median", "height_median",
        "aspect_ratio_median", "small_percent", "medium_percent", "large_percent",
        "boundary_touch_percent", "multi_object_image_percent", "objects_in_1_percent",
        "objects_in_2_3_percent", "objects_in_4plus_percent", "shift_flags",
    )
    group_by_class = {name: group for group, names in statistics["groups"].items() for name in names}
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        for class_name in AUDIT_CLASSES:
            shifts = statistics["classes"][class_name]["train_vs_valid"]["flags"]
            for split in ("train", "valid", "combined"):
                item = statistics["classes"][class_name][split]
                writer.writerow({
                    "class_name": class_name, "group": group_by_class[class_name], "split": split,
                    "image_count": item["image_count"], "object_count": item["object_count"],
                    "objects_per_image": item["objects_per_image"], "area_mean": item["normalized_area"]["mean"],
                    "area_median": item["normalized_area"]["median"], "area_q1": item["normalized_area"]["q1"],
                    "area_q3": item["normalized_area"]["q3"], "width_median": item["normalized_width"]["median"],
                    "height_median": item["normalized_height"]["median"], "aspect_ratio_median": item["aspect_ratio"]["median"],
                    "small_percent": item["size_percent"]["small"], "medium_percent": item["size_percent"]["medium"],
                    "large_percent": item["size_percent"]["large"], "boundary_touch_percent": item["boundary_touch_percent"],
                    "multi_object_image_percent": item["multi_object_image_percent"],
                    "objects_in_1_percent": item["object_crowdedness_percent"]["1"],
                    "objects_in_2_3_percent": item["object_crowdedness_percent"]["2-3"],
                    "objects_in_4plus_percent": item["object_crowdedness_percent"]["4+"],
                    "shift_flags": ";".join(shifts) if split == "combined" else "",
                })


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}%"


def write_summary(path: Path, statistics: dict[str, Any], manual: dict[str, Any], suspicious: list[dict[str, Any]]) -> None:
    lines = [
        "CLASS-FOCUSED DATA AUDIT (TRAIN + VALID ONLY)",
        "",
        "Combined structural statistics:",
        "Class      Objects Images MedianArea Small% Boundary% MultiObjectImages% ShiftFlags",
    ]
    for name in AUDIT_CLASSES:
        item = statistics["classes"][name]["combined"]
        flags = statistics["classes"][name]["train_vs_valid"]["flags"]
        lines.append(
            f"{name:<10} {item['object_count']:>7} {item['image_count']:>6} "
            f"{item['normalized_area']['median']:>10.3f} {pct(item['size_percent']['small']):>6} "
            f"{pct(item['boundary_touch_percent']):>9} {pct(item['multi_object_image_percent']):>18} "
            f"{','.join(flags) if flags else 'none'}"
        )
    lines.extend(["", "Evidence-based questions:"])
    answers = manual.get("answers", {})
    questions = (
        ("panda_small", "1. Is Panda weakness associated with small objects?"),
        ("panda_crowded", "2. Is Panda weakness associated with crowded scenes?"),
        ("panda_annotations", "3. Are Panda annotations visibly problematic?"),
        ("primates", "4. Is Monkeys/Gorilla confusion visually plausible or annotation-ambiguous?"),
        ("ungulates", "5. Are Goat/Camel/Deer distributions unusually similar?"),
        ("weak_smaller", "6. Are weak classes systematically smaller than strong classes?"),
        ("weak_crowded", "7. Are weak classes more crowded?"),
        ("weak_boundary", "8. Are weak classes more boundary-truncated?"),
        ("split_shifts", "9. Are there notable train/validation distribution shifts?"),
        ("quality_concerns", "10. Are there new annotation-quality concerns?"),
    )
    for key, question in questions:
        lines.append(question)
        lines.append(f"   {answers.get(key, 'Manual visual review pending.')}" )
    lines.extend([
        "",
        f"Suspicious review queue: {len(suspicious)} samples (review only; no edits or exclusions applied).",
        "",
        "At most two possible Experiment 03 directions:",
        *[f"- {item}" for item in manual.get("recommendations", ["Pending completion of manual visual review."])[:2]],
        "",
        "No model training, test-split access, annotation changes, or split changes occurred.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/class_data_audit"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    output = args.output_dir.resolve()
    if not root.is_dir():
        parser.error(f"Dataset root does not exist: {root}")
    if output == root or root in output.parents:
        parser.error("Output must remain outside the immutable raw dataset")
    output.mkdir(parents=True, exist_ok=True)
    hash_before = train_valid_hash(root)
    records, objects_by_class = scan(root)
    statistics = class_statistics(objects_by_class)
    write_json(output / "class_statistics.json", statistics)
    write_csv(output / "class_statistics.csv", statistics)
    visualizations = save_class_visuals(records, objects_by_class, output / "visualizations", args.seed)
    confusion_pairs = save_confusion_pairs(records, objects_by_class, output / "confusion_pairs")
    write_json(output / "visualizations.json", visualizations)
    write_json(output / "confusion_pairs.json", confusion_pairs)

    manual_path = output / "manual_review.json"
    manual = json.loads(manual_path.read_text(encoding="utf-8")) if manual_path.exists() else {}
    manifest = json.loads((PROJECT_ROOT / "data" / "manifests" / "excluded_samples.json").read_text(encoding="utf-8"))
    suspicious = [
        {
            "split": item["split"], "image_filename": item["image_filename"],
            "reason": "empty_annotation_with_visible_object", "evidence": item["reason"],
            "source": "previously confirmed exclusion manifest", "action": "review_only_no_change",
        }
        for item in manifest["samples"]
    ] + manual.get("suspicious_samples", [])
    write_json(output / "suspicious_samples.json", {"count": len(suspicious), "samples": suspicious})
    write_summary(output / "summary.txt", statistics, manual, suspicious)
    hash_after = train_valid_hash(root)
    if hash_after != hash_before:
        raise RuntimeError("Raw train/validation data changed during audit")
    integrity = {
        "raw_train_valid_sha256_before": hash_before,
        "raw_train_valid_sha256_after": hash_after,
        "raw_data_unchanged": True,
        "splits_constructed": list(SPLITS),
        "test_split_constructed": False,
        "training_performed": False,
        "optimizer_constructed": False,
        "annotation_or_split_changes": False,
        "record_counts": {split: sum(record["split"] == split for record in records) for split in SPLITS},
    }
    write_json(output / "integrity.json", integrity)
    print(f"records={len(records)} classes={len(AUDIT_CLASSES)}")
    print(f"class_visualizations={sum(len(paths) for value in visualizations.values() for key, paths in value.items() if key != 'overview')}")
    print(f"overviews={len(WEAK_CLASSES)} confusion_pairs={sum(len(paths) for paths in confusion_pairs.values())}")
    print(f"raw_train_valid_sha256={hash_before} unchanged=True test_accessed=False training=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
