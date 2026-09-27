#!/usr/bin/env python3
"""Read-only audit for a YOLO-format object-detection dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "valid", "test")


def describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def yaml_names(value: Any) -> dict[int, str]:
    if isinstance(value, list):
        return {index: str(name) for index, name in enumerate(value)}
    if isinstance(value, dict):
        result = {}
        for key, name in value.items():
            result[int(key)] = str(name)
        return dict(sorted(result.items()))
    return {}


def paired_files(images_dir: Path, labels_dir: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    images = {
        path.stem.casefold(): path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
    } if images_dir.is_dir() else {}
    labels = {
        path.stem.casefold(): path
        for path in labels_dir.glob("*.txt")
        if path.is_file()
    } if labels_dir.is_dir() else {}
    return images, labels


def write_tree(root: Path, split_counts: dict[str, dict[str, int]], output: Path) -> None:
    lines = [f"{root.name}/", "  data.yaml"]
    for split in SPLITS:
        counts = split_counts[split]
        lines.extend([
            f"  {split}/",
            f"    images/  ({counts['images']} files)",
            f"    labels/  ({counts['labels']} files)",
        ])
    other = sorted(p.name for p in root.iterdir() if p.name not in {*SPLITS, "data.yaml"})
    if other:
        lines.append("  Other top-level entries: " + ", ".join(other))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    root = args.dataset_root.expanduser().resolve()
    reports = args.reports_dir.resolve()
    if not root.is_dir():
        parser.error(f"Dataset root does not exist: {root}")
    yaml_path = root / "data.yaml"
    if not yaml_path.is_file():
        parser.error(f"Missing data.yaml: {yaml_path}")
    reports.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    class_names = yaml_names(config.get("names"))
    declared_nc = config.get("nc")
    top_level = sorted(p.name for p in root.iterdir())
    extension_counts = Counter(
        path.suffix.casefold() or "[no extension]" for path in root.rglob("*") if path.is_file()
    )

    path_keys = {"train": "train", "valid": "val", "test": "test"}
    declared_paths = {}
    for split, key in path_keys.items():
        raw = config.get(key)
        resolved = (yaml_path.parent / str(raw)).resolve() if raw is not None else None
        actual = (root / split / "images").resolve()
        declared_paths[split] = {
            "yaml_key": key,
            "declared": raw,
            "resolved_from_yaml": str(resolved) if resolved else None,
            "resolved_exists": bool(resolved and resolved.exists()),
            "actual_detected": str(actual),
            "actual_exists": actual.is_dir(),
        }

    split_counts: dict[str, dict[str, int]] = {}
    split_reports: dict[str, Any] = {}
    dimensions: dict[Path, tuple[int, int]] = {}
    corrupt: list[dict[str, str]] = []
    image_extensions: Counter[str] = Counter()
    resolutions: Counter[str] = Counter()
    image_records: list[tuple[str, Path]] = []
    label_records: list[tuple[str, Path]] = []
    filename_splits: defaultdict[str, list[str]] = defaultdict(list)

    for split in SPLITS:
        images, labels = paired_files(root / split / "images", root / split / "labels")
        split_counts[split] = {"images": len(images), "labels": len(labels)}
        for image in images.values():
            image_records.append((split, image))
            filename_splits[image.name.casefold()].append(split)
            image_extensions[image.suffix.casefold()] += 1
            try:
                with Image.open(image) as opened:
                    opened.verify()
                with Image.open(image) as opened:
                    width, height = opened.size
                dimensions[image] = (width, height)
                resolutions[f"{width}x{height}"] += 1
            except (OSError, ValueError, UnidentifiedImageError) as exc:
                corrupt.append({"path": str(image), "error": str(exc)})
        label_records.extend((split, label) for label in labels.values())

    evidence = {
        "data_yaml_present": yaml_path.is_file(),
        "names_declared": bool(class_names),
        "split_image_directories": sum((root / s / "images").is_dir() for s in SPLITS),
        "split_label_directories": sum((root / s / "labels").is_dir() for s in SPLITS),
        "label_text_files": len(label_records),
    }

    malformed: list[dict[str, Any]] = []
    invalid_boxes: list[dict[str, Any]] = []
    outside_boxes: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    empty_labels: list[str] = []
    missing_labels: list[str] = []
    orphan_labels: list[str] = []
    all_objects: Counter[int] = Counter()
    all_images_per_class: Counter[int] = Counter()
    discovered_classes: set[int] = set()
    object_counts_per_image: list[int] = []
    normalized_widths: list[float] = []
    normalized_heights: list[float] = []
    normalized_areas: list[float] = []
    pixel_widths: list[float] = []
    pixel_heights: list[float] = []
    pixel_areas: list[float] = []
    tiny_boxes = 0
    very_large_boxes = 0
    boundary_boxes = 0
    sample_rows: list[dict[str, Any]] = []

    for split in SPLITS:
        images, labels = paired_files(root / split / "images", root / split / "labels")
        missing_labels.extend(str(images[key]) for key in sorted(images.keys() - labels.keys()))
        orphan_labels.extend(str(labels[key]) for key in sorted(labels.keys() - images.keys()))
        split_objects: Counter[int] = Counter()
        split_images_per_class: Counter[int] = Counter()
        split_object_counts: list[int] = []

        for key, image in sorted(images.items()):
            label = labels.get(key)
            valid_in_image = 0
            classes_in_image: set[int] = set()
            if label is None:
                object_counts_per_image.append(0)
                split_object_counts.append(0)
                continue
            text = label.read_text(encoding="utf-8-sig", errors="replace")
            if not text.strip():
                empty_labels.append(str(label))
            seen_rows: set[str] = set()
            for line_number, raw_line in enumerate(text.splitlines(), 1):
                row = raw_line.strip()
                if not row:
                    continue
                canonical = " ".join(row.split())
                if canonical in seen_rows:
                    duplicate_rows.append({"file": str(label), "line": line_number, "row": row})
                seen_rows.add(canonical)
                fields = row.split()
                if len(fields) != 5:
                    malformed.append({"file": str(label), "line": line_number, "row": row, "reason": "expected 5 fields"})
                    continue
                try:
                    class_value = float(fields[0])
                    values = [float(value) for value in fields[1:]]
                except ValueError:
                    malformed.append({"file": str(label), "line": line_number, "row": row, "reason": "non-numeric field"})
                    continue
                if not class_value.is_integer() or not all(math.isfinite(v) for v in [class_value, *values]):
                    malformed.append({"file": str(label), "line": line_number, "row": row, "reason": "class is not an integer or values are non-finite"})
                    continue
                class_id = int(class_value)
                cx, cy, bw, bh = values
                reasons = []
                if class_id < 0 or (declared_nc is not None and class_id >= int(declared_nc)):
                    reasons.append("class ID outside declared range")
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                    reasons.append("normalized values outside allowed ranges")
                x1n, y1n, x2n, y2n = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
                if x1n < -1e-9 or y1n < -1e-9 or x2n > 1 + 1e-9 or y2n > 1 + 1e-9 or x2n <= x1n or y2n <= y1n:
                    reasons.append("invalid or out-of-bounds normalized geometry")
                if reasons:
                    invalid_boxes.append({"file": str(label), "line": line_number, "row": row, "reasons": reasons})
                discovered_classes.add(class_id)
                all_objects[class_id] += 1
                split_objects[class_id] += 1
                classes_in_image.add(class_id)
                valid_in_image += 1
                normalized_widths.append(bw)
                normalized_heights.append(bh)
                normalized_areas.append(bw * bh)
                if bw * bh < 0.001:
                    tiny_boxes += 1
                if bw * bh > 0.5:
                    very_large_boxes += 1
                size = dimensions.get(image)
                if size:
                    width, height = size
                    coords = [x1n * width, y1n * height, x2n * width, y2n * height]
                    pw, ph = bw * width, bh * height
                    pixel_widths.append(pw)
                    pixel_heights.append(ph)
                    pixel_areas.append(pw * ph)
                    if coords[0] < -1e-6 or coords[1] < -1e-6 or coords[2] > width + 1e-6 or coords[3] > height + 1e-6:
                        outside_boxes.append({"file": str(label), "line": line_number, "pixel_xyxy": coords})
                    if coords[0] <= 1 or coords[1] <= 1 or coords[2] >= width - 1 or coords[3] >= height - 1:
                        boundary_boxes += 1
                    if len(sample_rows) < 12:
                        sample_rows.append({
                            "split": split, "image": image.name, "dimensions": [width, height],
                            "raw": row, "class_id": class_id,
                            "class_name": class_names.get(class_id, "<undeclared>"),
                            "pixel_xyxy": coords,
                        })
            object_counts_per_image.append(valid_in_image)
            split_object_counts.append(valid_in_image)
            for class_id in classes_in_image:
                all_images_per_class[class_id] += 1
                split_images_per_class[class_id] += 1

        total_split_objects = sum(split_objects.values())
        split_reports[split] = {
            "images": len(images), "labels": len(labels), "objects": total_split_objects,
            "objects_per_class": {str(k): v for k, v in sorted(split_objects.items())},
            "images_per_class": {str(k): v for k, v in sorted(split_images_per_class.items())},
            "class_distribution_percent": {
                str(k): (100 * v / total_split_objects if total_split_objects else 0)
                for k, v in sorted(split_objects.items())
            },
            "objects_per_image": describe([float(v) for v in split_object_counts]),
        }

    # Hash all images once and report only groups spanning more than one split.
    hash_groups: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for split, image in image_records:
        try:
            hash_groups[sha256(image)].append({"split": split, "path": str(image)})
        except OSError:
            pass
    cross_split_hashes = [
        {"sha256": digest, "files": files}
        for digest, files in hash_groups.items()
        if len({item["split"] for item in files}) > 1
    ]
    duplicate_names = {
        name: splits for name, splits in sorted(filename_splits.items()) if len(set(splits)) > 1
    }

    total_images = len(image_records)
    stats_counts = Counter(object_counts_per_image)
    format_confirmed = (
        evidence["data_yaml_present"] and evidence["names_declared"]
        and evidence["split_image_directories"] == 3
        and evidence["split_label_directories"] == 3
        and evidence["label_text_files"] > 0
        and not malformed
    )
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "annotation_format": {"detected": "YOLO normalized cxcywh" if format_confirmed else "uncertain", "confirmed": format_confirmed, "evidence": evidence},
        "structure": {"top_level_entries": top_level, "extension_counts": dict(sorted(extension_counts.items())), "splits_present": {s: (root / s).is_dir() for s in SPLITS}},
        "data_yaml": {
            "path": str(yaml_path), "train": config.get("train"), "val": config.get("val"), "test": config.get("test"),
            "nc": declared_nc, "names": {str(k): v for k, v in class_names.items()}, "number_of_names": len(class_names),
            "declared_paths": declared_paths,
            "declared_count_matches_names": declared_nc == len(class_names),
            "declared_count_matches_discovered": declared_nc == len(discovered_classes),
        },
        "images": {
            "total": total_images, "per_split": {s: split_counts[s]["images"] for s in SPLITS},
            "extensions": dict(sorted(image_extensions.items())), "resolution_counts": dict(resolutions.most_common()),
            "unique_resolutions": len(resolutions), "most_common_resolution": resolutions.most_common(1)[0] if resolutions else None,
            "min_width": min((w for w, _ in dimensions.values()), default=None), "max_width": max((w for w, _ in dimensions.values()), default=None),
            "min_height": min((h for _, h in dimensions.values()), default=None), "max_height": max((h for _, h in dimensions.values()), default=None),
            "all_640x640": bool(dimensions) and all(size == (640, 640) for size in dimensions.values()),
            "unreadable_or_corrupt": corrupt,
        },
        "annotations": {
            "label_files": len(label_records), "malformed_rows": malformed, "invalid_boxes": invalid_boxes,
            "boxes_outside_image_after_conversion": outside_boxes, "duplicate_rows": duplicate_rows,
            "empty_label_files": empty_labels, "missing_label_files": missing_labels, "orphan_label_files": orphan_labels,
        },
        "objects": {
            "total": sum(all_objects.values()), "discovered_class_ids": sorted(discovered_classes), "discovered_class_count": len(discovered_classes),
            "objects_per_class": {str(k): v for k, v in sorted(all_objects.items())},
            "images_per_class": {str(k): v for k, v in sorted(all_images_per_class.items())},
            "images_with_zero_objects": stats_counts[0], "images_with_one_object": stats_counts[1],
            "images_with_multiple_objects": sum(v for k, v in stats_counts.items() if k > 1),
            "objects_per_image": describe([float(v) for v in object_counts_per_image]),
        },
        "bounding_boxes": {
            "definitions": {"tiny": "normalized area < 0.001", "very_large": "normalized area > 0.5", "touching_boundary": "pixel edge within 1 px of an image boundary"},
            "normalized_width": describe(normalized_widths), "normalized_height": describe(normalized_heights), "normalized_area": describe(normalized_areas),
            "pixel_width": describe(pixel_widths), "pixel_height": describe(pixel_heights), "pixel_area": describe(pixel_areas),
            "tiny_boxes": tiny_boxes, "very_large_boxes": very_large_boxes, "touching_boundary": boundary_boxes,
        },
        "splits": split_reports,
        "cross_split_checks": {
            "identical_image_paths": [], "duplicate_filenames": duplicate_names,
            "exact_duplicate_image_content": cross_split_hashes,
        },
    }

    json_path = reports / "dataset_audit.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_tree(root, split_counts, reports / "dataset_tree.txt")

    object_lines = [f"  {cid} ({class_names.get(cid, '<undeclared>')}): {count}" for cid, count in sorted(all_objects.items())]
    summary = [
        "DATASET AUDIT", "=============", f"Dataset root: {root}",
        f"Detected format: {report['annotation_format']['detected']}",
        f"Classes: declared={declared_nc}, names={len(class_names)}, discovered={len(discovered_classes)}",
        f"Images: total={total_images}, train={split_counts['train']['images']}, valid={split_counts['valid']['images']}, test={split_counts['test']['images']}",
        f"Objects: {sum(all_objects.values())}", "Objects per class:", *object_lines,
        f"Objects/image: min={report['objects']['objects_per_image']['min']}, max={report['objects']['objects_per_image']['max']}, mean={report['objects']['objects_per_image']['mean']:.4f}, median={report['objects']['objects_per_image']['median']}",
        f"Resolutions: unique={len(resolutions)}, most_common={report['images']['most_common_resolution']}, all_640x640={report['images']['all_640x640']}",
        f"Malformed rows: {len(malformed)}; invalid boxes: {len(invalid_boxes)}; outside after conversion: {len(outside_boxes)}",
        f"Missing labels: {len(missing_labels)}; orphan labels: {len(orphan_labels)}; empty labels: {len(empty_labels)}",
        f"Unreadable/corrupt images: {len(corrupt)}; duplicate annotation rows: {len(duplicate_rows)}",
        f"Cross-split duplicate filenames: {len(duplicate_names)}; cross-split exact-content groups: {len(cross_split_hashes)}",
        "", "Diagnostic box definitions:", "  Tiny: normalized area < 0.001", "  Very large: normalized area > 0.5", "  Touching boundary: pixel edge within 1 px of image boundary",
        "", "Note: declared YAML paths are resolved literally relative to data.yaml; see JSON for actual path checks.",
    ]
    (reports / "dataset_audit.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    samples = ["REPRESENTATIVE YOLO ANNOTATIONS", "===============================", ""]
    for sample in sample_rows:
        samples.extend([
            f"Split: {sample['split']}", f"Image: {sample['image']}", f"Dimensions: {sample['dimensions'][0]} x {sample['dimensions'][1]}",
            f"Raw YOLO: {sample['raw']}", f"Class: {sample['class_id']} -> {sample['class_name']}",
            "Pixel xyxy: " + ", ".join(f"{value:.3f}" for value in sample["pixel_xyxy"]), "",
        ])
    (reports / "sample_annotations.txt").write_text("\n".join(samples), encoding="utf-8")
    print(f"Audit complete: {json_path}")
    print(f"Detected format: {report['annotation_format']['detected']}")
    print(f"Images: {total_images}; objects: {sum(all_objects.values())}")
    return 0 if format_confirmed else 2


if __name__ == "__main__":
    raise SystemExit(main())
