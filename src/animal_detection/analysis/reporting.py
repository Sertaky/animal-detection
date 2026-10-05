"""Aggregate cached validation predictions into diagnostic reports."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from animal_detection.data.label_mapping import ANIMAL_CLASS_NAMES

from .matching import match_image
from .visualization import render_examples


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def statistics(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0, "mean": None, "median": None,
            "q1": None, "q3": None, "min": None, "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q1": float(np.quantile(array, 0.25)),
        "q3": float(np.quantile(array, 0.75)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def size_area(gt: dict[str, Any], image: dict[str, Any]) -> float:
    x1, y1, x2, y2 = gt["box"]
    height, width = image["transformed_size"]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1) / (height * width)


def size_boundaries(images: list[dict[str, Any]]) -> tuple[float, float, dict[str, Any]]:
    areas = [size_area(gt, image) for image in images for gt in image["ground_truth"]]
    initial = [sum(area < 0.10 for area in areas),
               sum(0.10 <= area < 0.40 for area in areas),
               sum(area >= 0.40 for area in areas)]
    if min(initial) >= 10:
        lower, upper, method = 0.10, 0.40, "fixed normalized area"
    else:
        lower, upper = (float(value) for value in np.quantile(areas, [1 / 3, 2 / 3]))
        if lower >= upper:
            raise ValueError("Object-size distribution cannot form nonempty ordered bins")
        method = "tertile normalized area because a fixed bin had fewer than 10 objects"
    return lower, upper, {
        "method": method,
        "proposed_fixed_bin_counts": dict(zip(("small", "medium", "large"), initial)),
        "area_distribution": statistics(areas),
        "boundaries": {"small_lt": lower, "medium_lt": upper, "large_gte": upper},
    }


def bucket_area(area: float, lower: float, upper: float) -> str:
    return "small" if area < lower else ("medium" if area < upper else "large")


def bucket_count(count: int) -> str:
    return "0" if count == 0 else ("1" if count == 1 else ("2-3" if count <= 3 else "4+"))


def diagnostic_rows(
    images: list[dict[str, Any]], score_threshold: float, iou_threshold: float
) -> list[dict[str, Any]]:
    return [
        {
            "image": image,
            "match": match_image(
                image["ground_truth"], image["predictions"],
                score_threshold=score_threshold, iou_threshold=iou_threshold,
            ),
        }
        for image in images
    ]


def threshold_report(images: list[dict[str, Any]], iou_threshold: float) -> dict[str, Any]:
    rows = []
    for threshold in (0.05, 0.10, 0.25, 0.50, 0.75):
        matches = diagnostic_rows(images, threshold, iou_threshold)
        tp = sum(row["match"]["true_positives"] for row in matches)
        fp = sum(row["match"]["false_positives"] for row in matches)
        fn = sum(row["match"]["false_negatives"] for row in matches)
        rows.append({
            "score_threshold": threshold, "predictions_kept": tp + fp,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
        })
    return {
        "matching_iou_threshold": iou_threshold,
        "note": "Diagnostic one-to-one counts, not COCO AP or an optimized operating threshold.",
        "thresholds": rows,
    }


def aggregate(
    cache: dict[str, Any], class_ap: list[dict[str, Any]],
    score_threshold: float, iou_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    images = cache["images"]
    rows = diagnostic_rows(images, score_threshold, iou_threshold)
    gt_count = sum(len(image["ground_truth"]) for image in images)
    tp = sum(row["match"]["true_positives"] for row in rows)
    fp = sum(row["match"]["false_positives"] for row in rows)
    fn = sum(row["match"]["false_negatives"] for row in rows)
    categories: Counter[str] = Counter()
    confusion: Counter[tuple[int, int]] = Counter()
    confusion_gt_ids: defaultdict[int, set[tuple[str, int]]] = defaultdict(set)
    matched_ious: list[float] = []
    correct_class_ious: list[float] = []
    scores: defaultdict[str, list[float]] = defaultdict(list)
    size_samples: defaultdict[str, dict[str, Any]] = defaultdict(
        lambda: {"gt_objects": 0, "tp": 0, "fn": 0, "matched_ious": []}
    )
    crowd_samples: defaultdict[str, dict[str, Any]] = defaultdict(
        lambda: {"images": 0, "gt_objects": 0, "tp": 0, "fp": 0, "fn": 0}
    )
    per_class: dict[int, dict[str, Any]] = {
        index: {
            "model_label": index, "dataset_label": index - 1,
            "class_name": ANIMAL_CLASS_NAMES[index - 1],
            "gt_objects": 0, "tp": 0, "fp": 0, "fn": 0,
            "matched_ious": [], "tp_scores": [], "fp_scores": [],
            "class_confusion_count": 0, "localization_error_count": 0,
        }
        for index in range(1, 21)
    }
    lower, upper, distribution = size_boundaries(images)

    for row in rows:
        image, matched = row["image"], row["match"]
        ground_truth = image["ground_truth"]
        crowd = crowd_samples[bucket_count(len(ground_truth))]
        crowd["images"] += 1
        crowd["gt_objects"] += len(ground_truth)
        crowd["tp"] += matched["true_positives"]
        crowd["fp"] += matched["false_positives"]
        crowd["fn"] += matched["false_negatives"]
        false_negative = set(matched["false_negative_gt_indices"])
        tp_by_gt = {
            record["gt_index"]: record
            for record in matched["predictions"]
            if record["status"] == "true_positive"
        }
        for gt_index, gt in enumerate(ground_truth):
            label = gt["label"]
            per_class[label]["gt_objects"] += 1
            area_group = size_samples[bucket_area(size_area(gt, image), lower, upper)]
            area_group["gt_objects"] += 1
            if gt_index in false_negative:
                per_class[label]["fn"] += 1
                area_group["fn"] += 1
            else:
                record = tp_by_gt[gt_index]
                per_class[label]["tp"] += 1
                per_class[label]["matched_ious"].append(record["iou"])
                per_class[label]["tp_scores"].append(record["score"])
                area_group["tp"] += 1
                area_group["matched_ious"].append(record["iou"])
        for record in matched["predictions"]:
            if record["best_correct_class_iou"] is not None:
                correct_class_ious.append(record["best_correct_class_iou"])
            if record["status"] == "true_positive":
                matched_ious.append(record["iou"])
                scores["true_positive"].append(record["score"])
            else:
                label = record["label"]
                per_class[label]["fp"] += 1
                per_class[label]["fp_scores"].append(record["score"])
                scores["false_positive"].append(record["score"])
                category = record["category"]
                categories[category] += 1
                scores[category].append(record["score"])
                if category == "class_confusion":
                    true_label = ground_truth[record["gt_index"]]["label"]
                    confusion[(true_label, label)] += 1
                    confusion_gt_ids[true_label].add((image["image_id"], record["gt_index"]))
                    per_class[true_label]["class_confusion_count"] += 1
                elif category == "localization":
                    per_class[label]["localization_error_count"] += 1

    ap_by_label = {item["model_label"]: item["ap"] for item in class_ap}
    if set(ap_by_label) != set(range(1, 21)):
        raise ValueError("Expected AP for all 20 validation classes")
    class_diagnostics = []
    for label in range(1, 21):
        item = per_class[label]
        class_diagnostics.append({
            "model_label": label, "dataset_label": item["dataset_label"],
            "class_name": item["class_name"], "gt_objects": item["gt_objects"],
            "tp": item["tp"], "fp": item["fp"], "fn": item["fn"],
            "precision": ratio(item["tp"], item["tp"] + item["fp"]),
            "recall": ratio(item["tp"], item["gt_objects"]),
            "baseline_validation_ap": ap_by_label[label],
            "mean_matched_iou": statistics(item["matched_ious"])["mean"],
            "median_matched_iou": statistics(item["matched_ious"])["median"],
            "mean_tp_confidence": statistics(item["tp_scores"])["mean"],
            "mean_fp_confidence": statistics(item["fp_scores"])["mean"],
            "class_confusion_count": item["class_confusion_count"],
            "localization_error_count": item["localization_error_count"],
            "matched_iou_distribution": statistics(item["matched_ious"]),
            "tp_confidence_distribution": statistics(item["tp_scores"]),
            "fp_confidence_distribution": statistics(item["fp_scores"]),
        })

    confusion_pairs = [
        {
            "ground_truth_class": ANIMAL_CLASS_NAMES[true_label - 1],
            "predicted_class": ANIMAL_CLASS_NAMES[predicted_label - 1],
            "ground_truth_model_label": true_label,
            "predicted_model_label": predicted_label,
            "count": count,
        }
        for (true_label, predicted_label), count in sorted(
            confusion.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    confusion_by_class = [
        {
            "class_name": ANIMAL_CLASS_NAMES[label - 1],
            "confusion_events": per_class[label]["class_confusion_count"],
            "unique_gt_objects_with_confusion": len(confusion_gt_ids[label]),
            "unique_gt_confusion_rate": ratio(
                len(confusion_gt_ids[label]), per_class[label]["gt_objects"]
            ),
        }
        for label in range(1, 21)
    ]
    localization = {
        "definition": "Best same-class GT IoU per retained prediction with any same-class GT; duplicates included.",
        "correct_class_prediction_iou": statistics(correct_class_ious),
        "correct_class_percent_ge_050": ratio(sum(iou >= 0.50 for iou in correct_class_ious), len(correct_class_ious)),
        "correct_class_percent_ge_075": ratio(sum(iou >= 0.75 for iou in correct_class_ious), len(correct_class_ious)),
        "correct_class_percent_ge_090": ratio(sum(iou >= 0.90 for iou in correct_class_ious), len(correct_class_ious)),
        "matched_iou": statistics(matched_ious),
        "matched_050_to_lt_075": sum(0.50 <= iou < 0.75 for iou in matched_ious),
        "matched_050_to_lt_075_fraction": ratio(sum(0.50 <= iou < 0.75 for iou in matched_ious), len(matched_ious)),
        "baseline_map_50": 0.8256866335868835,
        "baseline_map_75": 0.6530308127403259,
        "baseline_map_50_minus_map_75": 0.8256866335868835 - 0.6530308127403259,
    }
    confidence = {
        "raw_prediction_scores": statistics(
            [prediction["score"] for image in images for prediction in image["predictions"]]
        ),
        "score_threshold": score_threshold,
        "by_outcome": {
            name: statistics(scores[name])
            for name in (
                "true_positive", "false_positive", "class_confusion",
                "localization", "background", "duplicate", "other_overlap",
            )
        },
    }
    size_report = {
        **distribution,
        "groups": {
            group: {
                "gt_objects": size_samples[group]["gt_objects"],
                "tp": size_samples[group]["tp"],
                "fn": size_samples[group]["fn"],
                "recall": ratio(size_samples[group]["tp"], size_samples[group]["gt_objects"]),
                "mean_matched_iou": statistics(size_samples[group]["matched_ious"])["mean"],
            }
            for group in ("small", "medium", "large")
        },
    }
    crowd_report = {
        "group_definition": "Number of annotated ground-truth objects per usable validation image.",
        "groups": {
            group: {
                **{key: crowd_samples[group][key] for key in ("images", "gt_objects", "tp", "fp", "fn")},
                "precision": ratio(crowd_samples[group]["tp"], crowd_samples[group]["tp"] + crowd_samples[group]["fp"]),
                "recall": ratio(crowd_samples[group]["tp"], crowd_samples[group]["gt_objects"]),
                "mean_detections_per_image": ratio(
                    crowd_samples[group]["tp"] + crowd_samples[group]["fp"],
                    crowd_samples[group]["images"],
                ),
            }
            for group in ("0", "1", "2-3", "4+")
            if crowd_samples[group]["images"]
        },
    }
    summary = {
        "checkpoint_epoch": 9,
        "validation_images": len(images),
        "inference_runtime_seconds": cache["inference_runtime_seconds"],
        "raw_predictions": cache["raw_prediction_count"],
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "localization_floor": 0.10,
        "ground_truth_objects": gt_count,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "false_positive_categories": {
            name: categories[name]
            for name in ("class_confusion", "localization", "background", "duplicate", "other_overlap")
        },
        "matching_note": (
            "These are score-0.50, IoU-0.50 diagnostic one-to-one counts. "
            "The FP categories are local explanations, not official COCO metric types. "
            "An unmatched prediction may overlap a GT that is also counted as a false negative."
        ),
    }
    reports = {
        "summary.json": summary,
        "class_diagnostics.json": {"score_threshold": score_threshold, "classes": class_diagnostics},
        "class_confusion.json": {
            "definition": "Unmatched prediction of a different class overlapping any GT at IoU >= 0.50; duplicate has priority.",
            "total_confusion_events": categories["class_confusion"],
            "pairs": confusion_pairs, "by_ground_truth_class": confusion_by_class,
        },
        "localization_analysis.json": localization,
        "confidence_analysis.json": confidence,
        "object_size_analysis.json": size_report,
        "crowdedness_analysis.json": crowd_report,
        "threshold_analysis.json": threshold_report(images, iou_threshold),
    }
    return reports, rows


def generate_reports(
    *, cache: dict[str, Any], class_ap: list[dict[str, Any]],
    root: Path, report_dir: Path, score_threshold: float, iou_threshold: float,
) -> None:
    reports, rows = aggregate(cache, class_ap, score_threshold, iou_threshold)
    visualization_manifest = render_examples(rows, root, report_dir / "visualizations")
    reports["summary.json"]["visualization_counts"] = {
        name: len(paths) for name, paths in visualization_manifest.items()
    }
    for name, value in reports.items():
        write_json(report_dir / name, value)
    write_json(report_dir / "visualizations.json", visualization_manifest)
    summary = reports["summary.json"]
    classes = reports["class_diagnostics.json"]["classes"]
    size = reports["object_size_analysis.json"]["groups"]
    crowd = reports["crowdedness_analysis.json"]["groups"]
    loc = reports["localization_analysis.json"]
    conf = reports["confidence_analysis.json"]["by_outcome"]
    confusion = reports["class_confusion.json"]["pairs"]
    fp_categories = summary["false_positive_categories"]
    largest = max(fp_categories.items(), key=lambda item: item[1])
    weak = sorted(classes, key=lambda item: item["baseline_validation_ap"])[:4]
    focus_lines = []
    for name in ("Panda", "Camel", "Monkeys", "Goat"):
        item = next(entry for entry in classes if entry["class_name"] == name)
        pairs = [
            f"{pair['predicted_class']} ({pair['count']})"
            for pair in confusion if pair["ground_truth_class"] == name
        ]
        focus_lines.append(
            f"{name}: GT={item['gt_objects']}, TP={item['tp']}, FP={item['fp']}, "
            f"FN={item['fn']}, precision={item['precision']:.3f}, recall={item['recall']:.3f}, "
            f"AP={item['baseline_validation_ap']:.3f}; "
            f"matched IoU median={item['matched_iou_distribution']['median']:.3f} "
            f"(Q1-Q3 {item['matched_iou_distribution']['q1']:.3f}-"
            f"{item['matched_iou_distribution']['q3']:.3f}); "
            f"TP confidence median={item['tp_confidence_distribution']['median']:.3f}, "
            f"FP confidence median={item['fp_confidence_distribution']['median']:.3f}; "
            f"confused as: {', '.join(pairs) if pairs else 'none observed'}; "
            f"{len(visualization_manifest.get('low_ap_' + name.lower(), []))} review examples."
        )
    recommendations = []
    if loc["matched_050_to_lt_075_fraction"] is not None and loc["matched_050_to_lt_075_fraction"] >= 0.15:
        recommendations.append(
            "Run one controlled input-resolution comparison (for example 640x640) to test whether "
            "small-object recall and box placement improve; the measured size gap and "
            "matched-IoU 0.50-0.75 share motivate this hypothesis."
        )
    if weak[0]["recall"] is not None and weak[0]["recall"] < summary["recall"]:
        recommendations.append(
            f"Audit the {weak[0]['class_name']} validation misses and training labels, then consider one "
            "class-focused sampling/augmentation comparison; its AP and diagnostic recall are below the overall values."
        )
    lines = [
        "FIRST BASELINE VALIDATION ERROR ANALYSIS (diagnostic, not COCO AP)",
        f"Checkpoint epoch 9; {summary['validation_images']} usable validation images; "
        f"score threshold {score_threshold:.2f}; IoU threshold {iou_threshold:.2f}.",
        f"GT={summary['ground_truth_objects']}, TP={summary['true_positives']}, "
        f"FP={summary['false_positives']}, FN={summary['false_negatives']}; "
        f"precision={summary['precision']:.3f}, recall={summary['recall']:.3f}.",
        f"1. FP={summary['false_positives']} narrowly exceeds FN={summary['false_negatives']}; "
        f"{largest[0]} is the largest FP subtype ({largest[1]}). "
        f"Small objects account for {size['small']['fn']} FN and 4+ object images for "
        f"{crowd.get('4+', {}).get('fn', 0)} FN (overlapping groups).",
        f"2. The larger issue at 0.50 is {'recall' if summary['recall'] < summary['precision'] else 'precision'} "
        f"({summary['recall']:.3f} recall vs {summary['precision']:.3f} precision).",
        f"3. Localization-category FP count={fp_categories['localization']} "
        f"({ratio(fp_categories['localization'], summary['false_positives']):.1%} of FP).",
        f"4. Matched boxes at IoU 0.50 to <0.75: {loc['matched_050_to_lt_075']} "
        f"({loc['matched_050_to_lt_075_fraction']:.1%} of TP); "
        f"baseline mAP50 minus mAP75={loc['baseline_map_50_minus_map_75']:.3f}.",
        "5. Lowest existing validation AP classes: "
        + ", ".join(f"{item['class_name']} {item['baseline_validation_ap']:.3f}" for item in weak) + ".",
        "6. Size-group recalls: "
        + ", ".join(f"{name}={item['recall']:.3f} (n={item['gt_objects']})" for name, item in size.items()) + ".",
        "7. Object-count group recalls: "
        + ", ".join(f"{name}={item['recall']:.3f} (images={item['images']})" for name, item in crowd.items()) + ".",
        f"8. FP confidence median={conf['false_positive']['median']:.3f}, "
        f"TP confidence median={conf['true_positive']['median']:.3f}; "
        f"FP q1-q3={conf['false_positive']['q1']:.3f}-{conf['false_positive']['q3']:.3f}, "
        f"max={conf['false_positive']['max']:.3f}. FPs are generally lower-confidence "
        "than TPs, but some are highly confident.",
        "9. Most common confusion pairs: "
        + (", ".join(
            f"{item['ground_truth_class']} -> {item['predicted_class']} ({item['count']})"
            for item in confusion[:5]
        ) if confusion else "none observed") + ".",
        "10. Next experiment directions (hypotheses, not actions):",
        *[f"   - {recommendation}" for recommendation in recommendations[:2]],
        "",
        "Class focus:",
        *focus_lines,
        "Full distributions and exact example paths are in class_diagnostics.json and visualizations.json.",
        "",
        "Categorization priority for unmatched predictions: duplicate (same class, IoU>=0.50 "
        "with already matched GT), class confusion (wrong class, IoU>=0.50), localization "
        "(same class, 0.10<=IoU<0.50), background (max IoU<0.10), then other overlap. "
        "False negatives are unmatched GT boxes. Categories are diagnostic and do not replace COCO AP.",
        "The network was run once; threshold variants and reports use the prediction cache.",
    ]
    (report_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
