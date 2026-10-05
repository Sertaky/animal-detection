"""Deterministic, diagnostic matching independent of COCO AP calculation."""

from __future__ import annotations

from typing import Any


def box_iou(first: list[float], second: list[float]) -> float:
    """IoU of two pixel xyxy boxes; degenerate boxes have zero overlap."""
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def match_image(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    score_threshold: float = 0.50,
    iou_threshold: float = 0.50,
    localization_floor: float = 0.10,
) -> dict[str, Any]:
    """Greedy class-aware one-to-one matching, then explain unmatched predictions.

    Prediction indices refer to the original input list. Equal scores preserve
    their original order. Diagnostic labels are mutually exclusive:
    duplicate, class_confusion, localization, background, or other_overlap.
    False negatives are independently counted as unmatched ground-truth boxes.
    """
    if not 0 <= score_threshold <= 1:
        raise ValueError("score_threshold must be in [0, 1]")
    if not 0 <= localization_floor < iou_threshold <= 1:
        raise ValueError("Require 0 <= localization_floor < iou_threshold <= 1")

    retained = sorted(
        (index for index, prediction in enumerate(predictions)
         if prediction["score"] >= score_threshold),
        key=lambda index: (-predictions[index]["score"], index),
    )
    matched_gt: set[int] = set()
    records: list[dict[str, Any]] = []
    overlaps = {
        index: [
            box_iou(predictions[index]["box"], gt["box"]) for gt in ground_truth
        ]
        for index in retained
    }

    for index in retained:
        prediction = predictions[index]
        candidates = [
            (overlaps[index][gt_index], gt_index)
            for gt_index, gt in enumerate(ground_truth)
            if gt_index not in matched_gt
            and gt["label"] == prediction["label"]
            and overlaps[index][gt_index] >= iou_threshold
        ]
        record = {
            "prediction_index": index,
            "label": prediction["label"],
            "score": prediction["score"],
            "status": "false_positive",
            "category": None,
            "gt_index": None,
            "iou": None,
            "max_iou": max(overlaps[index], default=0.0),
            "best_correct_class_iou": max(
                (
                    overlaps[index][gt_index]
                    for gt_index, gt in enumerate(ground_truth)
                    if gt["label"] == prediction["label"]
                ),
                default=None,
            ),
        }
        if candidates:
            iou, gt_index = max(candidates, key=lambda candidate: (candidate[0], -candidate[1]))
            matched_gt.add(gt_index)
            record.update(status="true_positive", gt_index=gt_index, iou=iou)
        records.append(record)

    for record in records:
        if record["status"] == "true_positive":
            continue
        index = record["prediction_index"]
        label = record["label"]
        duplicate = [
            (overlaps[index][gt_index], gt_index)
            for gt_index in matched_gt
            if ground_truth[gt_index]["label"] == label
            and overlaps[index][gt_index] >= iou_threshold
        ]
        confusion = [
            (overlaps[index][gt_index], gt_index)
            for gt_index, gt in enumerate(ground_truth)
            if gt["label"] != label and overlaps[index][gt_index] >= iou_threshold
        ]
        localization = [
            (overlaps[index][gt_index], gt_index)
            for gt_index, gt in enumerate(ground_truth)
            if gt["label"] == label
            and localization_floor <= overlaps[index][gt_index] < iou_threshold
        ]
        if duplicate:
            record["category"] = "duplicate"
            record["iou"], record["gt_index"] = max(duplicate, key=lambda item: (item[0], -item[1]))
        elif confusion:
            record["category"] = "class_confusion"
            record["iou"], record["gt_index"] = max(confusion, key=lambda item: (item[0], -item[1]))
        elif localization:
            record["category"] = "localization"
            record["iou"], record["gt_index"] = max(localization, key=lambda item: (item[0], -item[1]))
        elif record["max_iou"] < localization_floor:
            record["category"] = "background"
        else:
            record["category"] = "other_overlap"

    return {
        "predictions": records,
        "false_negative_gt_indices": [
            index for index in range(len(ground_truth)) if index not in matched_gt
        ],
        "true_positives": len(matched_gt),
        "false_positives": len(records) - len(matched_gt),
        "false_negatives": len(ground_truth) - len(matched_gt),
    }
