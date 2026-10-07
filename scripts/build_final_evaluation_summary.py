#!/usr/bin/env python3
"""Assemble final reports from the frozen test prediction cache and benchmarks."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "final_evaluation"


def read(name: str) -> Any:
    return json.loads((REPORT_DIR / name).read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def collect_visualizations() -> dict[str, list[str]]:
    source_manifest = read("error_analysis/visualizations.json")
    mapping = {
        "high_confidence_correct_detections": "strongest_true_positives",
        "false_positives": "high_confidence_false_positives",
        "false_negatives": "missed_objects",
        "localization_errors": "localization",
        "class_confusions": "class_confusion",
        "small_object_cases": "small_object_cases",
        "crowded_scenes": "crowded_scenes",
    }
    destination = REPORT_DIR / "visualizations"
    destination.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, list[str]] = {}
    for desired, source_category in mapping.items():
        manifest[desired] = []
        for source_text in source_manifest[source_category]:
            source = Path(source_text)
            target = destination / source.name.replace(source_category, desired, 1)
            shutil.copy2(source, target)
            manifest[desired].append(target.relative_to(PROJECT_ROOT).as_posix())
    write_json(destination / "manifest.json", manifest)
    return manifest


def main() -> int:
    metrics = read("test_metrics.json")
    diagnostic = read("error_analysis/summary.json")
    sizes = read("error_analysis/object_size_analysis.json")["groups"]
    crowd = read("error_analysis/crowdedness_analysis.json")["groups"]
    localization = read("error_analysis/localization_analysis.json")
    confidence = read("error_analysis/confidence_analysis.json")["by_outcome"]
    confusion = read("error_analysis/class_confusion.json")
    classes = read("per_class_test_metrics.json")["classes"]
    comparison = read("validation_vs_test.json")
    integrity = read("integrity.json")
    checkpoint = read("checkpoint_verification.json")
    verification = read("verification.json")
    gpu = read("benchmark_cuda.json")
    cpu = read("benchmark_cpu.json")
    visuals = collect_visualizations()
    strongest = sorted(classes, key=lambda item: item["ap"], reverse=True)[:3]
    weakest = sorted(classes, key=lambda item: item["ap"])[:3]

    fp_categories = diagnostic["false_positive_categories"]
    fig, axis = plt.subplots(figsize=(8, 4.8))
    names = ["Class confusion", "Localization", "Background", "Duplicate", "Other overlap"]
    values = [fp_categories[key] for key in ("class_confusion", "localization", "background", "duplicate", "other_overlap")]
    axis.bar(names, values, color="#D95F02")
    axis.set_ylabel("False-positive count")
    axis.set_title("Held-out Test Diagnostic Error Breakdown (score/IoU = 0.50/0.50)")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "error_breakdown.png", dpi=180)
    plt.close(fig)

    summary = {
        "selected_model": "faster_rcnn_baseline_01",
        "selected_checkpoint": "checkpoints/faster_rcnn_baseline_01/best.pt",
        "selected_epoch": 9,
        "selection_basis": "Highest balanced validation result selected and locked before held-out test access; test did not influence selection.",
        "validation_metrics": {
            "map": 0.575484573841095, "map_50": 0.8256866335868835,
            "map_75": 0.6530308127403259, "mar_100": 0.6874443292617798,
        },
        "test_metrics": {key: metrics[key] for key in ("map", "map_50", "map_75", "mar_100")},
        "validation_to_test_deltas": {name: row["delta_test_minus_validation"] for name, row in comparison["metrics"].items()},
        "test_dataset": {
            "images": integrity["test_image_count"], "annotation_files": integrity["test_annotation_file_count"],
            "objects": integrity["test_object_count"], "zero_object_images": integrity["test_zero_object_count"],
        },
        "diagnostic_operating_point": {"score_threshold": 0.50, "iou_threshold": 0.50},
        "diagnostic": {
            "tp": diagnostic["true_positives"], "fp": diagnostic["false_positives"], "fn": diagnostic["false_negatives"],
            "precision": diagnostic["precision"], "recall": diagnostic["recall"],
            "size_recall": {name: sizes[name]["recall"] for name in ("small", "medium", "large")},
            "crowdedness_recall": {name: crowd[name]["recall"] for name in ("1", "2-3", "4+")},
            "false_positive_categories": fp_categories,
            "mean_matched_iou": localization["matched_iou"]["mean"],
            "median_matched_iou": localization["matched_iou"]["median"],
            "tp_confidence": confidence["true_positive"], "fp_confidence": confidence["false_positive"],
            "top_confusions": confusion["pairs"][:8],
        },
        "strongest_classes_by_test_ap": strongest,
        "weakest_classes_by_test_ap": weakest,
        "major_remaining_error_modes": [
            "Small-object recall remains much lower than medium/large recall.",
            "Class confusion is the largest diagnostic false-positive category.",
            "Scenes with four or more objects remain harder than one-object scenes.",
        ],
        "gpu_benchmark": gpu,
        "cpu_benchmark": cpu,
        "visualizations": {"count": sum(map(len, visuals.values())), "categories": visuals},
        "checkpoint_verification": checkpoint,
        "integrity": integrity,
        "verification": verification,
        "conclusion": (
            "The validation-selected baseline generalized at least as well on this held-out split for aggregate COCO metrics. "
            "Small-object and crowded-scene recall remain the principal limitations; no post-test model or threshold change was made."
        ),
        "test_used_for_selection_or_tuning": False,
        "training_occurred": False,
    }
    write_json(REPORT_DIR / "final_summary.json", summary)

    lines = [
        "FINAL HELD-OUT EVALUATION SUMMARY",
        "",
        "Selection was locked before test access: faster_rcnn_baseline_01, epoch 9, checkpoints/faster_rcnn_baseline_01/best.pt.",
        "Validation selected the model; held-out test results did not alter the checkpoint, threshold, preprocessing, or configuration.",
        "",
        f"Test set: {integrity['test_image_count']} images, {integrity['test_annotation_file_count']} annotation files, {integrity['test_object_count']} objects, {integrity['test_zero_object_count']} zero-object images.",
        f"COCO metrics: mAP={metrics['map']:.6f}, mAP50={metrics['map_50']:.6f}, mAP75={metrics['map_75']:.6f}, mAR100={metrics['mar_100']:.6f}.",
        f"Diagnostic @ score 0.50 / IoU 0.50: TP={diagnostic['true_positives']}, FP={diagnostic['false_positives']}, FN={diagnostic['false_negatives']}, precision={diagnostic['precision']:.6f}, recall={diagnostic['recall']:.6f}.",
        f"Size recall: small={sizes['small']['recall']:.6f}, medium={sizes['medium']['recall']:.6f}, large={sizes['large']['recall']:.6f}.",
        f"Scene recall: 1 object={crowd['1']['recall']:.6f}, 2-3={crowd['2-3']['recall']:.6f}, 4+={crowd['4+']['recall']:.6f}.",
        f"FP categories: class confusion={fp_categories['class_confusion']}, localization={fp_categories['localization']}, background={fp_categories['background']}, duplicate={fp_categories['duplicate']}, other overlap={fp_categories['other_overlap']}.",
        f"Matched IoU: mean={localization['matched_iou']['mean']:.6f}, median={localization['matched_iou']['median']:.6f}.",
        "",
        "Strongest test AP: " + ", ".join(f"{item['class_name']} {item['ap']:.6f}" for item in strongest) + ".",
        "Weakest test AP: " + ", ".join(f"{item['class_name']} {item['ap']:.6f}" for item in weakest) + ".",
        "Top confusion directions: " + ", ".join(f"{item['ground_truth_class']} -> {item['predicted_class']} ({item['count']})" for item in confusion['pairs'][:5]) + ".",
        "",
        f"GPU model-only (20 warmup, 100 timed): mean={gpu['model_only']['mean_latency_ms']:.3f} ms, median={gpu['model_only']['median_latency_ms']:.3f} ms, p95={gpu['model_only']['p95_latency_ms']:.3f} ms, {gpu['model_only']['images_per_second']:.3f} images/s.",
        f"GPU end-to-end: mean={gpu['end_to_end']['mean_latency_ms']:.3f} ms, median={gpu['end_to_end']['median_latency_ms']:.3f} ms, p95={gpu['end_to_end']['p95_latency_ms']:.3f} ms, {gpu['end_to_end']['images_per_second']:.3f} images/s.",
        f"GPU end-to-end peak memory: allocated={gpu['end_to_end']['peak_cuda_allocated_mib']:.3f} MiB, reserved={gpu['end_to_end']['peak_cuda_reserved_mib']:.3f} MiB.",
        f"CPU model-only (5 warmup, 20 timed): mean={cpu['model_only']['mean_latency_ms']:.3f} ms, median={cpu['model_only']['median_latency_ms']:.3f} ms, p95={cpu['model_only']['p95_latency_ms']:.3f} ms, {cpu['model_only']['images_per_second']:.3f} images/s.",
        "",
        "Conclusion: aggregate held-out metrics exceeded validation metrics, but small objects, crowded scenes, and class confusion remain the main limitations. No statistical-significance claim is made.",
        "No training, optimizer step, post-test checkpoint selection, or post-test threshold tuning occurred.",
        f"Full pytest: {verification['full_pytest']['passed']} passed, {verification['full_pytest']['failed']} failed in {verification['full_pytest']['runtime_seconds']:.2f} seconds.",
    ]
    (REPORT_DIR / "final_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
