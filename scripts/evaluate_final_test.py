#!/usr/bin/env python3
"""Run the locked baseline checkpoint's single final held-out test evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from animal_detection.analysis.reporting import generate_reports
from animal_detection.data import TorchvisionDetectionDataset, YoloDetectionDataset, get_eval_transforms
from animal_detection.data.label_mapping import ANIMAL_CLASS_NAMES
from animal_detection.data.validation import validate_model_target
from animal_detection.models import build_faster_rcnn

EXPECTED_CHECKPOINT_SHA256 = "890568ee1759b4941ea74d309f6dbc616f7bff52f963e53d2639613735b2447f"
EXPECTED_EPOCH = 9
EXPECTED_IMAGES = 300
EXPECTED_OBJECTS = 388


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def tree_hash(root: Path, relative_paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(relative_paths, key=lambda item: item.as_posix().casefold()):
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        with (root / path).open("rb") as source:
            while block := source.read(1024 * 1024):
                digest.update(block)
    return digest.hexdigest()


def raw_paths(root: Path) -> list[Path]:
    return [
        path.relative_to(root)
        for split in ("train", "valid", "test")
        for path in (root / split).rglob("*")
        if path.is_file()
    ]


def test_paths(root: Path) -> list[Path]:
    return [path.relative_to(root) for path in (root / "test").rglob("*") if path.is_file()]


def expected_mapping() -> dict[str, dict[str, Any]]:
    return {
        str(index): {"dataset_label": index - 1, "class_name": name}
        for index, name in enumerate(ANIMAL_CLASS_NAMES, start=1)
    }


def cpu_prediction(output: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: output[key].detach().cpu() for key in ("boxes", "scores", "labels")}


def cpu_target(target: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {"boxes": target["boxes"].detach().cpu(), "labels": target["labels"].detach().cpu()}


def scalar(result: dict[str, torch.Tensor], key: str) -> float:
    return float(result[key].item())


def plot_validation_vs_test(path: Path, validation: dict[str, float], test: dict[str, float]) -> None:
    labels = ["mAP", "mAP50", "mAP75", "mAR100"]
    keys = ["map", "map_50", "map_75", "mar_100"]
    x = np.arange(len(labels))
    width = 0.36
    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    axis.bar(x - width / 2, [validation[key] for key in keys], width, label="Validation")
    axis.bar(x + width / 2, [test[key] for key in keys], width, label="Held-out test")
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Metric value")
    axis.set_title("Locked Baseline: Validation vs Held-out Test")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_test_ap(path: Path, classes: list[dict[str, Any]]) -> None:
    fig, axis = plt.subplots(figsize=(11, 6.5))
    names = [item["class_name"] for item in classes]
    values = [item["ap"] for item in classes]
    axis.bar(np.arange(20), values, color="#2878B5")
    axis.set_xticks(np.arange(20), names, rotation=55, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Test AP@0.50:0.95")
    axis.set_title("Held-out Test AP by Class (Class-ID Order)")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_comparison(
    report_dir: Path, test_metrics: dict[str, Any], test_diag: dict[str, Any],
    test_size: dict[str, Any], test_crowd: dict[str, Any], test_loc: dict[str, Any],
    test_classes: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_dir = PROJECT_ROOT / "reports" / "experiments" / "faster_rcnn_baseline_01"
    validation_metrics = json.loads((baseline_dir / "summary.json").read_text(encoding="utf-8"))
    error_dir = baseline_dir / "error_analysis"
    validation_diag = json.loads((error_dir / "summary.json").read_text(encoding="utf-8"))
    validation_size = json.loads((error_dir / "object_size_analysis.json").read_text(encoding="utf-8"))
    validation_crowd = json.loads((error_dir / "crowdedness_analysis.json").read_text(encoding="utf-8"))
    validation_loc = json.loads((error_dir / "localization_analysis.json").read_text(encoding="utf-8"))
    validation_classes = json.loads((baseline_dir / "validation_class_metrics.json").read_text(encoding="utf-8"))["metrics"]

    rows: dict[str, dict[str, float | None]] = {}
    def add(name: str, valid: float | None, test: float | None) -> None:
        rows[name] = {"validation": valid, "test": test, "delta_test_minus_validation": None if valid is None or test is None else test - valid}

    add("map", validation_metrics["best_validation_map"], test_metrics["map"])
    add("map_50", validation_metrics["best_validation_map_50"], test_metrics["map_50"])
    add("map_75", validation_metrics["best_validation_map_75"], test_metrics["map_75"])
    add("mar_100", 0.6874443292617798, test_metrics["mar_100"])
    add("diagnostic_precision", validation_diag["precision"], test_diag["precision"])
    add("diagnostic_recall", validation_diag["recall"], test_diag["recall"])
    for group in ("small", "medium", "large"):
        add(f"{group}_recall", validation_size["groups"][group]["recall"], test_size["groups"][group]["recall"])
    for group in ("1", "2-3", "4+"):
        add(f"{group}_object_recall", validation_crowd["groups"][group]["recall"], test_crowd["groups"][group]["recall"])
    add("mean_matched_iou", validation_loc["matched_iou"]["mean"], test_loc["matched_iou"]["mean"])
    add("median_matched_iou", validation_loc["matched_iou"]["median"], test_loc["matched_iou"]["median"])
    validation_ap = {item["model_label"]: item["ap"] for item in validation_classes}
    class_rows = []
    for item in test_classes:
        valid = validation_ap[item["model_label"]]
        class_rows.append({
            "model_label": item["model_label"], "class_name": item["class_name"],
            "validation_ap": valid, "test_ap": item["ap"],
            "delta_test_minus_validation": item["ap"] - valid,
        })
    comparison = {
        "selected_model": "faster_rcnn_baseline_01",
        "selection_basis": "validation only; comparison is descriptive and does not alter selection",
        "metrics": rows,
        "per_class_ap": class_rows,
        "interpretation_note": "Deltas are descriptive test-minus-validation differences; no statistical significance is claimed.",
    }
    write_json(report_dir / "validation_vs_test.json", comparison)
    lines = [
        "LOCKED BASELINE: VALIDATION VS HELD-OUT TEST",
        "Model selection used validation only. Test-minus-validation deltas are descriptive; no statistical significance is claimed.",
        "",
        f"{'Metric':<28} {'Validation':>12} {'Test':>12} {'Delta':>12}",
        "-" * 67,
    ]
    for name, row in rows.items():
        lines.append(f"{name:<28} {row['validation']:>12.6f} {row['test']:>12.6f} {row['delta_test_minus_validation']:>+12.6f}")
    lines.extend(["", "Per-class AP (class-ID order):"])
    for item in class_rows:
        lines.append(f"{item['model_label']:>2} {item['class_name']:<10} valid={item['validation_ap']:.6f} test={item['test_ap']:.6f} delta={item['delta_test_minus_validation']:+.6f}")
    (report_dir / "validation_vs_test.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot_validation_vs_test(
        report_dir / "validation_vs_test_metrics.png",
        {"map": rows["map"]["validation"], "map_50": rows["map_50"]["validation"], "map_75": rows["map_75"]["validation"], "mar_100": rows["mar_100"]["validation"]},
        {key: test_metrics[key] for key in ("map", "map_50", "map_75", "mar_100")},
    )
    return comparison


def export_per_class(report_dir: Path, classes: list[dict[str, Any]]) -> None:
    fields = ["model_label", "dataset_label", "class_name", "gt_objects", "ap", "tp", "fp", "fn", "precision", "recall", "mean_matched_iou"]
    write_json(report_dir / "per_class_test_metrics.json", {"classes": classes})
    with (report_dir / "per_class_test_metrics.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: item[key] for key in fields} for item in classes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "faster_rcnn_baseline_01" / "best.pt")
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "Multi-Class Animal Detection.v1-yolov8")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "final_evaluation")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    lock_path = output_dir / "model_selection_lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError("Model-selection lock must exist before test evaluation")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["lock_state"] != "recorded_before_held_out_test_evaluation":
        raise ValueError("Model-selection lock state is invalid")
    cache_path = output_dir / "test_predictions.json"
    if cache_path.exists():
        raise FileExistsError("Refusing to rerun final held-out metrics: test prediction cache already exists")

    checkpoint_path = args.checkpoint.resolve()
    root = args.dataset_root.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint_sha_before = sha256_file(checkpoint_path)
    if checkpoint_sha_before != EXPECTED_CHECKPOINT_SHA256 or checkpoint_sha_before != lock["selected_checkpoint_sha256"]:
        raise ValueError("Selected checkpoint hash differs from the pre-test lock")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    config = checkpoint["configuration"]
    if (
        checkpoint["epoch"] != EXPECTED_EPOCH
        or config["architecture"] != "Faster R-CNN ResNet50-FPN"
        or config["pretrained_weights"] != "FasterRCNN_ResNet50_FPN_Weights.COCO_V1"
        or config["num_model_classes"] != 21
        or checkpoint["class_mapping"] != expected_mapping()
    ):
        raise ValueError("Checkpoint metadata is inconsistent with the locked selection")

    all_raw_paths = raw_paths(root)
    heldout_paths = test_paths(root)
    raw_hash_before = tree_hash(root, all_raw_paths)
    test_hash_before = tree_hash(root, heldout_paths)
    exclusion_path = PROJECT_ROOT / "data" / "manifests" / "excluded_samples.json"
    exclusion_hash_before = sha256_file(exclusion_path)

    raw_dataset = YoloDetectionDataset(root, "test", transforms=get_eval_transforms(size=(512, 512)))
    dataset = TorchvisionDetectionDataset(raw_dataset)
    image_files = list((root / "test" / "images").iterdir())
    label_files = list((root / "test" / "labels").glob("*.txt"))
    class_counts: Counter[int] = Counter()
    zero_object_count = 0
    object_count = 0
    for index in range(len(dataset)):
        image, target = dataset[index]
        validate_model_target(target, (int(image.shape[-2]), int(image.shape[-1])))
        count = int(target["labels"].numel())
        object_count += count
        zero_object_count += count == 0
        class_counts.update(int(value) for value in target["labels"].tolist())
    if len(dataset) != EXPECTED_IMAGES or len(label_files) != EXPECTED_IMAGES or object_count != EXPECTED_OBJECTS:
        raise ValueError(f"Unexpected test composition: images={len(dataset)}, labels={len(label_files)}, objects={object_count}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the final evaluation")
    model = build_faster_rcnn(num_classes=21, pretrained=False, min_size=512, max_size=512)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if model.roi_heads.box_predictor.cls_score.out_features != 21:
        raise ValueError("Model predictor does not have 21 outputs")
    model.to(device)
    model.eval()
    if model.training:
        raise RuntimeError("Model did not enter eval mode")
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", backend="pycocotools", class_metrics=True)
    images_cache: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for index in range(len(dataset)):
            image, target = dataset[index]
            prediction = model([image.to(device)])[0]
            prediction_cpu = cpu_prediction(prediction)
            target_cpu = cpu_target(target)
            metric.update([prediction_cpu], [target_cpu])
            images_cache.append({
                "image_id": Path(target["image_path"]).relative_to(root).as_posix(),
                "original_size": target["original_size"].tolist(),
                "transformed_size": target["size"].tolist(),
                "ground_truth": [
                    {"box": box, "label": label}
                    for box, label in zip(target["boxes"].tolist(), target["labels"].tolist())
                ],
                "predictions": [
                    {"box": box, "label": label, "score": score}
                    for box, label, score in zip(
                        prediction_cpu["boxes"].tolist(), prediction_cpu["labels"].tolist(), prediction_cpu["scores"].tolist()
                    )
                ],
            })
            if (index + 1) % 50 == 0:
                print(f"final_test_inference={index + 1}/{len(dataset)}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_runtime = time.perf_counter() - started
    result = metric.compute()
    classes = result["classes"].reshape(-1).tolist()
    ap_values = result["map_per_class"].reshape(-1).tolist()
    ar_values = result["mar_100_per_class"].reshape(-1).tolist()
    class_metrics = [
        {
            "model_label": int(label), "dataset_label": int(label) - 1,
            "class_name": ANIMAL_CLASS_NAMES[int(label) - 1],
            "ap": float(ap), "ar_100": float(ar),
        }
        for label, ap, ar in zip(classes, ap_values, ar_values)
        if 1 <= int(label) <= 20
    ]
    class_metrics.sort(key=lambda item: item["model_label"])
    if [item["model_label"] for item in class_metrics] != list(range(1, 21)):
        raise RuntimeError("COCO evaluator did not return all 20 classes")
    test_metrics = {
        "map": scalar(result, "map"), "map_50": scalar(result, "map_50"),
        "map_75": scalar(result, "map_75"), "mar_100": scalar(result, "mar_100"),
        "num_images": len(dataset), "num_batches": len(dataset),
        "runtime_seconds": inference_runtime, "per_class": class_metrics,
    }
    cache = {
        "schema_version": 1, "split": "test", "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": EXPECTED_EPOCH, "checkpoint_sha256": checkpoint_sha_before,
        "test_raw_sha256": test_hash_before, "preprocessing": ["Resize(512,512)", "ToTensor"],
        "image_count": len(images_cache),
        "raw_prediction_count": sum(len(item["predictions"]) for item in images_cache),
        "inference_runtime_seconds": inference_runtime, "images": images_cache,
    }
    write_json(cache_path, cache)
    write_json(output_dir / "test_metrics.json", test_metrics)
    write_json(output_dir / "test_class_metrics.json", {"checkpoint_epoch": EXPECTED_EPOCH, "metrics": class_metrics})

    error_dir = output_dir / "error_analysis"
    generate_reports(
        cache=cache, class_ap=class_metrics, root=root, report_dir=error_dir,
        score_threshold=0.50, iou_threshold=0.50,
        experiment_name="faster_rcnn_baseline_01_final_test",
        validation_metrics=test_metrics,
    )
    test_diag = json.loads((error_dir / "summary.json").read_text(encoding="utf-8"))
    test_size = json.loads((error_dir / "object_size_analysis.json").read_text(encoding="utf-8"))
    test_crowd = json.loads((error_dir / "crowdedness_analysis.json").read_text(encoding="utf-8"))
    test_loc = json.loads((error_dir / "localization_analysis.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((error_dir / "class_diagnostics.json").read_text(encoding="utf-8"))["classes"]
    ap_by_label = {item["model_label"]: item["ap"] for item in class_metrics}
    final_classes = [
        {
            "model_label": item["model_label"], "dataset_label": item["dataset_label"],
            "class_name": item["class_name"], "gt_objects": item["gt_objects"],
            "ap": ap_by_label[item["model_label"]], "tp": item["tp"], "fp": item["fp"],
            "fn": item["fn"], "precision": item["precision"], "recall": item["recall"],
            "mean_matched_iou": item["mean_matched_iou"],
        }
        for item in diagnostics
    ]
    export_per_class(output_dir, final_classes)
    comparison = make_comparison(output_dir, test_metrics, test_diag, test_size, test_crowd, test_loc, class_metrics)
    plot_test_ap(output_dir / "test_class_ap.png", class_metrics)

    test_hash_after = tree_hash(root, heldout_paths)
    raw_hash_after = tree_hash(root, all_raw_paths)
    checkpoint_sha_after = sha256_file(checkpoint_path)
    exclusion_hash_after = sha256_file(exclusion_path)
    if (test_hash_after != test_hash_before or raw_hash_after != raw_hash_before
            or checkpoint_sha_after != checkpoint_sha_before or exclusion_hash_after != exclusion_hash_before):
        raise RuntimeError("Integrity hash changed during final test evaluation")
    integrity = {
        "raw_dataset_sha256_before": raw_hash_before, "raw_dataset_sha256_after": raw_hash_after,
        "raw_dataset_file_count": len(all_raw_paths),
        "test_sha256_before": test_hash_before, "test_sha256_after": test_hash_after,
        "test_file_count": len(heldout_paths), "test_image_count": len(dataset),
        "test_annotation_file_count": len(label_files), "test_object_count": object_count,
        "test_zero_object_count": zero_object_count,
        "test_class_distribution": [
            {"model_label": index, "class_name": name, "objects": class_counts[index]}
            for index, name in enumerate(ANIMAL_CLASS_NAMES, start=1)
        ],
        "target_validation": "passed for all 300 transformed targets",
        "checkpoint_sha256_before": checkpoint_sha_before, "checkpoint_sha256_after": checkpoint_sha_after,
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "exclusion_manifest_sha256_before": exclusion_hash_before,
        "exclusion_manifest_sha256_after": exclusion_hash_after,
        "training_occurred": False, "optimizer_constructed": False, "optimizer_step_executed": False,
        "model_eval": True, "torch_no_grad": True,
    }
    write_json(output_dir / "integrity.json", integrity)
    write_json(output_dir / "checkpoint_verification.json", {
        "checkpoint": str(checkpoint_path), "exists": True, "loaded_successfully": True,
        "sha256": checkpoint_sha_before, "size_bytes": checkpoint_path.stat().st_size,
        "architecture": config["architecture"], "pretrained_weights": config["pretrained_weights"],
        "num_model_classes": 21, "class_mapping_unchanged": True,
        "selected_epoch": checkpoint["epoch"], "model_eval": True,
        "torch_no_grad": True, "optimizer_or_training_step": False,
        "internal_transform_min_size": list(model.transform.min_size),
        "internal_transform_max_size": model.transform.max_size,
        "device": str(device),
        "evaluation_peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "evaluation_peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
    })
    write_json(output_dir / "evaluation_state.json", {
        "locked_before_test": True, "held_out_metrics_run_count": 1,
        "test_used_for_model_selection": False, "post_test_checkpoint_change": False,
        "post_test_threshold_change": False, "comparison": comparison["interpretation_note"],
    })
    print(json.dumps({"test_metrics": test_metrics, "diagnostic": test_diag, "integrity": integrity}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
