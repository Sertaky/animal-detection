#!/usr/bin/env python3
"""Analyze one experiment's best checkpoint on the usable validation split only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from animal_detection.analysis.reporting import generate_reports
from animal_detection.data import TorchvisionDetectionDataset, YoloDetectionDataset, get_eval_transforms
from animal_detection.data.label_mapping import ANIMAL_CLASS_NAMES
from animal_detection.models import build_faster_rcnn


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def valid_integrity_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for folder in ("images", "labels"):
        for path in sorted((root / "valid" / folder).iterdir()):
            if path.is_file():
                digest.update(path.relative_to(root).as_posix().encode("utf-8"))
                digest.update(b"\0")
                with path.open("rb") as source:
                    while block := source.read(1024 * 1024):
                        digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_experiment(config_path: Path, checkpoint_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        config["architecture"] != "Faster R-CNN ResNet50-FPN"
        or config["pretrained_weights"] != "FasterRCNN_ResNet50_FPN_Weights.COCO_V1"
        or config["num_model_classes"] != 21
        or config["num_foreground_classes"] != 20
        or config["valid_transforms"] != [f"Resize({config['image_size']},{config['image_size']})", "ToTensor"]
        or config["valid_usable_samples"] != 299
    ):
        raise ValueError("Experiment configuration does not match the reviewed detection setup")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    checkpoint_config = checkpoint["configuration"]
    if checkpoint_config["architecture"] != config["architecture"] or checkpoint_config["num_model_classes"] != 21:
        raise ValueError("Checkpoint architecture/class count differs from experiment config")
    expected_mapping = {
        str(index): {"dataset_label": index - 1, "class_name": name}
        for index, name in enumerate(ANIMAL_CLASS_NAMES, start=1)
    }
    if checkpoint["class_mapping"] != expected_mapping:
        raise ValueError("Checkpoint class mapping differs from the current 20-class mapping")
    return config, checkpoint


def usable_validation_dataset(root: Path, config: dict[str, Any]) -> Subset:
    manifest = json.loads(Path(config["exclusions_manifest"]).read_text(encoding="utf-8"))
    excluded = [item["image_filename"] for item in manifest["samples"] if item["split"] == "valid"]
    if excluded != ["wolf-109-_jpg.rf.a384d38588191d796589048d6cb38e97.jpg"]:
        raise ValueError(f"Unexpected validation exclusion: {excluded}")
    image_size = int(config["image_size"])
    base = YoloDetectionDataset(root, "valid", transforms=get_eval_transforms(size=(image_size, image_size)))
    indices = [index for index, path in enumerate(base.image_paths) if path.name not in excluded]
    if len(base) != 300 or len(indices) != 299:
        raise ValueError("Validation split/exclusion count differs from baseline")
    excluded_label = base.labels_dir / (Path(excluded[0]).stem + ".txt")
    if excluded_label.stat().st_size != 0:
        raise ValueError("Excluded validation annotation is no longer empty")
    return Subset(TorchvisionDetectionDataset(base), indices)


def run_inference(
    dataset: Subset, root: Path, checkpoint: dict[str, Any], device: torch.device,
    image_size: int,
) -> tuple[list[dict[str, Any]], float]:
    model = build_faster_rcnn(
        num_classes=21, pretrained=False, min_size=image_size, max_size=image_size,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    images: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for position in range(len(dataset)):
            image, target = dataset[position]
            prediction = model([image.to(device)])[0]
            images.append({
                "image_id": Path(target["image_path"]).relative_to(root).as_posix(),
                "original_size": target["original_size"].tolist(),
                "transformed_size": target["size"].tolist(),
                "ground_truth": [
                    {"box": box, "label": label}
                    for box, label in zip(
                        target["boxes"].tolist(), target["labels"].tolist()
                    )
                ],
                "predictions": [
                    {"box": box, "label": label, "score": score}
                    for box, label, score in zip(
                        prediction["boxes"].cpu().tolist(),
                        prediction["labels"].cpu().tolist(),
                        prediction["scores"].cpu().tolist(),
                    )
                ],
            })
            if (position + 1) % 50 == 0:
                print(f"inference={position + 1}/299", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return images, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir", type=Path,
        default=PROJECT_ROOT / "reports" / "experiments" / "faster_rcnn_baseline_01",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=PROJECT_ROOT / "checkpoints" / "faster_rcnn_baseline_01" / "best.pt",
    )
    parser.add_argument("--score-threshold", type=float, default=0.50)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not 0 <= args.score_threshold <= 1:
        parser.error("score-threshold must be in [0, 1]")
    if args.iou_threshold != 0.50:
        parser.error("This baseline report uses the reviewed primary IoU threshold of 0.50")

    experiment_dir = args.experiment_dir.resolve()
    checkpoint_path = args.checkpoint.resolve()
    config, checkpoint = load_experiment(experiment_dir / "config.json", checkpoint_path)
    checkpoint_epoch = int(checkpoint["epoch"])
    root = Path(config["dataset_root"]).resolve()
    valid_hash_before = valid_integrity_hash(root)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    base_report_dir = experiment_dir / "error_analysis"
    report_dir = (
        base_report_dir if args.score_threshold == 0.50
        else base_report_dir / f"score_{args.score_threshold:.2f}".replace(".", "p")
    )
    cache_path = base_report_dir / "validation_predictions.json"
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            cache.get("checkpoint_sha256") != checkpoint_sha256
            or cache.get("valid_raw_sha256") != valid_hash_before
            or len(cache.get("images", [])) != 299
        ):
            raise ValueError("Existing prediction cache does not match checkpoint or validation data")
        print("Using existing validation prediction cache; no network inference rerun.", flush=True)
    else:
        dataset = usable_validation_dataset(root, config)
        images, runtime = run_inference(
            dataset, root, checkpoint, torch.device(args.device), int(config["image_size"])
        )
        if len(images) != 299 or len({item["image_id"] for item in images}) != 299:
            raise RuntimeError("Inference did not cover exactly 299 distinct validation images")
        cache = {
            "schema_version": 1,
            "split": "valid",
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint_epoch,
            "checkpoint_sha256": checkpoint_sha256,
            "valid_raw_sha256": valid_hash_before,
            "preprocessing": config["valid_transforms"],
            "image_count": len(images),
            "raw_prediction_count": sum(len(item["predictions"]) for item in images),
            "inference_runtime_seconds": runtime,
            "images": images,
        }
        write_json(cache_path, cache)
    existing_ap = json.loads((experiment_dir / "validation_class_metrics.json").read_text(encoding="utf-8"))
    if existing_ap["best_epoch"] != checkpoint_epoch:
        raise ValueError("Existing class AP report is not from the selected best checkpoint")
    generate_reports(
        cache=cache, class_ap=existing_ap["metrics"], root=root,
        report_dir=report_dir, score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold, experiment_name=config["experiment_name"],
        validation_metrics=checkpoint["validation_metrics"],
    )
    valid_hash_after = valid_integrity_hash(root)
    if valid_hash_after != valid_hash_before:
        raise RuntimeError("Validation raw data changed during analysis")
    summary_path = report_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["valid_raw_sha256_before"] = valid_hash_before
    summary["valid_raw_sha256_after"] = valid_hash_after
    summary["test_split_constructed"] = False
    summary["optimizer_used"] = False
    write_json(summary_path, summary)
    print(f"images=299 raw_predictions={cache['raw_prediction_count']} reports={report_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
