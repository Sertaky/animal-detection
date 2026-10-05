#!/usr/bin/env python3
"""Train Faster R-CNN and evaluate only on the validation split."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from animal_detection.data import (
    TorchvisionDetectionDataset,
    YoloDetectionDataset,
    detection_collate_fn,
    get_eval_transforms,
    get_train_transforms,
)
from animal_detection.data.label_mapping import ANIMAL_CLASS_NAMES
from animal_detection.engine import evaluate_map, train_one_epoch
from animal_detection.models import build_faster_rcnn
from animal_detection.utils import (
    append_training_history,
    save_checkpoint,
    write_training_history,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def limited_dataset(dataset: Dataset, max_batches: int | None, batch_size: int) -> Dataset:
    if max_batches is None:
        return dataset
    dataset_length = len(dataset)  # type: ignore[arg-type]
    sample_count = min(dataset_length, max_batches * batch_size)
    if sample_count == dataset_length:
        return dataset
    if sample_count == 1:
        indices = [0]
    else:
        indices = [
            round(position * (dataset_length - 1) / (sample_count - 1))
            for position in range(sample_count)
        ]
    return Subset(dataset, indices)


def json_configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def split_integrity_hash(dataset_root: Path) -> str:
    """Hash train and valid bytes only; never inspect the test split."""
    digest = hashlib.sha256()
    for split in ("train", "valid"):
        for folder in ("images", "labels"):
            for path in sorted((dataset_root / split / folder).iterdir()):
                if not path.is_file():
                    continue
                digest.update(path.relative_to(dataset_root).as_posix().encode("utf-8"))
                digest.update(b"\0")
                with path.open("rb") as source:
                    while block := source.read(1024 * 1024):
                        digest.update(block)
    return digest.hexdigest()


def excluded_dataset(dataset: YoloDetectionDataset, manifest: dict[str, Any]) -> Dataset:
    listed = {
        item["image_filename"]
        for item in manifest["samples"]
        if item["split"] == dataset.split
    }
    indices = [
        index for index, path in enumerate(dataset.image_paths)
        if path.name not in listed
    ]
    found = {path.name for path in dataset.image_paths} - {
        dataset.image_paths[index].name for index in indices
    }
    if found != listed:
        raise ValueError(f"Manifest mismatch for {dataset.split}: expected {sorted(listed)}, found {sorted(found)}")
    for item in manifest["samples"]:
        if item["split"] == dataset.split:
            label_path = dataset.labels_dir / (Path(item["image_filename"]).stem + ".txt")
            if label_path.stat().st_size != 0:
                raise ValueError(f"Excluded label is no longer empty: {label_path}")
    return Subset(TorchvisionDetectionDataset(dataset), indices)


def plot_history(history: list[dict[str, Any]], reports_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    for name, key, ylabel, title in (
        ("training_total_loss.png", "train_total_loss", "Mean total training loss", "Training loss"),
        ("validation_map.png", "validation_map", "Validation mAP@0.50:0.95", "Validation mAP"),
    ):
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.plot(epochs, [row[key] for row in history], marker="o")
        axis.set(xlabel="Epoch", ylabel=ylabel, title=title)
        axis.set_xticks(epochs)
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(reports_dir / name, dpi=150)
        plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--experiment-name", default="faster_rcnn_baseline_01")
    parser.add_argument(
        "--comparison-target",
        default=None,
        help="Name of the reference experiment for a controlled comparison.",
    )
    parser.add_argument(
        "--reference-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint whose parameter count must match during preflight.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate configuration/model/GPU forward pass without training.",
    )
    parser.add_argument(
        "--exclusions-manifest", type=Path,
        default=PROJECT_ROOT / "data" / "manifests" / "excluded_samples.json",
    )
    parser.add_argument("--step-size", type=int, default=7)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Explicitly limit batches for a controlled pipeline-validation run.",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Explicitly limit validation batches for a controlled run.",
    )
    parser.add_argument("--print-freq", type=int, default=20)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 64:
        parser.error("epochs and batch-size must be positive; image-size must be >= 64")
    if args.num_workers < 0:
        parser.error("num-workers cannot be negative")
    if args.max_train_batches is not None and args.max_train_batches < 1:
        parser.error("max-train-batches must be positive")
    if args.max_val_batches is not None and args.max_val_batches < 1:
        parser.error("max-val-batches must be positive")
    if args.max_train_batches is not None or args.max_val_batches is not None:
        parser.error("Baseline runs must use the complete usable train and valid splits")
    if args.step_size < 1 or not 0 < args.gamma < 1:
        parser.error("step-size must be positive and gamma must be in (0, 1)")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    seed_everything(args.seed)
    output_dir = args.output_dir.resolve()
    checkpoints_dir = output_dir / "checkpoints" / args.experiment_name
    reports_dir = output_dir / "reports" / "experiments" / args.experiment_name
    history_path = reports_dir / "training_history.json"
    class_metrics_path = reports_dir / "validation_class_metrics.json"
    config_path = reports_dir / "config.json"
    summary_path = reports_dir / "summary.json"
    if (checkpoints_dir / "last.pt").exists() or (
        history_path.exists() and json.loads(history_path.read_text(encoding="utf-8"))
    ):
        parser.error("Experiment already has completed history or a last checkpoint; refusing to overwrite it")
    raw_hash_before = split_integrity_hash(args.dataset_root)
    manifest = json.loads(args.exclusions_manifest.read_text(encoding="utf-8"))
    if len(manifest["samples"]) != 2:
        parser.error("Expected exactly the two reviewed exclusions")

    size = (args.image_size, args.image_size)
    train_base = YoloDetectionDataset(
        args.dataset_root,
        "train",
        transforms=get_train_transforms(size=size),
    )
    valid_base = YoloDetectionDataset(
        args.dataset_root,
        "valid",
        transforms=get_eval_transforms(size=size),
    )
    train_dataset = excluded_dataset(train_base, manifest)
    valid_dataset = excluded_dataset(valid_base, manifest)
    if (len(train_base), len(train_dataset), len(valid_base), len(valid_dataset)) != (1400, 1399, 300, 299):
        parser.error("Unexpected train/valid counts after applying exclusions")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=detection_collate_fn,
        generator=generator,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=detection_collate_fn,
    )

    configuration = {
        **json_configuration(args),
        "architecture": "Faster R-CNN ResNet50-FPN",
        "pretrained_weights": "FasterRCNN_ResNet50_FPN_Weights.COCO_V1",
        "num_foreground_classes": 20,
        "num_model_classes": 21,
        "all_parameters_trainable": True,
        "trainable_backbone_layers": 5,
        "optimizer": "SGD",
        "scheduler": {"type": "StepLR", "step_size": args.step_size, "gamma": args.gamma},
        "train_transforms": [f"Resize({args.image_size},{args.image_size})", "RandomHorizontalFlip(p=0.5)", "ToTensor"],
        "valid_transforms": [f"Resize({args.image_size},{args.image_size})", "ToTensor"],
        "precision": "FP32",
        "train_usable_samples": len(train_dataset),
        "valid_usable_samples": len(valid_dataset),
        "excluded_samples": len(manifest["samples"]),
        "exclusions_manifest": str(args.exclusions_manifest.resolve()),
        "model_selection_metric": "validation mAP@0.50:0.95; tie: mAP@0.50",
        "learning_rate_convention": "LR used during each epoch, before the end-of-epoch scheduler step",
        "split_usage": {"train": "optimization", "valid": "selection/evaluation", "test": "not accessed"},
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchvision", "torchmetrics", "pycocotools", "numpy", "matplotlib")
        },
        "raw_train_valid_sha256_before": raw_hash_before,
        "controlled_change": (
            {"variable": "image_size", "reference_value": 512, "experiment_value": args.image_size}
            if args.comparison_target else None
        ),
    }
    write_json(config_path, configuration)
    model = build_faster_rcnn(
        num_classes=21,
        pretrained=True,
        min_size=args.image_size,
        max_size=args.image_size,
        trainable_backbone_layers=5,
    ).to(device)
    optimizer = SGD(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Baseline requires all model parameters trainable")
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    reference_parameters = None
    if args.reference_checkpoint is not None:
        reference = torch.load(
            args.reference_checkpoint.resolve(), map_location="cpu", weights_only=False, mmap=True
        )
        parameter_names = {name for name, _ in model.named_parameters()}
        reference_parameters = sum(
            tensor.numel()
            for name, tensor in reference["model_state_dict"].items()
            if name in parameter_names
        )
        if reference_parameters != total_parameters:
            raise RuntimeError(
                f"Parameter count differs from reference: {total_parameters} != {reference_parameters}"
            )
        del reference
    configuration.update({
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "reference_parameter_count": reference_parameters,
    })
    write_json(config_path, configuration)

    if args.preflight_only:
        model.train()
        image, target = train_dataset[0]
        if tuple(image.shape) != (3, args.image_size, args.image_size):
            raise RuntimeError(f"Unexpected transformed image shape: {tuple(image.shape)}")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            losses = model(
                [image.to(device)],
                [{key: value.to(device) if torch.is_tensor(value) else value for key, value in target.items()}],
            )
        loss_values = {name: float(value.detach().cpu()) for name, value in losses.items()}
        if not loss_values or not all(np.isfinite(value) for value in loss_values.values()):
            raise RuntimeError(f"Preflight produced non-finite losses: {loss_values}")
        preflight = {
            "passed": True,
            "device": str(device),
            "train_base_samples": len(train_base),
            "train_usable_samples": len(train_dataset),
            "valid_base_samples": len(valid_base),
            "valid_usable_samples": len(valid_dataset),
            "excluded_samples": len(manifest["samples"]),
            "image_shape": list(image.shape),
            "target_box_count": int(target["boxes"].shape[0]),
            "losses": loss_values,
            "finite_losses": True,
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "all_parameters_trainable": total_parameters == trainable_parameters,
            "reference_parameter_count": reference_parameters,
            "parameter_count_matches_reference": reference_parameters == total_parameters,
            "gpu_peak_allocated_mib": (
                torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
            ),
            "gpu_peak_reserved_mib": (
                torch.cuda.max_memory_reserved(device) / (1024 ** 2) if device.type == "cuda" else 0.0
            ),
            "optimizer_step_performed": False,
            "test_split_constructed": False,
            "raw_train_valid_sha256": raw_hash_before,
        }
        write_json(reports_dir / "preflight.json", preflight)
        print(json.dumps(preflight, indent=2), flush=True)
        print("preflight_complete=True training_started=False", flush=True)
        return 0

    write_training_history(history_path, [])
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    class_mapping = {
        str(dataset_label + 1): {
            "dataset_label": dataset_label,
            "class_name": class_name,
        }
        for dataset_label, class_name in enumerate(ANIMAL_CLASS_NAMES)
    }
    best_map = float("-inf")
    best_map_50 = float("-inf")
    best_epoch = 0
    best_checkpoint: Path | None = None
    started = time.perf_counter()

    print(
        f"device={device} train_samples={len(train_dataset)} "
        f"valid_samples={len(valid_dataset)} epochs={args.epochs}"
    )
    print("training_test_split_accessed=False")
    try:
        for epoch_index in range(args.epochs):
            epoch = epoch_index + 1
            epoch_started = time.perf_counter()
            epoch_lr = float(optimizer.param_groups[0]["lr"])
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                epoch=epoch,
                print_freq=args.print_freq,
            )
            validation_metrics = evaluate_map(
                model,
                valid_loader,
                device,
                class_metrics=True,
                print_freq=50,
            )
            epoch_runtime = time.perf_counter() - epoch_started
            if train_metrics["num_samples"] != len(train_dataset) or validation_metrics["num_images"] != len(valid_dataset):
                raise RuntimeError("An epoch did not cover every usable training and validation image")
            record = {
                "epoch": epoch,
                "train_total_loss": train_metrics["total_loss"],
                "train_loss_classifier": train_metrics["loss_classifier"],
                "train_loss_box_reg": train_metrics["loss_box_reg"],
                "train_loss_objectness": train_metrics["loss_objectness"],
                "train_loss_rpn_box_reg": train_metrics["loss_rpn_box_reg"],
                "validation_map": validation_metrics["map"],
                "validation_map_50": validation_metrics["map_50"],
                "validation_map_75": validation_metrics["map_75"],
                "validation_mar_100": validation_metrics["mar_100"],
                "learning_rate": epoch_lr,
                "epoch_runtime_seconds": epoch_runtime,
                "training_runtime_seconds": train_metrics["runtime_seconds"],
                "validation_runtime_seconds": validation_metrics["runtime_seconds"],
                "gpu_peak_allocated_mib": (
                    torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                    if device.type == "cuda" else 0.0
                ),
                "gpu_peak_reserved_mib": (
                    torch.cuda.max_memory_reserved(device) / (1024 ** 2)
                    if device.type == "cuda" else 0.0
                ),
                "train_batches": train_metrics["num_batches"],
                "train_samples": train_metrics["num_samples"],
                "validation_batches": validation_metrics["num_batches"],
                "validation_images": validation_metrics["num_images"],
            }
            current_map = float(validation_metrics["map"])
            current_map_50 = float(validation_metrics["map_50"])
            improved = (current_map, current_map_50) > (best_map, best_map_50)
            scheduler.step()
            if improved:
                best_map = current_map
                best_map_50 = current_map_50
                best_epoch = epoch
                best_checkpoint = save_checkpoint(
                    checkpoints_dir / "best.pt",
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    validation_metrics=validation_metrics,
                    training_metrics=train_metrics,
                    configuration=configuration,
                    class_mapping=class_mapping,
                    scheduler=scheduler,
                )
                write_json(class_metrics_path, {
                    "best_epoch": epoch,
                    "metrics": validation_metrics["per_class"],
                })
            last_checkpoint = save_checkpoint(
                checkpoints_dir / "last.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                validation_metrics=validation_metrics,
                training_metrics=train_metrics,
                configuration=configuration,
                class_mapping=class_mapping,
                scheduler=scheduler,
            )
            history = append_training_history(history_path, record)
            print(
                f"epoch={epoch} complete loss={record['train_total_loss']:.6f} "
                f"map={current_map:.6f} map50={current_map_50:.6f} "
                f"lr_used={epoch_lr:.8g} runtime_seconds={epoch_runtime:.1f} "
                f"best_epoch={best_epoch}",
                flush=True,
            )
    except RuntimeError as exc:
        if device.type == "cuda" and "out of memory" in str(exc).casefold():
            print(f"CUDA OOM: {exc}", file=sys.stderr)
            torch.cuda.empty_cache()
            return 2
        raise

    raw_hash_after = split_integrity_hash(args.dataset_root)
    if raw_hash_after != raw_hash_before:
        raise RuntimeError("Train/valid raw data changed during the experiment")
    plot_history(history, reports_dir)
    best_record = history[best_epoch - 1]
    summary = {
        "experiment_name": args.experiment_name,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_map": best_record["validation_map"],
        "best_validation_map_50": best_record["validation_map_50"],
        "best_validation_map_75": best_record["validation_map_75"],
        "final_epoch_metrics": history[-1],
        "total_training_runtime_seconds": time.perf_counter() - started,
        "mean_epoch_runtime_seconds": sum(row["epoch_runtime_seconds"] for row in history) / len(history),
        "peak_gpu_allocated_mib": max(row["gpu_peak_allocated_mib"] for row in history),
        "peak_gpu_reserved_mib": max(row["gpu_peak_reserved_mib"] for row in history),
        "best_checkpoint_path": str(best_checkpoint),
        "last_checkpoint_path": str(last_checkpoint),
        "usable_training_count": len(train_dataset),
        "usable_validation_count": len(valid_dataset),
        "excluded_sample_count": len(manifest["samples"]),
        "raw_train_valid_sha256_before": raw_hash_before,
        "raw_train_valid_sha256_after": raw_hash_after,
        "test_split_accessed_by_training": False,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "all_parameters_trainable": total_parameters == trainable_parameters,
    }
    write_json(summary_path, summary)
    print(f"training_history={history_path}")
    print(f"class_metrics={class_metrics_path}")
    print(f"summary={summary_path}")
    print("training_complete=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
