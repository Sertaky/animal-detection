#!/usr/bin/env python3
"""Train Faster R-CNN and evaluate only on the validation split."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import SGD
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--image-size", type=int, default=512)
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

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    seed_everything(args.seed)
    output_dir = args.output_dir.resolve()
    checkpoints_dir = output_dir / "checkpoints"
    reports_dir = output_dir / "reports"
    history_path = reports_dir / "training_history.json"
    class_metrics_path = reports_dir / "validation_class_metrics.json"
    write_training_history(history_path, [])

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
    train_dataset = limited_dataset(
        TorchvisionDetectionDataset(train_base), args.max_train_batches, args.batch_size
    )
    valid_dataset = limited_dataset(
        TorchvisionDetectionDataset(valid_base), args.max_val_batches, args.batch_size
    )
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

    model = build_faster_rcnn(
        num_classes=21,
        pretrained=True,
        min_size=args.image_size,
        max_size=args.image_size,
    ).to(device)
    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    configuration = json_configuration(args)
    class_mapping = {
        str(dataset_label + 1): {
            "dataset_label": dataset_label,
            "class_name": class_name,
        }
        for dataset_label, class_name in enumerate(ANIMAL_CLASS_NAMES)
    }
    best_map = float("-inf")
    best_checkpoint: Path | None = None

    print(
        f"device={device} train_samples={len(train_dataset)} "
        f"valid_samples={len(valid_dataset)} epochs={args.epochs}"
    )
    print("test_split_accessed=False")
    try:
        for epoch_index in range(args.epochs):
            epoch = epoch_index + 1
            epoch_started = time.perf_counter()
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
            )
            epoch_runtime = time.perf_counter() - epoch_started
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
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_runtime_seconds": epoch_runtime,
                "training_runtime_seconds": train_metrics["runtime_seconds"],
                "validation_runtime_seconds": validation_metrics["runtime_seconds"],
                "gpu_peak_allocated_mib": train_metrics["gpu_peak_allocated_mib"],
                "gpu_peak_reserved_mib": train_metrics["gpu_peak_reserved_mib"],
                "train_batches": train_metrics["num_batches"],
                "train_samples": train_metrics["num_samples"],
                "validation_batches": validation_metrics["num_batches"],
                "validation_images": validation_metrics["num_images"],
            }
            append_training_history(history_path, record)
            class_metrics_path.parent.mkdir(parents=True, exist_ok=True)
            class_metrics_path.write_text(
                json.dumps(
                    {
                        "epoch": epoch,
                        "metrics": validation_metrics["per_class"],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            current_map = float(validation_metrics["map"])
            if current_map > best_map:
                best_map = current_map
                best_checkpoint = save_checkpoint(
                    checkpoints_dir / "faster_rcnn_best.pt",
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    validation_metrics=validation_metrics,
                    training_metrics=train_metrics,
                    configuration=configuration,
                    class_mapping=class_mapping,
                )
            print(json.dumps(record, indent=2))
            print(f"best_checkpoint={best_checkpoint}")
    except RuntimeError as exc:
        if device.type == "cuda" and "out of memory" in str(exc).casefold():
            print(f"CUDA OOM: {exc}", file=sys.stderr)
            torch.cuda.empty_cache()
            return 2
        raise

    print(f"training_history={history_path}")
    print(f"class_metrics={class_metrics_path}")
    print("training_complete=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
