#!/usr/bin/env python3
"""Benchmark steady-state inference for a locked Faster R-CNN checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from animal_detection.data import TorchvisionDetectionDataset, YoloDetectionDataset, get_eval_transforms
from animal_detection.data.label_mapping import ANIMAL_CLASS_NAMES
from animal_detection.models import build_faster_rcnn


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def summarize(latencies: list[float]) -> dict[str, float]:
    mean_ms = statistics.fmean(latencies)
    return {
        "mean_latency_ms": mean_ms,
        "median_latency_ms": statistics.median(latencies),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "images_per_second": 1000.0 / mean_ms,
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
    }


def time_calls(
    call: Callable[[int], None], device: torch.device, warmup: int, iterations: int,
) -> dict[str, Any]:
    with torch.inference_mode():
        for index in range(warmup):
            call(index)
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        latencies = []
        for index in range(iterations):
            synchronize(device)
            started = time.perf_counter()
            call(index)
            synchronize(device)
            latencies.append((time.perf_counter() - started) * 1000.0)
    result: dict[str, Any] = {
        "warmup_iterations": warmup, "timed_iterations": iterations,
        "batch_size": 1, **summarize(latencies),
    }
    if device.type == "cuda":
        result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        result["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        result["peak_cuda_allocated_mib"] = result["peak_cuda_allocated_bytes"] / 1024**2
        result["peak_cuda_reserved_mib"] = result["peak_cuda_reserved_bytes"] / 1024**2
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "faster_rcnn_baseline_01" / "best.pt")
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "Multi-Class Animal Detection.v1-yolov8")
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "final_evaluation")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("warmup must be nonnegative and iterations must be positive")

    checkpoint_path = args.checkpoint.resolve()
    root = args.dataset_root.resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    expected_mapping = {
        str(index): {"dataset_label": index - 1, "class_name": name}
        for index, name in enumerate(ANIMAL_CLASS_NAMES, start=1)
    }
    if checkpoint.get("epoch") != 9 or checkpoint.get("class_mapping") != expected_mapping:
        raise ValueError("Benchmark checkpoint differs from the locked epoch or class mapping")
    model = build_faster_rcnn(num_classes=21, pretrained=False, min_size=512, max_size=512)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()

    dataset = TorchvisionDetectionDataset(
        YoloDetectionDataset(root, args.split, transforms=get_eval_transforms(size=(512, 512)))
    )
    representative_count = min(10, len(dataset))
    host_images = [dataset[index][0] for index in range(representative_count)]
    device_images = [image.to(device) for image in host_images]

    model_only = time_calls(
        lambda index: model([device_images[index % representative_count]]),
        device, args.warmup, args.iterations,
    )

    def end_to_end(index: int) -> None:
        image, _ = dataset[index % len(dataset)]
        model([image.to(device)])

    end_to_end_result = time_calls(end_to_end, device, args.warmup, args.iterations)
    output = {
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": 9, "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "split": args.split, "image_size": [512, 512], "batch_size": 1,
        "representative_model_only_images": representative_count,
        "model_loading_included": False,
        "model_only_definition": "Preprocessed 3x512x512 tensors are already on the target device; timing includes the full torchvision detector call and its postprocessing.",
        "end_to_end_definition": "Timing includes image decode, RGB conversion, annotation parse, Resize(512,512), ToTensor, device transfer, detector call, and detector postprocessing; it excludes model loading.",
        "cuda_timing_synchronization": device.type == "cuda",
        "model_only": model_only, "end_to_end": end_to_end_result,
        "training_occurred": False, "optimizer_constructed": False,
    }
    output_dir = args.output_dir.resolve()
    tag = "cuda" if device.type == "cuda" else "cpu"
    write_json(output_dir / f"benchmark_{tag}.json", output)
    lines = [
        f"INFERENCE BENCHMARK ({tag.upper()})",
        f"Device: {output['device_name']}",
        f"Checkpoint: {checkpoint_path}",
        f"SHA-256: {output['checkpoint_sha256']}",
        f"Split: {args.split}; preprocessing: Resize(512,512) + ToTensor; batch size: 1",
        f"Warmup: {args.warmup}; timed iterations: {args.iterations}; model loading excluded",
        "CUDA synchronization surrounds each timed call." if device.type == "cuda" else "CPU wall-clock timing surrounds each call.",
        "",
    ]
    for name, result in (("Model only", model_only), ("End to end", end_to_end_result)):
        lines.append(
            f"{name}: mean={result['mean_latency_ms']:.3f} ms, median={result['median_latency_ms']:.3f} ms, "
            f"p95={result['p95_latency_ms']:.3f} ms, throughput={result['images_per_second']:.3f} images/s"
        )
        if device.type == "cuda":
            lines.append(
                f"{name}: peak allocated={result['peak_cuda_allocated_mib']:.3f} MiB, "
                f"peak reserved={result['peak_cuda_reserved_mib']:.3f} MiB"
            )
    (output_dir / f"benchmark_{tag}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
