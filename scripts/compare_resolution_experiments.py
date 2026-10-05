#!/usr/bin/env python3
"""Compare two completed detection experiments using their saved artifacts only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def delta(candidate: float | int, baseline: float | int) -> float:
    return float(candidate) - float(baseline)


def metric_pair(baseline: float | int, candidate: float | int) -> dict[str, float]:
    return {
        "baseline": float(baseline),
        "candidate": float(candidate),
        "delta": delta(candidate, baseline),
    }


def experiment(root: Path) -> dict[str, Any]:
    return {
        "config": load(root / "config.json"),
        "summary": load(root / "summary.json"),
        "history": load(root / "training_history.json"),
        "classes": load(root / "validation_class_metrics.json"),
        "errors": {
            name: load(root / "error_analysis" / name)
            for name in (
                "summary.json", "localization_analysis.json",
                "object_size_analysis.json", "crowdedness_analysis.json",
                "class_confusion.json",
            )
        },
    }


def plot_comparison(result: dict[str, Any], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline = result["artifacts"]["baseline_history"]
    candidate = result["artifacts"]["candidate_history"]
    epochs = [row["epoch"] for row in baseline]
    for filename, key, ylabel in (
        ("validation_map_by_epoch.png", "validation_map", "Validation mAP@0.50:0.95"),
        ("training_loss_by_epoch.png", "train_total_loss", "Mean training loss"),
        ("epoch_runtime_by_epoch.png", "epoch_runtime_seconds", "Epoch runtime (seconds)"),
    ):
        figure, axis = plt.subplots(figsize=(7.5, 4.5))
        axis.plot(epochs, [row[key] for row in baseline], marker="o", label="512 baseline")
        axis.plot(epochs, [row[key] for row in candidate], marker="o", label="640 candidate")
        axis.set(xlabel="Epoch", ylabel=ylabel)
        axis.set_xticks(epochs)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / filename, dpi=150)
        plt.close(figure)

    classes = result["per_class_ap"]
    figure, axis = plt.subplots(figsize=(10, 5.5))
    labels = [item["class_name"] for item in classes]
    values = [item["delta"] for item in classes]
    colors = ["#2374ab" if value >= 0 else "#d1495b" for value in values]
    axis.bar(labels, values, color=colors)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(ylabel="AP delta (640 - 512)", xlabel="Class")
    axis.tick_params(axis="x", rotation=55)
    figure.tight_layout()
    figure.savefig(output / "per_class_ap_delta.png", dpi=150)
    plt.close(figure)

    metric_names = ("map", "map_50", "map_75")
    display_names = ("mAP@0.50:0.95", "mAP@0.50", "mAP@0.75")
    x = list(range(len(metric_names)))
    width = 0.36
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.bar(
        [value - width / 2 for value in x],
        [result["best_metrics"][name]["baseline"] for name in metric_names],
        width, label="512 baseline",
    )
    axis.bar(
        [value + width / 2 for value in x],
        [result["best_metrics"][name]["candidate"] for name in metric_names],
        width, label="640 candidate",
    )
    axis.set(xticks=x, xticklabels=display_names, ylabel="Best validation metric", ylim=(0, 1))
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "comparison_map.png", dpi=150)
    plt.close(figure)

    size_names = ("small", "medium", "large")
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.bar(
        [value - width / 2 for value in x],
        [result["size_recall"][name]["baseline"] for name in size_names],
        width, label="512 baseline",
    )
    axis.bar(
        [value + width / 2 for value in x],
        [result["size_recall"][name]["candidate"] for name in size_names],
        width, label="640 candidate",
    )
    axis.set(xticks=x, xticklabels=[name.title() for name in size_names], ylabel="Diagnostic recall", ylim=(0, 1))
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "small_medium_large_recall_comparison.png", dpi=150)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    baseline = experiment(args.baseline_dir.resolve())
    candidate = experiment(args.candidate_dir.resolve())
    baseline_config, candidate_config = baseline["config"], candidate["config"]
    allowed = {
        "image_size", "experiment_name", "comparison_target", "reference_checkpoint",
        "preflight_only", "train_transforms", "valid_transforms", "controlled_change",
        "total_parameters", "trainable_parameters", "reference_parameter_count",
    }
    ignored = {"raw_train_valid_sha256_before"}
    differing = {
        key: {"baseline": baseline_config.get(key), "candidate": candidate_config.get(key)}
        for key in sorted(set(baseline_config) | set(candidate_config))
        if baseline_config.get(key) != candidate_config.get(key) and key not in ignored
    }
    unexpected = sorted(set(differing) - allowed)
    if unexpected:
        raise ValueError(f"Non-resolution configuration differences found: {unexpected}")
    if candidate_config["image_size"] != 640 or baseline_config["image_size"] != 512:
        raise ValueError("Expected a 512 baseline and 640 candidate")
    if baseline_config["raw_train_valid_sha256_before"] != candidate_config["raw_train_valid_sha256_before"]:
        raise ValueError("Train/valid raw hashes differ between experiments")

    bs, cs = baseline["summary"], candidate["summary"]
    be, ce = baseline["errors"]["summary.json"], candidate["errors"]["summary.json"]
    bl, cl = baseline["errors"]["localization_analysis.json"], candidate["errors"]["localization_analysis.json"]
    baseline_ap = {item["class_name"]: item["ap"] for item in baseline["classes"]["metrics"]}
    candidate_ap = {item["class_name"]: item["ap"] for item in candidate["classes"]["metrics"]}
    per_class = [
        {"class_name": name, **metric_pair(baseline_ap[name], candidate_ap[name])}
        for name in baseline_ap
    ]
    result = {
        "comparison": "640x640 minus 512x512; all other reviewed settings held constant",
        "baseline_experiment": baseline_config["experiment_name"],
        "candidate_experiment": candidate_config["experiment_name"],
        "configuration_differences": differing,
        "raw_train_valid_sha256": candidate_config["raw_train_valid_sha256_before"],
        "best_epoch": metric_pair(bs["best_epoch"], cs["best_epoch"]),
        "best_metrics": {
            key: metric_pair(bs[base_key], cs[base_key])
            for key, base_key in (
                ("map", "best_validation_map"),
                ("map_50", "best_validation_map_50"),
                ("map_75", "best_validation_map_75"),
            )
        },
        "best_mar_100": metric_pair(
            baseline["history"][bs["best_epoch"] - 1]["validation_mar_100"],
            candidate["history"][cs["best_epoch"] - 1]["validation_mar_100"],
        ),
        "runtime_and_memory": {
            "total_training_seconds": metric_pair(bs["total_training_runtime_seconds"], cs["total_training_runtime_seconds"]),
            "mean_epoch_seconds": metric_pair(bs["mean_epoch_runtime_seconds"], cs["mean_epoch_runtime_seconds"]),
            "peak_gpu_allocated_mib": metric_pair(bs["peak_gpu_allocated_mib"], cs["peak_gpu_allocated_mib"]),
            "peak_gpu_reserved_mib": metric_pair(bs["peak_gpu_reserved_mib"], cs["peak_gpu_reserved_mib"]),
        },
        "diagnostics_at_score_050_iou_050": {
            key: metric_pair(be[key], ce[key])
            for key in ("true_positives", "false_positives", "false_negatives", "precision", "recall")
        },
        "diagnostic_error_counts": {
            name: metric_pair(
                be["false_positive_categories"][name], ce["false_positive_categories"][name]
            ) for name in ("localization", "class_confusion")
        },
        "top_confusion_pairs": {
            "baseline": baseline["errors"]["class_confusion.json"]["pairs"][:10],
            "candidate": candidate["errors"]["class_confusion.json"]["pairs"][:10],
        },
        "localization": {
            "matched_mean_iou": metric_pair(bl["matched_iou"]["mean"], cl["matched_iou"]["mean"]),
            "matched_median_iou": metric_pair(bl["matched_iou"]["median"], cl["matched_iou"]["median"]),
            "matched_050_to_lt_075_fraction": metric_pair(
                bl["matched_050_to_lt_075_fraction"], cl["matched_050_to_lt_075_fraction"]
            ),
        },
        "size_recall": {
            name: metric_pair(
                baseline["errors"]["object_size_analysis.json"]["groups"][name]["recall"],
                candidate["errors"]["object_size_analysis.json"]["groups"][name]["recall"],
            ) for name in ("small", "medium", "large")
        },
        "crowdedness_recall": {
            name: metric_pair(
                baseline["errors"]["crowdedness_analysis.json"]["groups"][name]["recall"],
                candidate["errors"]["crowdedness_analysis.json"]["groups"][name]["recall"],
            ) for name in ("1", "2-3", "4+")
        },
        "per_class_ap": per_class,
        "artifacts": {
            "baseline_history": baseline["history"],
            "candidate_history": candidate["history"],
        },
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "comparison_vs_baseline_01.json", result)
    plot_comparison(result, output)
    ranked = sorted(per_class, key=lambda item: item["delta"], reverse=True)
    lines = [
        "CONTROLLED RESOLUTION COMPARISON: 640x640 vs 512x512",
        f"Baseline: {result['baseline_experiment']} (best epoch {bs['best_epoch']})",
        f"Candidate: {result['candidate_experiment']} (best epoch {cs['best_epoch']})",
        "Only reviewed experimental change: input resize 512 -> 640.",
        "",
        *[
            f"{name}: {item['baseline']:.6f} -> {item['candidate']:.6f} (delta {item['delta']:+.6f})"
            for name, item in result["best_metrics"].items()
        ],
        f"mAR@100: {result['best_mar_100']['baseline']:.6f} -> {result['best_mar_100']['candidate']:.6f} "
        f"(delta {result['best_mar_100']['delta']:+.6f})",
        "",
        "Diagnostic recall by object size:",
        *[
            f"  {name}: {item['baseline']:.6f} -> {item['candidate']:.6f} (delta {item['delta']:+.6f})"
            for name, item in result["size_recall"].items()
        ],
        "Diagnostic recall by objects per image:",
        *[
            f"  {name}: {item['baseline']:.6f} -> {item['candidate']:.6f} (delta {item['delta']:+.6f})"
            for name, item in result["crowdedness_recall"].items()
        ],
        "",
        "Largest class AP gains: " + ", ".join(f"{item['class_name']} {item['delta']:+.3f}" for item in ranked[:5]),
        "Largest class AP declines: " + ", ".join(f"{item['class_name']} {item['delta']:+.3f}" for item in ranked[-5:]),
        "",
        "Runtime and memory:",
        *[
            f"  {name}: {item['baseline']:.3f} -> {item['candidate']:.3f} (delta {item['delta']:+.3f})"
            for name, item in result["runtime_and_memory"].items()
        ],
    ]
    (output / "comparison_vs_baseline_01.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    diagnostics = result["diagnostics_at_score_050_iou_050"]
    localization = result["localization"]
    runtime = result["runtime_and_memory"]
    size = result["size_recall"]
    crowded = result["crowdedness_recall"]
    summary_lines = [
        "RESOLUTION EXPERIMENT: 512 VS 640",
        "",
        f"1. Overall mAP improved: No ({result['best_metrics']['map']['baseline']:.6f} -> "
        f"{result['best_metrics']['map']['candidate']:.6f}, delta {result['best_metrics']['map']['delta']:+.6f}).",
        f"2. mAP@0.75 improved: No ({result['best_metrics']['map_75']['baseline']:.6f} -> "
        f"{result['best_metrics']['map_75']['candidate']:.6f}, delta {result['best_metrics']['map_75']['delta']:+.6f}).",
        f"3. Small-object recall improved: Yes ({size['small']['baseline']:.6f} -> "
        f"{size['small']['candidate']:.6f}, delta {size['small']['delta']:+.6f}).",
        f"4. Crowded-scene (4+) recall improved: Yes ({crowded['4+']['baseline']:.6f} -> "
        f"{crowded['4+']['candidate']:.6f}, delta {crowded['4+']['delta']:+.6f}).",
        f"5. Mean/median matched IoU: {localization['matched_mean_iou']['baseline']:.6f}/"
        f"{localization['matched_median_iou']['baseline']:.6f} -> "
        f"{localization['matched_mean_iou']['candidate']:.6f}/"
        f"{localization['matched_median_iou']['candidate']:.6f}. "
        f"Localization-error count changed {result['diagnostic_error_counts']['localization']['baseline']:.0f} -> "
        f"{result['diagnostic_error_counts']['localization']['candidate']:.0f}.",
        "6. Largest AP gains: " + ", ".join(f"{item['class_name']} {item['delta']:+.3f}" for item in ranked[:5]) + ".",
        "7. Largest AP losses: " + ", ".join(f"{item['class_name']} {item['delta']:+.3f}" for item in reversed(ranked[-5:])) + ".",
        f"8. Runtime cost: mean epoch {runtime['mean_epoch_seconds']['baseline']:.3f} -> "
        f"{runtime['mean_epoch_seconds']['candidate']:.3f} seconds; total {runtime['total_training_seconds']['baseline']:.3f} -> "
        f"{runtime['total_training_seconds']['candidate']:.3f} seconds.",
        f"9. GPU-memory cost: allocated {runtime['peak_gpu_allocated_mib']['baseline']:.3f} -> "
        f"{runtime['peak_gpu_allocated_mib']['candidate']:.3f} MiB; reserved "
        f"{runtime['peak_gpu_reserved_mib']['baseline']:.3f} -> {runtime['peak_gpu_reserved_mib']['candidate']:.3f} MiB.",
        "10. Worth it: No for this fixed 10-epoch setup; targeted small/crowded recall gains did not offset "
        "lower AP, more false positives/class confusions, and higher runtime/memory cost.",
        "",
        f"At score/IoU 0.50, precision changed {diagnostics['precision']['baseline']:.6f} -> "
        f"{diagnostics['precision']['candidate']:.6f} and recall changed {diagnostics['recall']['baseline']:.6f} -> "
        f"{diagnostics['recall']['candidate']:.6f}.",
        "No Experiment 03 is recommended from this result alone; review the measured tradeoff first.",
        "No held-out test data or metrics were used.",
    ]
    (args.candidate_dir.resolve() / "summary.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )
    print(f"comparison={output / 'comparison_vs_baseline_01.json'} plots=6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
