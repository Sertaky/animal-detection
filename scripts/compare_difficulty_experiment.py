#!/usr/bin/env python3
"""Compare difficulty-aware sampling with Baseline 01 and Experiment 03."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

WEAK = ("Panda", "Monkeys", "Gorilla", "Goat", "Camel")
REFERENCE = ("Dog", "Wolf", "Rhino", "Lion")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def pair(reference: float | int, candidate: float | int) -> dict[str, float]:
    return {
        "reference": float(reference), "candidate": float(candidate),
        "delta": float(candidate) - float(reference),
    }


def artifacts(root: Path) -> dict[str, Any]:
    error = root / "error_analysis"
    summary = load(root / "summary.json")
    history = load(root / "training_history.json")
    return {
        "config": load(root / "config.json"), "summary": summary, "history": history,
        "best": history[summary["best_epoch"] - 1],
        "ap": {row["class_name"]: row["ap"] for row in load(root / "validation_class_metrics.json")["metrics"]},
        "error": load(error / "summary.json"),
        "size": load(error / "object_size_analysis.json"),
        "crowd": load(error / "crowdedness_analysis.json"),
        "localization": load(error / "localization_analysis.json"),
        "confidence": load(error / "confidence_analysis.json"),
    }


def compare(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    rs, cs = reference["summary"], candidate["summary"]
    re, ce = reference["error"], candidate["error"]
    rl, cl = reference["localization"], candidate["localization"]
    metrics = {
        "map": pair(rs["best_validation_map"], cs["best_validation_map"]),
        "map_50": pair(rs["best_validation_map_50"], cs["best_validation_map_50"]),
        "map_75": pair(rs["best_validation_map_75"], cs["best_validation_map_75"]),
        "mar_100": pair(reference["best"]["validation_mar_100"], candidate["best"]["validation_mar_100"]),
        "precision_score_050": pair(re["precision"], ce["precision"]),
        "recall_score_050": pair(re["recall"], ce["recall"]),
        "tp": pair(re["true_positives"], ce["true_positives"]),
        "fp": pair(re["false_positives"], ce["false_positives"]),
        "fn": pair(re["false_negatives"], ce["false_negatives"]),
        "class_confusion_count": pair(re["false_positive_categories"]["class_confusion"], ce["false_positive_categories"]["class_confusion"]),
        "localization_error_count": pair(re["false_positive_categories"]["localization"], ce["false_positive_categories"]["localization"]),
        "background_fp_count": pair(re["false_positive_categories"]["background"], ce["false_positive_categories"]["background"]),
        "duplicate_count": pair(re["false_positive_categories"]["duplicate"], ce["false_positive_categories"]["duplicate"]),
        "mean_matched_iou": pair(rl["matched_iou"]["mean"], cl["matched_iou"]["mean"]),
        "median_matched_iou": pair(rl["matched_iou"]["median"], cl["matched_iou"]["median"]),
        "mean_epoch_runtime_seconds": pair(rs["mean_epoch_runtime_seconds"], cs["mean_epoch_runtime_seconds"]),
        "total_runtime_seconds": pair(rs["total_training_runtime_seconds"], cs["total_training_runtime_seconds"]),
        "peak_gpu_allocated_mib": pair(rs["peak_gpu_allocated_mib"], cs["peak_gpu_allocated_mib"]),
        "peak_gpu_reserved_mib": pair(rs["peak_gpu_reserved_mib"], cs["peak_gpu_reserved_mib"]),
        "best_epoch": pair(rs["best_epoch"], cs["best_epoch"]),
    }
    for group in ("small", "medium", "large"):
        metrics[f"{group}_recall"] = pair(reference["size"]["groups"][group]["recall"], candidate["size"]["groups"][group]["recall"])
    for group, name in (("1", "one_object_recall"), ("2-3", "two_three_object_recall"), ("4+", "four_plus_object_recall")):
        metrics[name] = pair(reference["crowd"]["groups"][group]["recall"], candidate["crowd"]["groups"][group]["recall"])
    confidence = {
        outcome: {stat: pair(reference["confidence"]["by_outcome"][outcome][stat], candidate["confidence"]["by_outcome"][outcome][stat]) for stat in ("mean", "median")}
        for outcome in ("true_positive", "false_positive")
    }
    class_ap = {name: pair(reference["ap"][name], candidate["ap"][name]) for name in reference["ap"]}
    return {"metrics": metrics, "class_ap": class_ap, "confidence": confidence}


def text_comparison(title: str, comparison: dict[str, Any]) -> str:
    lines = [title, "", "Metric | Reference | Experiment 04 | Delta"]
    for name, values in comparison["metrics"].items():
        lines.append(f"{name} | {values['reference']:.6f} | {values['candidate']:.6f} | {values['delta']:+.6f}")
    lines.extend(["", "Class AP:"])
    for name, values in comparison["class_ap"].items():
        lines.append(f"{name} | {values['reference']:.6f} | {values['candidate']:.6f} | {values['delta']:+.6f}")
    lines.extend(["", "Confidence:"])
    for outcome, stats in comparison["confidence"].items():
        for stat, values in stats.items():
            lines.append(f"{outcome}_{stat} | {values['reference']:.6f} | {values['candidate']:.6f} | {values['delta']:+.6f}")
    return "\n".join(lines) + "\n"


def plots(output: Path, baseline: dict[str, Any], candidate: dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def plot(groups: tuple[str, ...], source: str, filename: str, title: str) -> None:
        before = baseline[source]["groups"]
        after = candidate[source]["groups"]
        x = list(range(len(groups)))
        width = 0.36
        figure, axis = plt.subplots(figsize=(7, 4.5))
        axis.bar([value - width / 2 for value in x], [before[group]["recall"] for group in groups], width, label="Baseline 01")
        axis.bar([value + width / 2 for value in x], [after[group]["recall"] for group in groups], width, label="Difficulty-aware")
        axis.set(xticks=x, xticklabels=groups, ylabel="Recall @ score/IoU 0.50", title=title, ylim=(0, 1))
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / filename, dpi=150)
        plt.close(figure)

    plot(("small", "medium", "large"), "size", "difficulty_recall_comparison.png", "Recall by object size")
    plot(("1", "2-3", "4+"), "crowd", "crowdedness_recall_comparison.png", "Recall by annotated objects per image")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--class-aware-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    args = parser.parse_args()
    baseline = artifacts(args.baseline_dir.resolve())
    class_aware = artifacts(args.class_aware_dir.resolve())
    candidate_root = args.candidate_dir.resolve()
    candidate = artifacts(candidate_root)
    hashes = {item["config"]["raw_train_valid_sha256_before"] for item in (baseline, class_aware, candidate)}
    if len(hashes) != 1 or any(item["config"]["image_size"] != 512 for item in (baseline, class_aware, candidate)):
        raise ValueError("Comparison experiments do not share raw data and 512x512 setup")
    vs_baseline = compare(baseline, candidate)
    vs_class = compare(class_aware, candidate)
    payload = {
        "baseline_experiment": baseline["config"]["experiment_name"],
        "candidate_experiment": candidate["config"]["experiment_name"],
        "controlled_change": candidate["config"]["controlled_change"],
        **vs_baseline,
    }
    write_json(candidate_root / "comparison_vs_baseline_01.json", payload)
    (candidate_root / "comparison_vs_baseline_01.txt").write_text(text_comparison("BASELINE 01 VS DIFFICULTY-AWARE SAMPLING", vs_baseline), encoding="utf-8")
    (candidate_root / "comparison_vs_class_aware_sampling_01.txt").write_text(text_comparison("CLASS-AWARE VS DIFFICULTY-AWARE SAMPLING", vs_class), encoding="utf-8")
    plots(candidate_root, baseline, candidate)
    sampling = load(candidate_root / "sampling_analysis.json")
    m = vs_baseline["metrics"]
    reference_deltas = ", ".join(f"{name} {vs_baseline['class_ap'][name]['delta']:+.6f}" for name in REFERENCE)
    effect = [
        "DIFFICULTY-AWARE SAMPLING EFFECT",
        "",
        f"1. Small-object-image exposure: {sampling['groups']['small_object_containing']['exposure_multiplier']:.3f}x ({sampling['groups']['small_object_containing']['sampled_occurrences']} draws / {sampling['groups']['small_object_containing']['raw_image_count']} raw images).",
        f"2. Crowded-scene exposure: {sampling['groups']['crowded']['exposure_multiplier']:.3f}x ({sampling['groups']['crowded']['sampled_occurrences']} draws / {sampling['groups']['crowded']['raw_image_count']} raw images).",
        f"3. Small-object recall delta: {m['small_recall']['delta']:+.6f}.",
        f"4. Four-plus-object recall delta: {m['four_plus_object_recall']['delta']:+.6f}.",
        f"5. Overall mAP delta: {m['map']['delta']:+.6f}.",
        f"6. Precision delta: {m['precision_score_050']['delta']:+.6f}.",
        f"7. Localization errors delta: {m['localization_error_count']['delta']:+.0f}; mean matched IoU delta {m['mean_matched_iou']['delta']:+.6f}.",
        f"8. Class-confusion count delta: {m['class_confusion_count']['delta']:+.0f}.",
        f"9. Strong reference AP deltas: {reference_deltas}.",
        f"10. Versus class-aware: mAP delta {vs_class['metrics']['map']['delta']:+.6f}, small recall delta {vs_class['metrics']['small_recall']['delta']:+.6f}, four-plus recall delta {vs_class['metrics']['four_plus_object_recall']['delta']:+.6f}, precision delta {vs_class['metrics']['precision_score_050']['delta']:+.6f}.",
        "11. Overall assessment: not beneficial. Exposure increased for the intended image groups, but small-object recall did not improve, crowded-scene recall and overall mAP declined, and all three primary outcomes were worse than class-aware sampling.",
        "",
        "No test metrics were used.",
    ]
    (candidate_root / "difficulty_sampling_effect.txt").write_text("\n".join(effect) + "\n", encoding="utf-8")
    print(f"comparison={candidate_root / 'comparison_vs_baseline_01.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
