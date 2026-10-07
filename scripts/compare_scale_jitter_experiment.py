#!/usr/bin/env python3
"""Generate Experiment 05 comparisons and scale-jitter effect reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REFERENCE_CLASSES = ("Dog", "Wolf", "Rhino", "Lion")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def artifacts(root: Path) -> dict[str, Any]:
    summary = load(root / "summary.json")
    history = load(root / "training_history.json")
    error = root / "error_analysis"
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


def pair(reference: float | int, candidate: float | int) -> dict[str, float]:
    return {"baseline": float(reference), "candidate": float(candidate), "delta": float(candidate) - float(reference)}


def compare(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    bs, cs = baseline["summary"], candidate["summary"]
    be, ce = baseline["error"], candidate["error"]
    bl, cl = baseline["localization"], candidate["localization"]
    metrics = {
        "map": pair(bs["best_validation_map"], cs["best_validation_map"]),
        "map_50": pair(bs["best_validation_map_50"], cs["best_validation_map_50"]),
        "map_75": pair(bs["best_validation_map_75"], cs["best_validation_map_75"]),
        "mar_100": pair(baseline["best"]["validation_mar_100"], candidate["best"]["validation_mar_100"]),
        "precision_score_050": pair(be["precision"], ce["precision"]),
        "recall_score_050": pair(be["recall"], ce["recall"]),
        "tp": pair(be["true_positives"], ce["true_positives"]),
        "fp": pair(be["false_positives"], ce["false_positives"]),
        "fn": pair(be["false_negatives"], ce["false_negatives"]),
        "class_confusion_count": pair(be["false_positive_categories"]["class_confusion"], ce["false_positive_categories"]["class_confusion"]),
        "localization_error_count": pair(be["false_positive_categories"]["localization"], ce["false_positive_categories"]["localization"]),
        "background_fp_count": pair(be["false_positive_categories"]["background"], ce["false_positive_categories"]["background"]),
        "duplicate_count": pair(be["false_positive_categories"]["duplicate"], ce["false_positive_categories"]["duplicate"]),
        "mean_matched_iou": pair(bl["matched_iou"]["mean"], cl["matched_iou"]["mean"]),
        "median_matched_iou": pair(bl["matched_iou"]["median"], cl["matched_iou"]["median"]),
        "mean_epoch_runtime_seconds": pair(bs["mean_epoch_runtime_seconds"], cs["mean_epoch_runtime_seconds"]),
        "total_runtime_seconds": pair(bs["total_training_runtime_seconds"], cs["total_training_runtime_seconds"]),
        "peak_gpu_allocated_mib": pair(bs["peak_gpu_allocated_mib"], cs["peak_gpu_allocated_mib"]),
        "peak_gpu_reserved_mib": pair(bs["peak_gpu_reserved_mib"], cs["peak_gpu_reserved_mib"]),
        "best_epoch": pair(bs["best_epoch"], cs["best_epoch"]),
    }
    for group in ("small", "medium", "large"):
        metrics[f"{group}_recall"] = pair(baseline["size"]["groups"][group]["recall"], candidate["size"]["groups"][group]["recall"])
    for group, name in (("1", "one_object_recall"), ("2-3", "two_three_object_recall"), ("4+", "four_plus_object_recall")):
        metrics[name] = pair(baseline["crowd"]["groups"][group]["recall"], candidate["crowd"]["groups"][group]["recall"])
    class_ap = {name: pair(value, candidate["ap"][name]) for name, value in baseline["ap"].items()}
    confidence = {
        outcome: {stat: pair(baseline["confidence"]["by_outcome"][outcome][stat], candidate["confidence"]["by_outcome"][outcome][stat]) for stat in ("mean", "median")}
        for outcome in ("true_positive", "false_positive")
    }
    return {"metrics": metrics, "class_ap": class_ap, "confidence": confidence}


def comparison_text(comparison: dict[str, Any]) -> str:
    lines = ["BASELINE 01 VS SCALE JITTER", "", "Metric | Baseline | Experiment 05 | Delta"]
    lines += [f"{name} | {value['baseline']:.6f} | {value['candidate']:.6f} | {value['delta']:+.6f}" for name, value in comparison["metrics"].items()]
    lines += ["", "Class AP:"]
    lines += [f"{name} | {value['baseline']:.6f} | {value['candidate']:.6f} | {value['delta']:+.6f}" for name, value in comparison["class_ap"].items()]
    lines += ["", "Confidence:"]
    for outcome, stats in comparison["confidence"].items():
        lines += [f"{outcome}_{name} | {value['baseline']:.6f} | {value['candidate']:.6f} | {value['delta']:+.6f}" for name, value in stats.items()]
    return "\n".join(lines) + "\n"


def plot_reports(output: Path, comparison: dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = ("small", "medium", "large")
    x = range(len(groups))
    width = 0.36
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar([v - width / 2 for v in x], [comparison["metrics"][f"{g}_recall"]["baseline"] for g in groups], width, label="Baseline 01")
    axis.bar([v + width / 2 for v in x], [comparison["metrics"][f"{g}_recall"]["candidate"] for g in groups], width, label="Scale jitter")
    axis.set(xticks=list(x), xticklabels=groups, ylabel="Recall @ score/IoU 0.50", title="Recall by object size", ylim=(0, 1))
    axis.grid(axis="y", alpha=0.25); axis.legend(); figure.tight_layout()
    figure.savefig(output / "size_recall_comparison.png", dpi=150); plt.close(figure)

    names = list(comparison["class_ap"])
    deltas = [comparison["class_ap"][name]["delta"] for name in names]
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.bar(names, deltas, color=["#2e8b57" if value >= 0 else "#b22222" for value in deltas])
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(ylabel="AP delta vs Baseline 01", title="Class AP change from scale jitter")
    axis.tick_params(axis="x", rotation=60); axis.grid(axis="y", alpha=0.25); figure.tight_layout()
    figure.savefig(output / "class_ap_delta_vs_baseline.png", dpi=150); plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--resolution-dir", type=Path, required=True)
    parser.add_argument("--class-aware-dir", type=Path, required=True)
    parser.add_argument("--difficulty-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    args = parser.parse_args()
    roots = {name: getattr(args, name).resolve() for name in ("baseline_dir", "resolution_dir", "class_aware_dir", "difficulty_dir", "candidate_dir")}
    experiments = {name: artifacts(path) for name, path in roots.items()}
    baseline, candidate = experiments["baseline_dir"], experiments["candidate_dir"]
    if baseline["config"]["raw_train_valid_sha256_before"] != candidate["config"]["raw_train_valid_sha256_before"]:
        raise ValueError("Raw train/validation hashes differ")
    comparison = compare(baseline, candidate)
    payload = {
        "baseline_experiment": baseline["config"]["experiment_name"],
        "candidate_experiment": candidate["config"]["experiment_name"],
        "controlled_change": candidate["config"]["controlled_change"], **comparison,
    }
    write_json(roots["candidate_dir"] / "comparison_vs_baseline_01.json", payload)
    (roots["candidate_dir"] / "comparison_vs_baseline_01.txt").write_text(comparison_text(comparison), encoding="utf-8")
    plot_reports(roots["candidate_dir"], comparison)

    labels = {
        "baseline_dir": "Baseline 01", "resolution_dir": "Experiment 02 (640)",
        "class_aware_dir": "Experiment 03 (class-aware)",
        "difficulty_dir": "Experiment 04 (difficulty-aware)",
        "candidate_dir": "Experiment 05 (scale jitter)",
    }
    previous = ["COMPARISON WITH PREVIOUS EXPERIMENTS", "", "Experiment | mAP | mAP50 | mAP75 | mAR100 | small recall | 4+ recall | precision | recall"]
    for key, label in labels.items():
        item = experiments[key]
        previous.append(
            f"{label} | {item['summary']['best_validation_map']:.6f} | {item['summary']['best_validation_map_50']:.6f} | "
            f"{item['summary']['best_validation_map_75']:.6f} | {item['best']['validation_mar_100']:.6f} | "
            f"{item['size']['groups']['small']['recall']:.6f} | {item['crowd']['groups']['4+']['recall']:.6f} | "
            f"{item['error']['precision']:.6f} | {item['error']['recall']:.6f}"
        )
    previous += ["", "Assessment uses all listed metrics, not a single-metric rank.", "Baseline 01 remains the best general configuration: Experiment 05 improves targeted recall but its effectively flat mAP, lower mAP75/precision, and increased localization/background errors prevent it from being a clear replacement.", "No test metrics were used."]
    (roots["candidate_dir"] / "comparison_with_previous_experiments.txt").write_text("\n".join(previous) + "\n", encoding="utf-8")

    aug = load(roots["candidate_dir"] / "augmentation_analysis.json")
    m = comparison["metrics"]
    stable_references = all(abs(comparison["class_ap"][name]["delta"]) <= 0.02 for name in REFERENCE_CLASSES)
    beneficial = (
        m["map"]["delta"] > 0.005 and m["small_recall"]["delta"] >= 0
        and m["four_plus_object_recall"]["delta"] >= 0 and m["map_75"]["delta"] >= 0
        and m["precision_score_050"]["delta"] >= -0.01
        and m["localization_error_count"]["delta"] <= 0 and stable_references
    )
    effect = [
        "SCALE-JITTER EFFECT", "",
        f"1. Scale use: 0.8={aug['scale_assignments']['0.8']}, 1.0={aug['scale_assignments']['1.0']}, 1.2={aug['scale_assignments']['1.2']}.",
        f"2. Size bins small/medium/large: {aug['size_bins']['before']['small']}/{aug['size_bins']['before']['medium']}/{aug['size_bins']['before']['large']} -> {aug['size_bins']['after']['small']}/{aug['size_bins']['after']['medium']}/{aug['size_bins']['after']['large']}.",
        f"3. Boxes clipped: {aug['boxes_clipped']}.", f"4. Boxes removed: {aug['boxes_completely_removed']}.",
        f"5. Zero-object transformed samples: {aug['images_becoming_zero_object']}.",
        f"6. Small recall delta: {m['small_recall']['delta']:+.6f}.",
        f"7. Medium/large recall deltas: {m['medium_recall']['delta']:+.6f}/{m['large_recall']['delta']:+.6f}.",
        f"8. Four-plus-object recall delta: {m['four_plus_object_recall']['delta']:+.6f}.",
        f"9. Overall mAP delta: {m['map']['delta']:+.6f}.", f"10. mAP75 delta: {m['map_75']['delta']:+.6f}.",
        f"11. Localization error delta: {m['localization_error_count']['delta']:+.0f}; mean matched IoU delta {m['mean_matched_iou']['delta']:+.6f}.",
        "12. Strong reference AP deltas: " + ", ".join(f"{name} {comparison['class_ap'][name]['delta']:+.6f}" for name in REFERENCE_CLASSES) + f"; stable within +/-0.02: {stable_references}.",
        f"13. Overall beneficial: {beneficial}. Targeted recall improved, but aggregate mAP was effectively flat while mAP75, precision, localization errors, background FPs, and strong-class stability worsened; Baseline 01 remains the better general configuration.", "", "No test metrics were used.",
    ]
    (roots["candidate_dir"] / "scale_jitter_effect.txt").write_text("\n".join(effect) + "\n", encoding="utf-8")
    print(f"comparison={roots['candidate_dir'] / 'comparison_vs_baseline_01.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
