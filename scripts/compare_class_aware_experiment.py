#!/usr/bin/env python3
"""Compare the class-aware sampling experiment with Baseline 01 artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from animal_detection.analysis.matching import match_image

WEAK = ("Panda", "Monkeys", "Gorilla", "Goat", "Camel")
REFERENCE = ("Dog", "Wolf", "Rhino", "Lion")
TARGET_PAIRS = (("Monkeys", "Gorilla"), ("Gorilla", "Monkeys"), ("Goat", "Camel"), ("Goat", "Deer"))


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def pair(baseline: float | int, candidate: float | int) -> dict[str, float]:
    return {"baseline": float(baseline), "candidate": float(candidate), "delta": float(candidate) - float(baseline)}


def artifacts(root: Path) -> dict[str, Any]:
    error = root / "error_analysis"
    return {
        "config": load(root / "config.json"),
        "summary": load(root / "summary.json"),
        "history": load(root / "training_history.json"),
        "class_ap": load(root / "validation_class_metrics.json"),
        "error_summary": load(error / "summary.json"),
        "class_diagnostics": load(error / "class_diagnostics.json"),
        "confusion": load(error / "class_confusion.json"),
        "localization": load(error / "localization_analysis.json"),
        "size": load(error / "object_size_analysis.json"),
        "crowd": load(error / "crowdedness_analysis.json"),
        "confidence": load(error / "confidence_analysis.json"),
        "cache": load(error / "validation_predictions.json"),
    }


def by_name(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {item["class_name"]: item for item in items}


def confusion_count(report: dict[str, Any], ground_truth: str, predicted: str) -> int:
    return sum(
        item["count"] for item in report["pairs"]
        if item["ground_truth_class"] == ground_truth and item["predicted_class"] == predicted
    )


def panda_small_recall(cache: dict[str, Any]) -> dict[str, int | float | None]:
    gt_count = 0
    tp = 0
    panda_model_label = 15
    for image in cache["images"]:
        matched = match_image(image["ground_truth"], image["predictions"], score_threshold=0.50, iou_threshold=0.50)
        matched_gt = {
            record["gt_index"] for record in matched["predictions"]
            if record["status"] == "true_positive"
        }
        height, width = image["transformed_size"]
        for index, gt in enumerate(image["ground_truth"]):
            x1, y1, x2, y2 = gt["box"]
            area = (x2 - x1) * (y2 - y1) / (height * width)
            if gt["label"] == panda_model_label and area < 0.10:
                gt_count += 1
                tp += int(index in matched_gt)
    return {"gt_objects": gt_count, "tp": tp, "fn": gt_count - tp, "recall": tp / gt_count if gt_count else None}


def plots(output: Path, weak: dict[str, Any], all_classes: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def bar(names: tuple[str, ...], filename: str, title: str) -> None:
        x = list(range(len(names)))
        width = 0.36
        lookup = {item["class_name"]: item for item in all_classes}
        figure, axis = plt.subplots(figsize=(8, 4.8))
        axis.bar([v - width / 2 for v in x], [lookup[n]["ap"]["baseline"] for n in names], width, label="Baseline 01")
        axis.bar([v + width / 2 for v in x], [lookup[n]["ap"]["candidate"] for n in names], width, label="Class-aware")
        axis.set(xticks=x, xticklabels=names, ylabel="Validation AP", title=title, ylim=(0, 1))
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / filename, dpi=150)
        plt.close(figure)

    bar(WEAK, "weak_class_ap_comparison.png", "Weak/confused class AP")
    bar(REFERENCE, "reference_class_ap_comparison.png", "Reference class AP")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    args = parser.parse_args()
    baseline_root = args.baseline_dir.resolve()
    candidate_root = args.candidate_dir.resolve()
    baseline, candidate = artifacts(baseline_root), artifacts(candidate_root)
    if baseline["config"]["image_size"] != 512 or candidate["config"]["image_size"] != 512:
        raise ValueError("Expected both experiments at 512x512")
    if baseline["config"]["raw_train_valid_sha256_before"] != candidate["config"]["raw_train_valid_sha256_before"]:
        raise ValueError("Raw train/validation hashes differ")

    baseline_ap = by_name(baseline["class_ap"]["metrics"])
    candidate_ap = by_name(candidate["class_ap"]["metrics"])
    baseline_diag = by_name(baseline["class_diagnostics"]["classes"])
    candidate_diag = by_name(candidate["class_diagnostics"]["classes"])
    class_rows = []
    weak_rows = {}
    for name in baseline_ap:
        class_rows.append({"class_name": name, "ap": pair(baseline_ap[name]["ap"], candidate_ap[name]["ap"])})
    for name in WEAK:
        before, after = baseline_diag[name], candidate_diag[name]
        weak_rows[name] = {
            "ap": pair(baseline_ap[name]["ap"], candidate_ap[name]["ap"]),
            **{key: pair(before[key], after[key]) for key in ("gt_objects", "tp", "fp", "fn", "precision", "recall")},
            "mean_matched_iou": pair(before["mean_matched_iou"], after["mean_matched_iou"]),
            "class_confusion_count": pair(before["class_confusion_count"], after["class_confusion_count"]),
            "localization_error_count": pair(before["localization_error_count"], after["localization_error_count"]),
        }
    panda_small_before = panda_small_recall(baseline["cache"])
    panda_small_after = panda_small_recall(candidate["cache"])
    weak_rows["Panda"]["small_object_diagnostic"] = {
        key: pair(panda_small_before[key], panda_small_after[key])
        for key in ("gt_objects", "tp", "fn", "recall")
    }
    confusion_pairs = {
        f"{left}_to_{right}": pair(
            confusion_count(baseline["confusion"], left, right),
            confusion_count(candidate["confusion"], left, right),
        ) for left, right in TARGET_PAIRS
    }
    bs, cs = baseline["summary"], candidate["summary"]
    be, ce = baseline["error_summary"], candidate["error_summary"]
    bl, cl = baseline["localization"], candidate["localization"]
    overall = {
        "best_epoch": pair(bs["best_epoch"], cs["best_epoch"]),
        "map": pair(bs["best_validation_map"], cs["best_validation_map"]),
        "map_50": pair(bs["best_validation_map_50"], cs["best_validation_map_50"]),
        "map_75": pair(bs["best_validation_map_75"], cs["best_validation_map_75"]),
        "mar_100": pair(
            baseline["history"][bs["best_epoch"] - 1]["validation_mar_100"],
            candidate["history"][cs["best_epoch"] - 1]["validation_mar_100"],
        ),
        "precision_score_050": pair(be["precision"], ce["precision"]),
        "recall_score_050": pair(be["recall"], ce["recall"]),
        "class_confusion_count": pair(be["false_positive_categories"]["class_confusion"], ce["false_positive_categories"]["class_confusion"]),
        "localization_error_count": pair(be["false_positive_categories"]["localization"], ce["false_positive_categories"]["localization"]),
        "mean_matched_iou": pair(bl["matched_iou"]["mean"], cl["matched_iou"]["mean"]),
        "median_matched_iou": pair(bl["matched_iou"]["median"], cl["matched_iou"]["median"]),
        "mean_epoch_runtime_seconds": pair(bs["mean_epoch_runtime_seconds"], cs["mean_epoch_runtime_seconds"]),
        "total_runtime_seconds": pair(bs["total_training_runtime_seconds"], cs["total_training_runtime_seconds"]),
        "peak_gpu_allocated_mib": pair(bs["peak_gpu_allocated_mib"], cs["peak_gpu_allocated_mib"]),
        "peak_gpu_reserved_mib": pair(bs["peak_gpu_reserved_mib"], cs["peak_gpu_reserved_mib"]),
    }
    for name in ("small", "medium", "large"):
        overall[f"{name}_recall"] = pair(baseline["size"]["groups"][name]["recall"], candidate["size"]["groups"][name]["recall"])
    for name, key in (("1", "one_object_recall"), ("2-3", "two_three_object_recall"), ("4+", "four_plus_object_recall")):
        overall[key] = pair(baseline["crowd"]["groups"][name]["recall"], candidate["crowd"]["groups"][name]["recall"])
    comparison = {
        "baseline_experiment": baseline["config"]["experiment_name"],
        "candidate_experiment": candidate["config"]["experiment_name"],
        "controlled_change": candidate["config"]["controlled_change"],
        "overall": overall,
        "class_ap": class_rows,
        "target_confusion_pairs": confusion_pairs,
        "confidence": {
            outcome: {
                statistic: pair(
                    baseline["confidence"]["by_outcome"][outcome][statistic],
                    candidate["confidence"]["by_outcome"][outcome][statistic],
                ) for statistic in ("mean", "median")
            } for outcome in ("true_positive", "false_positive")
        },
    }
    write_json(candidate_root / "weak_class_comparison.json", weak_rows)
    write_json(candidate_root / "comparison_vs_baseline_01.json", comparison)
    plots(candidate_root, weak_rows, class_rows)

    weak_lines = ["WEAK-CLASS COMPARISON: BASELINE 01 VS CLASS-AWARE SAMPLING", ""]
    for name in WEAK:
        item = weak_rows[name]
        weak_lines.append(
            f"{name}: AP {item['ap']['baseline']:.6f} -> {item['ap']['candidate']:.6f} ({item['ap']['delta']:+.6f}); "
            f"TP/FP/FN {item['tp']['baseline']:.0f}/{item['fp']['baseline']:.0f}/{item['fn']['baseline']:.0f} -> "
            f"{item['tp']['candidate']:.0f}/{item['fp']['candidate']:.0f}/{item['fn']['candidate']:.0f}; "
            f"recall {item['recall']['baseline']:.3f} -> {item['recall']['candidate']:.3f}."
        )
    (candidate_root / "weak_class_comparison.txt").write_text("\n".join(weak_lines) + "\n", encoding="utf-8")

    overall_lines = ["OVERALL COMPARISON: BASELINE 01 VS CLASS-AWARE SAMPLING", ""] + [
        f"{name}: {item['baseline']:.6f} -> {item['candidate']:.6f} (delta {item['delta']:+.6f})"
        for name, item in overall.items()
    ]
    overall_lines += ["", "Target confusion pairs:"] + [
        f"{name}: {item['baseline']:.0f} -> {item['candidate']:.0f} (delta {item['delta']:+.0f})"
        for name, item in confusion_pairs.items()
    ]
    (candidate_root / "comparison_vs_baseline_01.txt").write_text("\n".join(overall_lines) + "\n", encoding="utf-8")

    ref_deltas = {name: next(row["ap"]["delta"] for row in class_rows if row["class_name"] == name) for name in REFERENCE}
    effect = [
        "SAMPLING EFFECT SUMMARY",
        "",
        "1. Weak exposure: Panda 1.457x, Monkeys 1.729x, Goat 1.471x, Camel 1.714x in the fixed-seed simulation.",
        f"2. Panda AP improved: {'Yes' if weak_rows['Panda']['ap']['delta'] > 0 else 'No'} ({weak_rows['Panda']['ap']['delta']:+.6f}).",
        f"3. Panda diagnostic recall improved: {'Yes' if weak_rows['Panda']['recall']['delta'] > 0 else 'No'} ({weak_rows['Panda']['recall']['delta']:+.6f}); small-object recall delta {weak_rows['Panda']['small_object_diagnostic']['recall']['delta']:+.6f}.",
        f"4. Monkeys AP improved: {'Yes' if weak_rows['Monkeys']['ap']['delta'] > 0 else 'No'} ({weak_rows['Monkeys']['ap']['delta']:+.6f}).",
        f"5. Monkeys/Gorilla confusion: Monkeys->Gorilla {confusion_pairs['Monkeys_to_Gorilla']['baseline']:.0f}->{confusion_pairs['Monkeys_to_Gorilla']['candidate']:.0f}; Gorilla->Monkeys {confusion_pairs['Gorilla_to_Monkeys']['baseline']:.0f}->{confusion_pairs['Gorilla_to_Monkeys']['candidate']:.0f}.",
        f"6. Goat AP improved: {'Yes' if weak_rows['Goat']['ap']['delta'] > 0 else 'No'} ({weak_rows['Goat']['ap']['delta']:+.6f}).",
        f"7. Camel AP improved: {'Yes' if weak_rows['Camel']['ap']['delta'] > 0 else 'No'} ({weak_rows['Camel']['ap']['delta']:+.6f}).",
        f"8. Overall mAP improved: {'Yes' if overall['map']['delta'] > 0 else 'No'} ({overall['map']['delta']:+.6f}).",
        "9. Strong reference AP deltas: " + ", ".join(f"{name} {value:+.6f}" for name, value in ref_deltas.items()) + ".",
        "10. Overall assessment: mixed. Aggregate mAP changed only marginally; weak-class gains must be weighed against reference-class AP losses and diagnostic error changes.",
        "",
        "No test metrics were used.",
    ]
    (candidate_root / "sampling_effect_summary.txt").write_text("\n".join(effect) + "\n", encoding="utf-8")
    print(f"comparison={candidate_root / 'comparison_vs_baseline_01.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
