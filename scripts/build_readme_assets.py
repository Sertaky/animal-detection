#!/usr/bin/env python3
"""Build README presentation assets from frozen final-evaluation artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "reports" / "final_evaluation" / "visualizations"
OUTPUT = PROJECT_ROOT / "reports" / "readme"


def font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = ("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("arial.ttf", "DejaVuSans.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def montage(
    destination: Path,
    title: str,
    panels: list[tuple[str, str]],
    *,
    columns: int,
    tile_width: int = 410,
    tile_height: int = 452,
) -> None:
    rows = (len(panels) + columns - 1) // columns
    margin, gap, header, caption_height = 28, 18, 74, 42
    canvas_width = margin * 2 + columns * tile_width + (columns - 1) * gap
    canvas_height = margin * 2 + header + rows * (tile_height + caption_height) + (rows - 1) * gap
    canvas = Image.new("RGB", (canvas_width, canvas_height), "#f6f8fa")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 20), title, fill="#172b4d", font=font(34, bold=True))
    for index, (filename, caption) in enumerate(panels):
        row, column = divmod(index, columns)
        x = margin + column * (tile_width + gap)
        y = margin + header + row * (tile_height + caption_height + gap)
        with Image.open(SOURCE / filename) as original:
            fitted = ImageOps.fit(
                original.convert("RGB"), (tile_width, tile_height),
                method=Image.Resampling.LANCZOS,
            )
        canvas.paste(fitted, (x, y))
        draw.rectangle((x, y, x + tile_width - 1, y + tile_height - 1), outline="#d0d7de", width=2)
        draw.text((x + 8, y + tile_height + 8), caption, fill="#24292f", font=font(21, bold=True))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, optimize=True)


def experiment_plots() -> None:
    labels = ["Baseline\n512", "Resolution\n640", "Class-aware\nsampling", "Difficulty-aware\nsampling", "Scale\njitter"]
    map_values = [0.575485, 0.560321, 0.576391, 0.564713, 0.576239]
    colors = ["#1558a6"] + ["#73a2d6"] * 4
    figure, axis = plt.subplots(figsize=(10, 5.2))
    bars = axis.bar(np.arange(len(labels)), map_values, color=colors, width=0.68)
    axis.set_xticks(np.arange(len(labels)), labels)
    axis.set_ylim(0.53, 0.59)
    axis.set_ylabel("Validation mAP@0.50:0.95")
    axis.set_title("Controlled Experiment Comparison")
    axis.grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, map_values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.001, f"{value:.4f}", ha="center", va="bottom", fontsize=10)
    axis.text(0, 0.533, "selected", ha="center", va="bottom", color="#1558a6", weight="bold")
    figure.tight_layout()
    figure.savefig(OUTPUT / "experiment_map_comparison.png", dpi=180)
    plt.close(figure)

    small = [0.381579, 0.447368, 0.460526, 0.381579, 0.460526]
    crowded = [0.432836, 0.492537, 0.447761, 0.388060, 0.447761]
    x = np.arange(len(labels))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10, 5.6))
    axis.bar(x - width / 2, small, width, label="Small-object recall", color="#ee7733")
    axis.bar(x + width / 2, crowded, width, label="4+ object recall", color="#009988")
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 0.58)
    axis.set_ylabel("Validation recall")
    axis.set_title("Difficulty Recall Trade-offs")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, ncol=2, loc="upper center")
    figure.tight_layout()
    figure.savefig(OUTPUT / "experiment_recall_tradeoff.png", dpi=180)
    plt.close(figure)


def benchmark_plot() -> None:
    labels = ["GPU model-only", "GPU end-to-end", "CPU model-only", "CPU end-to-end"]
    values = [92.339, 102.101, 767.436, 1011.148]
    colors = ["#1558a6", "#4c88c7", "#9aa4b2", "#687386"]
    figure, axis = plt.subplots(figsize=(9.5, 4.8))
    bars = axis.barh(np.arange(len(labels)), values, color=colors)
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.invert_yaxis()
    axis.set_xlabel("Mean latency per image (ms), batch size 1")
    axis.set_title("Inference Latency")
    axis.grid(axis="x", alpha=0.22)
    for bar, value in zip(bars, values):
        axis.text(value + 14, bar.get_y() + bar.get_height() / 2, f"{value:.1f} ms", va="center", fontsize=10)
    axis.set_xlim(0, 1140)
    figure.tight_layout()
    figure.savefig(OUTPUT / "inference_benchmark.png", dpi=180)
    plt.close(figure)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    montage(
        OUTPUT / "final_detection_examples.png",
        "Representative held-out test detections",
        [
            ("high_confidence_correct_detections_01_horse-66-_jpg.rf.6e1670a6be8aa01b03b410409_56e4c61b.jpg", "Horse — confident match"),
            ("high_confidence_correct_detections_02_gorilla-192-_jpeg.rf.b1c0caed27a472a436dc3_2f56205c.jpg", "Gorilla — confident match"),
            ("high_confidence_correct_detections_03_panda-59-_jpg.rf.8c8ac1f53672ead35b0fa820d_7ecaa330.jpg", "Panda — confident match"),
            ("crowded_scenes_02_Zebra-Valid-329-_jpeg.rf.9b082f5d2ec8aeace_19857de1.jpg", "Zebras — crowded scene"),
            ("crowded_scenes_03_dog-39-_jpg.rf.fd4055542a11f4fe19b9d2684c6_60f25115.jpg", "Dogs — multiple objects"),
            ("crowded_scenes_01_Monkey-176-_jpeg.rf.2a21dc12a512131efb3382_795b117c.jpg", "Monkeys — mixed outcomes"),
        ],
        columns=3,
    )
    montage(
        OUTPUT / "failure_examples.png",
        "Representative held-out error cases",
        [
            ("small_object_cases_03_dog-18-_jpg.rf.440afac81e003136ad98cd711b4_6e5793ec.jpg", "Small/crowded class confusion"),
            ("crowded_scenes_01_Monkey-176-_jpeg.rf.2a21dc12a512131efb3382_795b117c.jpg", "Crowded-scene misses"),
            ("localization_errors_03_horse-27-_jpg.rf.32720670fc107ae6637996840_1f02487e.jpg", "Localization errors"),
            ("class_confusions_02_rhino-93-_jpg.rf.1f12f0be939aa997fd3e4ad92_47b0aabb.jpg", "Rhino / Hippo confusion"),
        ],
        columns=2,
        tile_width=520,
        tile_height=575,
    )
    experiment_plots()
    benchmark_plot()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
