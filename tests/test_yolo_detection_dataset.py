from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from animal_detection.data import ToTensor, YoloDetectionDataset
from animal_detection.data.yolo_detection_dataset import (
    YoloAnnotationError,
    _parse_yolo_annotation_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "Multi-Class Animal Detection.v1-yolov8"
EMPTY_TRAIN_STEM = "cow-19-_jpg.rf.79ab2f17459bf2b455ba932f730a35d2"
MULTI_TRAIN_STEM = "rat-38-_jpg.rf.445d47cfa6520071db240711501f1cf4"


def index_for_stem(dataset: YoloDetectionDataset, stem: str) -> int:
    return next(index for index, path in enumerate(dataset.image_paths) if path.stem == stem)


def first_nonempty_index(dataset: YoloDetectionDataset) -> int:
    return next(
        index
        for index, label_path in enumerate(dataset.label_paths)
        if label_path.read_text(encoding="utf-8-sig").strip()
    )


@pytest.mark.parametrize(
    ("split", "expected_length"),
    [("train", 1400), ("valid", 300), ("test", 300)],
)
def test_dataset_lengths(split: str, expected_length: int) -> None:
    assert len(YoloDetectionDataset(DATASET_ROOT, split)) == expected_length


def test_normal_sample_shapes_dtypes_and_bounds() -> None:
    dataset = YoloDetectionDataset(DATASET_ROOT, "train")
    image, target = dataset[first_nonempty_index(dataset)]
    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    boxes = target["boxes"]
    labels = target["labels"]
    assert isinstance(boxes, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert boxes.ndim == 2 and boxes.shape[1] == 4
    assert labels.shape == (boxes.shape[0],)
    assert boxes.dtype == torch.float32
    assert labels.dtype == torch.int64
    assert torch.all(boxes[:, 2] > boxes[:, 0])
    assert torch.all(boxes[:, 3] > boxes[:, 1])
    width, height = image.size
    assert torch.all(boxes[:, 0] >= 0) and torch.all(boxes[:, 2] <= width)
    assert torch.all(boxes[:, 1] >= 0) and torch.all(boxes[:, 3] <= height)
    assert torch.all((labels >= 0) & (labels <= 19))


def test_every_annotation_is_valid_and_all_class_ids_are_observed() -> None:
    observed_class_ids: set[int] = set()
    for split in ("train", "valid", "test"):
        dataset = YoloDetectionDataset(DATASET_ROOT, split)
        for index in range(len(dataset)):
            image, target = dataset[index]
            boxes = target["boxes"]
            labels = target["labels"]
            assert isinstance(boxes, torch.Tensor)
            assert isinstance(labels, torch.Tensor)
            assert boxes.shape == (labels.numel(), 4)
            if boxes.numel():
                assert torch.all(boxes[:, 2] > boxes[:, 0])
                assert torch.all(boxes[:, 3] > boxes[:, 1])
                assert torch.all(boxes[:, 0] >= 0) and torch.all(boxes[:, 2] <= image.width)
                assert torch.all(boxes[:, 1] >= 0) and torch.all(boxes[:, 3] <= image.height)
                assert torch.all((labels >= 0) & (labels <= 19))
                observed_class_ids.update(int(label) for label in labels.tolist())
            image.close()
    assert observed_class_ids == set(range(20))


def test_zero_object_sample_has_correct_empty_shapes() -> None:
    dataset = YoloDetectionDataset(DATASET_ROOT, "train")
    image, target = dataset[index_for_stem(dataset, EMPTY_TRAIN_STEM)]
    assert image.size == (640, 640)
    assert target["boxes"].shape == (0, 4)
    assert target["boxes"].dtype == torch.float32
    assert target["labels"].shape == (0,)
    assert target["labels"].dtype == torch.int64


def test_known_multi_object_sample_has_many_objects() -> None:
    dataset = YoloDetectionDataset(DATASET_ROOT, "train")
    _, target = dataset[index_for_stem(dataset, MULTI_TRAIN_STEM)]
    assert target["boxes"].shape[0] > 10
    assert target["boxes"].shape == (target["labels"].shape[0], 4)
    assert target["boxes"].dtype == torch.float32
    assert target["labels"].dtype == torch.int64


def test_indexing_and_metadata_are_deterministic() -> None:
    dataset = YoloDetectionDataset(DATASET_ROOT, "train")
    index = first_nonempty_index(dataset)
    _, first = dataset[index]
    _, second = dataset[index]
    assert first["image_path"] == second["image_path"]
    assert torch.equal(first["image_id"], second["image_id"])
    assert torch.equal(first["original_size"], second["original_size"])
    assert torch.equal(first["boxes"], second["boxes"])
    assert torch.equal(first["labels"], second["labels"])


def test_tensor_transform_preserves_target() -> None:
    dataset = YoloDetectionDataset(DATASET_ROOT, "train", transforms=ToTensor())
    image, target = dataset[first_nonempty_index(dataset)]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 640, 640)
    assert image.dtype == torch.float32
    assert 0 <= float(image.min()) <= float(image.max()) <= 1
    assert target["boxes"].dtype == torch.float32


def test_unsupported_split_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported split"):
        YoloDetectionDataset(DATASET_ROOT, "validation")


def test_missing_label_is_rejected(tmp_path: Path) -> None:
    images_dir = tmp_path / "train" / "images"
    labels_dir = tmp_path / "train" / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    Image.new("RGB", (10, 10)).save(images_dir / "sample.jpg")
    with pytest.raises(FileNotFoundError, match="Missing 1 label file"):
        YoloDetectionDataset(tmp_path, "train")


@pytest.mark.parametrize(
    ("row", "reason"),
    [
        ("0 0.5 0.5 0.2", "expected 5 fields"),
        ("1.0 0.5 0.5 0.2 0.2", "class_id must be an integer"),
        ("0 x 0.5 0.2 0.2", "box coordinates must be numeric"),
        ("20 0.5 0.5 0.2 0.2", "class_id must be in"),
        ("0 0.5 0.5 0 0.2", "width and height must be"),
        ("0 0.1 0.5 0.5 0.2", "box extends outside image bounds"),
    ],
)
def test_malformed_rows_report_context(tmp_path: Path, row: str, reason: str) -> None:
    label_path = tmp_path / "broken.txt"
    label_path.write_text(row + "\n", encoding="utf-8")
    with pytest.raises(YoloAnnotationError) as captured:
        _parse_yolo_annotation_file(label_path, 100, 100)
    message = str(captured.value)
    assert str(label_path) in message
    assert ":1:" in message
    assert row in message
    assert reason in message
