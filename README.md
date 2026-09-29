# Animal Detection

A standalone portfolio project for multi-class animal object detection using PyTorch.

## Status

Repository setup and dataset auditing are in progress. Model training and final model selection have not started, and no performance claims are made.

## Environment setup

The project uses a dedicated Conda environment named `animal-detection` with Python 3.12. After environment creation, activate it with:

```powershell
conda activate animal-detection
```

`requirements.txt` records the exact verified package versions. The CUDA 13.0 PyTorch wheels use PyTorch's package index, so install with:

```powershell
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

The environment sets `PYTHONNOUSERSITE=1` to prevent unrelated user-level Python packages from leaking into the project.

## Repository structure

```text
data/manifests/                 Generated lightweight manifests
data/processed/                 Generated processed data (ignored by Git)
reports/annotation_samples/     Generated annotation previews
scripts/                        Dataset inspection and visualization tools
src/animal_detection/           Project Python package
tests/                          Tests
Multi-Class Animal Detection.v1-yolov8/  Raw dataset (ignored by Git)
```

## Dataset policy

The raw dataset is expected at `Multi-Class Animal Detection.v1-yolov8/`. It is immutable: scripts only read its images, labels, and `data.yaml`. All generated reports and visualizations are written outside that directory.

## Audit the dataset

```powershell
python scripts/inspect_dataset.py --dataset-root "G:\animal-detection\Multi-Class Animal Detection.v1-yolov8"
```

## Visualize annotations

```powershell
python scripts/visualize_annotations.py --dataset-root "G:\animal-detection\Multi-Class Animal Detection.v1-yolov8" --split train --num-samples 8 --seed 42
```

## PyTorch Dataset Layer

`YoloDetectionDataset` reads the immutable YOLO annotations and converts normalized
`class_id cx cy width height` boxes in memory to pixel `x1 y1 x2 y2` coordinates.
Raw labels remain class IDs 0–19 at this stage; a future torchvision model adapter
must explicitly handle any background-label offset instead of changing the dataset
silently.

The dataset supports zero-object samples and detection transforms with an
`image, target` call signature. `detection_collate_fn` keeps images and targets as
sequences so samples with different object counts can share a DataLoader batch.

The cow and wolf samples identified by the audit have empty label files despite
visibly containing animals. They intentionally remain empty targets until a
separate annotation-override mechanism is designed; the raw labels are not edited.

Run a short DataLoader check with:

```powershell
python scripts/check_dataloader.py --dataset-root "G:\animal-detection\Multi-Class Animal Detection.v1-yolov8" --split train --batch-size 4 --num-workers 0
```

Install and run the development tests with:

```powershell
python -m pip install -r requirements-dev.txt --extra-index-url https://download.pytorch.org/whl/cu130
pytest -q
```

## Model-Facing Targets and Detection Transforms

Raw dataset labels remain unchanged at 0–19. Torchvision detector targets reserve
label 0 for background and use foreground labels 1–20; this shift is performed
explicitly by `adapt_target_for_torchvision`, never by `YoloDetectionDataset`.
The adapter also adds `area` and `iscrowd` without mutating the dataset target.

Paired transforms operate jointly on each image and target. Horizontal flips mirror
pixel `xyxy` coordinates, while resize scales both coordinate axes and updates the
current size metadata. The initial train preset uses 640×640 resize, a 50% horizontal
flip, and tensor conversion. The evaluation preset uses only 640×640 resize and
tensor conversion. No ImageNet normalization is applied because torchvision
detection models generally normalize inputs internally.

Verify the transform geometry numerically:

```powershell
python scripts/check_detection_transforms.py --dataset-root "G:\animal-detection\Multi-Class Animal Detection.v1-yolov8"
```

Generate transformed samples for visual review before training:

```powershell
python scripts/visualize_transformed_samples.py --dataset-root "G:\animal-detection\Multi-Class Animal Detection.v1-yolov8" --split train --num-samples 8 --seed 42
```

## Faster R-CNN Integration

The current detector integration uses torchvision Faster R-CNN with a ResNet50-FPN
backbone. COCO detector weights initialize the pretrained network, after which the
final predictor is replaced with 21 outputs: background label 0 plus 20 animal
foreground classes. Dataset labels 0–19 are explicitly adapted to model labels
1–20 before the forward pass.

This milestone performs only one smoke-test optimization step and one inference
pass. It is not a trainer, no checkpoint is saved, and no accuracy or performance
results exist yet. The default smoke resolution is 512×512 to provide a conservative
FP32 baseline for the 4 GB RTX 3050; this is not a final training configuration.

```powershell
python scripts/check_faster_rcnn.py --dataset-root "G:\animal-detection\Multi-Class Animal Detection.v1-yolov8" --batch-size 1 --num-workers 0
```

## Training and Validation

The reusable engine optimizes only on the `train` split and computes COCO-style
detection metrics only on the `valid` split. The `test` split remains untouched and
is not used for model selection. Validation tracks mAP@0.50:0.95, mAP@0.50,
mAP@0.75, mean recall, and class-wise AP where available through TorchMetrics.

The best-checkpoint criterion is validation mAP@0.50:0.95. Individual Faster R-CNN
losses, total loss, learning rate, runtime, and peak CUDA memory are written to
`reports/training_history.json`. The best checkpoint is written to
`checkpoints/faster_rcnn_best.pt`, which remains outside Git tracking. The initial
run is only a short pipeline-validation experiment, not final training or model
selection.

```powershell
python scripts/train_faster_rcnn.py --dataset-root "G:\animal-detection\Multi-Class Animal Detection.v1-yolov8" --epochs 1 --batch-size 1 --num-workers 0
```

For an explicitly bounded pipeline check, use `--max-train-batches` and
`--max-val-batches`; these limits are recorded in the experiment configuration.
