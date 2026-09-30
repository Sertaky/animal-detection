# Animal Detection

A standalone portfolio project for multi-class animal object detection using PyTorch.

## Status

Repository setup, dataset auditing, and the first full baseline experiment are complete. No test-set performance claim is made.

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
visibly containing animals. The raw parser continues to represent empty targets
faithfully. Full training and validation exclude these two known bad annotations
through `data/manifests/excluded_samples.json`; raw labels are never edited.

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
detection metrics only on the `valid` split. The training script does not construct
the `test` split, and it is not used for model selection. Validation tracks mAP@0.50:0.95, mAP@0.50,
mAP@0.75, mean recall, and class-wise AP where available through TorchMetrics.

The earlier bounded pipeline check is recorded in `reports/training_history.json`
with its checkpoint at `checkpoints/faster_rcnn_best.pt`.

```powershell
python scripts/train_faster_rcnn.py --dataset-root "G:\animal-detection\Multi-Class Animal Detection.v1-yolov8" --epochs 1 --batch-size 1 --num-workers 0
```

## First baseline experiment

The baseline uses Faster R-CNN ResNet50-FPN initialized with COCO V1 detector
weights and a new 21-class predictor (20 animals plus background). All parameters
are fine-tuned in FP32 with 512×512 inputs, batch size 1, and 10 epochs.
Training uses resize, 50% random horizontal flip, and tensor conversion;
validation uses resize and tensor conversion only. SGD uses learning rate 0.0025,
momentum 0.9, and weight decay 0.0005. StepLR decays the learning rate by 0.1
after epoch 7. The history records the rate used during each epoch.

The exclusion manifest removes one visibly annotated-missing cow image from the
1,400 training images and one wolf image from the 300 validation images, leaving
1,399 usable training images and 299 usable validation images. Selection uses
highest validation mAP@0.50:0.95, then mAP@0.50 for an exact tie. The held-out
test split is not part of training or selection.

Baseline reports are under `reports/experiments/faster_rcnn_baseline_01/`.
Ignored checkpoints are under `checkpoints/faster_rcnn_baseline_01/`.

### FIRST BASELINE VALIDATION RESULTS

After 10 full epochs, the best checkpoint was selected at epoch 9 by
validation mAP@0.50:0.95 = 0.575485. At that epoch, validation mAP@0.50
was 0.825687 and mAP@0.75 was 0.653031. These are validation results
for the first baseline, not test performance or a final model claim.
The complete epoch history, class-wise AP, configuration, summary, and
two unsmoothed plots are in the baseline report directory above.
