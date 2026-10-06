# Animal Detection

A standalone portfolio project for multi-class animal object detection using PyTorch.

## Status

Repository setup, dataset auditing, the first full baseline, and a controlled
640x640 resolution comparison are complete. No test-set performance claim is made.

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
pytest -q -m "not heldout"
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

The earlier integration check performed one smoke-test optimization step and one
inference pass without saving a checkpoint. The default smoke resolution was
512×512 to provide a conservative
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

## Baseline Validation Error Analysis

The epoch-9 best checkpoint was analyzed once on all 299 usable validation
images, with no random augmentation or weight updates. The primary diagnostic
matching uses score threshold 0.50 and class-aware IoU threshold 0.50.
At that operating point, 412 annotated objects yield 313 true positives,
103 false positives, and 99 false negatives. These counts are diagnostic
one-to-one matches, not COCO AP.

Class confusion is the largest false-positive subtype (57). Small objects
have 0.382 recall versus 0.892 for large objects; scenes with four or more
objects have 0.433 recall versus 0.871 for one-object scenes. Panda has
the lowest baseline AP and 16/36 objects matched at this operating point.
The analysis does not use test results.

See [the concise summary](reports/experiments/faster_rcnn_baseline_01/error_analysis/summary.txt),
[structured diagnostics](reports/experiments/faster_rcnn_baseline_01/error_analysis/summary.json),
and [visualization index](reports/experiments/faster_rcnn_baseline_01/error_analysis/visualizations.json).
The cached validation predictions and detailed per-class, threshold,
localization, confidence, size, crowdedness, and confusion reports are in the
same `error_analysis/` directory.

## Experiment 02: 640x640 resolution comparison

`faster_rcnn_resolution_640_01` changes only the train and validation resize
from 512x512 to 640x640. Architecture, COCO V1 initialization, 21 model
classes, all-trainable parameters, batch size 1, 10-epoch schedule, SGD and
StepLR settings, seed 42, transforms other than size, exclusion manifest,
usable splits, and model-selection rule match Baseline 01. A GPU preflight
confirmed a 3x640x640 tensor, finite training losses, no OOM, and the same
41,396,536 trainable parameters before the full run.

The best 640 checkpoint was epoch 9, with validation mAP = 0.560321,
mAP@0.50 = 0.809025, mAP@0.75 = 0.630468, and mAR@100 = 0.684877.
Relative to the 512 baseline, these changed by -0.015163, -0.016661,
-0.022563, and -0.002567 respectively. Mean epoch runtime increased from
412.301 to 468.981 seconds (+13.7%), while peak allocated GPU memory increased
from 737.231 to 934.290 MiB (+26.7%).

At the fixed diagnostic score/IoU thresholds of 0.50/0.50, 640 improved
small-object recall from 0.382 to 0.447 and 4+-object-scene recall from 0.433
to 0.493. Overall diagnostic recall rose slightly from 0.760 to 0.767, but
precision fell from 0.752 to 0.740, false positives rose from 103 to 111, and
class-confusion events rose from 57 to 71. Tiger, Buffalo, and Cow had the
largest class-AP gains; Wolf, Monkeys, and Rat had the largest declines.

Conclusion: 640x640 is not an overall validation improvement for this fixed
10-epoch setup. Its targeted small/crowded recall gains do not offset lower
COCO-style AP, more diagnostic false positives, and higher runtime/memory cost.
This is a validation-only experiment; no test-set claim is made.

See [the structured comparison](reports/experiments/faster_rcnn_resolution_640_01/comparison_vs_baseline_01.json),
[the concise comparison](reports/experiments/faster_rcnn_resolution_640_01/comparison_vs_baseline_01.txt),
and the plots in `reports/experiments/faster_rcnn_resolution_640_01/`.

## Class-Focused Data Audit

Before designing a third experiment, a train/validation-only data audit compared
the weak or confused Panda, Monkeys, Gorilla, Goat, and Camel classes with Dog,
Wolf, Rhino, and Lion references; Deer was included for the Goat/Deer confusion
context. The audit measures box geometry, fixed small/medium/large bins,
boundary contact, crowdedness, and train/validation shifts, and includes manual
review sheets without model predictions.

The clearest issue is Panda validation difficulty and class consistency. Panda
validation has 50.0% small objects and median normalized area 0.094 versus 0.535
in train; 47.2% of its validation objects occur in images with four or more
objects. Reviewed Panda samples also mix giant pandas and red pandas under one
label. Goat and Camel have similar scale/crowding distributions, while Deer is
larger and less crowded. Monkeys contains visually diverse primates, making
some Gorilla confusion plausible, but reviewed examples did not establish
systematic wrong labeling. Weak classes are not systematically more
boundary-truncated than the stronger references.

See [the audit summary](reports/class_data_audit/summary.txt),
[structured class statistics](reports/class_data_audit/class_statistics.json),
[the review-only suspicious sample queue](reports/class_data_audit/suspicious_samples.json),
and the visual indexes under `reports/class_data_audit/`. No raw files, labels,
splits, or exclusions were changed, and no test data or training was used.

## Experiment 03: Class-Aware Sampling

Experiment 03 tests one controlled change from Baseline 01: shuffled one-pass
training is replaced by a seeded `WeightedRandomSampler`. An image receives
weight 2.0 when it contains Panda, Monkeys, Goat, or Camel and weight 1.0
otherwise. Presence uses a maximum rule, so multiple weak classes do not stack.
Sampling uses replacement for exactly 1,399 draws per epoch. Architecture,
COCO initialization, 512x512 transforms, augmentation, optimizer, scheduler,
seed, epoch count, exclusions, and validation/model-selection policy remain
unchanged.

The fixed-seed simulation sampled 873 unique images, produced 526 repeated
draws, and drew any single image at most eight times. Weak-class-containing
images represented 31.88% of draws. Exposure was 1.457x for Panda, 1.729x for
Monkeys, 1.471x for Goat, and 1.714x for Camel.

The best checkpoint was epoch 10: validation mAP=0.576391, mAP@0.50=0.830832,
mAP@0.75=0.644557, and mAR@100=0.688296. Versus Baseline 01, overall mAP
changed only +0.000907. Camel AP improved +0.126750, Goat +0.041036, Panda
+0.014117, Gorilla +0.000702, and Monkeys -0.000420. All four reference
classes declined: Dog -0.027599, Wolf -0.082039, Rhino -0.049568, and Lion
-0.035365. Diagnostic recall improved, and the selected Monkeys/Gorilla and
Goat/Camel/Deer confusion pairs decreased, but precision, mAP@0.75, mean
matched IoU, total class-confusion errors, and localization-error count worsened.

Conclusion: the 2x class-aware policy produced useful Camel/Goat gains but is
not a clear overall improvement because aggregate AP was effectively flat and
strong-class degradation was consistent. No test metrics were used. See the
[sampling-effect summary](reports/experiments/faster_rcnn_class_aware_sampling_01/sampling_effect_summary.txt),
[weak-class comparison](reports/experiments/faster_rcnn_class_aware_sampling_01/weak_class_comparison.json),
and [overall comparison](reports/experiments/faster_rcnn_class_aware_sampling_01/comparison_vs_baseline_01.json).
