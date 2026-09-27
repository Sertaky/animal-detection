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
