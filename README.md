# Vegetable Leaf Disease Ensemble

This project trains and compares three transfer-learning classifiers—ResNet50,
EfficientNet-B0, and DenseNet121—on a vegetable-leaf disease dataset stored on
your computer.  It reports accuracy, macro precision, recall, F1-score,
confusion matrices, classification reports, and practical computational cost.
It also evaluates a weighted-probability ensemble and a majority-voting
ensemble against each individual architecture.

## 1. Prepare the dataset

The preferred folder layout is:

```text
dataset/
  train/
    healthy/
    tomato_early_blight/
    ...
  val/                 # `valid/` is also accepted
    healthy/
    tomato_early_blight/
  test/
    healthy/
    tomato_early_blight/
```

Every class folder must use exactly the same class names in every split.  If
your dataset is instead one folder with class folders only, use
`--auto-split`; the program creates reproducible stratified train/validation/
test splits without copying the images.

## 2. Install dependencies

Use a virtual environment in VS Code, then install a PyTorch build appropriate
for your CPU/CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/).
Install the remaining dependencies with:

```bash
pip install -r requirements.txt
```

## 3. Train all individual models and the ensemble

```bash
python ensemble_leaf_disease.py \
  --data-dir "C:/path/to/Vegetable_Leaf_Dataset" \
  --output-dir runs/vegetable_ensemble \
  --epochs 15 --batch-size 32 --image-size 224
```

For a flat dataset (one class folder per disease):

```bash
python ensemble_leaf_disease.py --data-dir "C:/path/to/dataset" --auto-split
```

On a CUDA GPU, use `--device cuda`; on a CPU, use `--device cpu` and reduce
`--batch-size` if memory is limited.  To resume or only evaluate trained
models, retain the output directory and add `--evaluate-only`.

## Outputs

The selected output directory contains:

* `checkpoints/*.pt` — best validation checkpoint for every architecture.
* `metrics_summary.csv` and `metrics_summary.json` — individual and ensemble
  comparison including computational cost.
* `classification_reports/*.csv` — per-class precision, recall, and F1.
* `confusion_matrices/*.png` — normalized confusion-matrix figures.
* `confusion_matrices/*_counts.csv` — raw, numeric confusion matrices.
* `training_history/*.csv` — epoch-by-epoch loss and accuracy.
* `ensemble_details.json` — validation-derived probability-fusion weights.
* `plots/model_comparison.png` — Python-generated metrics and inference-cost comparison.
* `experiment_report.txt` — Python-generated conclusion stating whether the best
  ensemble improves the best individual model's held-out macro F1.

Weighted fusion assigns higher weight to models with stronger validation macro
F1.  This avoids selecting weights using the held-out test data.  The test set
is used once for the final comparison, which lets you state whether ensemble
learning improved the reliability of your particular disease classifier.
All preparation, model training, fusion, metric calculation, confusion-matrix
creation, cost comparison, plots, and conclusion generation are performed by
`ensemble_leaf_disease.py`; no manual spreadsheet calculation is required.

## Useful options

```bash
python ensemble_leaf_disease.py --help
```

Important options include `--models resnet50 efficientnet_b0 densenet121`,
`--freeze-epochs 2`, `--num-workers 4`, `--learning-rate 0.0003`, and
`--seed 42`.
