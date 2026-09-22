"""Train, compare, and ensemble vegetable-leaf disease classifiers.

Expected input: ImageFolder-compatible class directories in train/val/test
splits (or a flat ImageFolder dataset with --auto-split).  The script is
intentionally self-contained so it can be run directly from VS Code.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from torchvision.models import (DenseNet121_Weights, EfficientNet_B0_Weights,
                                ResNet50_Weights)


MODEL_NAMES = ("resnet50", "efficientnet_b0", "densenet121")
@dataclass
class DataSplits:
    train: Dataset
    val: Dataset
    test: Dataset
    class_names: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vegetable leaf disease ensemble experiment")
    parser.add_argument("--data-dir", type=Path, required=True, help="Dataset root directory")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/vegetable_ensemble"))
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--freeze-epochs", type=int, default=2,
                        help="Train only new classification head for these initial epochs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--auto-split", action="store_true",
                        help="Split a flat class-folder dataset 70/15/15 (train/val/test)")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Skip training and evaluate checkpoints from --output-dir")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Do not download/use ImageNet initialization")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def image_transforms(size: int) -> tuple[Callable, Callable]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(), normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(int(size * 1.14)), transforms.CenterCrop(size),
        transforms.ToTensor(), normalize,
    ])
    return train_transform, eval_transform


class TransformSubset(Dataset):
    """Apply a transform after selecting examples from one ImageFolder."""

    def __init__(self, dataset: datasets.ImageFolder, indices: list[int], transform: Callable):
        self.dataset, self.indices, self.transform = dataset, indices, transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        path, label = self.dataset.samples[self.indices[index]]
        image = self.dataset.loader(path)
        return self.transform(image), label


def validate_split_classes(train: datasets.ImageFolder, val: datasets.ImageFolder,
                           test: datasets.ImageFolder) -> None:
    if train.classes != val.classes or train.classes != test.classes:
        raise ValueError("Class folders must be identical across train, val, and test splits.")
    if len(train.classes) < 2:
        raise ValueError("At least two disease classes are required.")


def build_datasets(data_dir: Path, size: int, auto_split: bool, seed: int) -> DataSplits:
    train_tf, eval_tf = image_transforms(size)
    if auto_split:
        base = datasets.ImageFolder(data_dir)
        labels = np.array(base.targets)
        indices = np.arange(len(base))
        train_idx, remainder_idx = train_test_split(
            indices, test_size=0.30, random_state=seed, stratify=labels
        )
        val_idx, test_idx = train_test_split(
            remainder_idx, test_size=0.50, random_state=seed,
            stratify=labels[remainder_idx],
        )
        return DataSplits(TransformSubset(base, train_idx.tolist(), train_tf),
                          TransformSubset(base, val_idx.tolist(), eval_tf),
                          TransformSubset(base, test_idx.tolist(), eval_tf), base.classes)

    val_dir = data_dir / "val" if (data_dir / "val").is_dir() else data_dir / "valid"
    required = {"train": data_dir / "train", "val/valid": val_dir, "test": data_dir / "test"}
    absent = [name for name, path in required.items() if not path.is_dir()]
    if absent:
        raise FileNotFoundError(f"Missing split folder(s): {', '.join(absent)}. Use --auto-split for a flat dataset.")
    raw_train, raw_val, raw_test = (datasets.ImageFolder(required[name]) for name in ("train", "val/valid", "test"))
    validate_split_classes(raw_train, raw_val, raw_test)
    return DataSplits(datasets.ImageFolder(required["train"], transform=train_tf),
                      datasets.ImageFolder(required["val/valid"], transform=eval_tf),
                      datasets.ImageFolder(required["test"], transform=eval_tf), raw_train.classes)


def make_loaders(splits: DataSplits, args: argparse.Namespace, device: torch.device) -> dict[str, DataLoader]:
    options = {"batch_size": args.batch_size, "num_workers": args.num_workers,
               "pin_memory": device.type == "cuda"}
    return {"train": DataLoader(splits.train, shuffle=True, **options),
            "val": DataLoader(splits.val, shuffle=False, **options),
            "test": DataLoader(splits.test, shuffle=False, **options)}


def build_model(name: str, num_classes: int, pretrained: bool) -> tuple[nn.Module, nn.Module]:
    if name == "resnet50":
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        head = model.fc
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        head = model.classifier[1]
    elif name == "densenet121":
        model = models.densenet121(weights=DenseNet121_Weights.DEFAULT if pretrained else None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        head = model.classifier
    else:
        raise ValueError(f"Unknown model: {name}")
    return model, head


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    probabilities, targets = [], []
    start = time.perf_counter()
    for images, labels in loader:
        outputs = model(images.to(device, non_blocking=True))
        probabilities.append(torch.softmax(outputs, dim=1).cpu().numpy())
        targets.append(labels.numpy())
    elapsed = time.perf_counter() - start
    return np.vstack(probabilities), np.concatenate(targets), elapsed


def train_one_model(name: str, model: nn.Module, head: nn.Module, loaders: dict[str, DataLoader],
                    args: argparse.Namespace, device: torch.device, checkpoint: Path) -> tuple[nn.Module, pd.DataFrame, float]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history, best_state, best_f1, started = [], None, -1.0, time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1:
            for parameter in model.parameters():
                parameter.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate * 0.3, weight_decay=args.weight_decay)
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for images, labels in loaders["train"]:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
        val_prob, val_true, _ = predict(model, loaders["val"], device)
        val_pred = val_prob.argmax(axis=1)
        val_scores = metric_values(val_true, val_pred)
        row = {"epoch": epoch, "train_loss": loss_sum / total, "train_accuracy": correct / total,
               "val_accuracy": val_scores["accuracy"], "val_macro_f1": val_scores["macro_f1"]}
        history.append(row)
        print(f"{name} | epoch {epoch:02d}/{args.epochs} | loss {row['train_loss']:.4f} | "
              f"val acc {row['val_accuracy']:.4f} | val macro F1 {row['val_macro_f1']:.4f}")
        if row["val_macro_f1"] > best_f1:
            best_f1, best_state = row["val_macro_f1"], copy.deepcopy(model.state_dict())
            torch.save({"model_name": name, "model_state": best_state}, checkpoint)
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), time.perf_counter() - started


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    macro = report["macro avg"]
    weighted = report["weighted avg"]
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "macro_precision": float(macro["precision"]),
            "macro_recall": float(macro["recall"]), "macro_f1": float(macro["f1-score"]),
            "weighted_precision": float(weighted["precision"]), "weighted_recall": float(weighted["recall"]),
            "weighted_f1": float(weighted["f1-score"])}


def save_evaluation(name: str, y_true: np.ndarray, probabilities: np.ndarray, class_names: list[str],
                    output_dir: Path, elapsed: float, train_seconds: float, params: int,
                    checkpoint: Optional[Path]) -> dict[str, float | str]:
    predictions = probabilities.argmax(axis=1)
    metrics = metric_values(y_true, predictions)
    report = pd.DataFrame(classification_report(y_true, predictions, target_names=class_names,
                                                output_dict=True, zero_division=0)).transpose()
    report.to_csv(output_dir / "classification_reports" / f"{name}.csv")
    raw_matrix = confusion_matrix(y_true, predictions)
    pd.DataFrame(raw_matrix, index=class_names, columns=class_names).to_csv(
        output_dir / "confusion_matrices" / f"{name}_counts.csv"
    )
    matrix = confusion_matrix(y_true, predictions, normalize="true")
    plt.figure(figsize=(max(8, len(class_names) * 0.65), max(6, len(class_names) * 0.55)))
    sns.heatmap(matrix, cmap="Blues", vmin=0, vmax=1, xticklabels=class_names, yticklabels=class_names,
                annot=len(class_names) <= 12, fmt=".2f")
    plt.xlabel("Predicted label"); plt.ylabel("True label"); plt.title(f"{name}: normalized test confusion matrix")
    plt.tight_layout(); plt.savefig(output_dir / "confusion_matrices" / f"{name}.png", dpi=180); plt.close()
    checkpoint_mb = checkpoint.stat().st_size / 1024**2 if checkpoint and checkpoint.exists() else 0.0
    metrics.update({"model": name, "parameters": params, "checkpoint_mb": checkpoint_mb,
                    "training_seconds": train_seconds, "test_inference_seconds": elapsed,
                    "test_ms_per_image": 1000 * elapsed / len(y_true)})
    return metrics


def save_comparison_plots(results: pd.DataFrame, output_dir: Path) -> None:
    """Create the accuracy/F1 and computational-cost comparison plots in Python."""
    labels = results["model"].str.replace("_", " ").str.title()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    results.plot(x="model", y=["accuracy", "macro_precision", "macro_recall", "macro_f1"],
                 kind="bar", ax=axes[0], ylim=(0, 1), rot=30)
    axes[0].set_xlabel("Model"); axes[0].set_ylabel("Score")
    axes[0].set_title("Test-set classification metrics")
    axes[0].legend(title="Metric", loc="lower right")
    axes[1].bar(labels, results["test_ms_per_image"], color="#4c78a8")
    axes[1].set_xlabel("Model"); axes[1].set_ylabel("Milliseconds per test image")
    axes[1].set_title("Inference computational cost")
    axes[1].tick_params(axis="x", rotation=30)
    for bar, value in zip(axes[1].patches, results["test_ms_per_image"]):
        axes[1].annotate(f"{value:.1f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                         ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "plots" / "model_comparison.png", dpi=180)
    plt.close(fig)


def write_experiment_report(results: pd.DataFrame, args: argparse.Namespace,
                            best_individual: dict, best_ensemble: dict, improvement: float) -> None:
    """Write a teacher-ready conclusion so no manual calculations are needed."""
    conclusion = "improved" if improvement > 0 else "did not improve"
    lines = [
        "VEGETABLE LEAF DISEASE ENSEMBLE EXPERIMENT",
        "=" * 44,
        f"Dataset directory: {args.data_dir}",
        f"Architectures compared: {', '.join(args.models)}",
        "Ensemble methods: validation-macro-F1 weighted probability fusion and majority voting.",
        "Metrics are measured on the held-out test set; weighted fusion weights are selected on validation data.",
        "",
        "MODEL COMPARISON",
        results.to_csv(index=False),
        f"Best individual model: {best_individual['model']} (macro F1 = {best_individual['macro_f1']:.4f}).",
        f"Best ensemble model: {best_ensemble['model']} (macro F1 = {best_ensemble['macro_f1']:.4f}).",
        f"Macro-F1 change: {improvement:+.4f}.",
        f"Conclusion: Ensemble learning {conclusion} held-out disease-classification reliability in this run.",
        "Computational cost is represented by parameter count, checkpoint size, training seconds, and test milliseconds/image.",
    ]
    (args.output_dir / "experiment_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.freeze_epochs < 0:
        raise ValueError("epochs and batch-size must be positive; freeze-epochs cannot be negative.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Use --device cpu or --device auto.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for folder in ("checkpoints", "classification_reports", "confusion_matrices", "training_history", "plots"):
        (args.output_dir / folder).mkdir(exist_ok=True)
    splits = build_datasets(args.data_dir, args.image_size, args.auto_split, args.seed)
    loaders = make_loaders(splits, args, device)
    print(f"Device: {device}; classes: {len(splits.class_names)}; samples train/val/test: "
          f"{len(splits.train)}/{len(splits.val)}/{len(splits.test)}")

    val_probabilities, test_probabilities, val_truth, test_truth, rows = {}, {}, None, None, []
    for name in args.models:
        checkpoint = args.output_dir / "checkpoints" / f"{name}.pt"
        model, head = build_model(name, len(splits.class_names), pretrained=not args.no_pretrained)
        model = model.to(device)
        params = sum(p.numel() for p in model.parameters())
        if args.evaluate_only:
            if not checkpoint.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model_state"])
            train_seconds = 0.0
        else:
            for parameter in model.parameters(): parameter.requires_grad = False
            for parameter in head.parameters(): parameter.requires_grad = True
            model, history, train_seconds = train_one_model(name, model, head, loaders, args, device, checkpoint)
            history.to_csv(args.output_dir / "training_history" / f"{name}.csv", index=False)
        val_prob, current_val_truth, _ = predict(model, loaders["val"], device)
        test_prob, current_test_truth, elapsed = predict(model, loaders["test"], device)
        if val_truth is None:
            val_truth, test_truth = current_val_truth, current_test_truth
        val_probabilities[name], test_probabilities[name] = val_prob, test_prob
        rows.append(save_evaluation(name, test_truth, test_prob, splits.class_names, args.output_dir,
                                    elapsed, train_seconds, params, checkpoint))

    # Derive fusion weights only from validation macro F1 to preserve the test set for final evaluation.
    validation_f1 = {name: metric_values(val_truth, probs.argmax(1))["macro_f1"] for name, probs in val_probabilities.items()}
    raw_weights = np.array([max(validation_f1[name], 1e-8) for name in args.models])
    weights = raw_weights / raw_weights.sum()
    stacked_test = np.stack([test_probabilities[name] for name in args.models])
    weighted_prob = np.tensordot(weights, stacked_test, axes=(0, 0))
    majority_labels = np.stack([p.argmax(1) for p in stacked_test]).T
    majority_pred = np.array([np.bincount(v, minlength=len(splits.class_names)).argmax() for v in majority_labels])
    majority_prob = np.eye(len(splits.class_names))[majority_pred]
    ensemble_cost = sum(row["test_inference_seconds"] for row in rows)
    total_params = sum(int(row["parameters"]) for row in rows)
    rows.append(save_evaluation("weighted_probability_ensemble", test_truth, weighted_prob, splits.class_names,
                                args.output_dir, ensemble_cost, sum(float(r["training_seconds"]) for r in rows), total_params, None))
    rows.append(save_evaluation("majority_vote_ensemble", test_truth, majority_prob, splits.class_names,
                                args.output_dir, ensemble_cost, 0.0, total_params, None))
    results = pd.DataFrame(rows)
    results.to_csv(args.output_dir / "metrics_summary.csv", index=False)
    with open(args.output_dir / "metrics_summary.json", "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    with open(args.output_dir / "ensemble_details.json", "w", encoding="utf-8") as handle:
        json.dump({"validation_macro_f1": validation_f1, "weighted_probability_fusion_weights": dict(zip(args.models, weights.tolist()))}, handle, indent=2)
    best_individual = max(rows[:len(args.models)], key=lambda row: float(row["macro_f1"]))
    best_ensemble = max(rows[len(args.models):], key=lambda row: float(row["macro_f1"]))
    improvement = float(best_ensemble["macro_f1"]) - float(best_individual["macro_f1"])
    save_comparison_plots(results, args.output_dir)
    write_experiment_report(results, args, best_individual, best_ensemble, improvement)
    print("\nFinal result")
    print(results[["model", "accuracy", "macro_precision", "macro_recall", "macro_f1", "test_ms_per_image", "parameters"]].to_string(index=False))
    print(f"Best individual: {best_individual['model']} (macro F1={best_individual['macro_f1']:.4f})")
    print(f"Best ensemble: {best_ensemble['model']} (macro F1={best_ensemble['macro_f1']:.4f}); change={improvement:+.4f}")
    print("Conclusion: ensemble learning improved held-out reliability." if improvement > 0 else "Conclusion: ensemble learning did not improve held-out reliability in this run.")


if __name__ == "__main__":
    main()
