"""Train a PyTorch image-only ResNet-18 baseline."""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_SOURCE_DIR = PROJECT_ROOT / "src" / "data"
if str(DATA_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_SOURCE_DIR))

from image_dataset import load_image_frame, make_torch_loader


DEFAULT_IMAGE_DIR = Path("E:/NNDL_Project_Data/Fakeddit/images")
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "final"
LABEL_COLUMNS = ("2_way_label", "3_way_label", "6_way_label")


def set_seed(seed: int) -> None:
    """Set reproducible random seeds for Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sample_frame(frame, sample_size: int | None, random_state: int):
    """Take a stratified subset for a quick experiment, when requested."""

    if sample_size is None or sample_size >= len(frame):
        return frame
    if sample_size < frame["label"].nunique():
        raise ValueError("--sample-size must be at least the number of label classes.")

    try:
        sampled, _ = train_test_split(
            frame,
            train_size=sample_size,
            stratify=frame["label"],
            random_state=random_state,
        )
    except ValueError:
        sampled = frame.sample(n=sample_size, random_state=random_state)
    return sampled.reset_index(drop=True)


def build_transform() -> transforms.Compose:
    """Return the standard preprocessing for pretrained ResNet-18."""

    weights = ResNet18_Weights.DEFAULT
    return weights.transforms()


def build_model(num_classes: int, freeze_backbone: bool) -> torch.nn.Module:
    """Build ImageNet-pretrained ResNet-18 with a new Fakeddit head."""

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    input_features = model.fc.in_features
    model.fc = torch.nn.Linear(input_features, num_classes)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.fc.parameters():
            parameter.requires_grad = True

    return model


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency weights for CrossEntropyLoss."""

    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    present = counts > 0
    weights = np.ones(num_classes, dtype=np.float64)
    weights[present] = len(labels) / (present.sum() * counts[present])
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(
    model: torch.nn.Module,
    loader,
    loss_function: torch.nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    freeze_backbone: bool = False,
) -> Dict[str, object]:
    """Run one training or evaluation epoch and collect predictions."""

    training = optimizer is not None
    model.train(training)
    if training and freeze_backbone:
        # Keep frozen BatchNorm layers in evaluation mode during head training.
        model.eval()
        model.fc.train()

    total_loss = 0.0
    total_rows = 0
    all_labels = []
    all_predictions = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            if batch is None:
                continue
            images, labels = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_function(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

            predictions = logits.argmax(dim=1)
            rows = labels.size(0)
            total_loss += loss.item() * rows
            total_rows += rows
            all_labels.extend(labels.detach().cpu().tolist())
            all_predictions.extend(predictions.detach().cpu().tolist())

    if total_rows == 0:
        raise RuntimeError("No readable images were available in a data loader.")

    return {
        "loss": total_loss / total_rows,
        "labels": np.asarray(all_labels, dtype=np.int64),
        "predictions": np.asarray(all_predictions, dtype=np.int64),
        "rows": total_rows,
    }


def classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
) -> Dict[str, object]:
    """Build classification metrics from collected labels and predictions."""

    label_values = list(range(num_classes))
    return {
        "rows": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=label_values,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=label_values
        ).tolist(),
        "labels_in_matrix": label_values,
    }


def train_and_evaluate(
    label_column: str,
    image_dir: Path,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    num_workers: int,
    sample_size: int | None,
    class_weight: str,
    freeze_backbone: bool,
    random_state: int,
    results_dir: Path,
    model_dir: Path,
    device_name: str,
) -> Dict[str, object]:
    """Train and evaluate the PyTorch ResNet-18 image baseline."""

    set_seed(random_state)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested with --device cuda, but no CUDA device is available. "
                "Use --device cpu or install a CUDA-enabled PyTorch build."
            )
        device = torch.device(device_name)
    transform = build_transform()

    train_frame = load_image_frame("train", label_column, image_dir)
    validation_frame = load_image_frame("validation", label_column, image_dir)
    test_frame = load_image_frame("test", label_column, image_dir)
    if train_frame.empty:
        raise RuntimeError(f"No usable training images found in {image_dir}.")
    if validation_frame.empty or test_frame.empty:
        raise RuntimeError("Validation and test splits must contain usable images.")

    full_train_frame = train_frame
    train_frame = sample_frame(train_frame, sample_size, random_state)
    num_classes = int(
        max(
            full_train_frame["label"].max(),
            validation_frame["label"].max(),
            test_frame["label"].max(),
        )
        + 1
    )

    train_loader = make_torch_loader(
        train_frame,
        transform,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        seed=random_state,
    )
    validation_loader = make_torch_loader(
        validation_frame, transform, batch_size, num_workers=num_workers
    )
    test_loader = make_torch_loader(
        test_frame, transform, batch_size, num_workers=num_workers
    )

    model = build_model(num_classes, freeze_backbone).to(device)
    weights = (
        compute_class_weights(train_frame["label"].to_numpy(), num_classes).to(device)
        if class_weight == "balanced"
        else None
    )
    loss_function = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
    )

    training_losses = []
    validation_losses = []
    last_validation = None
    for epoch in range(epochs):
        train_result = run_epoch(
            model,
            train_loader,
            loss_function,
            device,
            optimizer=optimizer,
            freeze_backbone=freeze_backbone,
        )
        validation_result = run_epoch(
            model, validation_loader, loss_function, device
        )
        training_losses.append(float(train_result["loss"]))
        validation_losses.append(float(validation_result["loss"]))
        last_validation = validation_result
        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"loss: {train_result['loss']:.4f} - "
            f"val_loss: {validation_result['loss']:.4f} - "
            f"val_accuracy: "
            f"{accuracy_score(validation_result['labels'], validation_result['predictions']):.4f}"
        )

    test_result = run_epoch(model, test_loader, loss_function, device)
    validation_metrics = classification_metrics(
        last_validation["labels"], last_validation["predictions"], num_classes
    )
    test_metrics = classification_metrics(
        test_result["labels"], test_result["predictions"], num_classes
    )

    majority_label = int(train_frame["label"].mode().iloc[0])
    metrics = {
        "task": "image_only_pytorch_resnet18_baseline",
        "architecture": "resnet18",
        "pretrained_weights": "imagenet",
        "label_column": label_column,
        "image_dir": str(image_dir),
        "device": str(device),
        "random_state": random_state,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "num_workers": num_workers,
        "freeze_backbone": freeze_backbone,
        "class_weight": class_weight,
        "available_rows": {
            "train": int(len(full_train_frame)),
            "validation": int(len(validation_frame)),
            "test": int(len(test_frame)),
        },
        "train_sample_size": int(len(train_frame)),
        "training_loss": training_losses,
        "validation_loss": validation_losses,
        "majority_baseline": {
            "label": majority_label,
            "validation_accuracy": float(
                (validation_frame["label"] == majority_label).mean()
            ),
            "test_accuracy": float((test_frame["label"] == majority_label).mean()),
        },
        "validation": validation_metrics,
        "test": test_metrics,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    suffix = label_column.replace("_", "")
    result_path = results_dir / f"image_baseline_{suffix}.json"
    model_path = model_dir / f"image_baseline_{suffix}.pt"
    metrics["result_path"] = str(result_path)
    metrics["model_path"] = str(model_path)
    checkpoint = {
        "architecture": "resnet18",
        "num_classes": num_classes,
        "label_column": label_column,
        "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
    }
    result_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save(checkpoint, model_path)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="Folder containing <submission_id>.jpg files.",
    )
    parser.add_argument(
        "--label-column",
        choices=("all",) + LABEL_COLUMNS,
        default="all",
        help="Label formulation to train, or 'all' to train all three.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional stratified training-image limit for a quick experiment.",
    )
    parser.add_argument(
        "--class-weight",
        choices=("none", "balanced"),
        default="none",
        help="Use balanced class weights as an optional comparison run.",
    )
    parser.add_argument(
        "--fine-tune-backbone",
        action="store_true",
        help="Unfreeze and fine-tune the pretrained ResNet-18 backbone.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Training device. 'auto' uses CUDA when available, otherwise CPU.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.sample_size is not None and args.sample_size < 1:
        parser.error("--sample-size must be at least 1")
    return args


if __name__ == "__main__":
    args = parse_args()
    label_columns = LABEL_COLUMNS if args.label_column == "all" else (args.label_column,)
    for label_column in label_columns:
        output = train_and_evaluate(
            label_column=label_column,
            image_dir=args.image_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            num_workers=args.num_workers,
            sample_size=args.sample_size,
            class_weight=args.class_weight,
            freeze_backbone=not args.fine_tune_backbone,
            random_state=args.random_state,
            results_dir=args.results_dir,
            model_dir=args.model_dir,
            device_name=args.device,
        )
        validation = output["validation"]
        test = output["test"]
        print(f"[{label_column}] Image baseline trained on {output['train_sample_size']:,} images.")
        print(f"  Device:              {output['device']}")
        print(f"  Validation accuracy: {validation['accuracy']:.4f}")
        print(f"  Test accuracy:       {test['accuracy']:.4f}")
        print(f"  Metrics: {output['result_path']}")
        print(f"  Model:   {output['model_path']}")
