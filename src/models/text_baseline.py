"""Train and evaluate a reproducible TF-IDF text-classification baseline."""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "final"
TEXT_COLUMN = "clean_title"
LABEL_COLUMNS = ("2_way_label", "3_way_label", "6_way_label")


def load_processed_split(split: str, label_column: str) -> Tuple[pd.Series, pd.Series]:
    """Load the normalized title text and selected label from one split."""

    path = PROCESSED_DIR / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run src/data/preprocess_dataset.py first."
        )

    frame = pd.read_csv(path, usecols=[TEXT_COLUMN, label_column])
    text = frame[TEXT_COLUMN].fillna("").astype(str)
    labels = pd.to_numeric(frame[label_column], errors="coerce")
    valid = labels.notna()
    return text.loc[valid].reset_index(drop=True), labels.loc[valid].astype(int).reset_index(drop=True)


def sample_split(
    text: pd.Series,
    labels: pd.Series,
    sample_size: int | None,
    random_state: int,
) -> Tuple[pd.Series, pd.Series]:
    """Optionally sample a split for a quick smoke test."""

    if sample_size is None or sample_size >= len(text):
        return text, labels

    indices = text.sample(n=sample_size, random_state=random_state).index
    return text.loc[indices].reset_index(drop=True), labels.loc[indices].reset_index(drop=True)


def build_pipeline(
    max_features: int,
    max_iter: int,
    random_state: int,
    class_weight: str | None,
) -> Pipeline:
    """Build the text baseline as one serializable scikit-learn pipeline."""

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight=class_weight,
                    max_iter=max_iter,
                    random_state=random_state,
                ),
            ),
        ]
    )


def evaluate(
    pipeline: Pipeline,
    text: pd.Series,
    labels: pd.Series,
) -> Dict[str, object]:
    """Return standard classification metrics for one split."""

    predictions = pipeline.predict(text)
    label_values = sorted(set(labels.tolist()).union(predictions.tolist()))
    return {
        "rows": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "classification_report": classification_report(
            labels,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=label_values).tolist(),
        "labels_in_matrix": label_values,
    }


def train_and_evaluate(
    label_column: str,
    max_features: int,
    max_iter: int,
    sample_size: int | None,
    random_state: int,
    class_weight: str | None,
    results_dir: Path,
    model_dir: Path,
) -> Dict[str, object]:
    """Train on the official training split and evaluate validation and test data."""

    train_text, train_labels = load_processed_split("train", label_column)
    validation_text, validation_labels = load_processed_split("validation", label_column)
    test_text, test_labels = load_processed_split("test", label_column)

    train_text, train_labels = sample_split(
        train_text, train_labels, sample_size, random_state
    )
    pipeline = build_pipeline(max_features, max_iter, random_state, class_weight)
    pipeline.fit(train_text, train_labels)

    majority_label = int(train_labels.mode().iloc[0])
    majority_baseline = {
        "label": majority_label,
        "validation_accuracy": float((validation_labels == majority_label).mean()),
        "test_accuracy": float((test_labels == majority_label).mean()),
    }

    metrics = {
        "task": "text_only_tfidf_logistic_regression",
        "label_column": label_column,
        "text_column": TEXT_COLUMN,
        "random_state": random_state,
        "train_sample_size": int(len(train_text)),
        "max_features": max_features,
        "max_iter": max_iter,
        "class_weight": class_weight,
        "majority_baseline": majority_baseline,
        "validation": evaluate(pipeline, validation_text, validation_labels),
        "test": evaluate(pipeline, test_text, test_labels),
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"text_baseline_{label_column.replace('_', '')}.json"
    model_path = model_dir / f"text_baseline_{label_column.replace('_', '')}.pkl"

    result_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with model_path.open("wb") as file:
        pickle.dump(pipeline, file)

    metrics["result_path"] = str(result_path)
    metrics["model_path"] = str(model_path)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label-column",
        choices=("all",) + LABEL_COLUMNS,
        default="all",
        help="Label formulation to train, or 'all' to train all three.",
    )
    parser.add_argument("--max-features", type=int, default=100_000)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument(
        "--class-weight",
        choices=("none", "balanced"),
        default="none",
        help="Use balanced class weights as an optional comparison run.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional training-row limit for a quick smoke test.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    label_columns = LABEL_COLUMNS if args.label_column == "all" else (args.label_column,)
    for label_column in label_columns:
        output = train_and_evaluate(
            label_column=label_column,
            max_features=args.max_features,
            max_iter=args.max_iter,
            sample_size=args.sample_size,
            random_state=args.random_state,
            class_weight=None if args.class_weight == "none" else args.class_weight,
            results_dir=args.results_dir,
            model_dir=args.model_dir,
        )
        validation = output["validation"]
        test = output["test"]
        print(f"[{label_column}] Text baseline trained on {output['train_sample_size']:,} rows.")
        print(f"  Validation accuracy: {validation['accuracy']:.4f}")
        print(f"  Test accuracy:       {test['accuracy']:.4f}")
        print(
            f"  Majority baseline:   "
            f"{output['majority_baseline']['test_accuracy']:.4f} test accuracy"
        )
        print(f"  Metrics: {output['result_path']}")
        print(f"  Model:   {output['model_path']}")
