# Multimodal Misinformation Detection

A multimodal deep-learning project for detecting misinformation in online posts by combining written content, associated images, and post metadata.

The project uses the Fakeddit dataset and is being developed in stages. The current implementation focuses on dataset exploration and reproducible preprocessing. Later stages will compare text-only, image-only, and multimodal models.

## Project objective

Online misinformation is often communicated through more than one medium. A misleading post may use sensational language, an unrelated image, or a combination of both. This project investigates whether combining textual and visual information can improve misinformation classification compared with using either modality alone.

The planned system will:

1. Explore the structure, quality, class balance, and relationships in the dataset.
2. Clean and standardize text, image references, metadata, labels, and comments.
3. Establish text-only and image-only baselines.
4. Train a multimodal model that fuses text and image representations.
5. Evaluate the models using reproducible classification metrics.
6. Provide a demonstration interface after the core experiments are complete.

Explainable AI is not part of the current project scope. Interpretability methods may be considered later, but the present work concentrates on data preparation, model performance, and comparison of modalities.

## Dataset

The project is designed around **Fakeddit**, a multimodal Reddit dataset containing submission text, image references, metadata, labels, and comments.

Each submission is identified by a submission ID. The main submission files and the comments file are joined through the submission ID fields.

The project uses the following raw files:

```text
data/raw/fakeddit/
├── multimodal_train.tsv
├── multimodal_validate.tsv
├── multimodal_test_public.tsv
└── all_comments.tsv
```

The submission data includes fields such as:

- title and cleaned title text
- image URL and image availability information
- subreddit, domain, author, score, and upvote ratio
- comment counts and timestamps
- two-way, three-way, and six-way classification labels

The comments file contains comment text, authorship information, parent relationships, submission IDs, and comment scores.

The raw dataset is not stored in this repository. It must be obtained separately and placed in the expected directory. The preprocessing pipeline currently validates image references and records image availability; it does not download every image automatically.

## Planned modeling workflow

The project will compare three modeling settings:

### Text-only

Text features will be extracted from the post title and related text. Planned experiments may include a simple baseline followed by a transformer-based text encoder such as BERT.

### Image-only

Available images will be transformed into model inputs. Planned experiments may include a conventional image baseline followed by a vision transformer such as ViT.

### Multimodal

The text and image representations will be combined with selected metadata before classification:

```text
Post text ──> text encoder ─────┐
                                ├──> fusion network ──> classifier
Image ─────> image encoder ─────┤
                                │
Metadata ───────────────────────┘
```

The exact model configuration, label formulation, and hyperparameters will be finalized after the data exploration and preprocessing stages.

## Repository structure

```text
NNDL Project/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/fakeddit/          # Locally stored source dataset; not committed
│   ├── processed/             # Generated cleaned datasets; not committed
│   └── splits/                # Optional generated split files; not committed
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   └── 02_data_preprocessing.ipynb
│
├── src/
│   ├── data/
│   │   ├── load_dataset.py
│   │   ├── preprocess_text.py
│   │   ├── preprocess_images.py
│   │   └── preprocess_dataset.py
│   ├── models/                # Future model implementations
│   ├── training/               # Future training scripts
│   ├── evaluation/            # Future metrics and comparisons
│   └── utils/                  # Shared utilities
│
├── app/ui/                    # Future demonstration interface
├── docs/                      # Project documentation and presentation material
├── experiments/               # Local experiment outputs; not committed by default
├── models/                    # Local checkpoints and final model files
└── results/                   # Local metrics, plots, and reports
```

## Current implementation

The current data pipeline:

- reads the official train, validation, and test TSV files
- normalizes missing values and text fields
- preserves the available two-way, three-way, and six-way labels
- standardizes timestamps and numeric metadata
- validates image URLs without downloading images
- adds text-length and image-availability features
- streams the large comments file instead of loading it entirely into memory
- generates a compact comment summary keyed by submission ID
- writes processed CSV files for later analysis and modeling

The notebooks provide code-based views of the dataset structure, label distributions, missing values, image references, comment coverage, and preprocessing outputs.

## Setup

Python 3.12 or newer is recommended for the pinned dependencies.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\\.venv\\Scripts\\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running preprocessing

After placing the raw Fakeddit files in `data/raw/fakeddit/`, run the main orchestration script from the project root:

```bash
python src/data/preprocess_dataset.py
```

This script coordinates text and image-reference preprocessing and creates outputs in `data/processed/`. The individual modules `preprocess_text.py` and `preprocess_images.py` are reusable components; they do not normally need to be run separately.

## Exploring the data

Start JupyterLab from the project root:

```bash
jupyter lab
```

Then open:

1. `notebooks/01_dataset_exploration.ipynb` for dataset statistics and quality checks.
2. `notebooks/02_data_preprocessing.ipynb` for preprocessing behavior and output inspection.

The notebooks expect the raw dataset and, where applicable, the generated processed files to exist in the paths described above.

## Evaluation plan

Future model experiments will use the official dataset splits where possible. Evaluation will include:

- accuracy
- precision, recall, and F1-score
- macro and weighted averages for imbalanced labels
- confusion matrices
- per-class performance
- comparison of text-only, image-only, and multimodal systems

Potential duplicate or near-duplicate content across splits will also be examined so that reported results are interpreted carefully.

## Reproducibility and version control

Source code, notebooks, documentation, and configuration files belong in Git. Raw datasets, generated CSV/TSV files, downloaded images, model checkpoints, plots, and experiment outputs are excluded through `.gitignore` because they can be very large or reproducible from the source files.

When new experiments are added, record their configuration, dataset split, label formulation, random seed, and evaluation results so that comparisons remain reproducible.

## Project status

Current stage: **data exploration and data preprocessing**.

The model implementations, training scripts, formal evaluation reports, and demonstration application are planned next and should not be considered complete yet.
