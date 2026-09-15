# Multimodal Misinformation Detection

This project investigates multimodal misinformation detection using the Fakeddit dataset. A data sample may contain Reddit submission text, an associated image, metadata, comments, and one or more classification labels.

## Current project phase

The current phase is limited to **data exploration and data preprocessing**. Model development, training, multimodal fusion, and the final application will be handled later.

Current objectives:

1. Understand the structure and columns of the available Fakeddit files.
2. Examine text, image references, metadata, comments, missing values, and label distributions.
3. Identify how records are connected through identifiers such as `submission_id`.
4. Clean and standardize the data without modifying the raw files.
5. Decide how to handle missing text, missing images, duplicate records, and invalid entries.
6. Create reproducible train, validation, and test preparation outputs.
7. Record the findings needed for the later modeling phase.

The first exploration and preprocessing pass is complete. The raw split IDs and row counts were preserved, and all three label columns remain available for the later label-selection decision.

## Dataset location

The currently available raw files are kept in `data/raw/fakeddit/`:

```text
data/raw/fakeddit/
├── multimodal_train.tsv
├── multimodal_validate.tsv
├── multimodal_test_public.tsv
└── all_comments.tsv
```

The raw dataset must remain unchanged. Processed files and generated split information will be stored under:

```text
data/
├── processed/
└── splits/
```

The exact usable columns and label scheme will be confirmed during dataset exploration. The project may use Fakeddit's 2-way, 3-way, or 6-way labels depending on data quality, class balance, and the final project scope.

## Initial dataset findings

- Train: 564,000 records
- Validation: 59,342 records
- Test: 59,319 records
- No duplicate IDs or ID overlap between the official splits
- Approximately 0.3% of records have missing image URLs
- The 6-way labels are imbalanced, with the smallest classes representing roughly 2–4% of a split
- The comments file contains 10.67 million parsed records; 353,300 current submissions have linked comments
- Duplicate titles occur across splits and will be monitored as a possible leakage concern

## Environment setup

Create and activate a virtual environment, then install the current-phase dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Launch the notebook environment with:

```powershell
jupyter lab
```

To regenerate the processed tables from the raw TSV files:

```powershell
python src/data/preprocess_dataset.py
```

This command streams the large comments file and writes only an aggregated comment summary. It does not modify the raw files.

## Current working areas

```text
notebooks/
├── 01_dataset_exploration.ipynb
└── 02_data_preprocessing.ipynb

src/data/
├── load_dataset.py
├── preprocess_text.py
├── preprocess_images.py
└── preprocess_dataset.py

data/processed/       # Cleaned split tables and comment summary
data/splits/          # Reserved for explicit split-ID exports
results/plots/        # Exploration visualizations
results/metrics/      # Data-quality and distribution summaries
```

## Data-handling principles

- Keep the original files in `data/raw/fakeddit/` unchanged.
- Use `id` in the submission tables and `submission_id` in the comments file to connect related records.
- Preserve punctuation and wording in `clean_title`; only Unicode and repeated whitespace are normalized.
- Keep missing metadata as missing values rather than replacing it with zero.
- Represent image availability explicitly instead of silently dropping records with missing URLs.
- Preserve the official train, validation, and test boundaries.
- Keep all 2-way, 3-way, and 6-way labels until the project chooses its final target.
- Avoid downloading or duplicating the full dataset until the required fields and sample scope are confirmed.
- Save preprocessing decisions and assumptions so that the later modeling stage is reproducible.
- Do not begin model training during the current phase.
