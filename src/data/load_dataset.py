"""Load the raw Fakeddit TSV files."""

from pathlib import Path
from typing import Iterable, Optional, Union

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "fakeddit"

SPLIT_FILES = {
    "train": RAW_DIR / "multimodal_train.tsv",
    "validation": RAW_DIR / "multimodal_validate.tsv",
    "test": RAW_DIR / "multimodal_test_public.tsv",
}
COMMENTS_FILE = RAW_DIR / "all_comments.tsv"


def read_split(
    split: str,
    usecols: Optional[Iterable[str]] = None,
    chunksize: Optional[int] = None,
) -> Union[pd.DataFrame, pd.io.parsers.readers.TextFileReader]:
    """Read one official dataset split without changing the raw file."""

    if split not in SPLIT_FILES:
        raise ValueError(f"Unknown split {split!r}; choose from {sorted(SPLIT_FILES)}")

    return pd.read_csv(
        SPLIT_FILES[split],
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=usecols,
        chunksize=chunksize,
    )


def read_comments(
    usecols: Optional[Iterable[str]] = None,
    chunksize: int = 500_000,
) -> pd.io.parsers.readers.TextFileReader:
    """Stream the large comments file in bounded chunks."""

    return pd.read_csv(
        COMMENTS_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=usecols,
        chunksize=chunksize,
    )
