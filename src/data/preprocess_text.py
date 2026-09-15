"""Text cleaning helpers for the Fakeddit submission fields."""

import re
import unicodedata

import pandas as pd


WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(value: object) -> str:
    """Normalize Unicode and whitespace while preserving punctuation and wording."""

    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    return WHITESPACE_RE.sub(" ", text).strip()


def prepare_text_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized titles and simple audit features."""

    result = frame.copy()
    for column in ("title", "clean_title"):
        if column in result.columns:
            result[column] = result[column].map(normalize_text)

    source_column = "clean_title" if "clean_title" in result.columns else "title"
    result["text_char_count"] = result[source_column].str.len().astype("int64")
    result["text_word_count"] = result[source_column].str.split().str.len().astype("int64")
    return result
