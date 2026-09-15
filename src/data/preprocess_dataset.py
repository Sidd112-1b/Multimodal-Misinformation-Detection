"""Create cleaned split tables and a compact comment-linkage summary."""

from collections import Counter
from pathlib import Path
from typing import Dict, Set

import pandas as pd

try:
    from load_dataset import COMMENTS_FILE, PROJECT_ROOT, read_comments, read_split
    from preprocess_images import prepare_image_columns
    from preprocess_text import prepare_text_columns
except ImportError:  # pragma: no cover - supports package-style imports too
    from .load_dataset import COMMENTS_FILE, PROJECT_ROOT, read_comments, read_split
    from .preprocess_images import prepare_image_columns
    from .preprocess_text import prepare_text_columns


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def prepare_chunk(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply conservative, model-agnostic cleaning to one raw chunk."""

    result = frame.copy()
    text_columns = ["author", "domain", "id", "linked_submission_id", "subreddit"]
    for column in text_columns:
        if column in result.columns:
            result[column] = result[column].astype(str).str.strip()

    result = prepare_text_columns(result)
    result = prepare_image_columns(result)

    for column in ["created_utc", "num_comments", "score", "upvote_ratio"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result["created_at"] = pd.to_datetime(
        result["created_utc"], unit="s", utc=True, errors="coerce"
    ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    for column in ["2_way_label", "3_way_label", "6_way_label"]:
        result[column] = pd.to_numeric(result[column], errors="coerce").astype("Int64")

    ordered_columns = [
        "id",
        "clean_title",
        "title",
        "text_char_count",
        "text_word_count",
        "image_url",
        "image_host",
        "image_url_valid",
        "image_available",
        "linked_submission_id",
        "author",
        "domain",
        "subreddit",
        "created_utc",
        "created_at",
        "num_comments",
        "score",
        "upvote_ratio",
        "2_way_label",
        "3_way_label",
        "6_way_label",
    ]
    return result[[column for column in ordered_columns if column in result.columns]]


def write_processed_splits() -> Dict[str, Set[str]]:
    """Write cleaned train, validation, and test CSVs and return their IDs."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    split_ids: Dict[str, Set[str]] = {}

    for split in ("train", "validation", "test"):
        output_path = PROCESSED_DIR / f"{split}.csv"
        if output_path.exists():
            output_path.unlink()

        ids: Set[str] = set()
        first_chunk = True
        for chunk in read_split(split, chunksize=100_000):
            cleaned = prepare_chunk(chunk)
            ids.update(cleaned["id"].tolist())
            cleaned.to_csv(output_path, mode="w" if first_chunk else "a", index=False, header=first_chunk)
            first_chunk = False
        split_ids[split] = ids

    return split_ids


def write_comment_summary(split_ids: Dict[str, Set[str]]) -> None:
    """Aggregate only comment records linked to the official multimodal splits."""

    all_ids = set().union(*split_ids.values())
    comment_counts = Counter()
    top_level_counts = Counter()
    max_ups = {}

    for chunk in read_comments(usecols=["submission_id", "isTopLevel", "ups"]):
        chunk["submission_id"] = chunk["submission_id"].astype(str).str.strip()
        chunk = chunk[chunk["submission_id"].isin(all_ids)]
        if chunk.empty:
            continue

        comment_counts.update(chunk["submission_id"].tolist())
        top_level_ids = chunk.loc[
            chunk["isTopLevel"].astype(str).str.strip().str.lower().eq("true"),
            "submission_id",
        ]
        top_level_counts.update(top_level_ids.tolist())

        chunk["ups_numeric"] = pd.to_numeric(chunk["ups"], errors="coerce")
        for submission_id, value in chunk.dropna(subset=["ups_numeric"]).groupby("submission_id")["ups_numeric"].max().items():
            max_ups[submission_id] = max(max_ups.get(submission_id, value), value)

    summary = pd.DataFrame({"id": sorted(comment_counts)})
    summary["comment_count"] = summary["id"].map(comment_counts).astype("int64")
    summary["top_level_comment_count"] = summary["id"].map(top_level_counts).fillna(0).astype("int64")
    summary["max_comment_ups"] = summary["id"].map(max_ups)
    summary.to_csv(PROCESSED_DIR / "comment_summary.csv", index=False)


if __name__ == "__main__":
    split_ids = write_processed_splits()
    write_comment_summary(split_ids)
    print(f"Wrote processed files to {PROCESSED_DIR}")
