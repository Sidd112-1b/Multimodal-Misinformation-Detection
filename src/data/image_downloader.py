"""Download Fakeddit images using the official dataset workflow.

Run this script from the destination directory. It creates an ``images``
subdirectory and stores each file as ``<submission_id>.jpg``. The official
Fakeddit script is sequential; this project version adds bounded concurrency,
retries, failure logging, and resume support for the large image collection.
"""

import argparse
import os
import shutil
import socket
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


socket.setdefaulttimeout(30)


def download_one(
    submission_id: str,
    image_url: str,
    output_dir: Path,
    retries: int,
) -> Optional[Tuple[str, str, str]]:
    """Download one image, returning failure details or ``None`` on success."""

    output_path = output_dir / f"{submission_id}.jpg"
    if output_path.exists() and output_path.stat().st_size > 0:
        return None

    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                image_url,
                headers={"User-Agent": "Fakeddit-image-downloader/1.0"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                with output_path.open("wb") as file:
                    shutil.copyfileobj(response, file)
            return None
        except Exception as error:
            try:
                output_path.unlink(missing_ok=True)
            except OSError:
                # The external drive can briefly disconnect while a file is
                # being written. Do not let cleanup terminate the whole run.
                pass
            if attempt == retries:
                return submission_id, image_url, str(error)

    return submission_id, image_url, "download failed"


def download_split(
    tsv_file: Path,
    workers: int,
    retries: int,
    batch_size: int,
    failure_log: Path,
) -> None:
    df = pd.read_csv(tsv_file, sep="\t")
    df = df.replace(np.nan, "", regex=True)
    df.fillna("", inplace=True)

    output_dir = Path("images")
    output_dir.mkdir(parents=True, exist_ok=True)
    failure_log.parent.mkdir(parents=True, exist_ok=True)
    if not failure_log.exists():
        failure_log.write_text("submission_id\timage_url\terror\n", encoding="utf-8")

    rows = df[["id", "image_url", "hasImage"]]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        with tqdm(total=len(rows), desc=tsv_file.name) as progress:
            for start in range(0, len(rows), batch_size):
                batch = rows.iloc[start : start + batch_size]
                futures = []
                for submission_id, image_url, has_image in batch.itertuples(index=False, name=None):
                    if str(has_image).strip().lower() == "true" and str(image_url).strip() not in ("", "nan"):
                        futures.append(
                            executor.submit(
                                download_one,
                                str(submission_id),
                                str(image_url).strip(),
                                output_dir,
                                retries,
                            )
                        )
                    else:
                        progress.update(1)

                for future in as_completed(futures):
                    failure = future.result()
                    if failure is not None:
                        try:
                            with failure_log.open("a", encoding="utf-8") as file:
                                file.write("\t".join(failure) + "\n")
                        except OSError:
                            # A logging failure must not interrupt image downloads.
                            pass
                    progress.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fakeddit image downloader")
    parser.add_argument("tsv_file", type=Path, help="Path to a Fakeddit TSV split")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2_000)
    parser.add_argument(
        "--failure-log",
        type=Path,
        default=None,
        help="Optional path for failed URL records; defaults beside the TSV file.",
    )
    args = parser.parse_args()
    failure_log = args.failure_log or args.tsv_file.parent / "images_download_failed.tsv"
    download_split(args.tsv_file, args.workers, args.retries, args.batch_size, failure_log)
    print("done")
