"""Image URL and availability helpers for Fakeddit records."""

from urllib.parse import urlparse

import pandas as pd


def _image_host(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    parsed = urlparse(str(value).strip())
    return parsed.netloc.lower()


def prepare_image_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add explicit image availability and host fields without downloading images."""

    result = frame.copy()
    url = result["image_url"].astype(str).str.strip()
    has_image = result["hasImage"].astype(str).str.strip().str.lower().eq("true")

    result["image_url"] = url
    result["image_host"] = url.map(_image_host)
    result["image_url_valid"] = url.str.match(r"^https?://[^\s]+$", na=False)
    result["image_available"] = has_image & result["image_url_valid"]
    return result
