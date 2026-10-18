"""Dataset and PyTorch DataLoader helpers for image models."""

from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset


ImageFile.LOAD_TRUNCATED_IMAGES = True
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def load_image_frame(
    split: str,
    label_column: str,
    image_dir: Path,
) -> pd.DataFrame:
    """Load one processed split and retain records with local image files."""

    path = PROCESSED_DIR / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run src/data/preprocess_dataset.py first."
        )

    frame = pd.read_csv(path, usecols=["id", label_column])
    frame["id"] = frame["id"].fillna("").astype(str).str.strip()
    frame["label"] = pd.to_numeric(frame[label_column], errors="coerce")
    frame = frame.dropna(subset=["label"])
    frame["label"] = frame["label"].astype(int)
    frame["image_path"] = frame["id"].map(lambda value: image_dir / f"{value}.jpg")
    available = frame["image_path"].map(
        lambda value: value.is_file() and value.stat().st_size > 0
    )
    frame = frame.loc[available, ["id", "label", "image_path"]]
    return frame.reset_index(drop=True)


class FakedditImageDataset(Dataset):
    """Read one image record and apply a torchvision transform."""

    def __init__(self, frame: pd.DataFrame, transform: Callable):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        try:
            with Image.open(row["image_path"]) as image:
                image = image.convert("RGB")
                tensor = self.transform(image)
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.as_tensor(tensor)
            return tensor, int(row["label"])
        except (OSError, ValueError, RuntimeError):
            return None


def drop_invalid_items(batch):
    """Drop unreadable images without terminating the training process."""

    valid = [item for item in batch if item is not None]
    if not valid:
        return None
    images, labels = zip(*valid)
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


def make_torch_loader(
    frame: pd.DataFrame,
    transform: Callable,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: Optional[int] = None,
) -> DataLoader:
    """Create a PyTorch DataLoader for image training or evaluation."""

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    dataset = FakedditImageDataset(frame, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=drop_invalid_items,
    )
