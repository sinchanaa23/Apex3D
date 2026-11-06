"""Shared Stanford Cars loader. Run scripts/prepare_dataset.py once first.

    from data.stanford_cars import StanfordCars

    train = StanfordCars("train", transform=my_transform, crop="box")
    val = StanfordCars("val", transform=eval_transform, crop="box")

Splits: train and val are a stratified 85/15 split of the original training images.
test is the original 8,041-image test set. Use it for final reporting only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Literal

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

MULTI_WORD_MAKES = ("AM General", "Aston Martin", "Land Rover")
BODY_TYPES = (
    "Extended Cab", "Regular Cab", "Crew Cab", "Quad Cab", "Super Cab", "Club Cab",
    "SUV", "Sedan", "Coupe", "Convertible", "Hatchback", "Wagon", "Minivan", "Van", "Cab",
)


def default_root() -> Path:
    base = Path(os.environ.get("APEX3D_DATA", Path.home() / ".apex3d" / "data"))
    return base / "stanford_cars"


def parse_class_name(name: str) -> dict[str, object]:
    """Split a class such as 'Aston Martin V8 Vantage Coupe 2012' into its parts."""
    words, year = name.rsplit(" ", 1)
    make = next((m for m in MULTI_WORD_MAKES if words.startswith(m + " ")), words.split(" ", 1)[0])
    rest = words[len(make) + 1 :]
    body = next((b for b in BODY_TYPES if rest.endswith(" " + b)), "")
    model = rest[: -len(body) - 1] if body else rest
    return {"class_name": name, "make": make, "model": model, "body_type": body, "year": int(year)}


def load_manifest(root: Path | None = None) -> pd.DataFrame:
    path = (root or default_root()) / "manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/prepare_dataset.py")
    return pd.read_csv(path, keep_default_na=False)


def load_class_names(root: Path | None = None) -> list[str]:
    return json.loads(((root or default_root()) / "classes.json").read_text(encoding="utf-8"))


BOX_MARGIN = 0.08


def box_with_margin(row, margin: float = BOX_MARGIN) -> tuple[int, int, int, int]:
    """The crop the classifier was trained on: the box plus a share of its size on each side.

    Matches cls_lib.crop_box in notebooks/, so every model sees identical crops.
    """
    width, height = int(row.x2) - int(row.x1) + 1, int(row.y2) - int(row.y1) + 1
    return (
        max(0, int(round(int(row.x1) - margin * width))),
        max(0, int(round(int(row.y1) - margin * height))),
        min(int(row.width), int(round(int(row.x2) + 1 + margin * width))),
        min(int(row.height), int(round(int(row.y2) + 1 + margin * height))),
    )


class StanfordCars(Dataset):
    """Returns (image, label). Images are RGB PIL images before the transform.

    crop="box" uses the annotated box plus an 8% margin, and crop="none" uses the full image.
    Reporting on detector crops, as the app does, needs its own boxes rather than crop="box".
    """

    def __init__(
        self,
        split: Literal["train", "val", "test"],
        transform: Callable | None = None,
        crop: Literal["none", "box"] = "none",
        margin: float = BOX_MARGIN,
        root: Path | None = None,
    ):
        self.root = root or default_root()
        frame = load_manifest(self.root)
        self.samples = frame[frame["split"] == split].reset_index(drop=True)
        if self.samples.empty:
            raise ValueError(f"Unknown or empty split: {split}")
        self.transform, self.crop, self.margin = transform, crop, margin
        self.classes = load_class_names(self.root)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        row = self.samples.iloc[index]
        with Image.open(self.root / row.path) as opened:
            image = opened.convert("RGB")
        if self.crop == "box":
            image = image.crop(box_with_margin(row, self.margin))
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row.label)
