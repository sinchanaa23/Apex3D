"""Shared code for the Stanford Cars classifier (owner: Person A).

Used by ``01_classifier_training.ipynb`` and ``01b_export_quantize.ipynb``, and
reusable by the backend: :func:`load_classifier` restores the exported bundle and
:func:`predict_crop` gives Top-k for one vehicle crop with the exact preprocessing
used at training time.

Kept in a module (not in the notebook) because Windows DataLoader workers use
``spawn`` and cannot unpickle classes defined in a notebook.
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BBOX_MARGIN = 0.08   # fraction of box size added on each side when cropping

# --------------------------------------------------------------------------- data


@dataclass
class Record:
    path: str          # absolute path to the original image
    label: int         # 0-based class index
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2 (inclusive, original pixels)


def find_stanford_root(base: Path) -> Path:
    """Locate the torchvision-style ``stanford_cars`` folder under ``base``."""
    for cand in [base, *base.rglob("stanford_cars")]:
        if (cand / "devkit" / "cars_meta.mat").exists():
            return cand
    raise FileNotFoundError(f"No stanford_cars folder with devkit/cars_meta.mat under {base}")


def _image_dir(root: Path, split: str) -> Path:
    d = root / f"cars_{split}"
    nested = d / f"cars_{split}"          # some mirrors nest the folder twice
    return nested if nested.is_dir() else d


def load_split(root: Path, split: str) -> list[Record]:
    """Parse the devkit .mat annotations. ``split`` is ``train`` or ``test``."""
    from scipy.io import loadmat

    ann_file = root / "devkit" / "cars_train_annos.mat" if split == "train" else root / "cars_test_annos_withlabels.mat"
    ann = loadmat(ann_file, squeeze_me=True)["annotations"]
    img_dir = _image_dir(root, split)
    out = []
    for a in ann:
        out.append(
            Record(
                path=str(img_dir / str(a["fname"])),
                label=int(a["class"]) - 1,
                bbox=(int(a["bbox_x1"]), int(a["bbox_y1"]), int(a["bbox_x2"]), int(a["bbox_y2"])),
            )
        )
    return out


def load_class_names(root: Path) -> list[str]:
    from scipy.io import loadmat

    return [str(n) for n in loadmat(root / "devkit" / "cars_meta.mat", squeeze_me=True)["class_names"]]


def parse_class_name(name: str) -> dict:
    """Split 'Make Model Type Year' into parts. Year is the trailing 4-digit token.

    Multi-word makes in Stanford Cars are handled explicitly; everything between make
    and year is kept together as 'model' (it includes the body type, e.g. 'Sedan').
    """
    multi_word_makes = ("AM General", "Aston Martin", "Land Rover", "Rolls-Royce")
    year_match = re.search(r"(\d{4})$", name)
    year = int(year_match.group(1)) if year_match else None
    rest = name[: year_match.start()].strip() if year_match else name
    make = next((m for m in multi_word_makes if rest.startswith(m + " ")), rest.split(" ")[0])
    model = rest[len(make):].strip()
    return {"name": name, "make": make, "model": model, "year": year}


def crop_box(img: Image.Image, bbox: Sequence[int], margin: float = BBOX_MARGIN) -> Image.Image:
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1 + 1, y2 - y1 + 1
    x1 = max(0, int(round(x1 - margin * w)))
    y1 = max(0, int(round(y1 - margin * h)))
    x2 = min(img.width, int(round(x2 + 1 + margin * w)))
    y2 = min(img.height, int(round(y2 + 1 + margin * h)))
    return img.crop((x1, y1, x2, y2))


def cache_crop(rec: Record, cache_dir: Path, max_side: int) -> str:
    """Crop the vehicle box once and store it downscaled so training is not I/O-bound.

    Returns the cached path. Downscaling keeps aspect ratio; ``max_side`` is chosen
    above the largest training resolution, so no information used later is lost.
    """
    out = cache_dir / (Path(rec.path).parent.name + "_" + Path(rec.path).stem + ".jpg")
    if not out.exists():
        img = Image.open(rec.path).convert("RGB")
        img = crop_box(img, rec.bbox)
        img.thumbnail((max_side, max_side), Image.BICUBIC)
        tmp = out.with_name(out.stem + f".{os.getpid()}.{threading.get_ident()}.tmp")
        img.save(tmp, format="JPEG", quality=95)
        os.replace(tmp, out)     # atomic: an interrupted run never leaves a half-written crop
    return str(out)


class CropDataset(Dataset):
    def __init__(self, paths: Sequence[str], labels: Sequence[int], transform):
        self.paths = list(paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]


# ------------------------------------------------------------------- transforms


def train_transform(size: int):
    return T.Compose(
        [
            T.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.75, 1.333), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.RandAugment(num_ops=2, magnitude=9),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            T.RandomErasing(p=0.25, scale=(0.02, 0.2)),   # also simulates occlusion
        ]
    )


def eval_transform(size: int):
    """Deterministic: resize the whole crop to a square (no centre-crop, so no part of
    the car is cut off; Stanford crops are wide and a centre crop would drop the ends)."""
    return T.Compose(
        [
            T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# ------------------------------------------------------------------------ model


def build_model(arch: str, num_classes: int, pretrained: bool = True, drop_path: float = 0.1):
    import timm

    return timm.create_model(arch, pretrained=pretrained, num_classes=num_classes, drop_path_rate=drop_path)


def save_bundle(path: Path, model, *, arch: str, img_size: int, class_names: list[str], temperature: float, thresholds: dict, extra: dict | None = None):
    """Everything the app needs to reproduce inference, in one file."""
    torch.save(
        {
            "arch": arch,
            "img_size": img_size,
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
            "bbox_margin": BBOX_MARGIN,
            "class_names": class_names,
            "temperature": float(temperature),
            "thresholds": thresholds,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "extra": extra or {},
        },
        path,
    )


@dataclass
class LoadedClassifier:
    model: torch.nn.Module
    transform: object
    class_names: list[str]
    temperature: float
    thresholds: dict
    img_size: int
    device: str
    bundle: dict


def load_classifier(path: str | Path, device: str | None = None) -> LoadedClassifier:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(bundle["arch"], len(bundle["class_names"]), pretrained=False, drop_path=0.0)
    model.load_state_dict(bundle["state_dict"])
    model.eval().to(device)
    return LoadedClassifier(
        model=model,
        transform=eval_transform(bundle["img_size"]),
        class_names=bundle["class_names"],
        temperature=bundle["temperature"],
        thresholds=bundle["thresholds"],
        img_size=bundle["img_size"],
        device=device,
        bundle=bundle,
    )


def confidence_state(probs_sorted: Sequence[float], thresholds: dict) -> str:
    """'confident' | 'ambiguous' | 'rejected' from calibrated Top-1 prob and margin."""
    p1 = float(probs_sorted[0])
    margin = p1 - float(probs_sorted[1])
    if p1 < thresholds["reject_below_p1"]:
        return "rejected"
    if p1 < thresholds["confident_p1"] or margin < thresholds["confident_margin"]:
        return "ambiguous"
    return "confident"


@torch.no_grad()
def predict_crop(clf: LoadedClassifier, crop: Image.Image, k: int = 5) -> dict:
    """Top-k for a vehicle crop (already cut out by the detector, margin included)."""
    x = clf.transform(crop.convert("RGB")).unsqueeze(0).to(clf.device)
    logits = clf.model(x).float() / clf.temperature
    probs = logits.softmax(-1)[0].cpu().numpy()
    idx = np.argsort(-probs)[:k]
    top = [{"class_id": int(i), "name": clf.class_names[i], "prob": float(probs[i]), **parse_class_name(clf.class_names[i])} for i in idx]
    return {"topk": top, "state": confidence_state(probs[idx], clf.thresholds)}
