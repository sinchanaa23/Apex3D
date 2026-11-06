"""Download Stanford Cars, verify it, and write images plus a manifest.

    python scripts/prepare_dataset.py

The data comes from three public mirrors, because the original Stanford URLs are unavailable:
    images and labels     Hugging Face tanganke/stanford_cars (Parquet)
    train boxes           GitHub BotechEngineering/StanfordCarsDatasetCSV
    test boxes, labels    cars_test_annos_withlabels.mat, checked against torchvision's MD5

The Parquet mirror drops the original filenames and stores rows out of file order, so rows cannot
be joined to annotations directly. Within each class, however, rows keep the original filename
order, so the join is made on (class, position within class). The script proves this on every run:
per-class counts must match and every box must fit inside its image. Any failed check stops the
script before a manifest is written.

The output folder defaults to ~/.apex3d/data/stanford_cars. Set APEX3D_DATA to change it.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
from scipy.io import loadmat
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from data.stanford_cars import parse_class_name  # noqa: E402

HF_BASE = "https://huggingface.co/datasets/tanganke/stanford_cars/resolve/main/data"
PARQUET = {
    "train": {"train-00000-of-00002.parquet": 503707028, "train-00001-of-00002.parquet": 484975142},
    "test": {"test-00000-of-00002.parquet": 513268211, "test-00001-of-00002.parquet": 473601440},
}
TRAIN_CSV_URL = "https://raw.githubusercontent.com/BotechEngineering/StanfordCarsDatasetCSV/main/cardatasettrain.csv"
TEST_MAT_URL = "https://huggingface.co/datasets/Shiym/ViT-FineTune/resolve/main/stanford_cars/cars_test_annos_withlabels.mat"
TEST_MAT_MD5 = "b0a2b23655a3edd16d84508592a98d10"
EXPECTED_ROWS = {"train": 8144, "test": 8041}
NUM_CLASSES = 196
VAL_FRACTION = 0.15
SEED = 42


def fetch(url: str, path: Path, size: int | None = None) -> None:
    if path.exists() and (size is None or path.stat().st_size == size):
        return
    print(f"Downloading {path.name}...", flush=True)
    temporary = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(url, temporary)
    if size is not None and temporary.stat().st_size != size:
        raise SystemExit(f"{path.name}: expected {size} bytes, got {temporary.stat().st_size}.")
    temporary.replace(path)


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def load_annotations(raw: Path) -> dict[str, pd.DataFrame]:
    train = pd.read_csv(raw / "cardatasettrain.csv").rename(columns={"image": "filename"})
    train["label"] = train["Class"] - 1

    mat_path = raw / "cars_test_annos_withlabels.mat"
    digest = md5(mat_path)
    if digest != TEST_MAT_MD5:
        raise SystemExit(f"Test annotations failed the MD5 check: {digest} != {TEST_MAT_MD5}.")
    records = loadmat(mat_path, squeeze_me=True)["annotations"]
    test = pd.DataFrame(
        {
            "x1": records["bbox_x1"].astype(int),
            "y1": records["bbox_y1"].astype(int),
            "x2": records["bbox_x2"].astype(int),
            "y2": records["bbox_y2"].astype(int),
            "label": records["class"].astype(int) - 1,
            "filename": records["fname"].astype(str),
        }
    )
    columns = ["filename", "label", "x1", "y1", "x2", "y2"]
    return {"train": train[columns], "test": test[columns]}


def extract_split(split: str, raw: Path, images: Path) -> tuple[pd.DataFrame, list[str]]:
    """Read Parquet in small batches and write each image's original bytes to a temporary name."""
    folder = images / split
    folder.mkdir(parents=True, exist_ok=True)
    rows, names, position = [], [], 0
    for name in sorted(PARQUET[split]):
        parquet = pq.ParquetFile(raw / name)
        if not names:
            metadata = json.loads(parquet.schema_arrow.metadata[b"huggingface"])
            names = metadata["info"]["features"]["label"]["names"]
        for batch in parquet.iter_batches(batch_size=32, columns=["image", "label"]):
            for image, label in zip(batch.column("image").to_pylist(), batch.column("label").to_pylist()):
                with Image.open(io.BytesIO(image["bytes"])) as opened:
                    width, height = opened.size
                (folder / f"row_{position:05d}.tmp").write_bytes(image["bytes"])
                rows.append((position, label, width, height))
                position += 1
    return pd.DataFrame(rows, columns=["row", "label", "width", "height"]), names


def join_split(split: str, rows: pd.DataFrame, annotations: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Join on class and position within class, then check the result. Stops on any failure."""
    if len(rows) != EXPECTED_ROWS[split] or len(annotations) != EXPECTED_ROWS[split]:
        raise SystemExit(f"{split}: expected {EXPECTED_ROWS[split]} rows, got {len(rows)} and {len(annotations)}.")
    if rows["label"].nunique() != NUM_CLASSES:
        raise SystemExit(f"{split}: expected {NUM_CLASSES} classes, got {rows['label'].nunique()}.")
    counts_rows = rows["label"].value_counts().sort_index()
    counts_annotations = annotations["label"].value_counts().sort_index()
    if not counts_rows.equals(counts_annotations):
        raise SystemExit(f"{split}: per-class counts differ between images and annotations.")

    left = rows.assign(position=rows.groupby("label").cumcount())
    ordered = annotations.sort_values(["label", "filename"], kind="stable")
    right = ordered.assign(position=ordered.groupby("label").cumcount())
    joined = left.merge(right, on=["label", "position"], validate="one_to_one").sort_values("filename")

    valid = (joined.x1 >= 0) & (joined.y1 >= 0) & (joined.x2 > joined.x1) & (joined.y2 > joined.y1)
    fits = (joined.x2 <= joined.width) & (joined.y2 <= joined.height)
    if not valid.all() or not fits.all():
        raise SystemExit(f"{split}: {int((~valid).sum())} invalid boxes and {int((~fits).sum())} boxes outside the image.")

    # What an incorrect join would score, to show that the check can actually fail.
    shuffled = rows.sample(frac=1, random_state=SEED).reset_index(drop=True)
    chance = float(((annotations.x2.values <= shuffled.width.values) & (annotations.y2.values <= shuffled.height.values)).mean())
    return joined, {"rows": len(joined), "boxes_inside_image": int(fits.sum()), "boxes_inside_image_if_shuffled": round(chance, 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_root = Path(os.environ.get("APEX3D_DATA", Path.home() / ".apex3d" / "data"))
    parser.add_argument("--root", type=Path, default=default_root)
    args = parser.parse_args()

    raw = args.root / "stanford_cars_raw"
    output = args.root / "stanford_cars"
    raw.mkdir(parents=True, exist_ok=True)
    for files in PARQUET.values():
        for name, size in files.items():
            fetch(f"{HF_BASE}/{name}", raw / name, size)
    fetch(TRAIN_CSV_URL, raw / "cardatasettrain.csv")
    fetch(TEST_MAT_URL, raw / "cars_test_annos_withlabels.mat")

    annotations = load_annotations(raw)
    images = output / "images"
    frames, report = [], {"test_annotations_md5": TEST_MAT_MD5, "splits": {}}
    class_names = None
    for split in ("train", "test"):
        print(f"Extracting {split}...", flush=True)
        rows, class_names = extract_split(split, raw, images)
        joined, report["splits"][split] = join_split(split, rows, annotations[split])
        for row, filename in zip(joined["row"], joined["filename"]):
            (images / split / f"row_{row:05d}.tmp").replace(images / split / filename)
        frames.append(joined.assign(split=split, path=[f"images/{split}/{name}" for name in joined["filename"]]))

    manifest = pd.concat(frames, ignore_index=True).drop(columns=["row", "position"])
    parsed = pd.DataFrame([parse_class_name(class_names[label]) for label in manifest["label"]])
    manifest = pd.concat([manifest, parsed], axis=1)

    # The validation set comes from train only, so that the test set stays unseen.
    train_index = manifest.index[manifest["split"] == "train"]
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=SEED)
    _, val_positions = next(splitter.split(np.zeros(len(train_index)), manifest.loc[train_index, "label"]))
    manifest.loc[train_index[val_positions], "split"] = "val"

    manifest.to_csv(output / "manifest.csv", index=False)
    (output / "classes.json").write_text(json.dumps(class_names, indent=1), encoding="utf-8")
    report["final_split_sizes"] = manifest["split"].value_counts().to_dict()
    report["val_fraction"], report["seed"] = VAL_FRACTION, SEED
    (output / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Dataset ready: {output}")


if __name__ == "__main__":
    main()
