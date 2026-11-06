"""Compute classical features for a whole dataset manifest, with a cache on disk."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from data.stanford_cars import box_with_margin

from .pipeline import extract_features


def _features_for(path: str, box: tuple[int, int, int, int] | None, enhance: bool) -> np.ndarray:
    cv2.setNumThreads(1)
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read {path}")
    if box is not None:
        x1, y1, x2, y2 = box
        image = image[y1:y2, x1:x2]
    return extract_features(image, enhance=enhance)


def compute_features(
    manifest: pd.DataFrame,
    root: Path,
    cache: Path,
    use_box: bool = True,
    enhance: bool = False,
    n_jobs: int = 2,
) -> np.ndarray:
    """Features in manifest order, from the same crops as the classifier. Delete the cache to recompute."""
    if cache.exists():
        stored = np.load(cache)
        if len(stored) == len(manifest):
            return stored
        raise ValueError(f"{cache} has {len(stored)} rows, not {len(manifest)}. Use another cache path.")

    features = Parallel(n_jobs=n_jobs, batch_size=64)(
        delayed(_features_for)(str(root / row.path), box_with_margin(row) if use_box else None, enhance)
        for row in manifest.itertuples()
    )
    stacked = np.vstack(features).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, stacked)
    return stacked
