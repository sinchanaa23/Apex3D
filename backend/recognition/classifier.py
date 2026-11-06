"""Adapter from Person A's checkpoint to the interface the evaluation scripts use."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "notebooks"))
import cls_lib  # noqa: E402

DEFAULT_CHECKPOINT = REPO / "checkpoints" / "apex3d_convnext_tiny.pt"


class Classifier:
    """predict_proba returns uncalibrated softmax, because calibration is applied by the caller."""

    def __init__(self, loaded: cls_lib.LoadedClassifier):
        self.loaded = loaded
        self.temperature = loaded.temperature
        self.thresholds = loaded.thresholds

    def _batch(self, images: list) -> torch.Tensor:
        return torch.stack([self.loaded.transform(image.convert("RGB")) for image in images]).to(self.loaded.device)

    @torch.inference_mode()
    def predict_proba(self, images: list) -> np.ndarray:
        return self.loaded.model(self._batch(images)).float().softmax(-1).cpu().numpy()

    @torch.inference_mode()
    def embed(self, images: list) -> np.ndarray:
        model = self.loaded.model
        features = model.forward_features(self._batch(images))
        return model.forward_head(features, pre_logits=True).float().cpu().numpy()


def load_classifier(device: str = "cpu", path: str | Path | None = None) -> Classifier:
    checkpoint = Path(path or os.environ.get("APEX3D_CHECKPOINT", DEFAULT_CHECKPOINT))
    if not checkpoint.exists():
        raise FileNotFoundError(f"{checkpoint} not found. Copy it from the team drive, or set APEX3D_CHECKPOINT.")
    return Classifier(cls_lib.load_classifier(checkpoint, device))
