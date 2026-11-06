"""Pretrained torchvision Faster R-CNN vehicle detectors, trained on COCO."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.transforms.functional import to_tensor

VARIANTS = {
    "mobilenet": (fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT),
    "resnet50": (fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT),
}
# COCO category ids: car, bus and truck. Pickups and large SUVs are often labelled truck.
VEHICLE_CATEGORIES = {3: "car", 6: "bus", 8: "truck"}


@dataclass
class Detection:
    box: tuple[float, float, float, float]
    score: float
    category: str


class VehicleDetector:
    def __init__(self, variant: str = "resnet50", device: str = "cpu", min_score: float = 0.05):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown detector: {variant}. Choose from {sorted(VARIANTS)}.")
        builder, weights = VARIANTS[variant]
        self.variant, self.device = variant, torch.device(device)
        self.model = builder(weights=weights, box_score_thresh=min_score).to(self.device).eval()

    @torch.inference_mode()
    def detect(self, image: Image.Image) -> list[Detection]:
        """Vehicles in an RGB PIL image, highest score first."""
        output = self.model([to_tensor(image).to(self.device)])[0]
        keep = np.isin(output["labels"].cpu().numpy(), list(VEHICLE_CATEGORIES))
        boxes = output["boxes"].cpu().numpy()[keep]
        scores = output["scores"].cpu().numpy()[keep]
        labels = output["labels"].cpu().numpy()[keep]
        found = [
            Detection(tuple(float(value) for value in box), float(score), VEHICLE_CATEGORIES[int(label)])
            for box, score, label in zip(boxes, scores, labels)
        ]
        return sorted(found, key=lambda item: -item.score)


def select_primary(found: list[Detection], min_score: float = 0.5) -> Detection | None:
    """The main vehicle in a photo: the largest confident box, not simply the top score.

    Rules are compared in results/detector_*.json, under selection_rules, and chosen on validation.
    """
    if not found:
        return None
    confident = [item for item in found if item.score >= min_score]
    if not confident:
        return max(found, key=lambda item: item.score)
    return max(confident, key=lambda item: (item.box[2] - item.box[0]) * (item.box[3] - item.box[1]))


def crop_to_detection(image: Image.Image, detection: Detection, margin: float = 0.08) -> Image.Image:
    """Crop with the same relative margin the classifier was trained with (8% per side)."""
    x1, y1, x2, y2 = detection.box
    pad_x, pad_y = (x2 - x1) * margin, (y2 - y1) * margin
    return image.crop((max(int(x1 - pad_x), 0), max(int(y1 - pad_y), 0), min(int(x2 + pad_x), image.width), min(int(y2 + pad_y), image.height)))
