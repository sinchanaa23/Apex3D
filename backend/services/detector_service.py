"""
Vehicle Detection Service using Faster R-CNN MobileNetV3 FPN, with a robust fallback.
Detects vehicles (cars, trucks, buses) in uploaded images and produces bounding boxes and crops.

Two ways to run the same network:
  * ONNX Runtime, from checkpoints/detector_mobilenet.onnx (made by scripts/export_detector_onnx.py). This is what a small server uses:
    it needs no PyTorch, which saves a few hundred megabytes of memory.
  * PyTorch and torchvision, when that file is absent and they are installed (a developer's machine).
The boxes are identical (checked in tests/test_detector_onnx.py). Everything after the raw boxes is shared.
"""

import os
import time
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
from PIL import Image

from backend.config import CHECKPOINTS_DIR, DEVICE, ORT_THREADS
from backend.models import BoundingBox

logger = logging.getLogger(__name__)

# COCO Class IDs for vehicles
# 3: car, 4: motorcycle, 6: bus, 8: truck
COCO_VEHICLE_LABELS = {
    3: "car",
    4: "motorcycle",
    6: "bus",
    8: "truck",
}


class VehicleDetector:
    _instance = None

    def __init__(self):
        self.model = None
        self.session = None
        self.backend = "none"  # "onnx", "torch" or "none" (the OpenCV fallback)
        self.device = DEVICE
        self.is_ready = False
        self._load_model()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = VehicleDetector()
        return cls._instance

    def _load_model(self):
        """Loads Faster R-CNN MobileNetV3 FPN: from ONNX when the file exists, otherwise from torchvision."""
        onnx_path = Path(os.getenv("APEX3D_DETECTOR_ONNX", CHECKPOINTS_DIR / "detector_mobilenet.onnx"))
        if onnx_path.is_file():
            try:
                import onnxruntime as ort

                options = ort.SessionOptions()
                options.enable_cpu_mem_arena = False  # hands buffers back after each photo instead of keeping them: less memory
                options.enable_mem_pattern = False
                if ORT_THREADS:
                    options.intra_op_num_threads = ORT_THREADS
                    options.inter_op_num_threads = 1
                self.session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
                self.backend, self.is_ready = "onnx", True
                logger.info("Faster R-CNN detector loaded from %s (ONNX Runtime, no PyTorch needed)", onnx_path)
                return
            except Exception as e:  # noqa: BLE001
                logger.warning("Could not load the ONNX detector %s: %s. Trying torchvision.", onnx_path, e)

        try:
            import torch
            from torchvision.models.detection import (
                fasterrcnn_mobilenet_v3_large_fpn,
                FasterRCNN_MobileNet_V3_Large_FPN_Weights,
            )
        except Exception:
            logger.warning("Neither detector_mobilenet.onnx nor torchvision is available. Detector will use the fallback heuristic.")
            return

        try:
            logger.info("Initializing Faster R-CNN MobileNetV3 Large FPN (torchvision)...")
            weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
            self.model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
            self.model.to(self.device)
            self.model.eval()
            self.backend, self.is_ready = "torch", True
            logger.info("Faster R-CNN detector loaded successfully on %s", self.device)
        except Exception as e:
            logger.warning(
                "Could not load pre-trained Faster R-CNN weights (offline or download error: %s). "
                "Detector will utilize OpenCV saliency/contour fallback.",
                e,
            )
            self.model = None
            self.is_ready = False

    def detect(
        self,
        image: Image.Image,
        score_threshold: float = 0.5,
    ) -> Tuple[Optional[BoundingBox], List[BoundingBox], float]:
        """
        Runs vehicle detection on PIL image.
        Returns:
            primary_box: BoundingBox or None
            all_boxes: List of BoundingBox
            inference_time_ms: float
        """
        start_time = time.perf_counter()
        width, height = image.size

        if self.is_ready:
            try:
                boxes, scores, labels = self._raw_detect(image)

                vehicle_boxes: List[BoundingBox] = []
                for box, score, label in zip(boxes, scores, labels):
                    if score >= score_threshold and label in COCO_VEHICLE_LABELS:
                        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                        # Clamp to image dimensions
                        x1 = max(0.0, min(x1, float(width)))
                        y1 = max(0.0, min(y1, float(height)))
                        x2 = max(0.0, min(x2, float(width)))
                        y2 = max(0.0, min(y2, float(height)))

                        norm = [
                            round(y1 / height, 4),
                            round(x1 / width, 4),
                            round(y2 / height, 4),
                            round(x2 / width, 4),
                        ]
                        vehicle_boxes.append(
                            BoundingBox(
                                x1=round(x1, 2),
                                y1=round(y1, 2),
                                x2=round(x2, 2),
                                y2=round(y2, 2),
                                confidence=round(float(score), 4),
                                label=COCO_VEHICLE_LABELS[label],
                                normalized=norm,
                            )
                        )

                elapsed_ms = (time.perf_counter() - start_time) * 1000

                if vehicle_boxes:
                    return choose_primary(vehicle_boxes), vehicle_boxes, round(elapsed_ms, 2)
                else:
                    return None, [], round(elapsed_ms, 2)

            except Exception as e:
                logger.error("Inference error in Faster R-CNN: %s. Falling back to heuristic.", e)

        # Fallback heuristic using OpenCV saliency/contours
        return self._heuristic_fallback_detect(image, start_time)

    def _raw_detect(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The network's raw output for one photo: boxes (n x 4, in pixels), scores and COCO labels."""
        if self.backend == "onnx":
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0
            boxes, labels, scores = self.session.run(None, {"image": np.ascontiguousarray(pixels)})
            return boxes, scores, labels

        import torch
        from torchvision.transforms import functional as TF

        tensor = TF.to_tensor(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            predictions = self.model(tensor)[0]
        return predictions["boxes"].cpu().numpy(), predictions["scores"].cpu().numpy(), predictions["labels"].cpu().numpy()

    def _heuristic_fallback_detect(
        self, image: Image.Image, start_time: float
    ) -> Tuple[Optional[BoundingBox], List[BoundingBox], float]:
        """
        OpenCV contour-based vehicle bounding box extraction fallback.
        """
        import cv2

        width, height = image.size
        np_img = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = (width * height) * 0.08  # Car should occupy at least 8% of frame
        candidate_boxes = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                # Check typical car aspect ratio (w/h typically 1.0 to 3.5)
                aspect = float(w) / max(1, h)
                if 0.8 <= aspect <= 4.0:
                    x1 = float(x)
                    y1 = float(y)
                    x2 = float(x + w)
                    y2 = float(y + h)
                    norm = [
                        round(y1 / height, 4),
                        round(x1 / width, 4),
                        round(y2 / height, 4),
                        round(x2 / width, 4),
                    ]
                    candidate_boxes.append(
                        BoundingBox(
                            x1=round(x1, 2),
                            y1=round(y1, 2),
                            x2=round(x2, 2),
                            y2=round(y2, 2),
                            confidence=0.85,
                            label="car",
                            normalized=norm,
                        )
                    )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if candidate_boxes:
            # Pick largest area
            primary = max(candidate_boxes, key=lambda b: (b.x2 - b.x1) * (b.y2 - b.y1))
            return primary, candidate_boxes, round(elapsed_ms, 2)

        # If no distinct bounding box found
        return None, [], round(elapsed_ms, 2)


def choose_primary(boxes: List[BoundingBox]) -> BoundingBox:
    """The main vehicle: the largest box among those already above the score threshold.

    The highest score often belongs to a tiny background car, which would make the app identify
    the wrong vehicle. Evidence is in results/detector_*.json under selection_rules.
    """
    return max(boxes, key=lambda b: (b.x2 - b.x1) * (b.y2 - b.y1))


def get_crop(
    image: Image.Image,
    bbox: Optional[BoundingBox] = None,
    force_full_frame: bool = False,
    padding_pct: float = 0.08,
) -> Tuple[Image.Image, str]:
    """
    Crops image to bounding box with optional padding.
    If bbox is None or force_full_frame is True, returns original image.
    Returns (cropped_image, crop_used_str).
    """
    if force_full_frame or bbox is None:
        return image, "full_frame"

    w, h = image.size
    pw = (bbox.x2 - bbox.x1) * padding_pct
    ph = (bbox.y2 - bbox.y1) * padding_pct

    x1 = max(0, int(bbox.x1 - pw))
    y1 = max(0, int(bbox.y1 - ph))
    x2 = min(w, int(bbox.x2 + pw))
    y2 = min(h, int(bbox.y2 + ph))

    if (x2 - x1) < 20 or (y2 - y1) < 20:
        return image, "full_frame"

    return image.crop((x1, y1, x2, y2)), "detected_bbox"

