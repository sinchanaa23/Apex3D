"""
Fine-Grained Vehicle Classifier Service for 196 Stanford Cars classes.
Supports PyTorch checkpoints, ONNX quantized models, and a feature-based fallback
to ensure zero-downtime frontend development while offline training proceeds.
"""

import os
import time
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from backend.config import (
    CHECKPOINTS_DIR,
    NOTEBOOKS_DIR,
    DEVICE,
    CONFIDENT_THRESHOLD,
    AMBIGUOUS_MARGIN_THRESHOLD,
    REJECTED_THRESHOLD,
)
from backend.data.stanford_cars_classes import (
    STANFORD_CARS_CLASSES,
    get_class_info,
)
from backend.models import VehiclePrediction

logger = logging.getLogger(__name__)


class ClassifierUnavailable(RuntimeError):
    """Raised when there is no trained model to answer with. The API reports it as a 503 with this message."""


class VehicleClassifier:
    _instance = None

    def __init__(self):
        self.device = DEVICE
        self.model = None
        self.onnx_session = None
        self.model_type = "none"
        self.model_path = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        self.reload_model()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = VehicleClassifier()
        return cls._instance

    def reload_model(self):
        """Scans candidate directories for trained checkpoints."""
        candidate_paths = [
            CHECKPOINTS_DIR / "convnext_stanford_cars.pth",
            CHECKPOINTS_DIR / "best_model.pth",
            CHECKPOINTS_DIR / "checkpoint.pth",
            NOTEBOOKS_DIR / "checkpoint.pth",
            CHECKPOINTS_DIR / "model_quantized.onnx",
            CHECKPOINTS_DIR / "model.onnx",
        ]

        for p in candidate_paths:
            if p.exists() and p.is_file():
                if p.suffix == ".onnx":
                    try:
                        import onnxruntime as ort
                        self.onnx_session = ort.InferenceSession(str(p))
                        self.model_type = "onnx"
                        self.model_path = str(p)
                        logger.info("Loaded ONNX classifier checkpoint from %s", p)
                        return
                    except Exception as e:
                        logger.warning("Failed loading ONNX model from %s: %s", p, e)
                elif p.suffix in [".pth", ".pt"]:
                    try:
                        checkpoint = torch.load(str(p), map_location=self.device)
                        if isinstance(checkpoint, torch.nn.Module):
                            self.model = checkpoint
                        elif isinstance(checkpoint, dict) and "model" in checkpoint:
                            self.model = checkpoint["model"]
                        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                            # If state dict only, let A supply instantiated model or custom loader
                            self.model = checkpoint["state_dict"]
                        
                        if hasattr(self.model, "eval"):
                            self.model.to(self.device)
                            self.model.eval()
                            self.model_type = "pytorch"
                            self.model_path = str(p)
                            logger.info("Loaded PyTorch classifier checkpoint from %s", p)
                            return
                    except Exception as e:
                        logger.warning("Failed loading PyTorch model from %s: %s", p, e)

        logger.warning(
            "No trained checkpoint found in checkpoints/ or notebooks/. Recognition will return an error "
            "until checkpoints/model_quantized.onnx is in place."
        )
        self.model_type = "none"
        self.model_path = None

    def classify(
        self, crop_image: Image.Image
    ) -> Tuple[List[VehiclePrediction], str, float, float, float, str]:
        """
        Classifies cropped vehicle image into Top-5 Stanford Cars classes.
        Returns:
            (predictions, confidence_state, confidence_score, margin, elapsed_ms, message)
        """
        start_time = time.perf_counter()

        # Check if PyTorch model is active
        if self.model_type == "pytorch" and self.model is not None and hasattr(self.model, "eval"):
            try:
                img_t = self.transform(crop_image.convert("RGB")).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.model(img_t)
                    probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                return self._process_probabilities(probs, start_time)
            except Exception as e:
                logger.error("Error executing PyTorch model: %s", e)
                raise ClassifierUnavailable(f"The classifier failed while running: {e}") from e

        # Check if ONNX session is active
        if self.model_type == "onnx" and self.onnx_session is not None:
            try:
                img_t = self.transform(crop_image.convert("RGB")).unsqueeze(0).numpy()
                input_name = self.onnx_session.get_inputs()[0].name
                raw_out = self.onnx_session.run(None, {input_name: img_t})[0]
                logits = torch.from_numpy(raw_out)
                probs = F.softmax(logits, dim=-1).squeeze(0).numpy()
                return self._process_probabilities(probs, start_time)
            except Exception as e:
                logger.error("Error executing ONNX model: %s", e)
                raise ClassifierUnavailable(f"The classifier failed while running: {e}") from e

        raise ClassifierUnavailable(
            "No trained classifier is loaded. Put model_quantized.onnx in the checkpoints/ folder "
            "(from Person A's handover zip) and restart the server."
        )

    def _process_probabilities(
        self, probs: np.ndarray, start_time: float
    ) -> Tuple[List[VehiclePrediction], str, float, float, float, str]:
        """Formats softmax vector into Top-5, calculates confidence state, margin, and timings."""
        top5_indices = np.argsort(probs)[::-1][:5]
        predictions: List[VehiclePrediction] = []

        for idx in top5_indices:
            p = float(probs[idx])
            info = get_class_info(int(idx))
            if info:
                predictions.append(
                    VehiclePrediction(
                        class_id=int(idx),
                        class_name=info["full_name"],
                        make=info["make"],
                        model=info["model"],
                        year=info["year"],
                        body_style=info["body_style"],
                        probability=round(p, 4),
                    )
                )

        top1_prob = predictions[0].probability if predictions else 0.0
        top2_prob = predictions[1].probability if len(predictions) > 1 else 0.0
        margin = round(top1_prob - top2_prob, 4)

        # Determine confidence state based on agreed threshold rules
        if top1_prob < REJECTED_THRESHOLD:
            confidence_state = "rejected"
            message = "Low confidence: Vehicle may be outside the Stanford Cars 196 taxonomy, or poorly visible."
        elif margin < AMBIGUOUS_MARGIN_THRESHOLD:
            confidence_state = "ambiguous"
            message = "Ambiguous classification: Multiple closely matching model years or trim levels detected."
        else:
            confidence_state = "confident"
            message = "High-confidence vehicle identification."

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return predictions, confidence_state, top1_prob, margin, elapsed_ms, message
