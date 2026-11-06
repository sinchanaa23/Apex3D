"""The classical pipeline in the format POST /api/classical returns (frontend/API_EXPECTATIONS.md)."""

from __future__ import annotations

import base64
import time

import cv2
import numpy as np

from .pipeline import hsv_histogram, normalize, run_pipeline

STEPS = (
    ("original", "Original", "The photo, resized for analysis."),
    ("clahe", "CLAHE", "Contrast equalised on the LAB lightness channel."),
    ("edges", "Canny edges", "Thresholds are set from the image's median brightness."),
    ("contours_roi", "Contours and ROI", "The largest contour and its bounding box. Busy backgrounds can pull this off the car."),
    ("roi_crop", "ROI crop", "The region selected by the contour."),
    ("hsv_hue", "HSV hue", "Colour, separated from brightness."),
    ("hsv_saturation", "HSV saturation", "How vivid the colour is."),
    ("hsv_value", "HSV value", "Brightness."),
    ("lbp", "LBP texture", "Local Binary Patterns describe fine texture."),
    ("hog", "HOG", "Edge directions. This was the strongest classical feature in our tests."),
)


def _data_url(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError("Could not encode a pipeline image.")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def api_response(image: np.ndarray) -> dict:
    """Takes a uint8 BGR image. Bad input raises ValueError with a message for the user."""
    started = time.perf_counter()
    result = run_pipeline(image)
    return {
        "steps": [{"key": key, "title": title, "image": _data_url(result.images[key]), "caption": caption} for key, title, caption in STEPS],
        "features": {
            **{name: round(value, 4) for name, value in result.geometry.items() if not name.startswith("hu_")},
            "hu_moments": [round(value, 3) for name, value in result.geometry.items() if name.startswith("hu_")],
            "mean_brightness": round(result.mean_luma, 1),
            "low_light": "yes" if result.low_light else "no",
            "hsv_histogram": [round(float(value), 4) for value in hsv_histogram(normalize(image))],
            "feature_vector_length": result.feature_length,
        },
        "timings_ms": {"total": round((time.perf_counter() - started) * 1000, 1)},
    }
