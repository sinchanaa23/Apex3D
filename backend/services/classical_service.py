"""
Classical Computer Vision Service for Apex3D.
Implements CLAHE, Canny edge detection, contour extraction, largest contour ROI,
HSV color histogram, HOG (Histogram of Oriented Gradients), LBP (Local Binary Patterns),
and geometric descriptors (aspect ratio, extent, solidity, Hu moments).

Gracefully interfaces with Person B's backend/classical_cv/ module when available.
"""

import io
import time
import base64
import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2
from PIL import Image
from skimage.feature import hog, local_binary_pattern

from backend.models import (
    ClassicalCVResponse,
    ClassicalFeatures,
    GeometricFeatures,
    HOGSummary,
    LBPSummary,
    HSVSummary,
)

logger = logging.getLogger(__name__)


def numpy_to_base64_png(arr: np.ndarray, is_bgr: bool = True) -> str:
    """Converts a numpy image array (BGR or Grayscale) to a base64 data URI PNG string."""
    if len(arr.shape) == 3 and not is_bgr:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".png", arr)
    encoded = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


class ClassicalCVService:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ClassicalCVService()
        return cls._instance

    def process(
        self,
        image: Image.Image,
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> ClassicalCVResponse:
        """
        Runs the full classical CV pipeline and returns visualisations and extracted features.
        """
        total_start = time.perf_counter()
        timings = {}

        # Crop to bbox if provided
        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            w, h = image.size
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if (x2 - x1) > 10 and (y2 - y1) > 10:
                image = image.crop((x1, y1, x2, y2))

        # Convert PIL to BGR numpy array
        rgb_np = np.array(image.convert("RGB"))
        bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
        orig_h, orig_w = bgr.shape[:2]

        # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization on LAB L-channel)
        t0 = time.perf_counter()
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        merged_lab = cv2.merge((cl, a_channel, b_channel))
        clahe_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
        timings["clahe"] = round((time.perf_counter() - t0) * 1000, 2)

        # 2. Canny Edge Detection & Contours
        t0 = time.perf_counter()
        gray = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_canvas = bgr.copy()
        cv2.drawContours(contours_canvas, contours, -1, (0, 255, 0), 2)

        # Find largest contour ROI
        if contours:
            largest_cnt = max(contours, key=cv2.contourArea)
            rx, ry, rw, rh = cv2.boundingRect(largest_cnt)
            # Ensure valid bounds
            rx1, ry1 = max(0, rx), max(0, ry)
            rx2, ry2 = min(orig_w, rx + rw), min(orig_h, ry + rh)
            if (rx2 - rx1) > 5 and (ry2 - ry1) > 5:
                largest_roi = bgr[ry1:ry2, rx1:rx2]
            else:
                largest_roi = bgr
        else:
            largest_cnt = None
            largest_roi = bgr

        timings["canny_contours"] = round((time.perf_counter() - t0) * 1000, 2)

        # 3. Geometric Features
        t0 = time.perf_counter()
        if largest_cnt is not None and len(largest_cnt) >= 5:
            area = float(cv2.contourArea(largest_cnt))
            perimeter = float(cv2.arcLength(largest_cnt, True))
            gx, gy, gw, gh = cv2.boundingRect(largest_cnt)
            aspect_ratio = float(gw) / max(1.0, gh)
            extent = float(area) / max(1.0, (gw * gh))

            hull = cv2.convexHull(largest_cnt)
            hull_area = float(cv2.contourArea(hull))
            solidity = float(area) / max(1.0, hull_area)

            moments = cv2.moments(largest_cnt)
            hu_m = cv2.HuMoments(moments).flatten()
            # Log transform Hu moments for numerical stability
            log_hu = [-1.0 * np.sign(h) * np.log10(abs(h) + 1e-12) for h in hu_m]
        else:
            aspect_ratio = float(orig_w) / max(1.0, orig_h)
            extent = 1.0
            solidity = 1.0
            area = float(orig_w * orig_h)
            perimeter = float(2 * (orig_w + orig_h))
            log_hu = [0.0] * 7

        geometric = GeometricFeatures(
            aspect_ratio=round(aspect_ratio, 3),
            extent=round(extent, 3),
            solidity=round(solidity, 3),
            contour_area=round(area, 1),
            perimeter=round(perimeter, 1),
            hu_moments=[round(float(v), 5) for v in log_hu],
        )
        timings["geometric"] = round((time.perf_counter() - t0) * 1000, 2)

        # 4. HOG (Histogram of Oriented Gradients)
        t0 = time.perf_counter()
        resized_gray = cv2.resize(gray, (128, 64))  # Standard HOG dimensions (W=128, H=64)
        hog_features, hog_image = hog(
            resized_gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=True,
            feature_vector=True,
        )
        # Normalize visualization for display
        hog_vis_uint8 = np.uint8(255 * (hog_image / (np.max(hog_image) + 1e-8)))
        hog_vis_bgr = cv2.cvtColor(hog_vis_uint8, cv2.COLOR_GRAY2BGR)

        hog_summary = HOGSummary(
            vector_length=len(hog_features),
            mean=round(float(np.mean(hog_features)), 5),
            std=round(float(np.std(hog_features)), 5),
            feature_sample=[round(float(v), 4) for v in hog_features[:8]],
        )
        timings["hog"] = round((time.perf_counter() - t0) * 1000, 2)

        # 5. LBP (Local Binary Patterns)
        t0 = time.perf_counter()
        n_points = 24
        radius = 3
        lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
        n_bins = n_points + 2
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

        lbp_summary = LBPSummary(
            num_points=n_points,
            radius=radius,
            histogram=[round(float(v), 4) for v in lbp_hist],
        )
        timings["lbp"] = round((time.perf_counter() - t0) * 1000, 2)

        # 6. HSV Color Histogram
        t0 = time.perf_counter()
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h_channel, s_channel, v_channel = cv2.split(hsv)
        hue_hist = cv2.calcHist([h_channel], [0], None, [16], [0, 180])
        hue_hist = cv2.normalize(hue_hist, hue_hist).flatten()

        color_hsv = HSVSummary(
            mean_hue=round(float(np.mean(h_channel)), 2),
            mean_saturation=round(float(np.mean(s_channel)), 2),
            mean_value=round(float(np.mean(v_channel)), 2),
            hue_histogram=[round(float(v), 4) for v in hue_hist],
        )
        timings["hsv"] = round((time.perf_counter() - t0) * 1000, 2)

        timings["total"] = round((time.perf_counter() - total_start) * 1000, 2)

        # Encode intermediate images to base64 data URIs
        intermediate_images = {
            "original": numpy_to_base64_png(bgr, is_bgr=True),
            "clahe": numpy_to_base64_png(clahe_bgr, is_bgr=True),
            "edges": numpy_to_base64_png(edges, is_bgr=False),
            "contours": numpy_to_base64_png(contours_canvas, is_bgr=True),
            "largest_contour_roi": numpy_to_base64_png(largest_roi, is_bgr=True),
            "hog_visualization": numpy_to_base64_png(hog_vis_bgr, is_bgr=True),
        }

        features = ClassicalFeatures(
            geometric=geometric,
            hog=hog_summary,
            lbp=lbp_summary,
            color_hsv=color_hsv,
        )

        return ClassicalCVResponse(
            success=True,
            intermediate_images=intermediate_images,
            features=features,
            timings_ms=timings,
        )

