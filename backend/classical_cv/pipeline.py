"""Classical computer-vision pipeline for Apex3D (P5 section 3A).

All functions take and return BGR uint8 images unless stated otherwise, matching OpenCV.

Two entry points:
    extract_features(image)  fixed-length vector for the classical and hybrid classifiers
    run_pipeline(image)      intermediate images and readable values for the app
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

FEATURE_SIZE = 128
LOW_LIGHT_LUMA = 70.0

HSV_BINS = (16, 8, 8)
LBP_POINTS = 8
LBP_RADIUS = 1
LBP_BINS = LBP_POINTS + 2
LBP_GRID = 4
HOG_ORIENTATIONS = 9
HOG_CELL = 16
HOG_BLOCK = 2
GEOMETRY_NAMES = (
    "aspect_ratio",
    "extent",
    "solidity",
    "area_fraction",
    "edge_density",
    "hu_1",
    "hu_2",
    "hu_3",
    "hu_4",
    "hu_5",
    "hu_6",
    "hu_7",
)


def validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Expected a uint8 NumPy image.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a 3-channel BGR image.")
    if min(image.shape[:2]) < 8:
        raise ValueError("Image is too small to analyse (minimum side is 8 pixels).")


def normalize(image: np.ndarray, size: int = FEATURE_SIZE) -> np.ndarray:
    """Resize to a fixed square so that features have a fixed length."""
    validate_image(image)
    interpolation = cv2.INTER_AREA if max(image.shape[:2]) > size else cv2.INTER_LINEAR
    return cv2.resize(image, (size, size), interpolation=interpolation)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, grid: int = 8) -> np.ndarray:
    """Equalise contrast on the LAB lightness channel, leaving colour unchanged."""
    lightness, a, b = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
    return cv2.cvtColor(cv2.merge((clahe.apply(lightness), a, b)), cv2.COLOR_LAB2BGR)


def mean_luma(image: np.ndarray) -> float:
    return float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean())


def is_low_light(image: np.ndarray, threshold: float = LOW_LIGHT_LUMA) -> bool:
    return mean_luma(image) < threshold


def adaptive_enhance(image: np.ndarray, threshold: float = LOW_LIGHT_LUMA) -> np.ndarray:
    """Apply CLAHE only when the image is dark. The hybrid experiments test whether this helps."""
    return apply_clahe(image) if is_low_light(image, threshold) else image


def canny_edges(image: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """Canny with thresholds set from the median intensity of the image."""
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    median = float(np.median(gray))
    low = int(max(0, (1.0 - sigma) * median))
    high = int(min(255, (1.0 + sigma) * median))
    return cv2.Canny(gray, low, max(high, low + 1))


def largest_contour(edges: np.ndarray) -> np.ndarray | None:
    """Largest external contour after closing gaps in the edge map."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea) if contours else None


def contour_roi(image: np.ndarray, contour: np.ndarray | None, min_area_fraction: float = 0.05) -> tuple[int, int, int, int]:
    """Bounding box (x1, y1, x2, y2) of a contour, or the full frame if it is too small."""
    height, width = image.shape[:2]
    if contour is not None:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h >= min_area_fraction * width * height:
            return x, y, x + w, y + h
    return 0, 0, width, height


def hsv_histogram(image: np.ndarray) -> np.ndarray:
    """Concatenated, normalised hue, saturation and value histograms."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ranges = ((0, 180), (0, 256), (0, 256))
    parts = []
    for channel, (bins, value_range) in enumerate(zip(HSV_BINS, ranges)):
        histogram = cv2.calcHist([hsv], [channel], None, [bins], list(value_range)).ravel()
        parts.append(histogram / max(histogram.sum(), 1.0))
    return np.concatenate(parts).astype(np.float32)


def lbp_map(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")


def lbp_histogram(image: np.ndarray, grid: int = LBP_GRID) -> np.ndarray:
    """Uniform LBP histograms over a spatial grid, which keeps coarse layout information."""
    lbp = lbp_map(image)
    height, width = lbp.shape
    parts = []
    for row in range(grid):
        for column in range(grid):
            cell = lbp[row * height // grid : (row + 1) * height // grid, column * width // grid : (column + 1) * width // grid]
            histogram, _ = np.histogram(cell, bins=LBP_BINS, range=(0, LBP_BINS))
            parts.append(histogram / max(histogram.sum(), 1))
    return np.concatenate(parts).astype(np.float32)


def hog_features(image: np.ndarray, visualize: bool = False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=(HOG_CELL, HOG_CELL),
        cells_per_block=(HOG_BLOCK, HOG_BLOCK),
        block_norm="L2-Hys",
        visualize=visualize,
        feature_vector=True,
    )


def geometric_features(image: np.ndarray, edges: np.ndarray | None = None, contour: np.ndarray | None = None) -> np.ndarray:
    """Shape descriptors of the largest contour. All values are zero if none is found."""
    edges = canny_edges(image) if edges is None else edges
    contour = largest_contour(edges) if contour is None else contour
    height, width = image.shape[:2]
    edge_density = float(np.count_nonzero(edges)) / edges.size
    if contour is None or cv2.contourArea(contour) <= 0:
        values = np.zeros(len(GEOMETRY_NAMES), dtype=np.float32)
        values[4] = edge_density
        return values
    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    hu = cv2.HuMoments(cv2.moments(contour)).ravel()
    # Hu moments span many orders of magnitude, so they are log-scaled.
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    shape = [w / max(h, 1), area / max(w * h, 1), area / max(hull_area, 1e-6), area / (width * height), edge_density]
    return np.nan_to_num(np.concatenate((shape, hu))).astype(np.float32)


def feature_blocks() -> dict[str, int]:
    """Length of each feature family, in the order used by extract_features."""
    cells = FEATURE_SIZE // HOG_CELL
    blocks = cells - HOG_BLOCK + 1
    return {
        "hsv": sum(HSV_BINS),
        "lbp": LBP_BINS * LBP_GRID**2,
        "hog": blocks * blocks * HOG_BLOCK * HOG_BLOCK * HOG_ORIENTATIONS,
        "geometry": len(GEOMETRY_NAMES),
    }


def extract_features(image: np.ndarray, enhance: bool = False) -> np.ndarray:
    """Fixed-length vector of HSV, LBP, HOG and geometric features."""
    resized = normalize(image)
    if enhance:
        resized = adaptive_enhance(resized)
    return np.concatenate(
        (hsv_histogram(resized), lbp_histogram(resized), hog_features(resized).astype(np.float32), geometric_features(resized))
    )


@dataclass
class PipelineResult:
    images: dict[str, np.ndarray]
    roi: tuple[int, int, int, int]
    low_light: bool
    mean_luma: float
    dominant_hue_bin: int
    geometry: dict[str, float]
    feature_length: int


def run_pipeline(image: np.ndarray, max_side: int = 640) -> PipelineResult:
    """Run every stage on an image and keep the intermediates for display."""
    validate_image(image)
    scale = max_side / max(image.shape[:2])
    display = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image.copy()

    enhanced = apply_clahe(display)
    edges = canny_edges(display)
    contour = largest_contour(edges)
    roi = contour_roi(display, contour)

    overlay = display.copy()
    if contour is not None:
        cv2.drawContours(overlay, [contour], -1, (255, 200, 0), 2)
    cv2.rectangle(overlay, roi[:2], roi[2:], (0, 200, 255), 2)

    _, hog_image = hog_features(normalize(display, 256), visualize=True)
    hog_view = cv2.normalize(hog_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    lbp_view = (lbp_map(display) / (LBP_BINS - 1) * 255).astype(np.uint8)
    hue, saturation, value = cv2.split(cv2.cvtColor(display, cv2.COLOR_BGR2HSV))

    geometry = geometric_features(display, edges, contour)
    return PipelineResult(
        images={
            "original": display,
            "clahe": enhanced,
            "edges": edges,
            "contours_roi": overlay,
            "roi_crop": display[roi[1] : roi[3], roi[0] : roi[2]],
            "hsv_hue": hue,
            "hsv_saturation": saturation,
            "hsv_value": value,
            "lbp": lbp_view,
            "hog": hog_view,
        },
        roi=roi,
        low_light=is_low_light(display),
        mean_luma=mean_luma(display),
        dominant_hue_bin=int(np.argmax(hsv_histogram(display)[: HSV_BINS[0]])),
        geometry={name: float(value) for name, value in zip(GEOMETRY_NAMES, geometry)},
        feature_length=sum(feature_blocks().values()),
    )
