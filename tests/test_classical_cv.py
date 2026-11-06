import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from classical_cv import api_response, apply_clahe, canny_edges, extract_features, feature_blocks, is_low_light, run_pipeline
from classical_cv.pipeline import contour_roi, geometric_features, hsv_histogram, largest_contour, lbp_histogram


@pytest.fixture
def scene():
    """A dark-blue rectangle with wheels on a grey background."""
    image = np.full((240, 400, 3), 130, dtype=np.uint8)
    cv2.rectangle(image, (80, 90), (320, 180), (160, 40, 20), -1)
    cv2.circle(image, (130, 185), 25, (15, 15, 15), -1)
    cv2.circle(image, (270, 185), 25, (15, 15, 15), -1)
    return image


def test_feature_vector_length_matches_blocks(scene):
    features = extract_features(scene)
    assert features.shape == (sum(feature_blocks().values()),)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_feature_length_does_not_depend_on_input_size(scene):
    small = cv2.resize(scene, (97, 53))
    assert extract_features(small).shape == extract_features(scene).shape


def test_different_images_give_different_features(scene):
    assert not np.allclose(extract_features(scene), extract_features(cv2.flip(scene, 0)))


def test_histograms_are_normalised(scene):
    assert hsv_histogram(scene)[:16].sum() == pytest.approx(1.0, abs=1e-5)
    assert lbp_histogram(scene)[:10].sum() == pytest.approx(1.0, abs=1e-5)


def test_clahe_raises_contrast_on_a_dim_gradient():
    gradient = np.tile(np.linspace(30, 70, 256).astype(np.uint8), (256, 1))
    image = cv2.merge((gradient, gradient, gradient))
    assert cv2.cvtColor(apply_clahe(image), cv2.COLOR_BGR2GRAY).std() > gradient.std()


def test_low_light_detection(scene):
    assert is_low_light((scene * 0.2).astype(np.uint8))
    assert not is_low_light(scene)


def test_contour_roi_finds_the_object(scene):
    x1, y1, x2, y2 = contour_roi(scene, largest_contour(canny_edges(scene)))
    assert x1 <= 90 and x2 >= 310
    assert y1 <= 100 and y2 >= 175


def test_roi_falls_back_to_full_frame_on_a_blank_image():
    blank = np.full((100, 200, 3), 128, dtype=np.uint8)
    assert contour_roi(blank, largest_contour(canny_edges(blank))) == (0, 0, 200, 100)


def test_geometry_on_a_blank_image_is_zero_not_nan():
    geometry = geometric_features(np.full((100, 100, 3), 128, dtype=np.uint8))
    assert np.isfinite(geometry).all()
    assert geometry[:4].sum() == 0


def test_pipeline_returns_every_stage(scene):
    result = run_pipeline(scene)
    assert set(result.images) == {
        "original", "clahe", "edges", "contours_roi", "roi_crop", "hsv_hue", "hsv_saturation", "hsv_value", "lbp", "hog",
    }
    assert all(view.dtype == np.uint8 and view.size > 0 for view in result.images.values())
    assert result.feature_length == extract_features(scene).size


def test_pipeline_downsizes_large_images():
    large = np.random.default_rng(0).integers(0, 255, (1600, 2400, 3), dtype=np.uint8)
    assert max(run_pipeline(large).images["original"].shape[:2]) == 640


def test_api_response_matches_the_frontend_contract(scene):
    response = api_response(scene)
    assert {step["key"] for step in response["steps"]} >= {"original", "clahe", "edges", "hog"}
    assert all(step["image"].startswith("data:image/jpeg;base64,") and step["title"] for step in response["steps"])
    for value in response["features"].values():
        assert isinstance(value, (int, float, str)) or all(isinstance(item, float) for item in value)
    assert len(response["features"]["hsv_histogram"]) <= 64
    assert response["timings_ms"]["total"] > 0


def test_api_response_rejects_bad_input():
    with pytest.raises(ValueError):
        api_response(np.zeros((3, 3, 3), dtype=np.uint8))


@pytest.mark.parametrize(
    "bad",
    [np.zeros((10, 10), dtype=np.uint8), np.zeros((10, 10, 3), dtype=np.float32), np.zeros((4, 4, 3), dtype=np.uint8), "not an image"],
)
def test_bad_input_is_rejected(bad):
    with pytest.raises(ValueError):
        extract_features(bad)
