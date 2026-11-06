"""Tests for evaluation logic. The stand-in classifier here only exercises code and writes to a temp folder."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from data.stanford_cars import parse_class_name
from evaluation import classifier_experiments as ex
from evaluation import confidence as conf
from evaluation.corruptions import CORRUPTIONS, apply


@pytest.fixture
def picture():
    return Image.fromarray(np.random.default_rng(0).integers(40, 220, (120, 200, 3), dtype=np.uint8))


@pytest.mark.parametrize("name", list(CORRUPTIONS))
@pytest.mark.parametrize("severity", [0, 1, 2])
def test_corruptions_keep_size_and_are_repeatable(picture, name, severity):
    first, second = apply(name, severity, picture, seed=5), apply(name, severity, picture, seed=5)
    assert first.size == picture.size and first.mode == "RGB"
    assert np.array_equal(first, second)
    assert not np.array_equal(first, picture)


def test_low_light_gets_darker_with_severity(picture):
    means = [np.asarray(apply("low_light", level, picture, 1)).mean() for level in range(3)]
    assert means[0] > means[1] > means[2] and means[0] < np.asarray(picture).mean()


def test_occlusion_covers_about_the_requested_area(picture):
    changed = np.any(np.asarray(apply("occlusion", 1, picture, 1)) != np.asarray(picture), axis=2).mean()
    assert changed == pytest.approx(0.25, abs=0.05)


def test_temperature_does_not_change_predictions_and_reduces_overconfidence():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 10, 2000)
    logits = rng.normal(0, 1, (2000, 10))
    logits[np.arange(2000), labels] += 2.0
    overconfident = conf.softmax(logits * 4)
    temperature = conf.fit_temperature(overconfident, labels)
    calibrated = conf.apply_temperature(overconfident, temperature)
    assert temperature > 1
    assert np.array_equal(calibrated.argmax(1), overconfident.argmax(1))
    assert conf.expected_calibration_error(calibrated, labels) < conf.expected_calibration_error(overconfident, labels)


def test_topk_accuracy():
    probabilities = np.array([[0.7, 0.2, 0.1], [0.5, 0.3, 0.2]])
    labels = np.array([0, 1])
    assert conf.topk_accuracy(probabilities, labels, 1) == 0.5
    assert conf.topk_accuracy(probabilities, labels, 2) == 1.0


def test_coverage_falls_as_threshold_rises():
    rng = np.random.default_rng(0)
    probabilities = rng.dirichlet(np.ones(5) * 0.3, 500)
    curve = conf.coverage_curve(probabilities, rng.integers(0, 5, 500))
    coverage = [row["coverage"] for row in curve]
    assert coverage[0] == 1.0 and coverage == sorted(coverage, reverse=True)


def test_choose_threshold_keeps_most_images_or_returns_none():
    curve = [
        {"threshold": 0.1, "coverage": 1.0, "accuracy_when_accepted": 0.8},
        {"threshold": 0.5, "coverage": 0.7, "accuracy_when_accepted": 0.95},
        {"threshold": 0.9, "coverage": 0.2, "accuracy_when_accepted": 0.99},
    ]
    assert conf.choose_threshold(curve, 0.95) == 0.5
    assert conf.choose_threshold(curve, 0.999) is None


def test_decide_states():
    states = conf.decide(np.array([0.9, 0.9, 0.5, 0.1]), np.array([0.6, 0.05, 0.3, 0.05]), 0.8, 0.2, 0.2)
    assert states.tolist() == ["confident", "ambiguous", "ambiguous", "rejected"]


def test_error_breakdown():
    classes = ["Audi A4 Sedan 2007", "Audi A4 Sedan 2012", "Audi TT Coupe 2012", "BMW M3 Coupe 2012"]
    labels = np.array([0, 0, 0, 0])
    predictions = np.array([0, 1, 2, 3])
    result = ex.error_breakdown(predictions, labels, classes, parse_class_name)
    assert result["errors"] == 3
    assert result["same_make_and_model_different_year"] == 1
    assert result["same_make_different_model"] == 1
    assert result["different_make"] == 1


def test_a_different_body_type_is_not_a_year_only_error():
    classes = ["Audi 100 Sedan 1994", "Audi 100 Wagon 1994"]
    result = ex.error_breakdown(np.array([1]), np.array([0]), classes, parse_class_name)
    assert result["same_make_and_model_different_year"] == 0
    assert result["same_make_different_model"] == 1


def test_confidence_experiment_uses_the_shipped_settings():
    probabilities = np.array([[0.9, 0.1], [0.6, 0.4], [0.3, 0.7], [0.51, 0.49]])
    labels = np.array([0, 1, 1, 0])
    thresholds = {"confident_p1": 0.8, "confident_margin": 0.3, "reject_below_p1": 0.55}
    result = ex.confidence_experiment(probabilities, labels, 1.0, thresholds)
    assert result["temperature"] == 1.0 and result["thresholds"] == thresholds
    assert result["test_states"]["confident"]["share"] == 0.25
    assert result["test_states"]["rejected"]["share"] == 0.25
    assert result["test_states"]["ambiguous"]["share"] == 0.5


def test_hybrid_detects_useful_classical_features_and_never_uses_test_to_choose():
    from classical_cv import feature_blocks

    rng = np.random.default_rng(0)
    labels = rng.integers(0, 5, 600)
    size = sum(feature_blocks().values())
    classical = rng.normal(0, 1, (600, size)).astype(np.float32)
    classical[:, :32] += np.eye(5)[labels].repeat(7, axis=1)[:, :32] * 3
    embeddings = rng.normal(0, 1, (600, 16)).astype(np.float32)
    masks = {"train": np.arange(600) < 400, "val": (np.arange(600) >= 400) & (np.arange(600) < 500), "test": np.arange(600) >= 500}
    result = ex.hybrid_experiment(embeddings, classical, labels, masks)
    assert result["fusion_helped_on_validation"]
    assert result["fusion_change_in_test_top1"] > 0
    assert set(result["test"]) == {"deep_only", result["chosen_fusion"]}


def test_failure_candidates_keep_only_errors_and_sort_by_confidence():
    rows = pd.DataFrame({"path": ["a", "b", "c"], "label": [0, 1, 1], "class_name": ["x", "y", "y"], "x1": 0, "y1": 0, "x2": 10, "y2": 10, "width": 10, "height": 20})
    probabilities = np.array([[0.9, 0.1], [0.6, 0.4], [0.95, 0.05]])
    table = ex.failure_candidates(rows, probabilities, ["x", "y"])
    assert table["path"].tolist() == ["c", "b"]
    assert table["box_area_share"].iloc[0] == 0.5


class StandIn:
    """Exercises the plumbing only. Its outputs are noise and go to a temporary folder."""

    def predict_proba(self, images):
        return np.random.default_rng(len(images)).dirichlet(np.ones(196), len(images))

    def embed(self, images):
        return np.random.default_rng(len(images)).normal(size=(len(images), 8))


def test_robustness_runs_every_condition(tmp_path, picture):
    for name in ("a.jpg", "b.jpg"):
        picture.save(tmp_path / name)
    rows = pd.DataFrame({"path": ["a.jpg", "b.jpg"], "label": [0, 1], "x1": 5, "y1": 5, "x2": 100, "y2": 100, "width": 200, "height": 120})
    result = ex.robustness_experiment(StandIn(), rows, tmp_path)
    assert len(result["conditions"]) == 18
    dark = [row for row in result["conditions"] if row["corruption"] == "low_light"]
    assert all("deep_with_adaptive_clahe" in row for row in dark)
