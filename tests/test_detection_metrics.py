import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from detection.metrics import average_precision, iou, mean_average_precision, pick_largest_confident, top_box_summary

BOX = (0, 0, 100, 100)
TRUTHS = {"a": BOX, "b": BOX}


def detection(image, score, box=BOX):
    return {"image": image, "score": score, "box": box}


def test_iou_values():
    assert iou(BOX, BOX) == 1.0
    assert iou(BOX, (200, 200, 300, 300)) == 0.0
    assert iou((0, 0, 100, 100), (50, 0, 150, 100)) == pytest.approx(1 / 3)


def test_iou_of_empty_boxes_is_zero():
    assert iou((5, 5, 5, 5), (5, 5, 5, 5)) == 0.0


def test_perfect_detections_score_one():
    result = mean_average_precision([detection("a", 0.9), detection("b", 0.8)], TRUTHS)
    assert result["map_50_95"] == pytest.approx(1.0)


def test_no_detections_score_zero():
    assert mean_average_precision([], TRUTHS)["map_50_95"] == 0.0


def test_missing_half_the_objects_limits_ap():
    # Recall stops at 0.5, so 51 of the 101 recall levels score precision 1.
    assert average_precision([detection("a", 0.9)], TRUTHS, 0.5) == pytest.approx(51 / 101)


def test_duplicate_detection_counts_as_false_positive():
    clean = average_precision([detection("a", 0.9), detection("b", 0.8)], TRUTHS, 0.5)
    duplicate = average_precision([detection("a", 0.9), detection("a", 0.85), detection("b", 0.8)], TRUTHS, 0.5)
    assert clean == pytest.approx(1.0)
    assert duplicate < clean


def test_higher_threshold_rejects_a_loose_box():
    loose = (0, 0, 100, 60)
    detections = [detection("a", 0.9, loose), detection("b", 0.9, loose)]
    assert average_precision(detections, TRUTHS, 0.5) == pytest.approx(1.0)
    assert average_precision(detections, TRUTHS, 0.75) == 0.0


def test_top_box_uses_the_highest_scoring_detection():
    summary = top_box_summary([detection("a", 0.2), detection("a", 0.9, (200, 200, 300, 300)), detection("b", 0.5)], TRUTHS)
    assert summary["mean_iou"] == pytest.approx(0.5)
    assert summary["images_with_no_vehicle_found"] == 0


def test_largest_confident_prefers_the_main_car_over_a_tiny_high_scoring_one():
    tiny = detection("a", 0.99, (0, 0, 5, 5))
    main = detection("a", 0.90, BOX)
    assert pick_largest_confident([tiny, main]) is main
    assert top_box_summary([tiny, main], {"a": BOX}, pick_largest_confident)["mean_iou"] == 1.0
    assert top_box_summary([tiny, main], {"a": BOX})["mean_iou"] == pytest.approx(25 / 10000)


def test_largest_confident_ignores_large_weak_boxes_and_falls_back_to_top_score():
    weak_large = detection("a", 0.2, (0, 0, 500, 500))
    small_strong = detection("a", 0.8, (0, 0, 5, 5))
    assert pick_largest_confident([weak_large, small_strong]) is small_strong
    weaker = detection("a", 0.1, (0, 0, 500, 500))
    assert pick_largest_confident([weak_large, weaker]) is weak_large


def test_app_selection_matches_the_evaluated_rule():
    pytest.importorskip("torchvision")
    from detection.detector import Detection, select_primary

    tiny, main = Detection((0, 0, 5, 5), 0.99, "car"), Detection((0, 0, 100, 100), 0.9, "car")
    assert select_primary([tiny, main]) is main
    assert select_primary([]) is None


def test_top_box_counts_missed_images_as_zero():
    summary = top_box_summary([detection("a", 0.9)], TRUTHS)
    assert summary["mean_iou"] == pytest.approx(0.5)
    assert summary["images_with_no_vehicle_found"] == 1
