"""Measure the deployed classifier on detector crops, which is what the app really classifies.

    python scripts/evaluate_detector_crops.py

Every other deep accuracy in the report uses annotated boxes, which a real photo doesn't have. This
uses the saved runs of scripts/evaluate_detector.py, and applies the app's rule: boxes must score at
least 0.5, and an image with none is classified on its full frame. A full detector run replaces a
sample automatically, and the results say which was used.

Writes results/detector_crop_accuracy.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from data.stanford_cars import default_root, load_manifest  # noqa: E402
from detection.metrics import box_area  # noqa: E402
from evaluation import classifier_experiments as ex  # noqa: E402
from evaluation import confidence as conf  # noqa: E402
from recognition.onnx_classifier import OnnxClassifier  # noqa: E402

MIN_SCORE = 0.5


def app_boxes(path: Path, rule: str) -> dict:
    """The box the app would use per image, or None for its full-frame fallback."""
    boxes = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        confident = [item for item in record["detections"] if item["score"] >= MIN_SCORE]
        if not confident:
            boxes[record["image"]] = None
        elif rule == "largest_confident":
            boxes[record["image"]] = max(confident, key=lambda item: box_area(item["box"]))["box"]
        else:
            boxes[record["image"]] = max(confident, key=lambda item: item["score"])["box"]
    return boxes


def margin_95(share: float, images: int) -> float:
    return 1.96 * float(np.sqrt(share * (1 - share) / images))


def main() -> None:
    root = default_root()
    manifest = load_manifest(root)
    test_rows = manifest[manifest["split"] == "test"]
    metrics = json.loads((ROOT / "results" / "classifier_metrics.json").read_text(encoding="utf-8"))
    temperature, thresholds = metrics["calibration"]["temperature"], metrics["thresholds"]
    classifier = OnnxClassifier(threads=4)

    output = {"experiment": "detector_crop_accuracy", "model": "checkpoints/model_quantized.onnx (deployed)", "min_score": MIN_SCORE, "runs": {}}
    annotated_cache: dict = {}
    for variant in ("mobilenet", "resnet50"):
        runs = sorted((root.parent / "detector_runs").glob(f"detector_{variant}_test_*.jsonl"), key=lambda path: path.stat().st_size)
        if not runs:
            continue
        run = runs[-1]
        chosen = app_boxes(run, "largest_confident")
        rows = test_rows[test_rows["path"].isin(chosen)]
        labels = rows["label"].to_numpy()
        key = tuple(rows["path"])
        if key not in annotated_cache:
            annotated_cache[key] = ex.run_batches(classifier.predict_proba, rows, root, "box")
        annotated = annotated_cache[key]
        crops = ex.run_batches(classifier.predict_proba, rows, root, "detector", chosen)
        top_score = ex.run_batches(classifier.predict_proba, rows, root, "detector", app_boxes(run, "top_score"))

        calibrated = conf.apply_temperature(crops, temperature)
        top, margin = conf.top_two(calibrated)
        states = conf.decide(top, margin, thresholds["confident_p1"], thresholds["confident_margin"], thresholds["reject_below_p1"])
        correct = calibrated.argmax(1) == labels
        result = ex.scores(crops, labels)
        output["runs"][variant] = {
            "detector_run": run.stem,
            "images": len(rows),
            "is_random_sample": len(rows) < len(test_rows),
            "images_with_no_box_used_full_frame": sum(chosen[path] is None for path in rows["path"]),
            "annotated_box": ex.scores(annotated, labels),
            "detector_crop": result,
            "detector_crop_top1_margin_95": margin_95(result["top1"], len(rows)),
            "top_score_rule": ex.scores(top_score, labels),
            "images_changed_by_rule": int((crops.argmax(1) != top_score.argmax(1)).sum()),
            "states_on_detector_crops": {
                state: {"share": float((states == state).mean()), "accuracy": float(correct[states == state].mean()) if (states == state).any() else None}
                for state in ("confident", "ambiguous", "rejected")
            },
        }
        print(variant, json.dumps(output["runs"][variant], indent=1), flush=True)
    ex.save_json(output, ROOT / "results" / "detector_crop_accuracy.json")


if __name__ == "__main__":
    main()
