"""Run the experiments that need Person A's classifier.

    python scripts/evaluate_with_classifier.py --from-artifacts-only
    python scripts/evaluate_with_classifier.py --device cuda

Needs from the team drive:
    artifacts/logits_{train,val,test}.npz and artifacts/embeddings_{train,val,test}.npz
    checkpoints/apex3d_convnext_tiny.pt, except with --from-artifacts-only

Using the saved logits and embeddings means hybrid, confidence and failure analysis describe exactly
the predictions in notebook 01. --from-artifacts-only runs just those, and needs no model. The full
run adds the crop ablations and robustness, which have to run the model on changed images.

Writes to results/: deep_eval.json, hybrid.json, confidence.json, failure_candidates.csv,
figures/failure_cases.jpg, and, in a full run, ablation_inference.json and robustness.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from classical_cv.batch import compute_features  # noqa: E402
from data.stanford_cars import default_root, load_class_names, load_manifest, parse_class_name  # noqa: E402
from detection.metrics import pick_largest_confident  # noqa: E402
from evaluation import classifier_experiments as ex  # noqa: E402
from evaluation import confidence as conf  # noqa: E402

CLASSICAL_CACHE = "classical_crop8.npy"


def load_artifact(directory: Path, stem: str, key: str, manifest) -> np.ndarray:
    """Saved arrays in manifest order. Stops if a file or label disagrees with the manifest.

    Train and test both contain files such as 00001.jpg, so each split is matched separately.
    """
    output = None
    for split in ("train", "val", "test"):
        path = directory / f"{stem}_{split}.npz"
        if not path.exists():
            raise SystemExit(f"{path} not found. Copy artifacts/ from the team drive.")
        with np.load(path) as saved:
            values, labels, files = saved[key], saved["labels"], saved["files"].astype(str)
        position = {name: index for index, name in enumerate(files)}
        rows = manifest[manifest["split"] == split]
        if set(position) != set(rows["filename"]):
            raise SystemExit(f"{path.name}: its files don't match the {split} split.")
        order = np.array([position[name] for name in rows["filename"]])
        if not np.array_equal(labels[order], rows["label"].to_numpy()):
            raise SystemExit(f"{path.name}: its labels don't match the manifest.")
        if output is None:
            output = np.zeros((len(manifest), values.shape[1]), dtype=np.float32)
        output[(manifest["split"] == split).to_numpy()] = values[order]
    return output


def load_detector_boxes(root: Path) -> tuple[dict, str | None]:
    """The box the app would choose per image, from the largest finished test-set detector run."""
    runs = sorted((root.parent / "detector_runs").glob("detector_*_test_*.jsonl"), key=lambda path: path.stat().st_size)
    if not runs:
        return {}, None
    boxes = {}
    for line in runs[-1].read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        boxes[record["image"]] = pick_largest_confident(record["detections"])["box"] if record["detections"] else None
    return boxes, runs[-1].stem


def from_saved_predictions(manifest, classes, root: Path, artifacts: Path, output: Path, temperature: float, thresholds: dict) -> None:
    labels = manifest["label"].to_numpy()
    masks = {split: (manifest["split"] == split).to_numpy() for split in ("train", "val", "test")}
    logits = load_artifact(artifacts, "logits", "logits", manifest)
    probabilities = conf.softmax(logits)
    test_probabilities, y_test = probabilities[masks["test"]], labels[masks["test"]]
    test_rows = manifest[masks["test"]]

    print("Deep evaluation...", flush=True)
    ex.save_json(
        {
            "experiment": "deep_evaluation",
            "source": "artifacts/logits_*.npz from notebook 01",
            "crop": "annotated box + 8% margin",
            "test_images": len(test_rows),
            "test": ex.scores(test_probabilities, y_test),
            "validation": ex.scores(probabilities[masks["val"]], labels[masks["val"]]),
            "error_breakdown": ex.error_breakdown(test_probabilities.argmax(1), y_test, classes, parse_class_name),
        },
        output / "deep_eval.json",
    )

    print("Confidence analysis...", flush=True)
    ex.save_json({"experiment": "confidence", **ex.confidence_experiment(test_probabilities, y_test, temperature, thresholds)}, output / "confidence.json")

    print("Failure candidates...", flush=True)
    calibrated = conf.apply_temperature(test_probabilities, temperature)
    failures = ex.failure_candidates(test_rows, calibrated, classes)
    columns = ["path", "class_name", "predicted", "confidence", "margin", "true_class_probability", "box_area_share", "width", "height"]
    output.mkdir(parents=True, exist_ok=True)
    failures[columns].to_csv(output / "failure_candidates.csv", index=False)
    ex.failure_sheet(failures, root, output / "figures" / "failure_cases.jpg")

    print("Hybrid experiment...", flush=True)
    embeddings = load_artifact(artifacts, "embeddings", "features", manifest)
    classical = compute_features(manifest, root, root.parent / "features" / CLASSICAL_CACHE)
    ex.save_json({"experiment": "hybrid", "embeddings": "artifacts/embeddings_*.npz", **ex.hybrid_experiment(embeddings, classical, labels, masks)}, output / "hybrid.json")


def with_model(classifier, manifest, root: Path, output: Path, robustness_images: int) -> None:
    test_rows = manifest[manifest["split"] == "test"]
    y_test = test_rows["label"].to_numpy()

    print("Crop ablations...", flush=True)
    test_box = ex.run_batches(classifier.predict_proba, test_rows, root, "box")
    ablation = {"experiment": "ablation_inference", "with_annotated_box_crop": ex.scores(test_box, y_test)}
    ablation["without_any_crop_full_image"] = ex.scores(ex.run_batches(classifier.predict_proba, test_rows, root, "none"), y_test)
    boxes, run_name = load_detector_boxes(root)
    if boxes:
        sample = test_rows[test_rows["path"].isin(boxes)]
        y_sample = sample["label"].to_numpy()
        ablation["detector_crop"] = {
            "detector_run": run_name,
            "selection_rule": "largest_confident",
            "images": len(sample),
            "with_detector_crop": ex.scores(ex.run_batches(classifier.predict_proba, sample, root, "detector", boxes), y_sample),
            "with_annotated_box_on_same_images": ex.scores(ex.run_batches(classifier.predict_proba, sample, root, "box"), y_sample),
            "images_with_no_detection_used_full_image": sum(boxes[path] is None for path in sample["path"]),
        }
    ex.save_json(ablation, output / "ablation_inference.json")

    print("Robustness...", flush=True)
    sample = test_rows.sample(min(robustness_images, len(test_rows)), random_state=42)
    ex.save_json({"experiment": "robustness", **ex.robustness_experiment(classifier, sample, root, load_classical_predict(root))}, output / "robustness.json")


def load_classical_predict(root: Path):
    import cv2
    import joblib

    from classical_cv import extract_features

    path = root.parent / "models" / "classical_baseline.joblib"
    settings = ROOT / "results" / "classical_baseline.json"
    if not path.exists() or not settings.exists():
        return None
    model = joblib.load(path)
    # The model must receive features prepared the same way as those it was trained on.
    enhance = json.loads(settings.read_text(encoding="utf-8"))["adaptive_clahe"]

    def predict(images):
        rows = [extract_features(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR), enhance=enhance) for image in images]
        return model.decision_function(np.vstack(rows))

    return predict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--robustness-images", type=int, default=1000)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--from-artifacts-only", action="store_true")
    args = parser.parse_args()

    root = default_root()
    manifest, classes = load_manifest(root), load_class_names(root)
    output = ROOT / "results"
    metrics = json.loads((output / "classifier_metrics.json").read_text(encoding="utf-8"))
    temperature, thresholds = metrics["calibration"]["temperature"], metrics["thresholds"]
    from_saved_predictions(manifest, classes, root, args.artifacts, output, temperature, thresholds)
    if not args.from_artifacts_only:
        from recognition.classifier import load_classifier

        with_model(load_classifier(args.device), manifest, root, output, args.robustness_images)
    print(f"Done. Results are in {output}", flush=True)


if __name__ == "__main__":
    main()
