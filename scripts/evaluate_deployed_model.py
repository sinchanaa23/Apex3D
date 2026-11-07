"""Run the experiments that need a classifier, using the deployed ONNX model, with no GPU and no PyTorch checkpoint.

    python scripts/evaluate_deployed_model.py --stages predictions confidence hybrid robustness

This does what scripts/evaluate_with_classifier.py does for hybrid, confidence and robustness, with the same
experiment code (backend/evaluation/classifier_experiments.py), but runs checkpoints/model_quantized.onnx, the model
the app serves. So the numbers describe what is deployed, including its static INT8 quantisation. The ONNX file has
the temperature baked in, so predictions are un-scaled first (see backend/recognition/onnx_classifier.py).

Stages (each writes its own file in results/, and later stages reuse the cached predictions):
    predictions   probabilities and embeddings for every image, cached under ~/.apex3d/features/
    confidence    results/confidence.json   coverage against accuracy at every threshold, and the shipped states
    hybrid        results/hybrid.json       deep alone against deep plus classical features
    robustness    results/robustness.json   accuracy under brightness, low light, blur, noise, resolution, occlusion
    ablation      results/ablation_inference.json   the annotated crop against no crop at all
    failures      results/failure_candidates.csv and results/figures/classifier_failures.jpg   the most confident wrong answers

The classical baseline used for the classical column of the robustness test is refitted here exactly as in
notebooks/02_classical_baseline.ipynb (cell 17), because that notebook's saved model is not in Git. Its published
results are not touched.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from classical_cv.batch import compute_features  # noqa: E402
from data.stanford_cars import default_root, load_class_names, load_manifest  # noqa: E402
from evaluation import confidence as conf  # noqa: E402
from evaluation import classifier_experiments as ex  # noqa: E402
from recognition.onnx_classifier import OnnxClassifier  # noqa: E402

RESULTS = ROOT / "results"


def masks_of(manifest):
    return {split: (manifest["split"] == split).to_numpy() for split in ("train", "val", "test")}


def predictions(classifier, manifest, root: Path, cache: Path):
    """Uncalibrated probabilities and 768-value embeddings for every image, in manifest order."""
    probabilities_path, embeddings_path = cache / "onnx_probabilities.npy", cache / "onnx_embeddings.npy"
    if probabilities_path.exists() and embeddings_path.exists():
        return np.load(probabilities_path), np.load(embeddings_path)

    cache.mkdir(parents=True, exist_ok=True)
    probabilities, embeddings, batch = [], [], []
    started = time.perf_counter()

    def flush():
        p, e = classifier.predict_and_embed(batch)
        probabilities.append(p)
        embeddings.append(e)
        batch.clear()

    for position, row in enumerate(manifest.itertuples()):
        batch.append(ex.load_image(root, row, "box"))
        if len(batch) == ex.BATCH_SIZE:
            flush()
        if position % 1600 == 0:
            print(f"  {position}/{len(manifest)} images, {time.perf_counter() - started:.0f}s", flush=True)
    if batch:
        flush()
    probabilities, embeddings = np.vstack(probabilities), np.vstack(embeddings)
    np.save(probabilities_path, probabilities)
    np.save(embeddings_path, embeddings)
    return probabilities, embeddings


def fit_classical_baseline(manifest, root: Path, cache: Path):
    """The classical model exactly as notebook 02 fits it for the final test: train plus validation, chosen C."""
    import joblib
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    settings = json.loads((RESULTS / "classical_baseline.json").read_text(encoding="utf-8"))
    enhance = bool(settings["adaptive_clahe"])
    features = compute_features(manifest, root, cache / ("classical_crop8_enhanced.npy" if enhance else "classical_crop8.npy"), enhance=enhance)
    masks = masks_of(manifest)
    development = masks["train"] | masks["val"]
    labels = manifest["label"].to_numpy()
    steps = [StandardScaler()]
    if features.shape[1] > 256:
        steps.append(PCA(256, random_state=42))
    steps.append(LinearSVC(C=settings["chosen_C"], dual=False, max_iter=2000, random_state=42))
    model = make_pipeline(*steps).fit(features[development], labels[development])
    accuracy = float((model.decision_function(features[masks["test"]]).argmax(1) == labels[masks["test"]]).mean())
    print(f"  classical baseline refitted: test top-1 {accuracy:.4f} (published: {settings['test']['top1']:.4f})", flush=True)
    model_dir = cache.parent / "models"
    model_dir.mkdir(exist_ok=True)
    joblib.dump(model, model_dir / "classical_baseline.joblib")
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", nargs="+", default=["predictions", "confidence", "hybrid", "robustness", "ablation", "failures"])
    parser.add_argument("--robustness-images", type=int, default=1000)
    args = parser.parse_args()

    root = default_root()
    manifest = load_manifest(root)
    cache = root.parent / "features"
    masks = masks_of(manifest)
    labels = manifest["label"].to_numpy()
    y_test = labels[masks["test"]]
    classifier = OnnxClassifier()
    metrics = json.loads((RESULTS / "classifier_metrics.json").read_text(encoding="utf-8"))

    print("Predictions...", flush=True)
    probabilities, embeddings = predictions(classifier, manifest, root, cache)
    test_scores = ex.scores(probabilities[masks["test"]], y_test)
    print(f"  deployed model, test: top-1 {test_scores['top1']:.4f}, top-5 {test_scores['top5']:.4f} "
          f"(published deployed: {metrics['test']['top1']:.4f})", flush=True)
    # A wrong class order or a join error would show up here as a collapse far below the published figure.
    if abs(test_scores["top1"] - metrics["test"]["top1"]) > 0.02:
        raise SystemExit("Top-1 differs from the published figure by more than 2 points. Stopping before writing results.")

    if "failures" in args.stages:
        print("Failure candidates (most confident wrong answers)...", flush=True)
        classes = load_class_names(root)
        test_rows = manifest[masks["test"]]
        calibrated = conf.apply_temperature(probabilities[masks["test"]], metrics["calibration"]["temperature"])
        candidates = ex.failure_candidates(test_rows, calibrated, classes)
        columns = ["path", "class_name", "predicted", "confidence", "margin", "true_class_probability", "box_area_share", "width", "height"]
        candidates[columns].to_csv(RESULTS / "failure_candidates.csv", index=False)
        ex.failure_sheet(candidates, root, RESULTS / "figures" / "classifier_failures.jpg")
        print(f"  {len(candidates)} wrong answers of {len(test_rows)} test images", flush=True)

    if "confidence" in args.stages:
        print("Confidence and threshold analysis...", flush=True)
        test_probabilities = probabilities[masks["test"]]
        temperature = metrics["calibration"]["temperature"]

        def nll(p):  # mean negative log-likelihood of the true class
            return float(-np.log(np.clip(p[np.arange(len(y_test)), y_test], 1e-12, None)).mean())

        ex.save_json(
            {"experiment": "confidence", "model": "checkpoints/model_quantized.onnx (deployed)",
             **ex.confidence_experiment(test_probabilities, y_test, temperature, metrics["thresholds"]),
             "nll_before": nll(test_probabilities), "nll_after": nll(conf.apply_temperature(test_probabilities, temperature))},
            RESULTS / "confidence.json",
        )

    classical = None
    if "hybrid" in args.stages:
        print("Hybrid experiment...", flush=True)
        classical = fit_classical_baseline(manifest, root, cache)
        ex.save_json(
            {"experiment": "hybrid", "embeddings": "penultimate layer of checkpoints/model_quantized.onnx (deployed)",
             **ex.hybrid_experiment(embeddings, classical, labels, masks)},
            RESULTS / "hybrid.json",
        )

    if "robustness" in args.stages:
        print("Robustness...", flush=True)
        if classical is None:
            fit_classical_baseline(manifest, root, cache)
        sys.path.insert(0, str(ROOT / "scripts"))
        from evaluate_with_classifier import load_classical_predict  # noqa: E402

        test_rows = manifest[masks["test"]]
        sample = test_rows.sample(min(args.robustness_images, len(test_rows)), random_state=42)
        ex.save_json(
            {"experiment": "robustness", "model": "checkpoints/model_quantized.onnx (deployed)",
             **ex.robustness_experiment(classifier, sample, root, load_classical_predict(root))},
            RESULTS / "robustness.json",
        )
    if "ablation" in args.stages:
        print("Crop ablation (the deployed model on the full image, no crop)...", flush=True)
        test_rows = manifest[masks["test"]]
        uncropped = ex.run_batches(classifier.predict_proba, test_rows, root, "none")
        ex.save_json(
            {"experiment": "ablation_inference", "model": "checkpoints/model_quantized.onnx (deployed)",
             "with_annotated_box_crop": test_scores, "without_any_crop_full_image": ex.scores(uncropped, y_test),
             "note": "Detector-crop rows need results from scripts/evaluate_detector.py on the full test set and are not part of this run."},
            RESULTS / "ablation_inference.json",
        )
    print(f"Done. Results are in {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
