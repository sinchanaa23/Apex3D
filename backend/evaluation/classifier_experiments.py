"""Experiments that need the trained classifier: hybrid, ablation, robustness, confidence, failures.

A classifier is any object with these two methods, which Person A's load_classifier() provides:

    predict_proba(images: list[PIL.Image]) -> np.ndarray   # N x 196, softmax probabilities
    embed(images: list[PIL.Image]) -> np.ndarray           # N x D, penultimate-layer features

Settings are chosen on validation only. Test results are computed last.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from classical_cv import feature_blocks
from classical_cv.pipeline import apply_clahe, is_low_light
from data.stanford_cars import box_with_margin

from . import confidence as conf
from .corruptions import CORRUPTIONS, apply

BATCH_SIZE = 32


def load_image(root: Path, row, crop: str, detections: dict | None = None) -> Image.Image:
    with Image.open(root / row.path) as opened:
        image = opened.convert("RGB")
    if crop == "box":
        return image.crop(box_with_margin(row))
    if crop == "detector" and detections and detections.get(row.path):
        x1, y1, x2, y2 = detections[row.path]
        # The detector's box gets the same margin as the annotated box did in training.
        return image.crop(
            box_with_margin(SimpleNamespace(x1=x1, y1=y1, x2=x2, y2=y2, width=image.width, height=image.height))
        )
    return image


def run_batches(function, rows: pd.DataFrame, root: Path, crop: str = "box", detections=None, change=None) -> np.ndarray:
    outputs, batch = [], []
    for position, row in enumerate(rows.itertuples()):
        image = load_image(root, row, crop, detections)
        batch.append(change(image, position) if change else image)
        if len(batch) == BATCH_SIZE:
            outputs.append(function(batch))
            batch = []
    if batch:
        outputs.append(function(batch))
    return np.vstack(outputs)


def scores(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "top1": conf.topk_accuracy(probabilities, labels, 1),
        "top5": conf.topk_accuracy(probabilities, labels, 5),
        "macro_f1": float(f1_score(labels, probabilities.argmax(1), average="macro", zero_division=0)),
    }


def linear_head(C: float):
    return make_pipeline(StandardScaler(), LinearSVC(C=C, dual=False, max_iter=2000, random_state=42))


def decision_scores(model, features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    decision = model.decision_function(features)
    return scores(decision, labels)


def hybrid_experiment(embeddings: np.ndarray, classical: np.ndarray, labels: np.ndarray, masks: dict) -> dict:
    """Deep features alone against deep plus classical, using the same kind of linear head.

    Using one head for both means any change comes from the classical features, and not from
    swapping the classifier. Every choice is made on validation before test is touched.
    """
    train, val, test = masks["train"], masks["val"], masks["test"]
    reducer = make_pipeline(StandardScaler(), PCA(64, random_state=42)).fit(classical[train])
    pieces, start = {}, 0
    for name, length in feature_blocks().items():
        pieces[name] = np.arange(start, start + length)
        start += length

    def fused(columns: np.ndarray | None = None) -> np.ndarray:
        if columns is None:
            return np.hstack((embeddings, reducer.transform(classical)))
        family = classical[:, columns]
        if family.shape[1] > 64:
            family = make_pipeline(StandardScaler(), PCA(64, random_state=42)).fit(family[train]).transform(family)
        return np.hstack((embeddings, family))

    candidates = {"deep_only": embeddings, "deep_plus_all_classical": fused()}
    candidates.update({f"deep_plus_{name}": fused(columns) for name, columns in pieces.items()})

    validation = {}
    for name, features in candidates.items():
        best = None
        for C in (0.001, 0.01, 0.1):
            result = decision_scores(linear_head(C).fit(features[train], labels[train]), features[val], labels[val])
            if best is None or result["top1"] > best["top1"]:
                best = {**result, "C": C}
        validation[name] = best

    deep_C = validation["deep_only"]["C"]
    best_fusion = max((name for name in validation if name != "deep_only"), key=lambda name: validation[name]["top1"])
    development = train | val
    test_results = {}
    for name, C in (("deep_only", deep_C), (best_fusion, validation[best_fusion]["C"])):
        model = linear_head(C).fit(candidates[name][development], labels[development])
        test_results[name] = decision_scores(model, candidates[name][test], labels[test])

    return {
        "validation": validation,
        "chosen_fusion": best_fusion,
        "fusion_helped_on_validation": validation[best_fusion]["top1"] > validation["deep_only"]["top1"],
        "test": test_results,
        "fusion_change_in_test_top1": test_results[best_fusion]["top1"] - test_results["deep_only"]["top1"],
    }


def confidence_experiment(test_probabilities, test_labels, temperature: float, thresholds: dict) -> dict:
    """Threshold analysis of what the app ships: the checkpoint's temperature and thresholds.

    Both were chosen on validation in notebook 01. Fitting different ones here would describe a
    system that isn't deployed. This adds the coverage curves and checks the shipped states.
    """
    calibrated = conf.apply_temperature(test_probabilities, temperature)
    top, margin = conf.top_two(calibrated)
    states = conf.decide(top, margin, thresholds["confident_p1"], thresholds["confident_margin"], thresholds["reject_below_p1"])
    correct = calibrated.argmax(1) == test_labels
    return {
        "temperature": temperature,
        "thresholds": thresholds,
        "ece_before": conf.expected_calibration_error(test_probabilities, test_labels),
        "ece_after": conf.expected_calibration_error(calibrated, test_labels),
        "test_coverage_curve": conf.coverage_curve(calibrated, test_labels),
        "test_margin_curve": conf.coverage_curve(calibrated, test_labels, rule="margin"),
        "test_states": {
            state: {"share": float((states == state).mean()), "accuracy": float(correct[states == state].mean()) if (states == state).any() else None}
            for state in ("confident", "ambiguous", "rejected")
        },
    }


def robustness_experiment(classifier, rows: pd.DataFrame, root: Path, classical_predict=None) -> dict:
    """Accuracy under each corruption. classical_predict is the classical model, if it is available."""
    labels = rows["label"].to_numpy()
    clean = run_batches(classifier.predict_proba, rows, root)
    output = {"images": len(rows), "deep_clean": scores(clean, labels), "conditions": []}
    if classical_predict:
        output["classical_clean"] = scores(run_batches(classical_predict, rows, root), labels)

    for name, (_, levels) in CORRUPTIONS.items():
        for severity, level in enumerate(levels):
            change = lambda image, position: apply(name, severity, image, seed=position)  # noqa: B023
            row = {"corruption": name, "severity": severity, "level": level}
            row["deep"] = scores(run_batches(classifier.predict_proba, rows, root, change=change), labels)
            if classical_predict:
                row["classical"] = scores(run_batches(classical_predict, rows, root, change=change), labels)
            if name == "low_light":
                # Adaptive CLAHE is the classical-plus-deep integration tested at this point.
                def enhanced(image, position):
                    dark = apply(name, severity, image, seed=position)
                    pixels = np.asarray(dark)[:, :, ::-1]
                    return Image.fromarray(apply_clahe(pixels)[:, :, ::-1].copy()) if is_low_light(pixels) else dark

                row["deep_with_adaptive_clahe"] = scores(run_batches(classifier.predict_proba, rows, root, change=enhanced), labels)
            output["conditions"].append(row)
    return output


def error_breakdown(predictions: np.ndarray, labels: np.ndarray, classes: list[str], parse) -> dict:
    """How wrong the wrong answers are. Year-only mistakes are milder than wrong-make mistakes."""
    parsed = [parse(name) for name in classes]
    counts = {"same_make_and_model_different_year": 0, "same_make_different_model": 0, "different_make": 0}
    wrong = predictions != labels
    for predicted, truth in zip(predictions[wrong], labels[wrong]):
        a, b = parsed[predicted], parsed[truth]
        if a["make"] != b["make"]:
            counts["different_make"] += 1
        # Body type counts here: Audi 100 Sedan and Audi 100 Wagon differ by more than the year.
        elif (a["model"], a["body_type"]) == (b["model"], b["body_type"]):
            counts["same_make_and_model_different_year"] += 1
        else:
            counts["same_make_different_model"] += 1
    total = max(int(wrong.sum()), 1)
    return {"errors": int(wrong.sum()), **counts, **{f"{key}_share": value / total for key, value in counts.items()}}


def failure_candidates(rows: pd.DataFrame, probabilities: np.ndarray, classes: list[str]) -> pd.DataFrame:
    """Wrong predictions with what's needed to study them, most confident first."""
    labels = rows["label"].to_numpy()
    predicted = probabilities.argmax(1)
    top, margin = conf.top_two(probabilities)
    table = rows.assign(
        predicted=[classes[index] for index in predicted],
        confidence=top,
        margin=margin,
        true_class_probability=probabilities[np.arange(len(labels)), labels],
        box_area_share=(rows.x2 - rows.x1) * (rows.y2 - rows.y1) / (rows.width * rows.height),
    )
    return table[predicted != labels].sort_values("confidence", ascending=False)


def failure_sheet(table: pd.DataFrame, root: Path, path: Path, columns: int = 4, rows: int = 3) -> None:
    """Contact sheet of failures, for a person to review and explain."""
    width, height = 300, 225
    sheet = Image.new("RGB", (width * columns, (height + 44) * rows), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(table.head(columns * rows).itertuples()):
        with Image.open(root / row.path) as opened:
            image = opened.convert("RGB")
        image.thumbnail((width, height))
        x, y = (index % columns) * width, (index // columns) * (height + 44)
        sheet.paste(image, (x, y + 42))
        draw.text((x + 3, y + 3), f"True: {row.class_name}"[:48], fill=(20, 110, 60))
        draw.text((x + 3, y + 21), f"Pred: {row.predicted} ({row.confidence:.2f})"[:48], fill=(170, 40, 30))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=88)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=float), encoding="utf-8")
