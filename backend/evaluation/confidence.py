"""Calibration and low-confidence decisions. Choose settings on validation, then report on test."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def topk_accuracy(probabilities: np.ndarray, labels: np.ndarray, k: int = 1) -> float:
    top = np.argpartition(-probabilities, k - 1, axis=1)[:, :k]
    return float((top == labels[:, None]).any(axis=1).mean())


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def fit_temperature(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Temperature that minimises negative log-likelihood. It cannot change the predicted class."""
    logits = np.log(np.clip(probabilities, 1e-12, 1))

    def loss(temperature: float) -> float:
        scaled = softmax(logits / temperature)
        return float(-np.log(np.clip(scaled[np.arange(len(labels)), labels], 1e-12, 1)).mean())

    return float(minimize_scalar(loss, bounds=(0.05, 10), method="bounded").x)


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    return softmax(np.log(np.clip(probabilities, 1e-12, 1)) / temperature)


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (confidence > low) & (confidence <= high)
        if inside.any():
            error += inside.mean() * abs(correct[inside].mean() - confidence[inside].mean())
    return float(error)


def top_two(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = -np.partition(-probabilities, 1, axis=1)
    return ordered[:, 0], ordered[:, 0] - ordered[:, 1]


def coverage_curve(probabilities: np.ndarray, labels: np.ndarray, thresholds=None, rule: str = "confidence") -> list[dict]:
    """Share of images accepted, and accuracy among them, at each threshold."""
    thresholds = np.linspace(0, 0.99, 34) if thresholds is None else thresholds
    confidence, margin = top_two(probabilities)
    signal = confidence if rule == "confidence" else margin
    correct = probabilities.argmax(axis=1) == labels
    rows = []
    for threshold in thresholds:
        accepted = signal >= threshold
        rows.append(
            {
                "threshold": round(float(threshold), 4),
                "coverage": float(accepted.mean()),
                "accuracy_when_accepted": float(correct[accepted].mean()) if accepted.any() else None,
            }
        )
    return rows


def choose_threshold(curve: list[dict], target_accuracy: float) -> float | None:
    """Lowest threshold that reaches the target, which keeps the most images."""
    for row in curve:
        if row["accuracy_when_accepted"] is not None and row["accuracy_when_accepted"] >= target_accuracy:
            return row["threshold"]
    return None


def decide(confidence: np.ndarray, margin: np.ndarray, accept: float, min_margin: float, reject: float) -> np.ndarray:
    """confident, ambiguous or rejected, for each image. These drive the app's three states."""
    states = np.full(len(confidence), "ambiguous", dtype=object)
    states[(confidence >= accept) & (margin >= min_margin)] = "confident"
    states[confidence < reject] = "rejected"
    return states
