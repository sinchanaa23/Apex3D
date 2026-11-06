"""Localisation metrics for one vehicle class, with boxes as (x1, y1, x2, y2)."""

from __future__ import annotations

import numpy as np

COCO_THRESHOLDS = tuple(round(0.5 + 0.05 * step, 2) for step in range(10))


def iou(a, b) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(right - left, 0) * max(bottom - top, 0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return float(intersection / union) if union > 0 else 0.0


def average_precision(detections: list[dict], truths: dict, threshold: float) -> float:
    """COCO-style 101-point AP.

    detections: dicts with image, score and box, for every image.
    truths: image id to its one annotated box. Stanford Cars annotates one car per image.
    Within an image, the best-overlapping unmatched detection above the threshold is the match,
    taken in score order. Each image has one truth, so only one detection can match.
    """
    if not truths:
        return 0.0
    ordered = sorted(detections, key=lambda item: -item["score"])
    matched: set = set()
    hits = np.zeros(len(ordered), dtype=bool)
    for position, detection in enumerate(ordered):
        image = detection["image"]
        if image in truths and image not in matched and iou(detection["box"], truths[image]) >= threshold:
            hits[position] = True
            matched.add(image)
    true_positives = np.cumsum(hits)
    recall = true_positives / len(truths)
    precision = true_positives / np.arange(1, len(ordered) + 1) if len(ordered) else np.array([])
    # Make precision never increase as recall rises, as COCO does.
    precision = np.maximum.accumulate(precision[::-1])[::-1] if len(ordered) else precision
    samples = []
    for level in np.linspace(0, 1, 101):
        reached = np.searchsorted(recall, level, side="left")
        samples.append(precision[reached] if reached < len(precision) else 0.0)
    return float(np.mean(samples))


def mean_average_precision(detections: list[dict], truths: dict) -> dict[str, float]:
    per_threshold = {threshold: average_precision(detections, truths, threshold) for threshold in COCO_THRESHOLDS}
    return {"ap50": per_threshold[0.5], "ap75": per_threshold[0.75], "map_50_95": float(np.mean(list(per_threshold.values())))}


def box_area(box) -> float:
    return max(box[2] - box[0], 0) * max(box[3] - box[1], 0)


def pick_top_score(items: list[dict]) -> dict:
    return max(items, key=lambda item: item["score"])


def pick_largest_confident(items: list[dict], min_score: float = 0.5) -> dict:
    """The largest box scoring at least min_score, or the top score if none does.

    Detectors often give tiny background cars scores as high as the main subject's, so the
    highest score alone can select the wrong car.
    """
    confident = [item for item in items if item["score"] >= min_score]
    return max(confident, key=lambda item: box_area(item["box"])) if confident else pick_top_score(items)


def pick_score_times_area(items: list[dict]) -> dict:
    return max(items, key=lambda item: item["score"] * box_area(item["box"]))


SELECTION_RULES = {
    "top_score": pick_top_score,
    "largest_confident": pick_largest_confident,
    "score_times_area": pick_score_times_area,
}


def top_box_summary(detections: list[dict], truths: dict, pick=pick_top_score) -> dict[str, float]:
    """IoU of the one vehicle a selection rule chooses in each image, which is what the app crops."""
    by_image: dict = {}
    for detection in detections:
        by_image.setdefault(detection["image"], []).append(detection)
    best = {image: pick(items) for image, items in by_image.items()}
    values = np.array([iou(best[image]["box"], box) if image in best else 0.0 for image, box in truths.items()])
    return {
        "images": int(len(values)),
        "mean_iou": float(values.mean()),
        "median_iou": float(np.median(values)),
        "share_iou_at_least_50": float((values >= 0.5).mean()),
        "share_iou_at_least_75": float((values >= 0.75).mean()),
        "images_with_no_vehicle_found": int(len(truths) - sum(image in best for image in truths)),
    }
