"""Measure the classical contour ROI against the annotated boxes, on the detector's sample.

    python scripts/evaluate_classical_roi.py --limit 500

Using the same seed as evaluate_detector.py gives the same images, so the results are comparable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from classical_cv.pipeline import canny_edges, contour_roi, largest_contour  # noqa: E402
from data.stanford_cars import default_root, load_manifest  # noqa: E402
from detection.metrics import iou  # noqa: E402

SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    root = default_root()
    rows = load_manifest(root).query("split == 'test'")
    if args.limit and args.limit < len(rows):
        rows = rows.sample(args.limit, random_state=SEED)

    values, full_frame = [], 0
    for row in rows.itertuples():
        image = cv2.imread(str(root / row.path))
        box = contour_roi(image, largest_contour(canny_edges(image)))
        values.append(iou(box, (row.x1, row.y1, row.x2, row.y2)))
        height, width = image.shape[:2]
        full_frame += (box[2] - box[0]) * (box[3] - box[1]) >= 0.9 * width * height
    values = np.array(values)

    # For reference: what simply returning the whole image would score, with no detection at all.
    whole = np.array([iou((0, 0, row.width, row.height), (row.x1, row.y1, row.x2, row.y2)) for row in rows.itertuples()])
    results = {
        "experiment": "classical_roi_evaluation",
        "method": "Largest contour after median-threshold Canny and morphological closing",
        "split": "test",
        "is_random_sample": bool(args.limit),
        "images": len(rows),
        "seed": SEED if args.limit else None,
        "mean_iou": float(values.mean()),
        "median_iou": float(np.median(values)),
        "share_iou_at_least_50": float((values >= 0.5).mean()),
        "share_iou_at_least_75": float((values >= 0.75).mean()),
        "share_covering_90_percent_of_frame": full_frame / len(rows),
        "whole_image_mean_iou": float(whole.mean()),
    }
    scope = f"sample{len(rows)}" if args.limit else "full"
    (ROOT / "results" / f"classical_roi_test_{scope}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
