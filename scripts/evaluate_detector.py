"""Measure detector IoU and mAP on Stanford Cars test images.

    python scripts/evaluate_detector.py --variant mobilenet --limit 500
    python scripts/evaluate_detector.py --variant resnet50 --device cuda

Detections are saved as the run goes, so an interrupted run resumes where it stopped.
A random sample is always labelled as a sample in the results. Without --limit, all 8,041 test
images are used.

Caveat on mAP: Stanford Cars annotates one car per image. A correct detection of a second car in
the background therefore counts as a false positive, so mAP here understates real precision.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from data.stanford_cars import default_root, load_manifest  # noqa: E402
from detection.detector import VehicleDetector  # noqa: E402
from detection.metrics import SELECTION_RULES, mean_average_precision, top_box_summary  # noqa: E402

SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default="resnet50", choices=["mobilenet", "resnet50"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, help="Evaluate a random sample of this many test images.")
    parser.add_argument("--split", default="test", choices=["test", "val"])
    args = parser.parse_args()

    root = default_root()
    rows = load_manifest(root).query("split == @args.split")
    if args.limit and args.limit < len(rows):
        rows = rows.sample(args.limit, random_state=SEED)

    scope = f"{args.split}_sample{len(rows)}" if args.limit else f"{args.split}_full"
    name = f"detector_{args.variant}_{scope}"
    progress_path = root.parent / "detector_runs" / f"{name}.jsonl"
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    finished = {}
    if progress_path.exists():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            finished[record["image"]] = record

    print(f"Loading {args.variant} on {args.device}...", flush=True)
    detector = VehicleDetector(args.variant, args.device)
    with progress_path.open("a", encoding="utf-8") as log:
        for position, row in enumerate(rows.itertuples(), 1):
            if row.path in finished:
                continue
            with Image.open(root / row.path) as opened:
                image = opened.convert("RGB")
            started = time.perf_counter()
            found = detector.detect(image)
            record = {
                "image": row.path,
                "seconds": time.perf_counter() - started,
                "detections": [{"box": item.box, "score": item.score, "category": item.category} for item in found],
            }
            finished[row.path] = record
            log.write(json.dumps(record) + "\n")
            log.flush()
            if position % 25 == 0:
                print(f"{position}/{len(rows)} images", flush=True)

    truths = {row.path: (row.x1, row.y1, row.x2, row.y2) for row in rows.itertuples()}
    records = [finished[image] for image in truths]
    detections = [{"image": record["image"], **item} for record in records for item in record["detections"]]
    # The first image includes start-up work, so it is left out of the timing.
    times = sorted(record["seconds"] for record in records[1:])
    results = {
        "experiment": "detector_evaluation",
        "detector": f"torchvision Faster R-CNN {args.variant} FPN, COCO weights, not retrained",
        "vehicle_classes": ["car", "bus", "truck"],
        "split": args.split,
        "is_random_sample": bool(args.limit),
        "images": len(rows),
        "seed": SEED if args.limit else None,
        "device": args.device,
        "seconds_per_image_median": times[len(times) // 2] if times else None,
        "selection_rule_in_app": "largest_confident",
        "selection_rules": {name: top_box_summary(detections, truths, pick) for name, pick in SELECTION_RULES.items()},
        "average_precision": mean_average_precision(detections, truths),
        "map_caveat": "One car is annotated per image, so correct detections of other cars count as false positives.",
    }
    output = ROOT / "results" / f"{name}.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
