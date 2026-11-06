"""Write docs/FAILURE_ANALYSIS.md and results/failure_cases.json from measured detector runs.

    python scripts/build_failure_cases.py

IoU values are computed here, never typed. The causes were written after looking at each image.
Green boxes are annotations, red are the app's detector crop, and blue are other methods' boxes.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from classical_cv.pipeline import canny_edges, contour_roi, largest_contour  # noqa: E402
from data.stanford_cars import default_root, load_manifest  # noqa: E402
from detection.metrics import iou, pick_largest_confident, pick_top_score  # noqa: E402

GREEN, RED, BLUE = (0, 200, 0), (235, 30, 30), (0, 110, 255)
FIGURES = ROOT / "results" / "figures" / "failures"


def main() -> None:
    root = default_root()
    manifest = load_manifest(root).set_index("path")
    run = root.parent / "detector_runs" / "detector_resnet50_test_sample500.jsonl"
    records = {r["image"]: r["detections"] for r in map(json.loads, run.read_text(encoding="utf-8").splitlines())}
    FIGURES.mkdir(parents=True, exist_ok=True)

    def truth(path):
        row = manifest.loc[path]
        return (row.x1, row.y1, row.x2, row.y2)

    def values(path):
        chosen, top = pick_largest_confident(records[path]), pick_top_score(records[path])
        return chosen, top, iou(chosen["box"], truth(path)), iou(top["box"], truth(path))

    def picture(path, name, boxes):
        with Image.open(root / path) as opened:
            image = opened.convert("RGB")
        scale = min(640 / max(image.size), 1.0)
        image = image.resize((int(image.width * scale), int(image.height * scale)))
        draw = ImageDraw.Draw(image)
        for box, colour in [(truth(path), GREEN), *boxes]:
            draw.rectangle([value * scale for value in box], outline=colour, width=4)
        target = FIGURES / f"{name}.jpg"
        image.save(target, quality=85)
        return target.relative_to(ROOT).as_posix()

    cases = []

    def add(key, title, path, boxes, measured, cause, change, status):
        cases.append(
            {
                "title": title,
                "image": picture(path, key, boxes),
                "image_id": path,
                "predicted": measured,
                "actual": manifest.loc[path].class_name,
                "explanation": f"{cause} {change}",
                "cause": cause,
                "improvement": change,
                "status": status,
            }
        )

    # The clearest example of the most common failure: top score misses, the chosen rule succeeds.
    examples = [path for path in records if records[path] and values(path)[3] < 0.05 and values(path)[2] > 0.9]
    background = max(examples, key=lambda path: values(path)[1]["score"])
    chosen, top, rule_iou, top_iou = values(background)
    total_wrong = sum(1 for path in records if records[path] and values(path)[3] < 0.5)
    total_after = sum(1 for path in records if records[path] and values(path)[2] < 0.5)
    add(
        "background_car",
        "A tiny background car outranks the main car",
        background,
        [(top["box"], RED)],
        f"Top-score box IoU {top_iou:.2f}, at score {top['score']:.2f}",
        f"The detector scores small distant cars as highly as the main subject, so choosing the top score selected a background car. This hit {total_wrong} of {len(records)} sampled test images, and the app would have identified the wrong car.",
        f"Fixed: the app now crops the largest box scoring at least 0.5, chosen on validation. This image rises to IoU {rule_iou:.2f}, and IoU failures fall to {total_after} of {len(records)}.",
        "Fixed",
    )

    path = "images/test/02111.jpg"
    chosen, top, rule_iou, top_iou = values(path)
    add(
        "weak_large_box",
        "The fix itself fails on a weak, large box",
        path,
        [(chosen["box"], RED), (top["box"], BLUE)],
        f"Chosen box IoU {rule_iou:.2f}, score {chosen['score']:.2f}. Top-score box IoU {top_iou:.2f}",
        "A large low-confidence box covering dark studio background just passes the 0.5 score limit and beats the correct box. Preferring large boxes has this cost.",
        "Possible improvement: test a higher score limit or score multiplied by area on validation. Score times area was close on validation, so it's the first candidate.",
        "Open: known trade-off of the fix",
    )

    path = "images/test/04814.jpg"
    chosen, top, rule_iou, top_iou = values(path)
    add(
        "close_subject_missed",
        "A very close car is missed in favour of a partial one",
        path,
        [(chosen["box"], RED)],
        f"Chosen box IoU {rule_iou:.2f}, score {chosen['score']:.2f}",
        "The main car fills the frame and shows its rear with a folded roof, a less common view, so no confident box covers it. The largest confident box is a sliver of another car at the edge.",
        "Possible improvement: ignore boxes cut off by the image edge when a central vehicle exists, or fall back to the full frame when the chosen box is narrow.",
        "Open",
    )

    path = "images/test/02234.jpg"
    chosen, top, rule_iou, top_iou = values(path)
    add(
        "open_doors",
        "Raised doors fall outside the detector's box",
        path,
        [(chosen["box"], RED)],
        f"Chosen box IoU {rule_iou:.2f}, score {chosen['score']:.2f}",
        "COCO training images rarely show scissor doors, so the box covers only the body, while the annotation includes the raised doors. The Ford GT example behaves the same way.",
        "Effect on classification is unmeasured until detector-crop accuracy is run. The car's body stays in the crop, so the harm may be small. Retraining the detector on Stanford boxes could address it.",
        "Open",
    )

    path = "images/test/05062.jpg"
    image = cv2.imread(str(root / path))
    roi = contour_roi(image, largest_contour(canny_edges(image)))
    chosen, _, rule_iou, _ = values(path)
    add(
        "classical_roi",
        "The classical ROI locks onto the background",
        path,
        [(roi, BLUE), (chosen["box"], RED)],
        f"Contour ROI IoU {iou(roi, truth(path)):.2f}. Detector IoU {rule_iou:.2f}",
        "The largest-contour method assumes the car has the strongest outline. Here the bridge does, so the ROI misses the car completely. It has no idea what a car is.",
        "This is why the app crops with the detector. The contour ROI stays as a visual explanation in the classical view, and isn't used for classification.",
        "By design: detector used instead",
    )

    public = [{key: case[key] for key in ("title", "image", "image_id", "predicted", "actual", "explanation")} for case in cases]
    (ROOT / "results" / "failure_cases.json").write_text(json.dumps(public, indent=2), encoding="utf-8")

    pairs = list(csv.DictReader((ROOT / "results" / "classifier_confused_pairs.csv").open(encoding="utf-8", newline="")))
    lines = [
        "# Failure analysis",
        "",
        "Generated by `scripts/build_failure_cases.py`. Green boxes are annotations, red are the app's detector crop, and blue are other methods' boxes.",
        f"Localisation cases come from a random sample of {len(records)} test images, using the ResNet-50 detector.",
        "",
    ]
    for number, case in enumerate(cases, 1):
        lines += [
            f"## {number}. {case['title']}",
            "",
            f"![{case['title']}](../{case['image']})",
            "",
            f"- **Image:** `{case['image_id']}`, {case['actual']}",
            f"- **Measured:** {case['predicted']}",
            f"- **Cause:** {case['cause']}",
            f"- **Improvement:** {case['improvement']}",
            f"- **Status:** {case['status']}",
            "",
        ]
    top_pairs = pairs[:5]
    lines += [
        f"## {len(cases) + 1}. Near-identical classes confuse the deep model",
        "",
        "These counts are from `results/classifier_confused_pairs.csv`, across all 8,041 test images. Image-level examples need the saved logits and aren't included yet.",
        "",
        "| Class A | Class B | Confusions |",
        "|---|---|---|",
        *[f"| {row['class_a']} | {row['class_b']} | {row['total']} |" for row in top_pairs],
        "",
        "- **Cause:** these are the same or very closely related vehicles. Examples are model years with little visual change, performance trims of one body, and rebadged vans. Some differences aren't visible from every angle.",
        "- **Improvement:** the app can show Top-5 and an ambiguous state rather than one firm answer. Training at a higher resolution or adding attention to parts may help, but neither has been tested.",
        "- **Status:** Open. This is the main remaining limit of the classifier.",
        "",
    ]
    (ROOT / "docs" / "FAILURE_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(cases) + 1} failure cases.")


if __name__ == "__main__":
    main()
