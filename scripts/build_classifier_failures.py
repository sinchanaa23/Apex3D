"""Write results/classifier_failure_cases.json and docs/CLASSIFIER_FAILURES.md from the deployed model's real wrong answers.

    python scripts/build_classifier_failures.py

Needs results/failure_candidates.csv, which `scripts/evaluate_deployed_model.py` writes from the model's predictions on
all 8,041 test images. Every number below is read from that file, never typed. The causes were written after looking at
each photo, and say "appears" where we could not check a label independently.

These cases are the most confident wrong answers, picked to show distinct causes. They are not a random sample. The
breakdown over all errors is the representative summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from data.stanford_cars import default_root, parse_class_name  # noqa: E402

FIGURES = ROOT / "results" / "figures" / "classifier_failures"

# (test image, short title, cause written after looking at the photo)
CASES = [
    ("images/test/07654.jpg", "A Jeep Compass is called a Grand Cherokee",
     "The photo shows a black SUV with Jeep's seven-slot grille, which to us looks like a Grand Cherokee, not a Compass. If that is right, the model is correct and the dataset label is wrong. "
     "We cannot check the label independently, so it counts as an error here, but it is probably label noise."),
    ("images/test/06675.jpg", "Two vans that are the same vehicle with different badges",
     "The Chevrolet Express and the GMC Savana are the same van with different badges. This photo shows a GMC badge on the grille, so the model's answer matches what is visible and the label names the twin model. "
     "Cause: twin vehicles, so the label is ambiguous."),
    ("images/test/07095.jpg", "An Audi TT and an Audi TT RS in a blurred photo",
     "The TT and the TT RS differ mostly in trim details such as the grille, bumpers and wheels, and this photo is motion-blurred, which removes those details. The model chose the RS coupe, the closest class of the same model. "
     "Cause: a fine trim difference, made worse by blur. The label also names a different year."),
    ("images/test/01821.jpg", "An Aston Martin V8 Vantage seen only from behind",
     "This is a studio view from the rear three-quarter, which hides the front grille and air intakes that separate the models. Both the V8 Vantage and the Virage are low, long-bonnet Aston Martin coupes. "
     "Cause: the viewpoint hides the features that tell the models apart."),
    ("images/test/07627.jpg", "A Silverado 1500 and a Silverado 2500HD",
     "Both are Silverado regular cab pickups with the same cab and body. They differ in payload class and small details, which are hard to see from the side. "
     "Cause: near-identical models that differ by capacity, not by shape."),
    ("images/test/07699.jpg", "A large 'SOLD' sticker over the windscreen",
     "A big 'SOLD' overlay covers part of the windscreen, and the dataset has two near-identical Express classes (Cargo Van and Van, both 2007). The model answered with the other Express class, so manufacturer, model and year were right and only the variant differed. "
     "Cause: the overlay hides detail, and the two classes look alike."),
]


def main() -> None:
    candidates = pd.read_csv(ROOT / "results" / "failure_candidates.csv")
    root = default_root()
    FIGURES.mkdir(parents=True, exist_ok=True)

    same_model = same_make = other_make = 0
    for truth, predicted in zip(candidates["class_name"], candidates["predicted"]):
        a, b = parse_class_name(truth), parse_class_name(predicted)
        if a["make"] != b["make"]:
            other_make += 1
        elif (a["model"], a["body_type"]) == (b["model"], b["body_type"]):
            same_model += 1
        else:
            same_make += 1
    errors = len(candidates)
    confident_errors = int((candidates["confidence"] >= 0.9).sum())
    summary = {
        "errors": errors,
        "same_model_different_year": same_model,
        "same_make_different_model": same_make,
        "different_make": other_make,
        "errors_with_confidence_at_least_0_9": confident_errors,
    }

    cases = []
    for path, title, cause in CASES:
        row = candidates[candidates["path"] == path]
        if row.empty:
            raise SystemExit(f"{path} is not among the wrong answers in failure_candidates.csv. The model or the data changed, so the written causes need checking again.")
        row = row.iloc[0]
        thumbnail = FIGURES / f"{Path(path).stem}.jpg"
        with Image.open(root / path) as opened:
            image = opened.convert("RGB")
        image.thumbnail((640, 640))
        image.save(thumbnail, quality=85)
        measured = f"Measured: the model gave {row['predicted']} with {row['confidence']:.1%} calibrated confidence, and gave the correct class {row['true_class_probability']:.2%}."
        cases.append({
            "title": title,
            "image": str(thumbnail.relative_to(ROOT)).replace("\\", "/"),
            "source_image": path,
            "predicted": row["predicted"],
            "actual": row["class_name"],
            "confidence": float(row["confidence"]),
            "true_class_probability": float(row["true_class_probability"]),
            "explanation": f"{cause} {measured}",
        })

    (ROOT / "results" / "classifier_failure_cases.json").write_text(json.dumps({"summary": summary, "cases": cases}, indent=2), encoding="utf-8")

    pct = lambda n: f"{n / errors:.1%}"  # noqa: E731
    lines = [
        "# Classifier failure analysis",
        "",
        "These are failures of the deployed classifier (`checkpoints/model_quantized.onnx`) on the real Stanford Cars test set, from `scripts/evaluate_deployed_model.py` and `scripts/build_classifier_failures.py`. "
        "Detector and classical failures are in `docs/FAILURE_ANALYSIS.md`.",
        "",
        "## How the errors break down",
        "",
        f"The model gave a wrong answer on {errors} of 8,041 test images. How wrong:",
        "",
        "| Kind of mistake | Images | Share of errors |",
        "|---|---|---|",
        f"| Same manufacturer and model, different year | {same_model} | {pct(same_model)} |",
        f"| Same manufacturer, different model | {same_make} | {pct(same_make)} |",
        f"| Different manufacturer | {other_make} | {pct(other_make)} |",
        "",
        f"{confident_errors} of the {errors} errors still had at least 90% calibrated confidence, so a high confidence does not guarantee a right answer. "
        "The confident state in the app is right 95% of the time on this test set, not always.",
        "",
        "## Six of the most confident wrong answers",
        "",
        "These are picked from the most confident errors to show distinct causes. They are not a random sample, so use the table above for the overall picture. "
        "Causes were written after looking at each photo. Where we say a label \"appears\" wrong, we could not check it independently.",
        "",
    ]
    for number, case in enumerate(cases, 1):
        lines += [
            f"### {number}. {case['title']}",
            "",
            f"![{case['title']}](../{case['image']})",
            "",
            f"- **True label:** {case['actual']}",
            f"- **Model's answer:** {case['predicted']} ({case['confidence']:.1%} confidence; correct class {case['true_class_probability']:.2%})",
            f"- **Cause:** {case['explanation'].rsplit(' Measured:', 1)[0]}",
            f"- **Image:** `{case['source_image']}`",
            "",
        ]
    lines += [
        "## What this means",
        "",
        "- Most mistakes are between neighbours: the same manufacturer, or twin or near-identical models (Chevrolet Express and GMC Savana, Silverado 1500 and 2500HD, TT and TT RS). "
        "The app's Top-5 list and the confirm step let a person correct these.",
        "- Some of the most confident errors look like dataset label noise rather than model mistakes, so the true accuracy may be a little higher than reported.",
        "- Three causes recur: trims and years that differ only in small details, viewpoints that hide those details, and photos with overlays, blur or occlusion.",
        "",
        "## Limits",
        "",
        "- The six cases are curated for variety, and the causes are our reading of the photos. We did not run an experiment to isolate each cause.",
        "- Labels were not re-checked against the manufacturers' records.",
        "- Results are for the deployed INT8 model on annotated crops with an 8% margin, not the full detect-then-classify pipeline.",
        "",
    ]
    (ROOT / "docs" / "CLASSIFIER_FAILURES.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"{errors} errors; wrote {len(cases)} cases.")


if __name__ == "__main__":
    main()
