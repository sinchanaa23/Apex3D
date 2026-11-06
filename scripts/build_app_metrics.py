"""Write eval_P5_metrics.json, the file behind GET /api/metrics and the Research Lab.

    python scripts/build_app_metrics.py

The sections follow frontend/API_EXPECTATIONS.md. Every value is read from results/. A section whose
source doesn't exist yet is the string "not available", which the page shows as such.
"""

from __future__ import annotations

import base64
import csv
import glob
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "backend"))
NOT_AVAILABLE = "not available"
RULE = "largest_confident"


def load(name: str) -> dict | None:
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def largest_run(pattern: str) -> tuple[dict, str] | tuple[None, None]:
    runs = [(json.loads(Path(path).read_text(encoding="utf-8")), Path(path).name) for path in glob.glob(str(RESULTS / pattern))]
    return max(runs, key=lambda run: run[0]["images"]) if runs else (None, None)


def section(source: str, data: dict) -> dict:
    path = RESULTS / source
    from datetime import datetime, timezone

    return {"source": f"results/{source}", "generated_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(), "data": data}


def classifier_section() -> dict | str:
    metrics = load("classifier_metrics")
    if not metrics:
        return NOT_AVAILABLE
    ablation = load("ablation_inference") or {}
    crop = ablation.get("detector_crop", {}).get("with_detector_crop", {})
    return section("classifier_metrics.json", {
        "model": metrics["model"]["arch"],
        "top1": metrics["test"]["top1"],
        "top5": metrics["test"]["top5"],
        "top1_detector_crop": crop.get("top1"),
        "top5_detector_crop": crop.get("top5"),
        "test_images": metrics["test"]["n"],
    })


def detection_section() -> dict | str:
    rows, source = [], None
    for label, pattern in (("Faster R-CNN ResNet-50 FPN v2", "detector_resnet50_test_*.json"), ("Faster R-CNN MobileNetV3 FPN", "detector_mobilenet_test_*.json")):
        run, name = largest_run(pattern)
        if run:
            source = source or name
            rows.append({"model": f"{label} ({run['images']} images)", "mean_iou": run["selection_rules"][RULE]["mean_iou"], "map": run["average_precision"]["map_50_95"], "cpu_ms": run["seconds_per_image_median"] * 1000 if run["device"] == "cpu" else None})
    return section(source, {"rows": rows}) if rows else NOT_AVAILABLE


def optimisation_section() -> dict | str:
    optimisation = load("optimization_metrics")
    if not optimisation:
        return NOT_AVAILABLE
    return section("optimization_metrics.json", {"rows": [
        {"variant": f"{name} (deployed)" if name == optimisation["deploy_variant"] else name, "size_mb": values["size_mb"], "cpu_ms": values["cpu_latency_ms_b1"], "gpu_ms": None, "top1": values["test_top1"]}
        for name, values in optimisation["variants"].items()
    ]})


def robustness_section() -> dict | str:
    robustness = load("robustness")
    if not robustness:
        return NOT_AVAILABLE
    return section("robustness.json", {"rows": [{"perturbation": row["corruption"], "severity": row["severity"] + 1, "top1": row["deep"]["top1"]} for row in robustness["conditions"]]})


def confidence_section() -> dict | str:
    confidence, metrics = load("confidence"), load("classifier_metrics")
    if confidence:
        return section("confidence.json", {
            "chosen_threshold": confidence["thresholds"]["confident_p1"],
            "rows": [{"threshold": row["threshold"], "coverage": row["coverage"], "accuracy": row["accuracy_when_accepted"]} for row in confidence["test_coverage_curve"] if row["accuracy_when_accepted"] is not None],
        })
    if metrics:
        states = metrics["test"]["states"]["confident"]
        chosen = metrics["thresholds"]["confident_p1"]
        return section("classifier_metrics.json", {"chosen_threshold": chosen, "rows": [{"threshold": chosen, "coverage": states["share"], "accuracy": states["accuracy"]}]})
    return NOT_AVAILABLE


def comparison_section() -> dict | str:
    classical, metrics, optimisation, hybrid = load("classical_baseline"), load("classifier_metrics"), load("optimization_metrics"), load("hybrid")
    rows = []
    if classical:
        rows.append({"approach": "Classical", "top1": classical["test"]["top1"], "top5": classical["test"]["top5"], "inference_ms": classical["per_image_ms_cpu"], "size_mb": classical["model_size_mb"]})
    if metrics and optimisation:
        variant = optimisation["variants"][optimisation["deploy_variant"]]
        rows.append({"approach": "Deep (deployed)", "top1": variant["test_top1"], "top5": variant["test_top5"], "inference_ms": variant["cpu_latency_ms_b1"], "size_mb": variant["size_mb"]})
    if hybrid:
        name = hybrid["chosen_fusion"]
        rows.append({"approach": f"Hybrid ({name})", "top1": hybrid["test"][name]["top1"], "top5": hybrid["test"][name]["top5"], "inference_ms": None, "size_mb": None})
    return section("classical_baseline.json", {"rows": rows}) if rows else NOT_AVAILABLE


def ablation_section() -> dict | str:
    classical, hybrid, ablation = load("classical_baseline"), load("hybrid"), load("ablation_inference")
    optimisation = load("optimization_metrics")
    if ablation:
        baseline = ablation["with_annotated_box_crop"]["top1"]
        rows = [{"removed": "Vehicle crop (classifies the full image)", "top1": ablation["without_any_crop_full_image"]["top1"], "note": "Deep model, test split"}]
        if "detector_crop" in ablation:
            crop = ablation["detector_crop"]
            rows.append({"removed": "Annotated box (uses the detector's box)", "top1": crop["with_detector_crop"]["top1"], "note": f"{crop['images']} test images"})
        if hybrid:
            rows.append({"removed": "Classical features from the best fusion", "top1": hybrid["test"]["deep_only"]["top1"], "note": "Linear head, test split"})
        return section("ablation_inference.json", {"baseline_top1": baseline, "rows": rows})
    if optimisation:
        baseline = optimisation["variants"]["onnx_fp32"]["test_top1"]
        return section("optimization_metrics.json", {"baseline_top1": baseline, "rows": [
            {"removed": f"FP32 precision: {name}", "top1": optimisation["variants"][name]["test_top1"], "note": "Deep model, test split"} for name in optimisation["summary"]
        ]})
    if classical:
        families = classical["validation_by_family"]
        return section("classical_baseline.json", {"baseline_top1": families[-1]["top1"], "rows": [{"removed": f"Everything except {row['features']}", "top1": row["top1"], "note": "Classical, validation"} for row in families[:-1]]})
    return NOT_AVAILABLE


def confused_pairs_section() -> dict | str:
    path = RESULTS / "classifier_confused_pairs.csv"
    if not path.exists():
        return NOT_AVAILABLE
    with path.open(encoding="utf-8", newline="") as file:
        return section(path.name, {"rows": [{"a": row["class_a"], "b": row["class_b"], "count": int(row["total"])} for row in csv.DictReader(file)]})


def thumbnail(path: Path, side: int = 480) -> str:
    from PIL import Image

    with Image.open(path) as opened:
        image = opened.convert("RGB")
    image.thumbnail((side, side))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def failure_section() -> dict | str:
    path = RESULTS / "failure_cases.json"
    if not path.exists():
        return NOT_AVAILABLE
    cases = json.loads(path.read_text(encoding="utf-8"))
    return section(path.name, {"cases": [
        {"title": case["title"], "image": thumbnail(ROOT / case["image"]), "predicted": case["predicted"], "actual": case["actual"], "explanation": case["explanation"]}
        for case in cases
    ]})


def main() -> None:
    output = {
        "classifier": classifier_section(),
        "detection": detection_section(),
        "optimisation": optimisation_section(),
        "robustness": robustness_section(),
        "confidence": confidence_section(),
        "comparison": comparison_section(),
        "ablation": ablation_section(),
        "confused_pairs": confused_pairs_section(),
        "failure_cases": failure_section(),
    }
    (ROOT / "eval_P5_metrics.json").write_text(json.dumps(output, indent=1), encoding="utf-8")
    print({key: "ready" if value != NOT_AVAILABLE else NOT_AVAILABLE for key, value in output.items()})


if __name__ == "__main__":
    main()
