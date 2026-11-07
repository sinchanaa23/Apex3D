"""Write eval_P5_<TeamID>.md from the saved results, with metrics in P5's order.

    python scripts/build_eval_report.py --team T01 --usns USN1,USN2,USN3,USN4 [--team-name NAME --members "Name - USN; Name - USN"]

Every number comes from a file in results/. A missing result is reported as missing, so the file can
be generated at any time. Without --team and --usns, a clearly marked draft is written.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MISSING = "Not available yet"
RULE = "largest_confident"


def load(name: str) -> dict | None:
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def largest_run(pattern: str) -> dict | None:
    """The run with the most images, so that a full run replaces a sample automatically."""
    runs = [json.loads(Path(path).read_text(encoding="utf-8")) for path in glob.glob(str(RESULTS / pattern))]
    return max(runs, key=lambda run: run["images"]) if runs else None


SCREENSHOTS = {
    "landing": ("01-landing.png", "Landing page. The headline, the two ways in (Identify a car, See the evaluation) and the 3D car that opens the app."),
    "identity": ("02-identity.png", "Identity page for a red Audi R8. The detector box (95%) and the classifier's answer sit on the photo. The right side gives the confidence ring (100%, margin 99.9%, state Confident), the Manufacturer, Model and Year as separate fields, and buttons to confirm or reject. Timings: detection 1.97 s, classification 280 ms, total 2.25 s."),
    "top5": ("10-identity-top5.png", "A second Audi R8 photo, a harder case. The model answers Audi R8 Coupe 2012 at 53.2% with a 39.3% lead over the runner-up, just above the app's 0.52 confidence threshold. The Top 5 list shows what it considered: two Lamborghini coupes (13.9% and 10.9%), then two sedans (7.7% and 0.8%). The detector found the car with 99% confidence. Timings: detection 2.21 s, classification 357 ms, total 2.57 s."),
    "failures": ("11-lab-failure-cases.png", "Research Lab, failure cases. Three detector failures with the photo, the box each method chose, and the cause: a tiny background car outranks the main Ford Mustang (top-score box IoU 0.00), a large low-confidence box beats the right box on an Aston Martin V8 Vantage (chosen box IoU 0.28), and a very close FIAT 500 seen from behind is missed in favour of a sliver of another car (IoU 0.03)."),
    "references": ("03-references.png", "Reference Studio. After the user confirms the car, the app lists reference photos of that exact model (here a front view from Wikimedia Commons, CC BY 4.0, flagged 'Title matches model'). The clearest one is picked as the reconstruction input, and the user can switch to their own photo or continue to the 3D Studio."),
    "studio": ("04-studio-showcase.png", "3D Studio, showcase model. A 3D car in the style of the matched car (Audi R8 Coupe 2012), painted with the colour taken from the photo (#ba283e) and given coupe proportions. It can be downloaded as GLB, OBJ or STL. The Experimental mesh tab reads 'not built' until the user asks for it."),
    "mesh": ("05-studio-mesh.png", "3D Studio, experimental mesh. TripoSR rebuilt a rough mesh from one reference photo. The validation panel is measured on the file: watertight and manifold, but 97 degenerate faces and 4 connected components, so asset readiness is 'Needs repair'. The page warns that the unseen sides are inferred and the scale is unknown."),
    "robustness": ("06-lab-robustness.png", "Research Lab, robustness. Top-1 of the deployed model at three severities of each condition. Brightness, low light and noise cost a few points; heavy blur, low resolution and large occlusion cost the most."),
    "confidence": ("07-lab-confidence.png", "Research Lab, confidence and rejection. Raising the confidence threshold lowers coverage (orange) and raises accuracy on the images kept (blue). The vertical line marks the threshold the app uses (0.52)."),
    "comparison": ("08-lab-comparison.png", "Research Lab, classical against deep against hybrid, on the same split, with Top-1, Top-5, inference time and size. Adding classical features to the deep model gave no gain."),
    "ablation": ("09-lab-ablation.png", "Research Lab, ablation study. Each row removes or swaps one component of the pipeline and shows the Top-1 and the change against the baseline."),
}


def figure(key: str) -> list[str]:
    """A screenshot with its description, when the image file is in docs/screenshots/."""
    name, caption = SCREENSHOTS[key]
    if not (ROOT / "docs" / "screenshots" / name).exists():
        return []
    return [f"![{caption.split('.')[0]}](docs/screenshots/{name})", "", f"*{caption}*", ""]


def pct(value) -> str:
    return f"{value * 100:.2f}%" if value is not None else MISSING


def scope(run: dict) -> str:
    return f"random sample of {run['images']} test images" if run.get("is_random_sample") else f"all {run['images']} test images"


def table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def build(usns: str, team: str, team_name: str | None = None, members: str | None = None, live_url: str | None = None) -> str:
    classical, metrics, hybrid = load("classical_baseline"), load("classifier_metrics"), load("hybrid")
    optimisation = load("optimization_metrics")
    resnet = largest_run("detector_resnet50_test_*.json")
    mobilenet = largest_run("detector_mobilenet_test_*.json")
    validation_run = largest_run("detector_mobilenet_val_*.json")
    roi = largest_run("classical_roi_test_*.json")
    confidence, robustness, ablation = load("confidence"), load("robustness"), load("ablation_inference")
    detector_crops, baselines, preprocessing = load("detector_crop_accuracy"), load("baselines_metrics"), load("preprocessing_check")
    heading = f"# Evaluation results: P5, team {team_name} ({team})" if team_name else f"# Evaluation results: P5, team {team}"
    out = [usns, "", heading, ""]
    if members:
        out += ["**Team members:** " + members, ""]
    out += ["**Repository:** https://github.com/sinchanaa23/Apex3D", ""]
    if live_url:
        out += [f"**Live demo:** {live_url} (sign up on first use; the first visit after a sleep takes about a minute)", ""]
    out += ["Generated by `scripts/build_eval_report.py` from `results/`.", ""]

    out += ["## 1. Classification performance (Top-1 and Top-5)", ""]
    rows = []
    if classical:
        rows.append(["Classical (HOG, LBP, HSV, geometry + linear SVM)", pct(classical["test"]["top1"]), pct(classical["test"]["top5"]), pct(classical["test"]["macro_f1"]), classical["test_images"]])
    else:
        rows.append(["Classical", MISSING, MISSING, MISSING, "-"])
    if metrics:
        test = metrics["test"]
        rows.append([f"Deep ({metrics['model']['arch']})", pct(test["top1"]), pct(test["top5"]), pct(test["macro_f1"]), test["n"]])
    else:
        rows.append(["Deep", MISSING, MISSING, MISSING, "-"])
    if optimisation:
        variant = optimisation["deploy_variant"]
        values = optimisation["variants"][variant]
        rows.append([f"Deep, deployed ({variant})", pct(values["test_top1"]), pct(values["test_top5"]), "-", optimisation["test_images"]])
    if hybrid:
        name = hybrid["chosen_fusion"]
        rows.append([f"Hybrid ({name}, chosen on validation)", pct(hybrid["test"][name]["top1"]), pct(hybrid["test"][name]["top5"]), pct(hybrid["test"][name]["macro_f1"]), classical["test_images"] if classical else "-"])
    else:
        rows.append(["Hybrid", MISSING + " (needs the saved embeddings)", MISSING, MISSING, "-"])
    out += [table(["Approach", "Top-1", "Top-5", "Macro F1", "Test images"], rows), "", "All use annotated-box crops with an 8% margin on the unseen test split, with the same train and validation images. Chance is 0.51% Top-1 and 2.55% Top-5.", ""]
    if metrics:
        test = metrics["test"]
        out += [f"Deep make-level Top-1 is {pct(test['make_top1'])}. Of the deep model's errors, {pct(test['errors_same_make'])} are within the same make and {pct(test['errors_same_make_model_diff_year'])} differ only by year.", ""]
    if hybrid:
        out += [f"Deep features alone with the same linear head: {pct(hybrid['test']['deep_only']['top1'])} Top-1. Change from adding classical features: {hybrid['fusion_change_in_test_top1'] * 100:+.2f} percentage points.", ""]

    if detector_crops and detector_crops["runs"]:
        out += ["### The full pipeline: classifying detector crops", "", "The rows above use annotated boxes, which a real photo doesn't have. Here the deployed classifier receives the crop the app would use, on the same images each time.", ""]
        rows = []
        for label, name in (("MobileNetV3 (used by the app)", "mobilenet"), ("ResNet-50", "resnet50")):
            run = detector_crops["runs"].get(name)
            if run:
                rows.append([label, pct(run["annotated_box"]["top1"]), f"{pct(run['detector_crop']['top1'])} ± {run['detector_crop_top1_margin_95'] * 100:.1f}", pct(run["detector_crop"]["top5"]), pct(run["top_score_rule"]["top1"]), f"{'random sample of ' if run['is_random_sample'] else 'all '}{run['images']}"])
        out += [table(["Detector", "Annotated box Top-1", "Detector crop Top-1 (95% margin)", "Detector crop Top-5", "Top-1 with the old top-score rule", "Test images"], rows), ""]
        out += ["An image with no box scoring at least 0.5 is classified on its full frame, as in the app. On this sample, detector crops did not measurably lower accuracy compared with annotated boxes. Choosing the wrong detected car did.", ""]
    if baselines:
        out += ["### Backbone comparison", "", "Trained with the same split, crops, augmentation and schedule. Validation chose the model; test is shown for reporting.", ""]
        rows = [[values["arch"], f"{values['params_m']:.1f} M", pct(values["val_top1"]), pct(values["test_top1"]), pct(values["test_top5"]), pct(values["test_macro_f1"]), f"{values['cpu_fp32_ms']:.0f} ms"] for values in baselines["models"].values()]
        out += [table(["Backbone", "Parameters", "Validation Top-1", "Test Top-1", "Test Top-5", "Test macro F1", "CPU time per image"], rows), ""]
    if preprocessing:
        modes = preprocessing["modes"]
        out += ["### Why two Top-1 figures appear for the deployed model", "", preprocessing["explanation"], "", table(["Preprocessing", "Test Top-1", "Test Top-5", "Wrong images"], [[mode["name"], pct(mode["top1"]), pct(mode["top5"]), mode["wrong"]] for mode in modes]), ""]

    out += ["## 2. Object localisation (IoU)", ""]
    rows = []
    for label, run in (("Faster R-CNN ResNet-50 FPN v2", resnet), ("Faster R-CNN MobileNetV3 FPN", mobilenet)):
        if run:
            top = run["selection_rules"][RULE]
            rows.append([label, f"{top['mean_iou']:.3f}", f"{top['median_iou']:.3f}", pct(top["share_iou_at_least_50"]), pct(top["share_iou_at_least_75"]), scope(run)])
        else:
            rows.append([label, MISSING, MISSING, MISSING, MISSING, "-"])
    if roi:
        rows.append(["Classical contour ROI", f"{roi['mean_iou']:.3f}", f"{roi['median_iou']:.3f}", pct(roi["share_iou_at_least_50"]), pct(roi["share_iou_at_least_75"]), scope(roi)])
    out += [table(["Method", "Mean IoU", "Median IoU", "IoU ≥ 0.5", "IoU ≥ 0.75", "Images"], rows), ""]
    if roi:
        out += [f"Returning the whole image with no detection at all gives a mean IoU of {roi['whole_image_mean_iou']:.3f}, so that's the bar a useful method has to beat. The contour ROI covered at least 90% of the frame on {pct(roi['share_covering_90_percent_of_frame'])} of images.", ""]
    out += ["### Choosing which detected vehicle to crop", "", "Photos often contain several cars, and tiny background cars can score as high as the main subject. The rule was chosen on validation, then reported on test.", ""]
    rows = []
    for label, run in (("Validation, MobileNetV3", validation_run), ("Test, MobileNetV3", mobilenet), ("Test, ResNet-50", resnet)):
        if run:
            for rule, values in run["selection_rules"].items():
                rows.append([label, f"{rule} (used)" if rule == RULE else rule, f"{values['mean_iou']:.3f}", pct(values["share_iou_at_least_50"]), run["images"]])
    if rows:
        out += [table(["Run", "Selection rule", "Mean IoU", "IoU ≥ 0.5", "Images"], rows), ""]

    out += ["## 3. Object detection (mAP)", ""]
    rows = []
    for label, run in (("Faster R-CNN ResNet-50 FPN v2", resnet), ("Faster R-CNN MobileNetV3 FPN", mobilenet)):
        if run:
            ap = run["average_precision"]
            rows.append([label, f"{ap['map_50_95']:.3f}", f"{ap['ap50']:.3f}", f"{ap['ap75']:.3f}", scope(run)])
        else:
            rows.append([label, MISSING, MISSING, MISSING, "-"])
    out += [table(["Detector", "mAP@[.5:.95]", "AP@.5", "AP@.75", "Images"], rows), "", "Caveat: Stanford Cars annotates one car per image, so correct detections of other cars count as false positives and these values understate real precision. The vehicle class combines COCO car, bus and truck, and the detectors were not retrained.", ""]

    out += ["## 4. Model efficiency (size and inference time)", ""]
    rows = []
    if optimisation:
        for name, values in optimisation["variants"].items():
            label = f"{name} (deployed)" if name == optimisation["deploy_variant"] else name
            rows.append([label, f"{values['size_mb']:.1f} MB", f"{values['cpu_latency_ms_b1']:.1f} ms", pct(values["test_top1"]), "Intel i5-12500H, batch 1"])
    else:
        rows.append(["Deep classifier", MISSING, MISSING, MISSING, "-"])
    cost = load("hybrid_cost")
    if cost:
        rows.append([f"Hybrid ({cost['fusion']}): deployed deep model + geometry features + linear head", f"{cost['total_size_mb']:.1f} MB", f"{cost['total_inference_ms']:.1f} ms", pct(hybrid["test"][cost["fusion"]]["top1"]) if hybrid else "-", "see note below"])
    if classical:
        rows.append(["Classical model, including feature extraction", f"{classical['model_size_mb']:.1f} MB", f"{classical['per_image_ms_cpu']:.0f} ms", pct(classical["test"]["top1"]), classical["cpu"]])
    for label, run in (("ResNet-50 detector", resnet), ("MobileNetV3 detector", mobilenet)):
        if run and run.get("seconds_per_image_median"):
            rows.append([label, "-", f"{run['seconds_per_image_median'] * 1000:.0f} ms (median)", "-", f"{run['device']}, Intel i5-1135G7"])
    out += [table(["Model", "Size", "Time per image", "Test Top-1", "Hardware"], rows), "", "Optimisation: static INT8 quantization of the MatMul layers. Timings come from different laptops, so don't compare times between the two CPUs directly.", ""]
    if cost:
        out += [f"Hybrid cost: the deep model's {cost['deep_model']['size_mb']:.1f} MB and {cost['deep_model']['cpu_latency_ms_b1']:.1f} ms are Person A's figures. Computing the geometry features ({cost['geometry_features_ms_median']:.1f} ms) and applying the linear head ({cost['linear_head_ms_median']:.1f} ms, {cost['linear_head_size_mb']:.1f} MB) were timed here on {cost['test_images_timed']} test crops, so the hybrid total mixes two machines. The extra steps are small next to the deep model.", ""]

    out += ["## 5. Robustness (performance under varying conditions)", ""]
    if robustness:
        out += [f"Evaluated on {robustness['images']} randomly sampled test images. Clean deep Top-1: {pct(robustness['deep_clean']['top1'])}.", ""]
        rows = [[row["corruption"], row["level"], pct(row["deep"]["top1"]), pct(row["classical"]["top1"]) if "classical" in row else "-", pct(row["deep_with_adaptive_clahe"]["top1"]) if "deep_with_adaptive_clahe" in row else "-"] for row in robustness["conditions"]]
        out += [table(["Condition", "Level", "Deep Top-1", "Classical Top-1", "Deep + adaptive CLAHE"], rows), ""]
    else:
        out += [f"{MISSING}. Needs the trained checkpoint.", ""]

    out += figure("robustness")
    out += ["## 6. Confidence (prediction confidence and threshold analysis)", ""]
    if metrics:
        calibration, thresholds, states = metrics["calibration"], metrics["thresholds"], metrics["test"]["states"]
        out += [f"Temperature scaling (T = {calibration['temperature']:.3f}) was fitted on validation, reducing validation calibration error from {calibration['val_ece_before']:.3f} to {calibration['val_ece_after']:.3f}. Test calibration error is {metrics['test']['ece_calibrated']:.3f}. Thresholds were chosen on validation: confident at probability ≥ {thresholds['confident_p1']} and margin ≥ {thresholds['confident_margin']}, rejected below {thresholds['reject_below_p1']}.", ""]
        out += [table(["App state", "Share of test images", "Accuracy"], [[state, pct(values["share"]), pct(values["accuracy"])] for state, values in states.items()]), ""]
        outside = load("out_of_taxonomy")
        if outside:
            out += ["### Photos outside the 196 classes", "", "The classifier always names one of its 196 classes, so only the confidence rule protects against inputs it was never taught. "
                    f"Measured on the Caltech-101 test split: {outside['source']}.", ""]
            labels = {"in_class_uncropped": "Stanford Cars photos, uncropped (for comparison)", "other_vehicles": "Other vehicles (airplanes, motorbikes, helicopters, boats)", "other_things": "Other things (animals, objects, scenes)"}
            out += [table(["Photos", "Images", "Rejected", "Ambiguous", "Confident"], [[labels[name], values["images"], pct(values["rejected"]), pct(values["ambiguous"]), pct(values["confident"])] for name, values in outside["groups"].items()]), "",
                    f"Separating in-class from outside photos by top probability alone gives an AUROC of {outside['separation_auroc']:.3f}.", "", "Limits: " + " ".join(outside["limits"]), ""]
        else:
            out += ["Out-of-taxonomy vehicles were not part of this test, so rejection of unknown cars isn't measured here.", ""]
    else:
        out += [f"{MISSING}.", ""]
    if confidence:
        curve = [row for row in confidence["test_coverage_curve"] if row["accuracy_when_accepted"] is not None][::6]
        out += ["### Coverage against accuracy (test)", "", table(["Confidence threshold", "Images accepted", "Accuracy when accepted"], [[row["threshold"], pct(row["coverage"]), pct(row["accuracy_when_accepted"])] for row in curve]), ""]

    out += figure("confidence")
    out += ["## Comparison: classical, deep and hybrid", "", "See section 1.", ""]
    out += figure("comparison")
    if classical:
        out += ["### Classical feature families (validation)", ""]
        out += [table(["Features", "Dimensions", "Top-1", "Top-5"], [[row["features"], row["dimensions"], pct(row["top1"]), pct(row["top5"])] for row in classical["validation_by_family"]]), ""]
        clahe = classical["validation_clahe"]
        change = (clahe["with"]["top1"] - clahe["without"]["top1"]) * 100
        out += [f"Adaptive CLAHE changed classical validation Top-1 by {change:+.2f} percentage points, on 1,222 validation images. A change this size may be noise.", ""]

    out += ["## Ablation study", ""]
    ablation_figure = figure("ablation")
    rows = []
    if classical:
        best = classical["validation_by_family"][-1]
        rows += [[f"Classical: {row['features']} features only", "Validation", pct(row["top1"]), f"{(row['top1'] - best['top1']) * 100:+.2f}"] for row in classical["validation_by_family"][:-1]]
    if resnet:
        rules = resnet["selection_rules"]
        rows.append([f"Detector: top score instead of largest confident ({resnet['images']} images, mean IoU)", "Test", f"{rules['top_score']['mean_iou']:.3f}", f"{rules['top_score']['mean_iou'] - rules[RULE]['mean_iou']:+.3f} IoU"])
    if detector_crops and detector_crops["runs"].get("mobilenet"):
        run = detector_crops["runs"]["mobilenet"]
        rows.append([f"Full pipeline: top-score rule instead of largest confident ({run['images']} images)", "Test", pct(run["top_score_rule"]["top1"]), f"{(run['top_score_rule']['top1'] - run['detector_crop']['top1']) * 100:+.2f}"])
        rows.append([f"Full pipeline: annotated box instead of detector crop ({run['images']} images)", "Test", pct(run["annotated_box"]["top1"]), f"{(run['annotated_box']['top1'] - run['detector_crop']['top1']) * 100:+.2f}"])
    if baselines:
        models = baselines["models"]
        rows += [[f"Backbone: {values['arch']} instead of ConvNeXt-Tiny", "Test", pct(values["test_top1"]), f"{(values['test_top1'] - models['convnext_tiny']['test_top1']) * 100:+.2f}"] for name, values in models.items() if name != "convnext_tiny"]
    if optimisation:
        rows += [[f"Deep: {name} instead of ONNX FP32", "Test", pct(optimisation["variants"][name]["test_top1"]), f"{values['top1_change_vs_onnx_fp32'] * 100:+.2f}"] for name, values in optimisation["summary"].items()]
    if ablation:
        reference = ablation["with_annotated_box_crop"]["top1"]
        rows.append(["Deep: no crop, full image", "Test", pct(ablation["without_any_crop_full_image"]["top1"]), f"{(ablation['without_any_crop_full_image']['top1'] - reference) * 100:+.2f}"])
        if "detector_crop" in ablation:
            crop = ablation["detector_crop"]
            value, same = crop["with_detector_crop"]["top1"], crop["with_annotated_box_on_same_images"]["top1"]
            rows.append([f"Deep: detector crop, not annotated box ({crop['images']} images)", "Test", pct(value), f"{(value - same) * 100:+.2f}"])
    if hybrid:
        rows += [[f"Hybrid: {name}", "Validation", pct(values["top1"]), f"{(values['top1'] - hybrid['validation']['deep_only']['top1']) * 100:+.2f}"] for name, values in hybrid["validation"].items() if name != "deep_only"]
    out += [table(["Component changed", "Split", "Result", "Change (points)"], rows) if rows else MISSING, "", "Classical rows compare against all classical features, optimisation rows against ONNX FP32, no-crop against the annotated box, full-pipeline rows against the app's detector crop, backbone rows against ConvNeXt-Tiny, and hybrid rows against deep features alone.", ""]
    if not ablation:
        out += [f"Deep crop ablations: {MISSING}. They need the trained checkpoint.", ""]

    analysis = ROOT / "docs" / "FAILURE_ANALYSIS.md"
    out += ablation_figure
    footprint, onnx_parity = load("deployment_footprint"), load("detector_onnx_parity")
    out += ["## Deployment and serving footprint", ""]
    if footprint:
        out += [f"Live: {footprint['live_url']}. {footprint['hosting']}", "", f"Measured on {footprint['machine']}", ""]
        rows = [[v["configuration"], f"{v['memory_after_startup_mb']} MB", f"{v['memory_peak_mb']} MB", f"{v['recognition_seconds'][0]:.2f} to {v['recognition_seconds'][1]:.2f} s", v["how"]] for v in footprint["server"]]
        out += [table(["Server configuration", "Memory after start-up", "Memory at peak", "Time per recognition", "How measured"], rows), "",
                "The detector was moved from PyTorch to ONNX Runtime (the same network, exported by `scripts/export_detector_onnx.py`) and the classifier's preprocessing from torchvision to NumPy and Pillow, so the server no longer loads PyTorch. "
                "That cut memory at start-up from 577 MB to 207 MB and at peak from 816 MB to 257 MB, which is what lets it fit a 512 MB host.", ""]
        rows = [[m["cpu_threads"], f"{m['seconds']} s ({m['seconds'] / 60:.1f} min)", f"{m['memory_peak_mb']} MB", m["note"]] for m in footprint["mesh_builder"]]
        out += ["3D mesh builder (TripoSR on CPU), one mesh:", "", table(["CPU threads", "Time", "Whole-server memory at peak", "Note"], rows), "",
                f"Online: {'on' if footprint['mesh_builder_online'] else 'off'}. {footprint['mesh_builder_online_reason']}", ""]
    else:
        out += [MISSING + " (needs results/deployment_footprint.json).", ""]
    if onnx_parity:
        n, box, gt = onnx_parity["photos"], onnx_parity["chosen_box_iou_onnx_vs_pytorch"], onnx_parity["mean_iou_with_true_box"]
        out += [f"The ONNX detector is not bit-identical to PyTorch. On {n} random test photos (`scripts/check_detector_onnx.py`):", "",
                table(["Measure", "PyTorch detector", "ONNX detector"],
                      [["Mean IoU with the true box", f"{gt['pytorch']:.4f}", f"{gt['onnx']:.4f}"],
                       ["Photos classified correctly", f"{onnx_parity['correct']['pytorch']} of {n}", f"{onnx_parity['correct']['onnx']} of {n}"],
                       ["Photos with no box found", onnx_parity["no_box_found"]["pytorch"], onnx_parity["no_box_found"]["onnx"]]]), "",
                f"The chosen box matches the PyTorch box with a mean IoU of {box['mean']:.4f} (lowest {box['min']:.3f}; {pct(box['share_at_least_0.99'])} of photos at IoU 0.99 or better), and {onnx_parity['same_final_class']} of {n} photos get the same final car class. "
                "The detector figures in sections 2 and 3 were measured with the PyTorch detector.", ""]
    out += ["## Failure cases", ""]
    detector_cases = load("failure_cases") or []
    classifier_failures = load("classifier_failure_cases")
    if detector_cases:
        out += ["### Detector and classical-ROI failures", ""]
        out += figure("failures")
        out += [table(["Case", "Result", "Cause"], [[case["title"], f"{case['predicted']} (true: {case['actual']})", case["explanation"].replace("|", "/")] for case in detector_cases]), ""]
    if classifier_failures:
        summary = classifier_failures["summary"]
        errors = summary["errors"]
        out += ["### Classifier failures (deployed model, real test set)", "",
                f"{errors} of 8,041 test images were wrong: {pct(summary['same_model_different_year'] / errors)} same manufacturer and model with a different year, "
                f"{pct(summary['same_make_different_model'] / errors)} same manufacturer with a different model, {pct(summary['different_make'] / errors)} a different manufacturer. "
                f"{summary['errors_with_confidence_at_least_0_9']} errors still had at least 90% confidence.", "",
                "The cases below are the most confident wrong answers, picked to show distinct causes. They are not a random sample.", ""]
        out += [table(["Case", "True label", "Model's answer", "Cause"], [[case["title"], case["actual"], case["predicted"], case["explanation"].replace("|", "/")] for case in classifier_failures["cases"]]), ""]
    out += ["Full write-ups with images: " + ", ".join(f"`{name}`" for name in ("docs/FAILURE_ANALYSIS.md", "docs/CLASSIFIER_FAILURES.md") if (ROOT / name).exists()) + "." if analysis.exists() else f"{MISSING}.", ""]
    app_screens = [line for key in ("landing", "identity", "top5", "references", "studio", "mesh") for line in figure(key)]
    if app_screens:
        out += ["## The application", "", "Screenshots of the running app. A screen recording of the same flow is in `docs/demo/demo.mp4`.", ""] + app_screens
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--team")
    parser.add_argument("--team-name", help="Team name, shown in the heading.")
    parser.add_argument("--members", help="Full names with USNs, shown under the heading.")
    parser.add_argument("--live-url", help="Public address of the running app, shown under the heading.")
    parser.add_argument("--usns", help="Comma-separated USNs, written as the first line.")
    args = parser.parse_args()
    if args.team and args.usns:
        path, text = ROOT / f"eval_P5_{args.team}.md", build(args.usns, args.team, args.team_name, args.members, args.live_url)
    else:
        path, text = ROOT / "eval_P5_DRAFT.md", build("DRAFT: rerun with --team and --usns before submitting", "not set")
    path.write_text(text, encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
