"""Prepare the Hackathon Technical Solution Report (template: "Hackathon Technical Solution Report").

    python scripts/build_report_facts.py            writes report/FACTS_SHEET.md, creates report/report_P5_T25.md if it is missing,
                                                    and refreshes the data tables inside it
    python scripts/build_report_facts.py --reset    throws away report/report_P5_T25.md and starts from a fresh skeleton

What this does and does not do. The hackathon limits AI-written content in the final report to 30% and requires AI help to be
declared. So this script writes no explanation, reasoning or analysis. It writes:
  * report/FACTS_SHEET.md: for each template section, the measured facts with the file each came from, the repository modules
    to point at, and the figures that exist. It is a reference for the team to write from, not text to paste.
  * report/report_P5_T25.md: the template's headings, with every explanation left as a [WRITE: ...] slot for the team, and the
    numeric tables filled in from the real result files. Those tables sit between <!-- AUTO:name --> markers, which is the only
    part this script ever rewrites, so the team's prose is safe to re-run over.
Nothing is invented: every number comes from a file in results/ or from git.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORT_DIR = ROOT / "report"
REPORT = REPORT_DIR / "report_P5_T25.md"
FACTS = REPORT_DIR / "FACTS_SHEET.md"

USNS = "1RUA24CSE0461, 1RUA24CSE0467, 1RUA24CSE0445, 1RUA24CSE0453"
REPO_URL = "https://github.com/sinchanaa23/Apex3D"
LIVE_URL = "https://apex3d-xvuc.onrender.com"
TEAM_NAME = "NoharaFam"
MEMBERS = [
    ("Sinchana R Kambar", "1RUA24CSE0461"),
    ("Sohini Dutta", "1RUA24CSE0467"),
    ("Shreyas Muniraju Kolar", "1RUA24CSE0445"),
    ("Shubanck Prasoon Guptha", "1RUA24CSE0453"),
]
TITLE = "Intelligent Fine-Grained Vehicle Recognition and Analysis System"


def load(name: str):
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def pct(value, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%" if value is not None else "n/a"


def table(head: list[str], rows: list[list], align: list[str] | None = None) -> str:
    align = align or [":---"] + [":---:"] * (len(head) - 1)
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# The data tables (all numbers from results/)
# ---------------------------------------------------------------------------

def table_main() -> str:
    classical, classifier, hybrid, cost, optimisation = load("classical_baseline.json"), load("classifier_metrics.json"), load("hybrid.json"), load("hybrid_cost.json"), load("optimization_metrics.json")
    rows = []
    if classical:
        rows.append(["**Classical baseline** (HSV, LBP, HOG, geometry + linear SVM)", pct(classical["test"]["top1"]), pct(classical["test"]["top5"]), pct(classical["test"]["macro_f1"]), f"{classical['per_image_ms_cpu']:.1f}", f"{classical['model_size_mb']:.1f}"])
    if classifier and optimisation:
        deployed = optimisation["variants"][optimisation["deploy_variant"]]
        rows.append([f"**Deep learning only** (ConvNeXt-Tiny, deployed {optimisation['deploy_variant']})", pct(deployed["test_top1"]), pct(deployed["test_top5"]), pct(classifier["test"].get("macro_f1")), f"{deployed['cpu_latency_ms_b1']:.1f}", f"{deployed['size_mb']:.1f}"])
    if hybrid and cost:
        name = hybrid["chosen_fusion"]
        rows.append([f"**Hybrid** (deep features + classical geometry features, `{name}`)", pct(hybrid["test"][name]["top1"]), pct(hybrid["test"][name]["top5"]), pct(hybrid["test"][name]["macro_f1"]), f"{cost['total_inference_ms']:.1f}", f"{cost['total_size_mb']:.1f}"])
    return table(["Approach / Model Variant", "Top-1 Accuracy", "Top-5 Accuracy", "Macro F1", "Latency (ms / image, CPU)", "Model size (MB)"], rows)


def table_efficiency() -> str:
    optimisation = load("optimization_metrics.json")
    if not optimisation:
        return "_results/optimization_metrics.json not found._"
    rows = [[f"{name}{' (deployed)' if name == optimisation['deploy_variant'] else ''}", f"{v['size_mb']:.1f}", f"{v['cpu_latency_ms_b1']:.1f}", pct(v["test_top1"])] for name, v in optimisation["variants"].items()]
    return table(["Variant", "Size (MB)", "CPU latency (ms, batch 1)", "Test Top-1"], rows)


def table_robustness() -> str:
    data = load("robustness.json")
    if not data:
        return "_results/robustness.json not found._"
    rows = [[r["corruption"], r["level"], pct(r["deep"]["top1"]), pct(r["classical"]["top1"]) if "classical" in r else "-"] for r in data["conditions"]]
    return f"Measured on {data['images']} randomly sampled test images; clean deep Top-1 on that sample is {pct(data['deep_clean']['top1'])}.\n\n" + table(["Condition", "Level", "Deep Top-1", "Classical Top-1"], rows)


def table_confidence() -> str:
    classifier = load("classifier_metrics.json")
    if not classifier:
        return "_results/classifier_metrics.json not found._"
    states = classifier["test"]["states"]
    t = classifier["thresholds"]
    rows = [[state, pct(v["share"]), pct(v["accuracy"])] for state, v in states.items()]
    return f"Thresholds chosen on validation: confident at probability >= {t['confident_p1']} with margin >= {t['confident_margin']}, rejected below {t['reject_below_p1']}.\n\n" + table(["App state", "Share of test images", "Accuracy"], rows)


def table_oot() -> str:
    data = load("out_of_taxonomy.json")
    if not data:
        return "_results/out_of_taxonomy.json not found._"
    labels = {"in_class_uncropped": "Stanford Cars photos (in the 196 classes), uncropped", "other_vehicles": "Other vehicles (airplanes, motorbikes, helicopters, boats)", "other_things": "Other things (animals, objects, scenes)"}
    rows = [[labels.get(name, name), v["images"], pct(v["rejected"], 1), pct(v["ambiguous"], 1), pct(v["confident"], 1)] for name, v in data["groups"].items()]
    return table(["Photos", "Images", "Rejected", "Ambiguous", "Confident"], rows) + f"\n\nAUROC for separating in-class from outside photos by top probability: {data['separation_auroc']:.3f}."


def table_ablation() -> str:
    ablation, hybrid, baselines, crops = load("ablation_inference.json"), load("hybrid.json"), load("baselines_metrics.json"), load("detector_crop_accuracy.json")
    rows = []
    if ablation:
        base = ablation["with_annotated_box_crop"]["top1"]
        rows.append(["Vehicle crop removed (classify the full image)", pct(ablation["without_any_crop_full_image"]["top1"]), f"{(ablation['without_any_crop_full_image']['top1'] - base) * 100:+.2f}"])
    run = ((crops or {}).get("runs") or {}).get("mobilenet")
    if run:
        rows.append([f"Annotated box replaced by the detector's box ({run['images']} test images; annotated box scored {pct(run['annotated_box']['top1'], 1)} on them)", pct(run["detector_crop"]["top1"]), f"{(run['detector_crop']['top1'] - run['annotated_box']['top1']) * 100:+.2f}"])
        rows.append([f"Detector's box chosen by top score instead of the app's rule ({run['images']} images)", pct(run["top_score_rule"]["top1"]), f"{(run['top_score_rule']['top1'] - run['annotated_box']['top1']) * 100:+.2f}"])
    if hybrid:
        rows.append(["Classical geometry features removed from the hybrid", pct(hybrid["test"]["deep_only"]["top1"]), f"{(hybrid['test']['deep_only']['top1'] - hybrid['test'][hybrid['chosen_fusion']]['top1']) * 100:+.2f}"])
    if baselines:
        ref = baselines["models"]["convnext_tiny"]["test_top1"]
        rows += [[f"Backbone: {v['arch']} instead of ConvNeXt-Tiny", pct(v["test_top1"]), f"{(v['test_top1'] - ref) * 100:+.2f}"] for k, v in baselines["models"].items() if k != "convnext_tiny"]
    return table(["Component changed", "Test Top-1", "Change vs. reference (points)"], rows) + "\n\nReference for each row: the deployed deep model with annotated-box crops, except the hybrid row, which compares against the hybrid itself."


def table_failures() -> str:
    data = load("classifier_failure_cases.json")
    if not data:
        return "_results/classifier_failure_cases.json not found._"
    s = data["summary"]
    rows = [[c["title"], c["actual"], c["predicted"], pct(c["confidence"], 1), pct(c["true_class_probability"], 2)] for c in data["cases"]]
    head = f"The deployed model was wrong on {s['errors']} of 8,041 test images; {s['errors_with_confidence_at_least_0_9']} of those errors had confidence of at least 90%.\n\n"
    return head + table(["Case", "True label", "Model's answer", "Confidence", "Probability of the true class"], rows) + "\n\nThe cases are the most confident wrong answers, picked to show different causes; they are not a random sample. Causes are for the team to write."


BLOCKS = {
    "table-main": table_main,
    "table-efficiency": table_efficiency,
    "table-robustness": table_robustness,
    "table-confidence": table_confidence,
    "table-oot": table_oot,
    "table-ablation": table_ablation,
    "table-failures": table_failures,
}


def refresh(text: str) -> str:
    """Rewrite every <!-- AUTO:name --> ... <!-- /AUTO --> block; leave everything else exactly as the team wrote it."""
    commit = git("rev-parse", "--short", "HEAD")

    def fill(match: re.Match) -> str:
        name = match.group(1)
        if name == "commit":
            return f"<!-- AUTO:commit -->`{commit}`<!-- /AUTO -->"
        if name not in BLOCKS:
            return match.group(0)
        return f"<!-- AUTO:{name} -->\n{BLOCKS[name]()}\n<!-- /AUTO -->"

    return re.sub(r"<!-- AUTO:([\w-]+) -->.*?<!-- /AUTO -->", fill, text, flags=re.S)


# ---------------------------------------------------------------------------
# The report skeleton: the template's headings, data filled in, every explanation left to the team
# ---------------------------------------------------------------------------

FIGURES = [
    ("../results/figures/classical_stages.png", "classical pipeline stages on one photo"),
    ("../results/plots/training_curves.png", "training curves of the deployed model"),
    ("../results/plots/baselines_comparison.png", "the three backbones compared"),
    ("../results/plots/calibration_coverage.png", "calibration and coverage against accuracy"),
    ("../docs/screenshots/02-identity.png", "the app's identity page: box, make, model, year, confidence"),
    ("../docs/screenshots/05-studio-mesh.png", "the 3D studio's experimental mesh and its validation"),
]


def skeleton() -> str:
    figures = "\n\n".join(f"![{what}]({path})\n\n*[WRITE: one line saying what this figure shows]*" for path, what in FIGURES)
    return f"""# Hackathon Technical Solution Report

| Field | Details |
| :--- | :--- |
| **Problem ID** | P5 |
| **Problem Title** | {TITLE} |
| **Team Name** | {TEAM_NAME} |
| **Team Members** | {"; ".join(f"{n} ({u})" for n, u in MEMBERS)} |
| **GitHub Repository** | {REPO_URL} , commit evaluated: <!-- AUTO:commit -->`x`<!-- /AUTO --> |
| **Live Demo / Deployment** | {LIVE_URL} (Render, free plan) |

---

## 1. Problem Understanding & Solution Strategy

* **Core Technical Challenge:** [WRITE: the main technical barrier and the failure modes you targeted, with the constraints you worked under]
* **Proposed Methodology:** [WRITE: 2 to 3 sentences on your end-to-end strategy and why you chose it over a single static model]

---

## 2. Technical Architecture & Implementation

> *Reference repository modules; do not paste code. Module map: see report/FACTS_SHEET.md.*

### 2.1 Classical Algorithmic Baseline

* **Algorithms Utilized:** [WRITE]
* **Adaptive Selection Logic:** [WRITE]

### 2.2 Deep Learning Pipeline

* **Model Architecture:** [WRITE: the model you chose and why]
* **Loss Functions & Training:** [WRITE: loss, schedule, augmentation and data used]

### 2.3 Hybrid Integration & Adaptive Mechanism

* **Fusion Strategy:** [WRITE: how the classical and deep parts connect]
* **Decision Logic:** [WRITE: how the system decides between a confident answer, an ambiguous one and a rejection]

---

## 3. Experimental Benchmarks & Evaluation

### 3.1 Quantitative Benchmarks

<!-- AUTO:table-main -->
<!-- /AUTO -->

* **Analysis:** [WRITE: 2 to 3 sentences on the trade-offs between accuracy and speed that the table shows]

Additional measured tables (delete any you do not need):

**Model optimisation (size and speed)**

<!-- AUTO:table-efficiency -->
<!-- /AUTO -->

**Robustness to illumination, blur, noise, resolution and occlusion**

<!-- AUTO:table-robustness -->
<!-- /AUTO -->

**Confidence and rejection**

<!-- AUTO:table-confidence -->
<!-- /AUTO -->

**Photos outside the 196 classes**

<!-- AUTO:table-oot -->
<!-- /AUTO -->

### 3.2 Visual Outputs & Intermediate Representations

{figures}

### 3.3 Ablation Study

<!-- AUTO:table-ablation -->
<!-- /AUTO -->

* [WRITE: what the ablation shows about each component]

---

## 4. Failure Modes & Limitations

* **Observed Failure Case:** [WRITE: at least one scenario where the system went wrong]
* **Root Cause & Mitigation:** [WRITE: why it happened and what you would change]

Measured facts about the failures:

<!-- AUTO:table-failures -->
<!-- /AUTO -->

* **Limitations and ethics:** [WRITE: the limits of the evaluation and of the system, and the ethical points: photos uploaded by users, accounts with unverified emails, label noise in the dataset, the dataset's age and coverage]

---

## 5. Deployment & Quickstart Guide

### Quickstart Commands

```bash
# 1. Clone the repository and install the dependencies
git clone {REPO_URL}
cd Apex3D
pip install -r requirements.txt
# 2. Build the frontend once, then start the app and open http://localhost:8000 (sign up on first use)
cd frontend && npm ci && npm run build && cd ..
python start_server.py
```

[TODO: run these commands on a clean clone and correct them before submitting]

### Deployment Format

* [WRITE: how the app is packaged and where it runs (Docker image, ONNX runtime, the live URL)]

---

## AI Assistance Declaration

[WRITE: state honestly which parts of the project and of this report AI tools helped with, which tools, and your estimate of the share of this report's text that is AI-written. The policy allows AI help with system design if declared, and caps AI content in the report at 30%.]

Facts the team can build the declaration from: Claude (Anthropic), used through Claude Code, was used on this repository for the React frontend, the FastAPI integration and login system, the evaluation scripts that measured the deployed model, the metric and report builders, the deployment files and the repository documentation. [TODO: add what each other member used, and which parts were done without AI]

---

### Student Submission Checklist

- [ ] Working GitHub repository link included with an updated `README.md`.
- [ ] No raw code pasted into this report.
- [ ] All table metrics match test runs reproducible via the repository.
- [ ] All team member details accurately recorded.
- [ ] AI assistance declared, and the AI share of this report is at most 30%.
"""


# ---------------------------------------------------------------------------
# The facts sheet
# ---------------------------------------------------------------------------

MODULES = [
    ("Classical CV pipeline (preprocessing, edges, contours, colour, texture, ROI, geometry)", "backend/classical_cv/pipeline.py, backend/classical_cv/api.py, backend/classical_cv/batch.py"),
    ("Classical baseline: features, linear SVM, tuning", "notebooks/02_classical_baseline.ipynb, results/classical_baseline.json"),
    ("Deep classifier training (ConvNeXt-Tiny, transfer learning)", "notebooks/01_classifier_training.ipynb, notebooks/cls_lib.py, results/classifier_metrics.json"),
    ("Optimisation: quantisation and ONNX export", "notebooks/01b_export_quantize.ipynb, results/optimization_metrics.json"),
    ("Backbone comparison", "notebooks/01c_baseline_models.ipynb, results/baselines_metrics.json"),
    ("Vehicle detector (Faster R-CNN) and box selection", "backend/detection/detector.py, backend/detection/metrics.py, results/detector_*.json"),
    ("ONNX classifier used by the app", "backend/recognition/onnx_classifier.py, checkpoints/model_quantized.onnx"),
    ("Confidence, calibration and thresholds", "backend/evaluation/confidence.py, results/confidence.json"),
    ("Hybrid, robustness and ablation experiments", "backend/evaluation/classifier_experiments.py, backend/evaluation/corruptions.py, scripts/evaluate_deployed_model.py"),
    ("Out-of-taxonomy test", "scripts/evaluate_out_of_taxonomy.py, results/out_of_taxonomy.json"),
    ("API, login, job queue", "backend/main.py, backend/auth.py, backend/api/, backend/services/"),
    ("3D reconstruction (TripoSR bridge, mesh library)", "backend/reconstruction/, backend/services/mesh_library.py"),
    ("Frontend", "frontend/src/ (pages: Upload, Identity, Classical, References, Studio, Lab)"),
    ("Evaluation report generator", "scripts/build_eval_report.py, scripts/build_app_metrics.py, eval_P5_T25.md"),
]

FIGURE_FILES = [
    ("results/figures/classical_stages.png", "the classical stages on one photo"),
    ("results/figures/classical_confusion.png", "classical model confusion"),
    ("results/figures/classical_families.png", "classical feature families compared"),
    ("results/plots/training_curves.png", "training curves"),
    ("results/plots/confusion_matrix.png", "deep model confusion matrix"),
    ("results/plots/calibration_coverage.png", "calibration and coverage"),
    ("results/plots/baselines_comparison.png", "three backbones compared"),
    ("results/plots/eda_overview.png", "dataset overview"),
    ("results/plots/augmentations.png", "training augmentations"),
    ("results/figures/classifier_failures.jpg", "sheet of confident wrong answers"),
    ("docs/screenshots/02-identity.png", "app: identity page"),
    ("docs/screenshots/10-identity-top5.png", "app: identity with top-5 list"),
    ("docs/screenshots/05-studio-mesh.png", "app: 3D mesh and validation"),
]


def facts_sheet() -> str:
    classifier, classical, hybrid, cost, optimisation = load("classifier_metrics.json"), load("classical_baseline.json"), load("hybrid.json"), load("hybrid_cost.json"), load("optimization_metrics.json")
    detectors = {name: load(f"detector_{name}_test_sample500.json") for name in ("resnet50", "mobilenet")}
    crops = load("detector_crop_accuracy.json")
    out = ["# Facts sheet for the report", "",
           "A reference to write from. It holds measured numbers with their source files, repository modules and the figures that exist. It contains no explanation on purpose: the explanations in the report should be the team's own words (the hackathon caps AI-written content at 30%). Regenerate with `python scripts/build_report_facts.py`.", ""]

    out += ["## Header", "", f"- Problem: P5, {TITLE}", f"- Team ID: T25. USNs: {USNS}", f"- Repository: {REPO_URL}", f"- Commit to quote (HEAD when this was generated): `{git('rev-parse', 'HEAD')}`. Push everything and regenerate right before submitting so the hash is the final one.",
            "- Live URL: {LIVE_URL} (Render free plan: accounts reset when it restarts, it sleeps after about 15 minutes idle, and the mesh builder is off). Checked from outside: /api/health answers with detector onnx and mesh_builder false, protected routes answer 401, /docs is hidden. Sign-up and recognition on the live site were not tested by the tool that wrote this.".format(LIVE_URL=LIVE_URL), ""]

    if classifier:
        d, m, t = classifier["dataset"], classifier["model"], classifier["training"]
        out += ["## Section 1: the problem and the data", "",
                f"- Dataset: {d['name']}, {d['classes']} classes; split used here: {d['train']} train, {d['val']} validation, {d['test']} test. Input to the classifier: {d['input']}. (`results/classifier_metrics.json`, dataset)",
                f"- Fine-grained difficulty, measured: of the deep model's errors, {pct(classifier['test']['errors_same_make'])} are within the same make and {pct(classifier['test']['errors_same_make_model_diff_year'])} are the same make and model with a different year. Make-level Top-1 is {pct(classifier['test']['make_top1'])}.",
                f"- What the rubric and problem statement ask for: make, model and year, confidence, bounding box, low-confidence handling, comparison of classical, deep and hybrid, ablation, one optimisation technique, unseen test images, at least three failure cases.", ""]
        out += ["## Section 2: architecture facts", "", "### Repository modules to reference", "", table(["Part", "Where"], [[a, b] for a, b in MODULES], [":---", ":---"]), ""]
        out += ["### Deep model and training (`results/classifier_metrics.json`)", "",
                f"- Architecture: {m['arch']}, {m['params_m']:.1f} M parameters, input {m['img_size']} px, pretrained: {m['pretrained']}.",
                f"- Schedule: {t['head_epochs']} head-only epochs, then fine-tuning up to {t['max_ft_epochs']} epochs; {t['epochs_run']} epochs run, best at epoch {t['best_epoch']}. Learning rate {t['lr_head']} (head) and {t['lr_ft']} (fine-tune), layer decay {t['layer_decay']}, weight decay {t['weight_decay']}, label smoothing {t['label_smoothing']}, mixed precision {t['amp']}.",
                f"- Augmentation: {t['augmentation']}.",
                f"- Hardware used for training: {classifier['environment'].get('gpu')}.", ""]
    if classical:
        out += ["### Classical baseline (`results/classical_baseline.json`)", "",
                f"- Features: HSV colour histogram, LBP texture, HOG, and 12 geometric features; linear SVM, chosen C = {classical['chosen_C']}; adaptive CLAHE used: {classical['adaptive_clahe']}.",
                f"- Test Top-1 {pct(classical['test']['top1'])}, Top-5 {pct(classical['test']['top5'])}, macro F1 {pct(classical['test']['macro_f1'])}; {classical['per_image_ms_cpu']:.1f} ms per image, {classical['model_size_mb']:.1f} MB.",
                "- Feature families on validation: " + ", ".join(f"{v['features']} {pct(v['top1'])}" for v in classical["validation_by_family"]) + ".", ""]
    if hybrid and cost:
        name = hybrid["chosen_fusion"]
        out += ["### Hybrid (`results/hybrid.json`, `results/hybrid_cost.json`)", "",
                f"- Choice made on validation: `{name}` (deep embedding of the deployed model plus the classical geometry features, linear head).",
                f"- Test Top-1: deep only {pct(hybrid['test']['deep_only']['top1'])}, hybrid {pct(hybrid['test'][name]['top1'])}. The fusion did not help on validation or on test.",
                f"- Cost of the hybrid: {cost['total_size_mb']:.1f} MB and about {cost['total_inference_ms']:.1f} ms (deep model's figures are from another laptop, the extra steps were timed here).", ""]
    for name, run in detectors.items():
        if run:
            rule = run["selection_rules"]
            out.append(f"- Detector `{name}`: " + "; ".join(f"{k}: mean IoU {v['mean_iou']:.3f}" for k, v in rule.items()) + f" ({run['images']} test images).")
    run = ((crops or {}).get("runs") or {}).get("mobilenet")
    if run:
        out.append(f"- Full detect-then-classify path (`results/detector_crop_accuracy.json`, {run['images']} test images): annotated box Top-1 {pct(run['annotated_box']['top1'], 1)}, detector crop Top-1 {pct(run['detector_crop']['top1'], 1)}, with the top-score rule {pct(run['top_score_rule']['top1'], 1)}.")
    out += [""]

    out += ["## Section 3: figures that exist in the repository", "", table(["File", "Exists", "What it is"], [[p, "yes" if (ROOT / p).exists() else "**missing**", w] for p, w in FIGURE_FILES], [":---", ":---:", ":---"]), ""]
    if optimisation:
        out += ["Optimisation technique applied: quantisation (see the efficiency table in the report; deployed variant: `" + optimisation["deploy_variant"] + "`).", ""]

    out += ["## Section 4: failures", "",
            "- Classifier failure cases: `results/classifier_failure_cases.json` and `docs/CLASSIFIER_FAILURES.md` (measured confidence and the probability given to the true class for each case).",
            "- Detector and classical-ROI failure cases: `results/failure_cases.json` and `docs/FAILURE_ANALYSIS.md`.",
            "- Facts to be honest about (all measured or stated in the repository): label noise in the dataset is possible (one of the most confident errors looks like a wrong label); the 92.02% and 92.18% Top-1 figures differ by 13 images because of how the image is resized (`results/preprocessing_check.json`); the hybrid did not beat the deep model; the out-of-taxonomy test uses Caltech-101 photos only, so it does not cover newer car models; accounts have no email verification; uploaded photos are processed on the server.", ""]

    out += ["## Section 5: commands", "",
            "- Local run: `pip install -r requirements.txt`, then `cd frontend && npm ci && npm run build`, then `python start_server.py` and open http://localhost:8000. A login is required: sign up on first use (`APEX3D_REQUIRE_LOGIN=0` turns it off for local work).",
            "- Docker: `docker build -t apex3d .` then `docker run -p 10000:10000 apex3d` (`Dockerfile` at the repository root).",
            "- Reproduce the numbers: `scripts/evaluate_deployed_model.py`, `scripts/evaluate_out_of_taxonomy.py`, `scripts/measure_hybrid_cost.py`, `scripts/build_app_metrics.py`, `scripts/build_eval_report.py` (needs the dataset from `scripts/prepare_dataset.py`).",
            "- **Not yet verified from a clean clone.** The template's checklist says the metrics must be reproducible from the repository. Run the commands on a fresh clone before submitting.", ""]

    out += ["## Only the team can supply", "",
            "- The AI-assistance declaration and your honest estimate of the AI share of the report.",
            "- Each member's own part, for the viva.", "",
            "## Building the PDF", "",
            "Write the `[WRITE: ...]` and `[TODO: ...]` slots in `report/report_P5_T25.md`, then run `python scripts/build_report_pdf.py`. It refuses to build while any slot is left (use `--draft` to preview), checks the template's headings, lists any percentage in your text that is not in the result files, and writes `report/report_P5_T25.pdf`.", ""]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reset", action="store_true", help="overwrite the report with a fresh skeleton (loses the team's prose)")
    args = parser.parse_args()

    REPORT_DIR.mkdir(exist_ok=True)
    FACTS.write_text(facts_sheet(), encoding="utf-8")
    print(f"Wrote {FACTS.relative_to(ROOT)}")
    created = args.reset or not REPORT.exists()
    text = skeleton() if created else REPORT.read_text(encoding="utf-8")
    REPORT.write_text(refresh(text), encoding="utf-8")
    print(f"{'Created' if created else 'Refreshed the data tables in'} {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
