# Facts sheet for the report

A reference to write from. It holds measured numbers with their source files, repository modules and the figures that exist. It contains no explanation on purpose: the explanations in the report should be the team's own words (the hackathon caps AI-written content at 30%). Regenerate with `python scripts/build_report_facts.py`.

## Header

- Problem: P5, Intelligent Fine-Grained Vehicle Recognition and Analysis System
- Team ID: T25. USNs: 1RUA24CSE0461, 1RUA24CSE0467, 1RUA24CSE0445, 1RUA24CSE0453
- Repository: https://github.com/sinchanaa23/Apex3D
- Commit to quote (HEAD when this was generated): `0e3a14aa941a3f98098bef9706f4c1db6baa4a4d`. Push everything and regenerate right before submitting so the hash is the final one.
- Live URL: https://apex3d-xvuc.onrender.com (Render free plan: accounts reset when it restarts, it sleeps after about 15 minutes idle, and the mesh builder is off). Checked from outside: /api/health answers with detector onnx and mesh_builder false, protected routes answer 401, /docs is hidden. Sign-up and recognition on the live site were not tested by the tool that wrote this.

## Section 1: the problem and the data

- Dataset: Stanford Cars, 196 classes; split used here: 6922 train, 1222 validation, 8041 test. Input to the classifier: ground-truth box crop + 8% margin. (`results/classifier_metrics.json`, dataset)
- Fine-grained difficulty, measured: of the deep model's errors, 63.48% are within the same make and 6.22% are the same make and model with a different year. Make-level Top-1 is 97.15%.
- What the rubric and problem statement ask for: make, model and year, confidence, bounding box, low-confidence handling, comparison of classical, deep and hybrid, ablation, one optimisation technique, unseen test images, at least three failure cases.

## Section 2: architecture facts

### Repository modules to reference

| Part | Where |
| :--- | :--- |
| Classical CV pipeline (preprocessing, edges, contours, colour, texture, ROI, geometry) | backend/classical_cv/pipeline.py, backend/classical_cv/api.py, backend/classical_cv/batch.py |
| Classical baseline: features, linear SVM, tuning | notebooks/02_classical_baseline.ipynb, results/classical_baseline.json |
| Deep classifier training (ConvNeXt-Tiny, transfer learning) | notebooks/01_classifier_training.ipynb, notebooks/cls_lib.py, results/classifier_metrics.json |
| Optimisation: quantisation and ONNX export | notebooks/01b_export_quantize.ipynb, results/optimization_metrics.json |
| Backbone comparison | notebooks/01c_baseline_models.ipynb, results/baselines_metrics.json |
| Vehicle detector (Faster R-CNN) and box selection | backend/detection/detector.py, backend/detection/metrics.py, results/detector_*.json |
| ONNX classifier used by the app | backend/recognition/onnx_classifier.py, checkpoints/model_quantized.onnx |
| Confidence, calibration and thresholds | backend/evaluation/confidence.py, results/confidence.json |
| Hybrid, robustness and ablation experiments | backend/evaluation/classifier_experiments.py, backend/evaluation/corruptions.py, scripts/evaluate_deployed_model.py |
| Out-of-taxonomy test | scripts/evaluate_out_of_taxonomy.py, results/out_of_taxonomy.json |
| API, login, job queue | backend/main.py, backend/auth.py, backend/api/, backend/services/ |
| 3D reconstruction (TripoSR bridge, mesh library) | backend/reconstruction/, backend/services/mesh_library.py |
| Frontend | frontend/src/ (pages: Upload, Identity, Classical, References, Studio, Lab) |
| Evaluation report generator | scripts/build_eval_report.py, scripts/build_app_metrics.py, eval_P5_T25.md |

### Deep model and training (`results/classifier_metrics.json`)

- Architecture: convnext_tiny.fb_in22k_ft_in1k, 28.0 M parameters, input 224 px, pretrained: ImageNet-22k -> ImageNet-1k (timm).
- Schedule: 2 head-only epochs, then fine-tuning up to 30 epochs; 32 epochs run, best at epoch 28. Learning rate 0.001 (head) and 0.0003 (fine-tune), layer decay 0.9, weight decay 0.05, label smoothing 0.1, mixed precision fp16.
- Augmentation: RandomResizedCrop(0.6-1), HFlip, RandAugment(2,9), RandomErasing(0.25).
- Hardware used for training: NVIDIA GeForce RTX 3050 Laptop GPU.

### Classical baseline (`results/classical_baseline.json`)

- Features: HSV colour histogram, LBP texture, HOG, and 12 geometric features; linear SVM, chosen C = 0.01; adaptive CLAHE used: False.
- Test Top-1 17.32%, Top-5 36.81%, macro F1 15.85%; 14.8 ms per image, 2.5 MB.
- Feature families on validation: hsv 3.11%, lbp 4.09%, hog 14.40%, geometry 1.47%, all 16.78%.

### Hybrid (`results/hybrid.json`, `results/hybrid_cost.json`)

- Choice made on validation: `deep_plus_geometry` (deep embedding of the deployed model plus the classical geometry features, linear head).
- Test Top-1: deep only 92.18%, hybrid 92.09%. The fusion did not help on validation or on test.
- Cost of the hybrid: 35.8 MB and about 22.6 ms (deep model's figures are from another laptop, the extra steps were timed here).

- Detector `resnet50`: top_score: mean IoU 0.811; largest_confident: mean IoU 0.931; score_times_area: mean IoU 0.934 (500 test images).
- Detector `mobilenet`: top_score: mean IoU 0.787; largest_confident: mean IoU 0.916; score_times_area: mean IoU 0.916 (500 test images).
- Full detect-then-classify path (`results/detector_crop_accuracy.json`, 500 test images): annotated box Top-1 91.8%, detector crop Top-1 92.4%, with the top-score rule 80.6%.

## Section 3: figures that exist in the repository

| File | Exists | What it is |
| :--- | :---: | :--- |
| results/figures/classical_stages.png | yes | the classical stages on one photo |
| results/figures/classical_confusion.png | yes | classical model confusion |
| results/figures/classical_families.png | yes | classical feature families compared |
| results/plots/training_curves.png | yes | training curves |
| results/plots/confusion_matrix.png | yes | deep model confusion matrix |
| results/plots/calibration_coverage.png | yes | calibration and coverage |
| results/plots/baselines_comparison.png | yes | three backbones compared |
| results/plots/eda_overview.png | yes | dataset overview |
| results/plots/augmentations.png | yes | training augmentations |
| results/figures/classifier_failures.jpg | yes | sheet of confident wrong answers |
| docs/screenshots/02-identity.png | yes | app: identity page |
| docs/screenshots/10-identity-top5.png | yes | app: identity with top-5 list |
| docs/screenshots/05-studio-mesh.png | yes | app: 3D mesh and validation |

Optimisation technique applied: quantisation (see the efficiency table in the report; deployed variant: `onnx_int8_static_matmul`).

## Section 4: failures

- Classifier failure cases: `results/classifier_failure_cases.json` and `docs/CLASSIFIER_FAILURES.md` (measured confidence and the probability given to the true class for each case).
- Detector and classical-ROI failure cases: `results/failure_cases.json` and `docs/FAILURE_ANALYSIS.md`.
- Facts to be honest about (all measured or stated in the repository): label noise in the dataset is possible (one of the most confident errors looks like a wrong label); the 92.02% and 92.18% Top-1 figures differ by 13 images because of how the image is resized (`results/preprocessing_check.json`); the hybrid did not beat the deep model; the out-of-taxonomy test uses Caltech-101 photos only, so it does not cover newer car models; accounts have no email verification; uploaded photos are processed on the server.

## Section 5: commands

- Local run: `pip install -r requirements.txt`, then `cd frontend && npm ci && npm run build`, then `python start_server.py` and open http://localhost:8000. A login is required: sign up on first use (`APEX3D_REQUIRE_LOGIN=0` turns it off for local work).
- Docker: `docker build -t apex3d .` then `docker run -p 10000:10000 apex3d` (`Dockerfile` at the repository root).
- Reproduce the numbers: `scripts/evaluate_deployed_model.py`, `scripts/evaluate_out_of_taxonomy.py`, `scripts/measure_hybrid_cost.py`, `scripts/build_app_metrics.py`, `scripts/build_eval_report.py` (needs the dataset from `scripts/prepare_dataset.py`).
- **Not yet verified from a clean clone.** The template's checklist says the metrics must be reproducible from the repository. Run the commands on a fresh clone before submitting.

## Only the team can supply

- The AI-assistance declaration and your honest estimate of the AI share of the report.
- Each member's own part, for the viva.

## Building the PDF

Write the `[WRITE: ...]` and `[TODO: ...]` slots in `report/report_P5_T25.md`, then run `python scripts/build_report_pdf.py`. It refuses to build while any slot is left (use `--draft` to preview), checks the template's headings, lists any percentage in your text that is not in the result files, and writes `report/report_P5_T25.pdf`.
