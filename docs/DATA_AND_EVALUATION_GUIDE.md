# Person B: data and evaluation guide

Everything here ran on the real dataset unless it says otherwise. Results are in `results/`, and the final tables are in `eval_P5_DRAFT.md`.

## Person A: last results still needed

Hybrid, full robustness and crop ablations need your files, which aren't in Git. Nothing has produced them yet: `results/hybrid.json`, `robustness.json` and `ablation_inference.json` don't exist.

**The quick option:** share `outputs\apex3d_person_a_handover.zip` with Person B. Hybrid and confidence then run from the saved embeddings without any model.

**About test image 00010 (your note 9):** this was the wrong-car bug. The detector found the main car at 0.87, but a 17×12 px background car scored 0.95 and was chosen. That crop was too small, so the app used the full frame. `detector_service.py` now picks the largest box above the score threshold.

**Or run it all on your GPU laptop:**

```bash
python scripts/prepare_dataset.py
```

```bash
python scripts/evaluate_with_classifier.py --device cuda
```

```bash
python scripts/evaluate_detector.py --variant resnet50 --device cuda
```

Run the detector on all 8,041 images before the second command if you can, so that the crop ablation uses the full set. The first command downloads about 2 GB with no login. `evaluate_with_classifier.py` expects `artifacts/*.npz` and `checkpoints/apex3d_convnext_tiny.pt` where your notebook saved them. It also needs `~/.apex3d/models/classical_baseline.joblib` for the classical robustness column. That column is skipped if the file is missing, and everything else still runs.

**Then rebuild the reports and push `results/`:**

```bash
python scripts/build_failure_cases.py
```

```bash
python scripts/build_eval_report.py --team <TeamID> --usns USN1,USN2,USN3,USN4
```

```bash
python scripts/build_app_metrics.py
```

The scripts stop with an error if the saved files or labels disagree with the dataset.

## Person C: detection in the app

**Already changed in `backend/services/detector_service.py`:**
- The app now crops the largest box above the score threshold (`choose_primary`), not the top score. In our 500-image sample, the top score missed the main car in 68 images, usually for a tiny car in the background.
- The crop margin is now 8%, the margin the classifier was trained with.

**Keep `backend/data/__init__.py` using relative imports.** Its `backend.data` import stopped every Person B script from starting. `tests/test_person_b_integration.py` now guards against this.

**Metrics.** `scripts/build_app_metrics.py` writes `eval_P5_metrics.json`, one of the paths your route reads.

## Person D: Research Lab

Your `/api/metrics` sections come from `eval_P5_metrics.json`. Sections not yet measured say `"not available"`. Failure cases include their images. `results/figures/` has the classical figures.

## Reproducing Person B's results

```bash
python scripts/prepare_dataset.py
```

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/02_classical_baseline.ipynb
```

```bash
python scripts/evaluate_detector.py --variant mobilenet --limit 400 --split val
```

```bash
python scripts/evaluate_detector.py --variant resnet50 --limit 500
```

```bash
python scripts/evaluate_classical_roi.py --limit 500
```

```bash
python -m pytest tests/test_classical_cv.py tests/test_dataset.py tests/test_detection_metrics.py tests/test_evaluation.py
```

**About the dataset:** the Hugging Face mirror stores rows out of file order without filenames, so joining them directly to annotations silently gets labels wrong. `prepare_dataset.py` joins by class and position within class, then proves it: all 16,185 boxes fit their images, against about 53% if shuffled, and the test annotations match torchvision's MD5. Its train and validation split is identical to `results/split.json`, and the notebook asserts this. Crops match the classifier: annotated box plus 8%.

## What to say in the viva

- **Classical vs deep:** classical reaches about 17% and deep about 92% on the same test images and crops. HOG carries most of the classical signal, because shape separates cars better than colour.
- **CLAHE:** it didn't help on validation, so it's off in the classical model. We tested it instead of assuming it would help. Its low-light effect on the deep model is still to be measured.
- **Biggest finding:** the detector's top score often picks a tiny background car. Choosing the largest confident box lifted mean IoU from 0.78 to 0.91 on validation. See `docs/FAILURE_ANALYSIS.md` for the case where that rule fails.
- **Why mAP looks low:** only one car per image is annotated, so correct detections of other cars count as false positives.
- **Sample sizes:** detector numbers come from samples of 400 and 500 images, unless a full run has replaced them.
