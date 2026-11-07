# Person B: data and evaluation guide

Everything here ran on the real dataset unless it says otherwise. Results are in `results/`, and the final tables are in `eval_P5_T25.md`.

## Current state of the evaluation

Every P5 metric now has a measured result. The final model is `checkpoints/model_quantized.onnx`, sha256 `0167758b...41fc27`, and hybrid, robustness and confidence were measured on it with `scripts/evaluate_deployed_model.py`. An independent rerun of that model reproduced its 642 test errors exactly.

**About test image 00010 (Person A's note 9):** this was the wrong-car bug. The detector found the main car at 0.87, but a 17×12 px background car scored 0.95 and was chosen. That crop was too small, so the app used the full frame. `detector_service.py` now picks the largest box above the score threshold.

**Worth doing if time allows:** detector results come from samples of 400 and 500 images. On the GPU laptop, this would run the app's detector on all 8,041 images. The reports then use the full run automatically:

```bash
python scripts/evaluate_detector.py --variant mobilenet --device cuda
```

```bash
python scripts/evaluate_detector_crops.py
```

**Rebuild the reports after any result changes:**

```bash
python scripts/build_failure_cases.py
```

```bash
python scripts/build_eval_report.py --team T25 --usns 1RUA24CSE0461,1RUA24CSE0467,1RUA24CSE0445,1RUA24CSE0453
```

```bash
python scripts/build_app_metrics.py
```

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
- **CLAHE:** it didn't help the classical model on validation, so it's off. On the deep model under simulated low light, it lowered Top-1 at every level tested, by 0.5 to 2.2 points on 1,000 images. We tested it instead of assuming it would help.
- **Hybrid:** adding classical features to the deep features didn't help: 92.09% against 92.18% with the same linear head on test. On validation the best fusion only tied. The useful integration is the pipeline around the model, not feature fusion. Report this as a real negative result.
- **Biggest finding:** the detector's top score often picks a tiny background car. Choosing the largest confident box lifted mean IoU from 0.78 to 0.91 on validation. On 500 test images, it lifted full-pipeline Top-1 from 80.6% to 92.4% (± 2.3). See `docs/FAILURE_ANALYSIS.md` for the case where that rule fails.
- **Detector crops:** on those images, the annotated box scored 91.8% and the app's detector crop 92.4%. That difference is within the margin of error, so say that detector crops didn't measurably hurt, not that they helped.
- **Why two accuracy figures:** the deployed model scores 92.18% in Person A's notebook and 92.02% with the app's preprocessing. We reproduced the 92.02% exactly but couldn't fully explain the 13-image gap, so quote about 92.0-92.2%.
- **Why mAP looks low:** only one car per image is annotated, so correct detections of other cars count as false positives.
- **Sample sizes:** detector numbers come from samples of 400 and 500 images, unless a full run has replaced them.
