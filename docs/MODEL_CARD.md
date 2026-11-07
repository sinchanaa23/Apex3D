# Model card: Apex3D Stanford Cars classifier

Owner: Person A. Every number here is copied from a file produced by a notebook or script. The file is named next to each section.

## Model details

| | |
|---|---|
| Task | Fine-grained vehicle classification: make, model and year, one of 196 Stanford Cars classes |
| Architecture | ConvNeXt-Tiny (Liu et al., 2022), 28.0M parameters, 224×224 RGB input |
| Initial weights | `timm` `convnext_tiny.fb_in22k_ft_in1k` (ImageNet-22k pretraining, ImageNet-1k fine-tune; Facebook ConvNeXt weights, MIT licence) |
| Training code | `notebooks/01_classifier_training.ipynb` (with `notebooks/cls_lib.py`) |
| Optimisation code | `notebooks/01b_export_quantize.ipynb`, `scripts/export_app_classifier.py` |
| Files | `checkpoints/apex3d_convnext_tiny.pt` (PyTorch, 112 MB, includes class names, temperature and thresholds); `checkpoints/model_quantized.onnx` (INT8, the file the app loads) |
| Trained on | One RTX 3050 Laptop GPU (4 GB), about 40 minutes of compute, 2026-11-06 |

## Intended use

- **For:** identifying cars in photos uploaded to the Apex3D research demonstration, then helping the user find reference images and start a 3D draft.
- **Input:** a crop around one vehicle. During training this was the annotated box plus an 8% margin, so use the same margin on detector boxes.
- **Not for:** vehicles outside the 196 classes (anything newer than the 2012 model year, vans and trucks not in the list, non-US-market models), surveillance or vehicle tracking, law enforcement, insurance, or any decision about a person. Number plates are not read.

## Training data

Stanford Cars (Krause et al., 2013). There are 16,185 images: 8,144 train and 8,041 test. Each class is labelled as *Make Model Type Year*.
- **Source:** Kaggle mirror `rickyyyyyyy/torchvision-stanford-cars`. The original download is offline, and the mirror named in the proposal returned 403 Forbidden.
- **Split:** the 8,144 training images were split 85/15 into train and validation (stratified, seed 42; `results/split.json`). Validation was used for every tuning decision, and the test set was used once.
- **Test-label check:** the test labels come from a third-party file. Every class is present, per-class train/test counts correlate at 0.993, and validation and test Top-1 differ by 0.0002.
- **Licence:** generally described as non-commercial research use. Apex3D is therefore a research demonstration.

## Training procedure

The model trained on box crops (cached at a longest side of 448 px). Augmentation was RandomResizedCrop (scale 0.6–1), horizontal flip, RandAugment (2 operations, magnitude 9) and random erasing.

Training ran in two stages:
1. **Head only, 2 epochs:** AdamW, learning rate 1e-3.
2. **Full fine-tune:** AdamW with layer-wise learning-rate decay 0.9, peak learning rate 3e-4, cosine schedule with 1 warm-up epoch, weight decay 0.05, label smoothing 0.1, FP16 mixed precision. Batch 32 with 2 accumulation steps (effective 64). Early stopping on validation Top-1 with patience 5. The best epoch was 28 of 32.

After training, a temperature T = 0.617 was fitted on validation (temperature scaling).

## Evaluation (`results/classifier_metrics.json`, `results/optimization_metrics.json`)

Test set: 8,041 unseen images, ground-truth box crops.

| Metric | PyTorch FP32 | Deployed INT8 ONNX |
|---|---|---|
| Top-1 | 92.20% | 92.18% |
| Top-5 | 98.42% | 98.51% |
| Make-level Top-1 | 97.15% | — |
| Macro F1 | 0.921 | — |
| Calibration error (ECE) after temperature scaling | 0.024 | — |
| Size | 112 MB | 34.6 MB |
| CPU latency, batch 1, i5-12500H, 6 threads | 49.2 ms | 19.4 ms |
| GPU latency, batch 1, RTX 3050 FP32 | 5.2 ms | — |

**Why ConvNeXt-Tiny** (`results/baselines_metrics.json`, notebook 01c, same recipe and same test set):

| Model | Test Top-1 |
|---|---|
| ConvNeXt-Tiny | 92.20% |
| EfficientNetV2-S | 87.13% |
| ResNet-50 | 81.58% |

The choice was made on validation (92.23%, 86.50% and 83.55% respectively).

**Confidence states** (thresholds chosen on validation). A prediction is *rejected* when the calibrated Top-1 probability is below 0.52. It is *ambiguous* when the gap between the Top-1 and Top-2 probabilities is below 0.34. Otherwise it is *confident*.

| State | Share of test images | Accuracy |
|---|---|---|
| confident | 93.5% | 95.0% |
| ambiguous | 1.3% | 59.8% |
| rejected | 5.2% | 49.2% |

**Errors:** 63.5% of errors are within the same make, and 6.2% get the make and model right but the year wrong. The most-confused pairs are in `results/classifier_confused_pairs.csv`, and per-class precision, recall and F1 are in `results/classifier_per_class.csv`.

## Limitations

- **Closed set.** The model always picks one of 196 classes. Rejection thresholds were tuned on in-taxonomy cars only, so they do **not** guarantee that unknown vehicles are rejected. That needs its own test on non-Stanford images (Person B).
- **Ground-truth crops.** All numbers above use ground-truth crops. Accuracy on detector crops, full frames, low light, blur and occlusion is measured separately (Person B's robustness study) and may be lower.
- **Year.** The year is the labelled model year, not a learned estimate of the vehicle's generation. Neighbouring model years can look identical.
- **Bias.** The dataset leans towards US-market cars up to 2012, so performance on other regions' vehicles is likely lower. We have no labels to measure this.
- **Optimistic validation figures.** The thresholds were chosen on the same validation set used for early stopping and temperature fitting. Quote the *test* state accuracies above, not the validation ones.

## Ethical considerations

Uploads can contain people and number plates. The app deletes them after processing and does no plate recognition. Car designs and trademarks may be protected, and a recognised identity or a generated mesh grants no rights to the design.
