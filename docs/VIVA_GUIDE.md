# Presentation and viva guide

For explaining the project to a teacher. Everything here points to a real file in the repository and a real number from `results/`. If you can't explain a line, open the file it names and read it before you present.

**The 30-second version.** Apex3D takes a photo of a car and says the manufacturer, model and year (196 Stanford Cars classes). A Faster R-CNN finds the car, a fine-tuned ConvNeXt-Tiny names it, and a confidence rule says whether to trust the answer. We compared it with a classical computer-vision model (17.3% Top-1) and with a hybrid of the two, and the deep model reached 92.2% Top-1. A React website and a FastAPI server wrap it, and the app also shows a 3D car.

---

## 1. "Show where you trained the model"

| What | Where |
|---|---|
| **The training code** | `notebooks/01_classifier_training.ipynb`, with the helper functions in `notebooks/cls_lib.py`. A Colab version is `notebooks/01_classifier_training_colab.ipynb` |
| Its output | `results/classifier_metrics.json`, `results/classifier_history.csv`, plots in `results/plots/` (`training_curves.png`, `confusion_matrix.png`) |
| The trained model used by the app | `checkpoints/model_quantized.onnx`. The INT8 model comes from `notebooks/01b_export_quantize.ipynb`; `scripts/export_app_classifier.py` writes the file the app loads, with the temperature already divided into the output so that a plain softmax gives calibrated probabilities (`results/app_classifier_export.json`) |
| The other two deep models we compared | `notebooks/01c_baseline_models.ipynb` (ResNet-50, EfficientNetV2-S) |
| The classical model | `notebooks/02_classical_baseline.ipynb` |

**What the training notebook does, in order:** set the seed and check the hardware, get the dataset, parse the annotations, check the labels, explore the data, build the data loaders (crop each car using its ground-truth box plus an 8% margin), then train in two stages and calibrate.

- **Model:** `convnext_tiny.fb_in22k_ft_in1k` from the timm library, 28.0 million parameters, pretrained on ImageNet-22k then ImageNet-1k, 224 by 224 input. This is transfer learning.
- **Stage 1, 2 epochs:** only the new 196-class head is trained (learning rate 0.001).
- **Stage 2, up to 30 epochs:** the whole network is fine-tuned (peak learning rate 0.0003). We ran 32 epochs in all and kept the best one, epoch 28, chosen by validation Top-1, with early stopping after 5 epochs without improvement.
- **Settings:** AdamW optimiser, weight decay 0.05, cosine learning-rate schedule with 1 warm-up epoch, layer-wise learning-rate decay 0.9, label smoothing 0.1, gradient clipping at norm 1.0, mixed-precision (fp16), batch 32 (effective 64), seed 42.
- **Augmentation:** random resized crop (scale 0.6 to 1), horizontal flip, RandAugment, random erasing.
- **Data split:** 6,922 training, 1,222 validation and 8,041 test images. The test set is used only for the final report.
- **Hardware:** an RTX 3050 laptop GPU. Stage 2 took about 183 minutes.

---

## 2. "What formulas have you used?"

Say the idea first, then the formula.

**Classification and training**

```
softmax:             p_k = exp(z_k) / sum_j exp(z_j)
cross-entropy with label smoothing (e = 0.1, K = 196 classes):
                     loss = - sum_k q_k * log p_k ,   q_k = (1 - e) if k is the true class, else e / K
cosine schedule:     lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T))
layer-wise decay:    earlier layers get a smaller learning rate:  lr_layer = lr * 0.9 ^ (depth below the head)
Top-k accuracy:      share of images whose true class is among the k highest probabilities
macro F1:            F1_c = 2 * P_c * R_c / (P_c + R_c) for each class, then the mean over classes
```

**Confidence (the part that handles "low-confidence predictions")**

```
temperature scaling: p = softmax(z / T), with T chosen to minimise the negative log-likelihood on VALIDATION data
                     NLL = - (1/N) * sum_i log p_i(true class)        our T = 0.617
expected calibration error (15 bins):
                     ECE = sum_b (|B_b| / N) * | accuracy(B_b) - mean confidence(B_b) |
                     validation 0.208 -> 0.034 after scaling;  test 0.175 -> 0.023
decision rule:       confident  if p1 >= 0.52 and (p1 - p2) >= 0.34
                     rejected   if p1 < 0.52
                     ambiguous  otherwise
```

Temperature scaling never changes which class wins; it only makes the probability honest. Code: `backend/evaluation/confidence.py`.

**Detection**

```
IoU = area(A and B) / area(A or B)            overlap between a predicted and a true box
AP  = area under the precision-recall curve at an IoU threshold
mAP@[.5:.95] = mean of AP over IoU thresholds 0.50, 0.55, ..., 0.95        code: backend/detection/metrics.py
box choice:  among boxes with score >= 0.5, take the one with the largest area
```

**Classical computer vision** (`backend/classical_cv/pipeline.py`)

```
CLAHE:   contrast-limited adaptive histogram equalisation on the lightness channel (clip limit 2.0, 8x8 tiles)
Canny:   low = (1 - 0.33) * median brightness,  high = (1 + 0.33) * median brightness
HOG:     histograms of gradient direction, weighted by gradient strength, in small cells
LBP:     for each pixel, code = sum_p s(g_p - g_centre) * 2^p ,  s(x) = 1 if x >= 0 else 0
HSV:     colour histograms of hue, saturation and value
Hu moments, aspect ratio, area ratios, edge density: 12 shape features
model:   standardise  z = (x - mean)/std  ->  PCA to 256 dimensions  ->  linear SVM
SVM:     minimise  0.5*||w||^2 + C * sum max(0, 1 - y * (w . x))^2     (squared hinge, one-vs-rest, C = 0.01)
```

**Model optimisation (quantisation)** (`notebooks/01b_export_quantize.ipynb`)

```
q = clip( round(x / s) + z , 0 , 255 )        x ~ s * (q - z)         s = scale, z = zero point
```

We used static INT8 quantisation, calibrated on 256 training-split crops, with per-channel weights, applied to the matrix multiplications only. Result: 112 MB to 34.6 MB, 49.2 ms to 19.4 ms per image, Top-1 92.20% to 92.18%. Quantising every layer instead lost 6 points, which is why we kept to the matrix multiplications.

**Hybrid:** the deep model's 768-value embedding is joined to the 12 classical geometry features and given to a linear head. It did not help (92.09% against 92.18%), and we report that.

---

## 3. "How have you connected the model to the frontend?"

The model never runs in the browser. The browser sends the photo to our server, and the server runs the model.

```
Browser (React)                                    Server (FastAPI, Python)
frontend/src/components/PhotoUpload.tsx            backend/api/routes_recognition.py   POST /api/recognize
  -> frontend/src/api/client.ts  recognize()  ----> 1. check the file, decode the image
     (multipart upload, shrinks photo to 2048 px)   2. backend/services/detector_service.py   find the car, choose the box
  <- JSON answer  <---------------------------------  3. crop the box with a margin
frontend/src/api/adapter.ts   maps it to types      4. backend/services/classifier_service.py  ONNX Runtime runs
frontend/src/state/session.tsx   keeps it               checkpoints/model_quantized.onnx, softmax, Top-5
frontend/src/pages/Identity.tsx   draws the box,       5. apply the confidence rule  -> confident / ambiguous / rejected
  Manufacturer / Model / Year, Top-5, confidence ring   6. return boxes, Top-5, probability, margin, timings
```

- **Show it live:** open the website, press F12, choose the **Network** tab, upload a photo, and click the `recognize` request. The Response tab shows the JSON the model produced: `predictions` (Top-5 with make, model, year, probability), `bounding_box`, `confidence_state`, `confidence_score`, `margin`, `timings_ms`.
- **Other connections:** the Classical page calls `POST /api/classical`; References call `GET /api/references`; the 3D Studio calls `POST /api/reconstruct` and then asks `GET /api/jobs/{id}`; the Research Lab calls `GET /api/metrics`. The full table is in `docs/ARCHITECTURE.md`, and the exact request and response formats are in `docs/API_CONTRACT.md`.
- **Login:** every page except the landing page needs an account (`backend/auth.py`). Passwords are stored as scrypt hashes.

---

## 4. "Where did the results come from?"

Nothing was typed in by hand. Each number comes from a script or notebook that wrote a file in `results/`; one script gathers them for the app and the report.

| Result | Made by | File |
|---|---|---|
| Top-1, Top-5, macro F1, calibration, thresholds | `notebooks/01_classifier_training.ipynb` (the deployed model re-checked by `scripts/evaluate_deployed_model.py`) | `results/classifier_metrics.json` |
| Classical baseline, feature families | `notebooks/02_classical_baseline.ipynb` | `results/classical_baseline.json` |
| Quantisation, size, speed | `notebooks/01b_export_quantize.ipynb` | `results/optimization_metrics.json` |
| ResNet-50 and EfficientNetV2-S | `notebooks/01c_baseline_models.ipynb` | `results/baselines_metrics.json` |
| Hybrid comparison and its cost | `scripts/evaluate_deployed_model.py`, `scripts/measure_hybrid_cost.py` | `results/hybrid.json`, `results/hybrid_cost.json` |
| Robustness (brightness, low light, blur, noise, resolution, occlusion) | `scripts/evaluate_deployed_model.py` with `backend/evaluation/corruptions.py` | `results/robustness.json` |
| Confidence and coverage curve | `scripts/evaluate_deployed_model.py` | `results/confidence.json` |
| Photos outside the 196 classes (Caltech-101) | `scripts/evaluate_out_of_taxonomy.py` | `results/out_of_taxonomy.json` |
| Detector accuracy (IoU, mAP) | `scripts/evaluate_detector.py`, `scripts/evaluate_detector_crops.py` | `results/detector_*.json` |
| Ablation | the scripts above | `results/ablation_inference.json`, `results/detector_crop_accuracy.json` |
| Failure cases | `scripts/build_classifier_failures.py`, `scripts/build_failure_cases.py` | `results/classifier_failure_cases.json`, `results/failure_cases.json` |
| **Everything gathered** | `scripts/build_app_metrics.py` | `eval_P5_metrics.json`, shown by the Research Lab through `GET /api/metrics` |
| The evaluation report | `scripts/build_eval_report.py` | `eval_P5_T25.md` |

**Why the test results are fair:** the test set was never used to choose anything. Temperature, thresholds, the fusion choice and the early-stopping epoch were all chosen on the validation set. Validation and test accuracy agree within 0.02 points (92.23% against 92.20%), which is what you expect when the model has not overfitted.

---

## 5. Other questions to expect

**Which dataset, and how big?** Stanford Cars: 16,185 images, 196 classes of make, model and year. We used the dataset's own bounding boxes for training.

**Why ConvNeXt-Tiny?** We trained three backbones with the same recipe: ConvNeXt-Tiny 92.20%, EfficientNetV2-S 87.13%, ResNet-50 81.58% (`results/baselines_metrics.json`). The best one on validation was chosen.

**Why detect first?** Classifying the whole photo scored 86.1%; cropping to the car scored about 92%. With the detector's own box it scored 92.4% on a 500-photo sample.

**Is the hybrid better?** No, and we say so. Classical alone is 17.3%, deep alone 92.2%, the fusion 92.1%. The classical stages are still useful as an explanation view and as the baseline the rubric asks for.

**What are the failure cases?** Look-alike cars (a Jeep Compass called a Grand Cherokee, a Chevrolet Express called a GMC Savana), the detector picking a tiny background car before we changed the rule, and heavy blur or tiny images (52.7% under heavy blur, 9.0% at 32 pixels). See `docs/CLASSIFIER_FAILURES.md`.

**What about a car that is not in the 196?** The model must answer one of them, so the confidence rule is the only protection. On Caltech-101 photos it rejected 91.5% of other vehicles and 98.3% of other things, but 8.5% and 1.7% still got a confident wrong answer. Modern cars were not tested.

**What is the 3D part?** The 3D Studio shows a designed showcase car recoloured from your photo. TripoSR (a single-image 3D model) can build a rough mesh in 4 to 5 minutes, but it needs about 1.25 GB of memory, so it runs when the app is hosted locally and is switched off on the free Render site.

**Where is it deployed?** https://apex3d-xvuc.onrender.com on Render's free plan, as a Docker container (`Dockerfile`, `render.yaml`). It sleeps when idle, so open it a few minutes early, and accounts reset when it restarts.

**Did you use AI?** Answer honestly. AI assistance (Claude) helped build much of the code, the evaluation scripts, the documents and the report, and the report declares it. Be ready to explain your own part in your own words, and to open the file for anything you say. The rubric gives the most marks to the demo and viva.

---

## 6. A 3-minute live demo

1. Open the live site and sign up (the landing page is public). Press F12 and open the **Network** tab.
2. **Upload** a clear car photo. Point out the box, the Manufacturer, Model and Year, the confidence state and the Top-5. Show the `recognize` request and its JSON.
3. Upload a harder or non-car photo and show the ambiguous or rejected state.
4. Show **Classical**: the stage images.
5. Show **References** (licensed photos), then the **3D Studio** showcase car and a download.
6. Open the **Research Lab**: classical against deep against hybrid, robustness, confidence curve, ablation, failure cases.
7. Finish by opening `notebooks/01_classifier_training.ipynb` for the training, and `eval_P5_T25.md` for the results.

**Backups:** a screen recording is at `docs/demo/demo.mp4`, and every screen is in `docs/screenshots/`. To run locally instead: `python start_server.py`, then open http://localhost:8000.
