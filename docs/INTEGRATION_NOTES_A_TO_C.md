# Integration notes: Person A → Person C

> **Status 2026-11-07, re-checked on `main` (`918d627`):**
> - **Fixed:** items 1–5, 7 and 8.
>   - The class list now comes from `results/class_names.json`, and 8 of 8 test photos show the correct name.
>   - There are no fake fallbacks.
>   - The rembg step was removed.
>   - Thresholds are 0.52/0.34, and metrics are served.
>   - Crop margin is 0.08, and the `image_url` fetch was removed.
> - **Still open (minor):**
>   - Item 6: the app's own ONNX session in `classifier_service.py` doesn't set `intra_op_num_threads`, so classification takes about 32 ms instead of about 20 ms.
>   - The app resizes bilinearly while training used bicubic (about 0.1–0.2 points; see `eval_P5_T25.md`).
>   - New: 4 tests in `tests/test_api.py` fail on a fresh clone. Use `with TestClient(app) as client:` so startup creates the database.
>
> The original findings below are kept for the record.

Tested 2026-11-06 on the GPU laptop. I merged `origin/main` (frontend), `origin/backend-c` and `person-a` into a **local, unpushed** branch, ran `python start_server.py`, and sent real requests. Everything below was observed, not guessed.

## Works

- The classifier loader finds `checkpoints/model_quantized.onnx` automatically ("Loaded ONNX classifier checkpoint").
- `/api/recognize`: my model predicted the **correct class ID for 5 of 5** Stanford test photos.
- `/api/reconstruct`: calls Person A's `reconstruct()` and gets a real TripoSR mesh (80,456 faces, watertight). The validator and GLB/OBJ/STL export work, and the job finished in 42 s from a cold start.

## Must fix before the demo (in `backend-c`)

### 1. The class list is wrong: 115 of 196 names are in the wrong position
`backend/data/stanford_cars_classes.py` has **197** entries, and the order drifts from index 59 onwards. The model outputs class IDs in the dataset's own order (`devkit/cars_meta.mat`), so the app shows the wrong car even when the model is right:

| Photo | True class | Model's class ID | App displayed |
|---|---|---|---|
| test 00001 | Suzuki Aerio Sedan 2007 | 180 ✔ | **Spyker C8 Coupe 2009** |
| test 00100 | Dodge Magnum Wagon 2008 | 91 ✔ | **Dodge Durango SUV 2004** |
| test 01000 | Toyota Camry Sedan 2012 | 186 ✔ | **Toyota Sequoia SUV 2012** |

**Fix:** build the list from `results/class_names.json` (committed on `person-a`), which was generated from `cars_meta.mat`, instead of a hand-typed list:
```python
import json
from backend.config import PROJECT_ROOT
_CLASSES = json.loads((PROJECT_ROOT / "results" / "class_names.json").read_text())
STANFORD_CARS_CLASSES = [c["name"] for c in _CLASSES]   # index == model output index
# each entry also has "make", "model", "year"
```

### 2. Fake fallbacks show invented results as if they were real
This breaks the team's honesty rule: no fake progress, and no stored results shown as live.
- `classifier_service._feature_based_classify` returns **random** Top-5 classes with made-up probabilities whenever no checkpoint loads.
- `reconstruction_service._run_reconstruction` catches *every* exception from `reconstruct()` and returns a **procedural box car**, which is then exported and validated as if it were a result.

**Fix:** if the model is missing, return a clear "classifier not loaded" error. If `reconstruct()` raises `ReconstructionError`, mark the job failed with `str(e)`; the message is safe to show users.

### 3. Background removal downloads a non-commercial model
`_remove_background` calls `rembg.remove(img)` with no session. rembg 2.0.87 then downloads **BRIA RMBG-2.0 (1 GB, CC BY-NC, non-commercial)**. Person A's `reconstruct()` already removes the background with the Apache-2.0 **u2net** model, so delete this step and pass the image straight to `reconstruct()`. (I bypassed it only in my local test branch.)

### 4. The confidence thresholds are not the validated ones
They were chosen on validation in notebook 01 (`results/classifier_metrics.json` → `thresholds`). Your decision logic maps onto them exactly when you set:
```
APEX3D_REJECTED_THRESHOLD=0.52
APEX3D_MARGIN_THRESHOLD=0.34
```
The current defaults (0.20 and 0.15) mark many wrong answers as "confident". The probabilities from `model_quantized.onnx` are already calibrated, because the temperature is baked into the model graph. So a plain softmax is correct, exactly as your code does now.

### 5. `/api/metrics` says "not available"
The real files are `results/classifier_metrics.json` and `results/optimization_metrics.json`. Add them to `METRICS_CANDIDATE_PATHS` in `backend/config.py`.

## Should fix

6. **ONNX threads:** create the session with `SessionOptions().intra_op_num_threads = 6`. That gives 19–22 ms per image instead of 83–100 ms. The default uses this CPU's slow efficiency cores (measured in `results/optimization_metrics.json` → `thread_sweep_fp32_ms`).
7. **Crop margin:** the classifier was trained on box + **8%** margin, but `get_crop` uses `padding_pct=0.05`. Change it to 0.08 to match training.
8. **Server-side `image_url` fetch** in `/api/reconstruct`: once the tunnel is public, anyone can make the laptop fetch any URL, including `http://localhost`. Allow only `upload.wikimedia.org`, or remove the field and let the frontend upload files.
9. **The 3D job hit no detector box on test 00010** (`crop_used: full_frame`), but the classifier was still right. This is worth a look in Person B's detector metrics.

## How to run with Person A's models

1. Get these from the team drive into `checkpoints/`: `model_quantized.onnx`, plus `apex3d_convnext_tiny.pt` if you want the PyTorch path.
2. Run `scripts/setup_triposr.ps1` (or set `APEX3D_TRIPOSR_DIR`), then start with `python start_server.py`.
3. Requirements: `requirements-ml.txt` (Person A) plus your `requirements.txt`.
