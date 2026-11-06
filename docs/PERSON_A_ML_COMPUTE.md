# Person A: ML and compute: status and handover

Owner: Person A (GPU laptop). Branch: `person-a`.
Every number in this file comes from a saved file in `results/`. If a file and this page disagree, the file wins.

## 1. GPU laptop report (hour-0 requirement)

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU, **4 GB VRAM** (about 0.7 GB already used by the desktop) |
| Driver / CUDA | 592.00 / CUDA 13.1 driver; PyTorch 2.11.0 built for CUDA 12.8 |
| CPU / RAM | Intel Core i5-12500H (4 performance + 8 efficiency cores, 16 threads) / 16 GB |
| Python | 3.12.4 in `.venv` |

**4 GB is below the 6 GB the proposal assumed.** What this changed:
- TripoSR runs with `chunk_size=2048` (default 8192). Measured peak VRAM is 2.3 GB, so the default marching-cubes resolution of 256 still fits.
- The classifier trains at batch 32 with 2 accumulation steps (effective batch 64), FP16 mixed precision, at 224 px. The 384 px option from the proposal was not tried: it would roughly triple training time on this GPU.
- Training and 3D generation **cannot run at the same time**. See the GPU queue in §6.

## 2. Environment setup

```
powershell -ExecutionPolicy Bypass -File scripts\setup_gpu_env.ps1
```
This creates `.venv` (Python 3.12), installs CUDA PyTorch from the PyTorch index plus `requirements-ml.txt`, and clones TripoSR at a pinned commit into `third_party/` (gitignored).

Dataset (about 1.9 GB, gitignored):
```
kaggle datasets download rickyyyyyyy/torchvision-stanford-cars -p data/raw --unzip
```

### Deviations from the proposal, and why

| Proposal | What we did | Why |
|---|---|---|
| Kaggle mirror `jessicali9530/stanford-cars-dataset` | `rickyyyyyyy/torchvision-stanford-cars` | The planned mirror returned 403 Forbidden on 2026-11-06. The new one has the same 16,185 images and includes `cars_test_annos_withlabels.mat`. Notebook 01 §2 and §4 verify the file counts and the test labels. |
| TripoSR's `torchmcubes` (compiled C++/CUDA) | `backend/reconstruction/mcubes_shim.py` (scikit-image marching cubes) | Avoids a C++ build on Windows and on the backup laptop. `tests/test_reconstruction_shim.py` checks it against TripoSR's own helper on an analytic sphere: position, radius, watertightness and outward normals. |
| rembg default background removal | rembg **`u2net`** model pinned explicitly | rembg 2.0.87 now defaults to BRIA RMBG-2.0, which is **CC BY-NC (non-commercial)** and a 1 GB download. u2net is Apache-2.0 and is the model TripoSR was released with. |
| ONNX INT8 quantization | Dynamic **and** static (calibrated) INT8 compared | Measured on this CPU, dynamic INT8 was slower than FP32. Static INT8 is both faster and smaller. Calibration uses training-split crops only. |

## 3. 3D reconstruction (TripoSR) — `backend/reconstruction/`

Hour-1 checkpoint met: **TripoSR generated real meshes from car photos on this laptop's GPU.** The measured records are in `results/triposr_test.json`, and sample meshes are in `outputs/triposr_test/` (share these through the team drive for Person D's viewer).

| Run | Preprocess | Model load | Inference | Mesh extraction | Total | Peak VRAM | Faces |
|---|---|---|---|---|---|---|---|
| First call (cold) | 5.2 s | 8.7 s | 1.3 s | 8.2 s | 23.6 s | 2.32 GB | 80k |
| Later calls (model loaded) | about 1.1 s | 0 | about 1.4 s | about 7.7 s | **about 10.2 s** | 2.32 GB | about 85k |

Three Stanford Cars test images; settings: `chunk_size=2048`, `mc_resolution=256`, u2net background removal, vertex colours.
Two of the three meshes were watertight. Watertightness is not guaranteed, so Person C's validation must check it every time.

### Interface for Person C

```python
from backend.reconstruction import reconstruct, ReconstructionError, ReconstructionSettings, release_gpu

try:
    result = reconstruct(image_path_or_pil)           # blocking: call from the job worker
except ReconstructionError as e:
    job.fail(str(e))                                   # message is safe to show the user

result.mesh                 # trimesh.Trimesh, Y-up (glTF/Three.js), unitless, centred near the origin
result.mesh.export("mesh.glb")   # also .obj, .stl
result.processed_image      # the exact background-removed image fed to TripoSR (show it in the UI)
result.summary()            # JSON-safe: device, timings_s, settings, peak VRAM, vertex/face counts, notes
```

- Only one reconstruction runs at a time (internal lock). Queue jobs in front of it.
- The model stays loaded on the GPU after the first call (peak use while generating is 2.3 GB). Call `release_gpu()` before anything else needs the GPU.
- If CUDA runs out of memory, the job is retried on the CPU, and `summary()["notes"]` says so.
- `notes` always includes "Single-view reconstruction: unseen sides are inferred." and "Scale is unknown." The UI must show both.

## 4. Classifier — `notebooks/01_classifier_training.ipynb`

The notebook ran top to bottom on this laptop with every output saved (2026-11-06). Source for all numbers: `results/classifier_metrics.json`.

| Item | Value |
|---|---|
| Model | ConvNeXt-Tiny (`timm convnext_tiny.fb_in22k_ft_in1k`), 28.0M parameters, 224 px, 112 MB checkpoint |
| Data | 6,922 train / 1,222 val (stratified 85/15, seed 42) / 8,041 test; ground-truth box crop with 8% margin |
| Training | 2 head-only epochs, then full fine-tune: AdamW, layer decay 0.9, cosine LR, label smoothing 0.1, FP16 AMP, effective batch 64. Best epoch 28 of 32 (val Top-1) |
| **Test Top-1 / Top-5** | **92.20% / 98.42%** (8,041 unseen images, GT crops) |
| Validation Top-1 / Top-5 | 92.23% / 98.85% |
| Make-level Top-1 (test) | 97.15% |
| Macro precision / recall / F1 (test) | see `results/classifier_per_class.csv`; macro F1 0.921 |
| Errors that are year-only (same make + model) | 6.2% of errors; 63.5% of errors are within the same make |
| Calibration | Temperature T = 0.617 (label smoothing made raw scores under-confident). Val ECE 0.208 → 0.034; test ECE 0.024 |
| Confidence states (thresholds chosen on val) | confident: p1 ≥ 0.52 and margin ≥ 0.34; rejected: p1 < 0.52 |
| States on test | confident 93.5% of images (95.0% accurate) · ambiguous 1.3% (59.8%) · rejected 5.2% (49.2%) |
| Test-label check | 196/196 classes present; per-class train/test count correlation 0.993; val − test Top-1 = +0.0002 |
| Latency, batch 1 | GPU FP32 5.2 ms · GPU FP16 5.9 ms · CPU FP32 56 ms (12 threads) |
| Reload check | Bundle reloaded the way the app does it agrees with the notebook on 100% of 200 test crops |

**Training time.** Each stage-2 epoch took about 63 s (6,922 images), so the whole run takes about 40 minutes of compute. The saved `stage2_min` (183) and `notebook_runtime_min` (188) are inflated because **the laptop went to sleep during epoch 28** (that epoch shows 9,175 s in `results/classifier_history.csv`). Training resumed correctly after waking.

**Lightweight enough for the RTX 3050?** Yes, as measured. Peak training VRAM was 2.06 GB at batch 32 (gradient checkpointing was not needed), out of 4 GB.

**Caveats to say in the viva:** thresholds are tuned on the same validation set used for early stopping and temperature, so quote the *test* state accuracies (above), not the validation ones. Validation contains only in-taxonomy cars, so out-of-taxonomy rejection is **not** measured here.

## 5. Optimisation — `notebooks/01b_export_quantize.ipynb`

Results are filled in from `results/optimization_metrics.json` after the run (see §7).

## 6. GPU queue (Person A decides what runs)

| When | GPU job | Notes |
|---|---|---|
| Training | `01_classifier_training.ipynb` only | No TripoSR while it runs. Others work on CPU. |
| After training | Embeddings and logits export (part of notebook 01) | Handed to Person B |
| Then | TripoSR samples and tuning | `release_gpu()` before any other GPU job |
| Judging | Live app only: recognition plus queued 3D | 01b shows the ONNX classifier runs on the CPU, leaving the GPU for 3D |

## 7. Hand-overs

| To | What | Where |
|---|---|---|
| B, C | Checkpoint bundle: weights, architecture, input size, normalisation, class names, temperature, thresholds | `checkpoints/apex3d_convnext_tiny.pt` (team drive). Load with `notebooks/cls_lib.load_classifier` and predict with `cls_lib.predict_crop` |
| B, C | Class list with parsed make / model / year | `results/class_names.json` |
| B | Exact train/val split used for every model | `results/split.json` |
| B | Logits and 768-d embeddings for train/val/test | `artifacts/logits_*.npz`, `artifacts/embeddings_*.npz` (team drive) |
| B, D | Metrics JSON and plots | `results/classifier_metrics.json`, `results/optimization_metrics.json`, `results/plots/` |
| C | `reconstruct(image) → mesh` | `backend/reconstruction/` (§3) |
| D | Real sample meshes | `outputs/triposr_test/*/mesh.glb` (team drive) |

**Crop convention for the app:** the classifier was trained on the annotated box plus an **8% margin** on each side (`bbox_margin` in the bundle). Apply the same margin to the detector box before calling `predict_crop`.

## 8. Hosting (judging day)

1. Plug the laptop in. In Windows Settings → System → Power, set screen and sleep to **Never** while plugged in, and set closing the lid to "Do nothing". I have not changed these settings; you change them yourself.
2. Close Docker Desktop, the WSL VM and other heavy apps. They hold about 5 GB of memory, and Windows DataLoader workers previously failed with "paging file is too small" (error 1455).
3. Start the server (Person C's one-command startup), then run `scripts\run_tunnel.ps1 -Port <port>`. It prints the public `trycloudflare.com` URL.
4. The URL changes every time the tunnel restarts, so update the stable link page whenever it does.

The tunnel has **not been tested yet**. It needs `cloudflared` installed and a running server.
