# Running the full app: frontend, backend and Person A's model

Written by Person D after merging `backend-c`, `person-a` and the frontend. Commands are for Windows PowerShell, run from the repo root.

## 1. Get the trained model (not in Git)

Person A's model files are too large for Git. Ask Person A for `apex3d_person_a_handover.zip` (made by `scripts/package_handover.ps1`) and **unzip it at the repo root**. Paths are preserved. It puts these in place:

| File | Used for |
|---|---|
| `checkpoints/model_quantized.onnx` | The classifier the backend loads automatically (34.6 MB, temperature built in) |
| `outputs/triposr_test/*/mesh.glb` | Real sample meshes, handy for testing the 3D viewer |
| `checkpoints/apex3d_convnext_tiny.pt`, `artifacts/*.npz` | Not needed by the app (PyTorch bundle, and embeddings for Person B) |

Without `model_quantized.onnx` the backend does not use the trained model. See "Known problems" below for what it does instead.

## 2. Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt onnxruntime
```

For real 3D meshes also run `scripts\setup_triposr.ps1` and `pip install -r requirements-ml.txt` (Person A's notes: `docs/PERSON_A_ML_COMPUTE.md`). `onnxruntime` is missing from `requirements.txt`, so it is added by hand above.

## 3. Build the frontend

```powershell
cd frontend
npm install
npm run build
cd ..
```

The backend serves `frontend/dist` on its own, so there is one address for everything.

## 4. Start

Use the confidence thresholds Person A validated (`results/classifier_metrics.json`). The defaults in the backend (0.20 and 0.15) call many wrong answers confident.

```powershell
$env:APEX3D_REJECTED_THRESHOLD = "0.52"
$env:APEX3D_MARGIN_THRESHOLD = "0.34"
# The app needs a login by default. Create an account on /signup the first time.
# $env:APEX3D_REQUIRE_LOGIN = "0"            # switch the login requirement off for local work
# $env:APEX3D_CORS_ORIGINS = "https://example.org"   # extra origins allowed to call the API with cookies
python start_server.py
```

Open http://localhost:8000. For public access, `scripts\run_tunnel.ps1 -Port 8000` prints a `trycloudflare.com` address.

For frontend work with hot reload, run the backend as above and, in another window, `cd frontend; npm run dev`. The dev server proxies `/api` to port 8000.

## 5. Research Lab numbers

The Research Lab asks the backend's `/api/metrics` first. Whatever the backend has no data for is filled from Person A's saved files, `results/classifier_metrics.json` and `results/optimization_metrics.json`. `npm run build` copies those two files into `frontend/dist/results/`, and `npm run dev` serves them from `results/`. So Person A's real numbers show up without any backend change, and anything nobody has produced yet still says "Not available".

Rebuild the frontend (`npm run build`) after those files change. Person B's sections appear once the backend serves their results, for example by adding them to `notebooks/metrics.json` (see `frontend/API_NOTES.md`).

## Known problems that are in backend files

These are in Person C's files, and are listed in Person A's `docs/INTEGRATION_NOTES_A_TO_C.md` with the fixes. They decide whether the demo shows the right car. Status when this was written:

| Problem | Effect in the app |
|---|---|
| **Class list is wrong** (197 entries, 115 in the wrong place) | The model is right but the app shows another car's name. Build the list from `results/class_names.json` |
| **Fake fallbacks** (random classifier when no model loads, a box-shaped car when 3D fails) | Invented results look real. They should return an error instead |
| **`rembg` downloads a non-commercial model** | Delete the extra background-removal step. `reconstruct()` already does it with an Apache-2.0 model |
| **Thresholds** | Set the two environment variables in step 4 |
| **`/api/metrics` path** | See step 5 |

## What the frontend already does about the model

- Shows whatever state the backend returns (confident, ambiguous or rejected) and never replaces it, so correct thresholds in the backend decide what users see.
- Uploads reference images as files from the browser. It no longer sends web addresses for the backend to fetch (Person A's note 8).
- Shows an error message, not a made-up result, if the backend answers with an error.
