# Apex3D — Recognise. Reconstruct. Create.

**P5: Intelligent Fine-Grained Vehicle Recognition and Analysis System**  
*Computer Vision Hackathon 2026*  
**Repository:** [https://github.com/sinchanaa23/Apex3D](https://github.com/sinchanaa23/Apex3D)

---

## Overview

Apex3D identifies the fine-grained make, model, and year of a vehicle in an uploaded photograph (196 Stanford Cars classes), extracts intermediate classical computer vision features (CLAHE, Canny, Contours, HOG, LBP, and Geometric analysis), retrieves verified CC-licensed reference viewpoints from Wikimedia Commons, and coordinates 3D mesh reconstruction with automated geometry validation.

---

## System Architecture

```
                       [React + Three.js Frontend]
                                    │
                                    ▼ (HTTP / REST)
                       [FastAPI Backend Service]
                       (Port 8000 / Tunnel Host)
                                    │
        ┌───────────────┬───────────┴───────────┬───────────────┐
        ▼               ▼                       ▼               ▼
 [Vehicle Detector] [Classifier]         [Classical CV]   [3D Queue Worker]
  Torchvision R-CNN  ConvNeXt-Tiny (196)  CLAHE, Canny,    SQLite Jobs DB
  + Heuristic Fallback + ONNX/PyTorch     HOG, LBP, Geo    TripoSR / Trimesh
                                                │               │
                                                ▼               ▼
                                         [Wikimedia API]  [Asset Exporter]
                                         CC-BY Retrieval  GLB, OBJ, STL
```

---

## Quickstart (One Command)

### 1. Requirements & Setup

- Python 3.12 or 3.13
- Install dependencies:

```bash
pip install -r requirements.txt
# Or run with existing environment:
python start_server.py
```

### 2. Start the Backend

```bash
python start_server.py
```

- **Interactive API Documentation (Swagger UI):** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc Documentation:** [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Check:** [http://localhost:8000/api/health](http://localhost:8000/api/health)

---

## Running the full app (model, 3D reconstruction, frontend)

Step-by-step guide: [docs/RUN_FULL_APP.md](docs/RUN_FULL_APP.md). In short:

1. Put Person A's `model_quantized.onnx` in `checkpoints/` (not in Git, ask Person A for the file).
2. `scripts\setup_triposr.ps1` (or clone TripoSR to a short path and set `APEX3D_TRIPOSR_DIR`), then `pip install -r requirements-ml.txt`. The TripoSR weights (about 1.7 GB, `stabilityai/TripoSR`) download on first use.
3. `cd frontend && npm install && npm run build`, then `python start_server.py` and open http://localhost:8000.

## Development log (Person D)

Newest first. Each stage of the frontend and integration work is added here.

- **Showcase on every job page (done):** the page for a reconstruction job (`/studio/<job id>`) opens on the showcase car, with tabs to switch to the experimental TripoSR mesh or catalog model. The car is there while the mesh is still building, and on old links. `GET /api/jobs/{id}` now also returns `class_name`, so the car's body style is known even after a page reload.
- **Showcase covers all 196 classes (done):** every Stanford Cars class maps to a body style (sedan, coupe, SUV, hatchback, wagon, van or pickup), including the 14 labels that name a trim instead, such as "Corvette ZR1" or "Challenger SRT8". A test (`frontend/src/lib/adapt.test.ts`) checks all 196 names against `results/class_names.json`, so a new or renamed class cannot silently fall back to the default shape.
- **Showcase car in the 3D Studio (done):** single-photo reconstruction (TripoSR) cannot give a clean car structure, so the Studio now opens on a designed showcase car instead: the landing page's Car Concept model (Khronos, CC BY 4.0, credited on the page) with the same contour-grid look. It is adapted to the user's photo in two measured ways only: paint colour, read from the car in the photo in the browser (`frontend/src/lib/adapt.ts`), and body proportions from the confirmed Stanford Cars body style. The page says plainly that it is not a 3D copy of their car. The adapted car is the Studio's deliverable: GLB, OBJ and STL downloads are written in the browser (`frontend/src/lib/exportCar.ts`) with the photo colour and proportions applied. The TripoSR mesh is still available as "Build experimental mesh", and its page has a back link to the showcase.
- **Better meshes for matched cars (done):** once the user confirms a car, the References page picks the clearest reference photo of that exact model (a three-quarter or side view whose title matches) and the Studio builds the mesh from it. The finished mesh is cached per car (`backend/storage/mesh_cache/<class_id>/`), so the next request for the same car is instant. Licensed models in `catalog/<class_id>.glb` are shown instantly, labelled "Catalog model". The Studio always says what the mesh came from: catalog model, reference photo of the matched car, or the user's own photo. `POST /api/reconstruct` takes two new optional fields, `class_id` and `source`, and `GET /api/jobs/{id}` returns `mesh_source`.
- **3D Studio restyle and real TripoSR meshes (done):** TripoSR is installed and the website generates real meshes from photos (about 3 minutes per car on CPU, first run also downloads about 1.7 GB of weights and a 176 MB background-removal model). The mesh viewer now uses the landing page's look: dark glossy body, orange contour grid that follows the mesh, scan sweep and pulse, white edge lines, glowing floor, slow turntable that pauses while dragging. A toggle shows the mesh's original colours. Meshes with no normals get them computed on load. Photo quality matters: a clean side or three-quarter photo gives a car-shaped mesh, while open doors or busy backgrounds give lumpy results.
- **Fake 3D fallback removed:** `backend/services/reconstruction_service.py` no longer invents a box-shaped car when TripoSR fails. The job fails with a clear error instead. Background removal now happens once, inside `backend/reconstruction`.
- **Person A's model connected:** classifier loads `checkpoints/model_quantized.onnx`. Class names come from `results/class_names.json`, and the default thresholds are the validated 0.52 / 0.34.
- **Person B's work connected:** `/api/classical` returns Person B's 10-stage pipeline. The Research Lab reads `eval_P5_metrics.json` plus Person A's `results/*.json`.
- **Frontend:** React 19 + Vite + Tailwind, landing page with a 3D hero, HUD-style inner pages, upload on the page, Research Lab with real numbers. Tests: `cd frontend && npx vitest run`.

---

## API Endpoints Summary

Full specifications and schemas are in [docs/API_CONTRACT.md](docs/API_CONTRACT.md).

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/recognize` | Vehicle detection, crop extraction, Top-5 prediction, confidence state |
| `POST` | `/api/classical` | CLAHE, Canny, contours, ROI, HOG, LBP, and geometric descriptors |
| `GET` | `/api/references?class_id=...` | Wikimedia Commons CC-licensed references with deduplication |
| `POST` | `/api/references/upload` | User reference photo upload |
| `POST` | `/api/reconstruct` | Enqueue background 3D reconstruction job |
| `GET` | `/api/jobs/{id}` | Real-time status, execution stage, logs, and progress |
| `GET` | `/api/assets/{id}/{file}` | Download generated GLB, OBJ, STL meshes and validation JSON |
| `GET` | `/api/metrics` | Saved offline notebook evaluation results (honesty principle) |
| `GET` | `/api/health` | GPU availability and queue status |

---

## Team Integration & Handovers

- **Person A (ML & Compute):**
  - Place fine-tuned checkpoint (`checkpoint.pth`, `best_model.pth`, or `model.onnx`) in `checkpoints/`. The classifier service auto-detects and loads it without code modification.
  - Export `reconstruct(image) -> trimesh.Trimesh` in `backend/reconstruction/`. The 3D worker dynamically hooks into it.
  - Save training metrics in `notebooks/metrics.json`.
- **Person B (Classical CV & Evaluation):**
  - Place custom classical CV algorithms in `backend/classical_cv/`.
  - Save evaluation and comparison results in `eval_P5_metrics.json`.
- **Person C (Backend & Retrieval - Current Role):**
  - FastAPI service, background job queue, SQLite persistence, trimesh validation & multi-format export, Wikimedia Commons retrieval.
- **Person D (Frontend):**
  - Connect your Vite/React app to `http://localhost:8000/api`.
  - For single-command production hosting, run `npm run build` inside `frontend/`. The backend will automatically serve `frontend/dist` on the root URL!

---

## Running Automated Tests

Run the end-to-end integration test suite:

```bash
python -m pytest -v tests/test_api.py
```

