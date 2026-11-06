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

