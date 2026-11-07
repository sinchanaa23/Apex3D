# Apex3D architecture

**Status: describes the code as built** (checked against `backend/`, `frontend/src/` and `Dockerfile` at the commit shown in the report). The app runs locally and has a Docker image; it is deployed on Render's free plan at https://apex3d-xvuc.onrender.com. Numbers quoted here come from `results/` and `eval_P5_T25.md`.

## System

```mermaid
flowchart LR
    subgraph Client["Browser: phone or laptop"]
        UI["React 19 app (Vite, Tailwind)<br/>Log in / Sign up, Upload, Identity, Classical,<br/>References, 3D Studio, Research Lab"]
        V3D["three.js viewers<br/>showcase car (Khronos Car Concept GLB),<br/>TripoSR mesh viewer, GLB/OBJ/STL export"]
    end

    subgraph Server["FastAPI server: one process, one container"]
        GATE["Login check on every route<br/>except /api/auth and /api/health"]
        AUTH["Accounts<br/>scrypt password hashes,<br/>HttpOnly session cookie"]

        subgraph REC["POST /api/recognize"]
            DET["Detector<br/>Faster R-CNN MobileNetV3 FPN<br/>largest box scoring at least 0.5"]
            CROP["Crop with margin<br/>or full frame if no box"]
            CLS["Classifier<br/>ConvNeXt-Tiny, INT8 ONNX Runtime<br/>196 classes, temperature scaling"]
            CONF["Confidence rule<br/>confident, ambiguous or rejected<br/>(probability 0.52, margin 0.34)"]
        end

        CCV["POST /api/classical<br/>10-stage OpenCV and scikit-image<br/>analysis view"]
        REFS["GET /api/references<br/>Wikimedia Commons, perceptual-hash<br/>dedup, viewpoint, licence"]

        subgraph R3D["POST /api/reconstruct"]
            LIB["Mesh library<br/>catalog or cached mesh<br/>returns at once"]
            Q["Job queue<br/>one job at a time"]
            TSR["TripoSR with u2net<br/>background removal"]
            VAL["trimesh validation<br/>and GLB, OBJ, STL export"]
        end

        CLEAN["Retention task<br/>deletes uploaded photos and<br/>old results after 2 hours"]
        MET["GET /api/metrics and /results/*.json<br/>saved evaluation results"]
    end

    DB[("SQLite<br/>users, sessions, jobs,<br/>reference cache")]
    FS[("Files<br/>uploads, job outputs,<br/>mesh cache")]
    WM["Wikimedia Commons API"]

    subgraph Offline["Offline evaluation, not part of the running app"]
        NB["Notebooks 01, 01b, 01c, 02<br/>training, quantisation,<br/>backbones, classical baseline"]
        SC["scripts/evaluate_*.py<br/>robustness, hybrid, confidence,<br/>out-of-taxonomy, ablation, failures"]
        RES["results/*.json,<br/>eval_P5_metrics.json,<br/>eval_P5_T25.md"]
    end

    UI --> GATE
    UI --- V3D
    GATE --> AUTH
    GATE --> REC
    GATE --> CCV
    GATE --> REFS
    GATE --> R3D
    GATE --> MET
    DET --> CROP --> CLS --> CONF
    REFS --> WM
    LIB -. "miss" .-> Q --> TSR --> VAL
    AUTH --> DB
    Q --> DB
    REFS --> DB
    VAL --> FS
    LIB --> FS
    CLEAN --> FS
    NB -->|"model_quantized.onnx"| CLS
    NB --> RES
    SC --> RES
    RES --> MET
```

The deployed model file is `checkpoints/model_quantized.onnx`, produced by notebook `01b` from the model trained in `01`. The server serves the built React app as well as the API, so one address exposes the whole application.

## What the live recognition path does, and what it does not

The live path is **detector, crop, classifier, confidence rule**. The classical stages run in their own endpoint (`/api/classical`) so users can inspect them, and the same classical features are the baseline in the evaluation. Fusing classical geometry features with the deep embedding was tested (`results/hybrid.json`) and did not improve accuracy (92.18% deep only, 92.09% fused on test), so the fusion is **not** in the production path. The report states this plainly rather than presenting the fusion as a gain.

## Which page uses which endpoint

| Page | Route | Endpoint | Shows |
|---|---|---|---|
| Log in, Sign up | `/login`, `/signup` | `POST /api/auth/login`, `POST /api/auth/signup`, `GET /api/auth/me`, `POST /api/auth/logout` | Account forms, session state |
| Home | `/` | None | What the app does and its limits |
| Upload | `/upload` | `POST /api/recognize` | File checks, preview, elapsed time |
| Vehicle Identity | `/identity` | Result of the upload | Box, Manufacturer, Model, Year, confidence state, Top-5, correction |
| Classical CV | `/classical` | `POST /api/classical` | Intermediate images and features |
| Reference Studio | `/references` | `GET /api/references`, `POST /api/references/upload` | References by viewpoint, licence, input choice |
| 3D Studio | `/studio`, `/studio/:jobId` | `POST /api/reconstruct`, `GET /api/jobs/{id}`, `GET /api/assets/{id}/{file}` | Showcase model, queue, logs, mesh viewer, validation, downloads |
| Research Lab | `/lab` | `GET /api/metrics`, `/results/*.json` | Saved evaluation results, or "not available" |

Every route above except the login pages and the landing page (`/`, with the showcase car it loads from `/models/`) needs a logged-in user. `GET /api/health` is the only other public API route.

## How a photo moves through the app

```mermaid
sequenceDiagram
    actor User
    participant App as React app
    participant API as FastAPI
    participant M as Detector and classifier

    User->>App: Sign up or log in
    App->>API: POST /api/auth/login
    API-->>App: Session cookie
    User->>App: Choose a photo
    App->>App: Check file, rotate upright, resize
    App->>API: POST /api/recognize
    API->>M: Detect, crop, classify
    M-->>API: Box, Top-5, probability, margin
    API-->>App: Result, confidence state, timings
    User->>App: Confirm or correct the identity
    App->>API: GET /api/references?class_id
    API-->>App: Candidates with licences
    User->>App: Pick one input view
    App->>API: POST /api/reconstruct
    alt A catalog or cached mesh exists for the class
        API-->>App: Finished job at once
    else Otherwise
        API-->>App: Job id (queued)
        loop Every 1.5 s until finished
            App->>API: GET /api/jobs/{id}
            API-->>App: Status, queue place, logs
        end
    end
    App->>API: GET the GLB and validation report
    API-->>App: Generated files
    App-->>User: Showcase car, mesh viewer, validation, downloads
```

## Data the server keeps

| What | Where | How long |
|---|---|---|
| Accounts: name, email, scrypt password hash | SQLite `users` | Until deleted by hand |
| Sessions: SHA-256 of the cookie token | SQLite `sessions` | 7 days, then ignored and removed at the next login |
| Photos sent to `/api/recognize` | Memory only | Not written to disk |
| Photos sent to `/api/reconstruct`, and photos a user uploads as a reference | `backend/storage/uploads/`, `backend/storage/references/upload_*` | Deleted automatically after `APEX3D_RETENTION_HOURS` (default 2), checked at start-up and every 15 minutes. A photo needed by a queued or running job is kept until the job ends. |
| Finished meshes and validation reports | `backend/storage/jobs/` | Same retention |
| Job records (status, logs) | SQLite `jobs` | Kept; the files they point at are removed |
| Reference images fetched from Wikimedia Commons, and meshes cached per class | `backend/storage/references/`, `backend/storage/mesh_cache/` | Kept (not user photos) |

## Deployment

| Piece | State |
|---|---|
| `Dockerfile` | The lean image Render builds: the frontend and a server without PyTorch (the detector is exported to ONNX during the build). `deploy/Dockerfile.full` adds TripoSR |
| `render.yaml` | Render Blueprint on the free plan. The server needs about 260 MB, so it fits; the mesh builder does not (see `docs/DEPLOY_RENDER.md`) |
| `deploy/huggingface/` | Hugging Face Space version. Docker Spaces need a paid subscription (see `docs/DEPLOY_HUGGINGFACE.md`) |
| Laptop and tunnel | `scripts/run_tunnel.ps1` exposes the local server through a Cloudflare quick tunnel |
| Public URL | https://apex3d-xvuc.onrender.com (Render free plan: no disk, so accounts reset on restart; sleeps when idle; no mesh builder) |
