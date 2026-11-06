# Apex3D architecture

**Status: planned design.** Only the frontend exists so far. The backend parts are taken from the project proposal and should be updated if the build differs.

## System

```mermaid
flowchart TB
    subgraph Browser["Phone or laptop browser"]
        UI["React app<br/>Upload, Identity, Classical,<br/>References, 3D Studio, Research Lab"]
    end

    TUN["Cloudflare tunnel<br/>(public URL)"]

    subgraph Host["Team GPU laptop"]
        API["FastAPI<br/>API + built frontend"]
        DB[("SQLite job records<br/>+ local asset files")]

        subgraph Recognition
            DET["Detector<br/>torchvision R-CNN"]
            CLS["Classifier<br/>notebook checkpoint"]
            HYB["Confidence and<br/>hybrid logic"]
            CCV["Classical CV"]
        end

        subgraph Assets["3D assets"]
            REC["Reconstruction interface"]
            TSR["TripoSR on GPU"]
            VAL["Validation and export<br/>trimesh"]
        end

        RET["Reference retrieval"]
    end

    subgraph Offline["Offline notebooks"]
        NB["Training and evaluation"]
        OUT["Checkpoint + metrics JSON"]
    end

    WM["Wikimedia Commons API"]

    UI <--> TUN <--> API
    API --> DB
    API --> DET --> CLS --> HYB
    API --> CCV --> HYB
    API --> RET --> WM
    RET --> CLS
    API --> REC --> TSR --> VAL --> DB
    NB --> OUT
    OUT --> CLS
    OUT -->|"read by /api/metrics"| API
```

FastAPI serves both the API and the built frontend, so one tunnel exposes the whole app.

## Which page uses which endpoint

| Page | Route | Endpoint | Shows |
|---|---|---|---|
| Home | `/` | None | What the app does and its limits |
| Upload | `/upload` | `POST /api/recognize` | File checks, preview, elapsed time |
| Vehicle Identity | `/identity` | Result of the upload | Box, confidence state, Top-5, correction |
| Classical CV | `/classical` | `POST /api/classical` | Intermediate images and features |
| Reference Studio | `/references` | `GET /api/references`, `POST /api/references/upload` | References by viewpoint, licence, input choice |
| 3D Studio | `/studio`, `/studio/:jobId` | `POST /api/reconstruct`, `GET /api/jobs/{id}`, `GET /api/assets/{id}/{file}` | Queue, logs, viewer, validation, downloads |
| Research Lab | `/lab` | `GET /api/metrics` | Saved notebook results, or "not available" |

## How a photo moves through the app

```mermaid
sequenceDiagram
    actor User
    participant App as React app
    participant API as FastAPI
    participant GPU as Models

    User->>App: Choose a photo
    App->>App: Check file, rotate upright, resize
    App->>API: POST /api/recognize
    API->>GPU: Detect, then classify the crop
    GPU-->>API: Box, Top-5, confidence state
    API-->>App: Result with timings
    User->>App: Confirm or correct the identity
    App->>API: GET /api/references?class_id
    API-->>App: Candidates with licences
    User->>App: Pick one input view
    App->>API: POST /api/reconstruct
    API-->>App: Job id
    loop Every 1.5 s until finished
        App->>API: GET /api/jobs/{id}
        API-->>App: Real status, queue place, logs
    end
    App->>API: GET the GLB and validation report
    API-->>App: Generated files
    App-->>User: Viewer, validation, downloads
```
