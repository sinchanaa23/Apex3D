# Apex3D Team Workplan

**P5, Computer Vision Hackathon 2026**

Four people, one GPU laptop, and about 8.5 hours of estimated work. The P5 requirements come first, and the 3D features are built on top of them.

## How the work is split

| Person | Role | Owns | Viva topic |
|---|---|---|---|
| **A** | GPU laptop owner, ML and compute | Classifier training, optimisation, TripoSR, live hosting | Model choice, training, quantization, 3D limits |
| **B** | Classical CV and evaluation | Classical pipeline, detection metrics, hybrid, ablation, robustness | Classical choices, metrics, failure cases |
| **C** | Backend and retrieval | FastAPI, inference service, mesh export, reference retrieval | Architecture, data flow, retrieval filtering |
| **D** | Frontend and presentation | React app, 3D viewer, phone testing, design doc, demo script | Interface decisions, demo walkthrough |

> **The biggest risk is the GPU queue.** Training and 3D generation can't comfortably share one GPU, so Person A decides what runs when. Everyone else works on CPU and sends A any GPU jobs they need.

## Schedule by hour

Hours are estimates counted from when you start. If your deadline is less than 8.5 hours away, use the cut order later in this plan.

| Hours | GPU queue | A · GPU | B · CV | C · API | D · UI |
|---|---|---|---|---|---|
| 0–1 | 3D test | TripoSR test | Dataset check | Repo, API contract | App shell, upload |
| 1–2 | Classifier training | Training notebook | Classical CV, detector metrics | Recognition API | App shell, upload |
| 2–3.5 | Classifier training | Training notebook | Classical CV, detector metrics | Recognition API | Identity, classical view |
| 3.5–5 | Evaluation, embeddings | Export, quantize | Hybrid, ablation, robustness | 3D jobs, export | 3D Studio, Research Lab |
| 5–5.5 | 3D tuning, samples | 3D quality, samples | Hybrid, ablation, robustness | Retrieval | 3D Studio, Research Lab |
| 5.5–6 | 3D tuning, samples | 3D quality, samples | Hybrid, ablation, robustness | Retrieval | Reference Studio |
| 6–6.5 | 3D tuning, samples | 3D quality, samples | Eval file, failure cases | Retrieval | Reference Studio |
| 6.5–7.5 | Live app only | Host, tunnel | Eval file, failure cases | Harden, bad input | Phone testing |
| 7.5–8.5 | Live app only | Model card, rehearse | Rehearse | README, rehearse | Demo script |

## Checkpoints

| By | Everyone can see | If it fails |
|---|---|---|
| **1h** | TripoSR makes a real mesh from a car photo. Dataset files and test labels are verified. Repo and API contract are pushed. | Mesh fails: A gets one more hour while training starts. Test labels unusable: hold out part of the train set and tell the team now. |
| **3.5h** | A checkpoint from the notebook is in shared storage, with real validation results. The API serves it. | Training is slow: ship the best saved checkpoint and retrain later if time allows. |
| **5h** | Upload to identity works from the real frontend, using the real API. | This is the core of P5. Pull C or D off 3D work until it passes. |
| **6.5h** | Public URL opens on a phone. **Feature freeze.** | Cut retrieval to manual upload and fix hosting first. |
| **7.5h** | A teammate outside the build runs a new photo end to end. **Code freeze.** | Fix bugs only. |
| **8.5h** | Everything is submitted and the three-minute demo is rehearsed. | Late submission costs 5 points per 15 minutes, so submit what works. |

## Person A: GPU laptop owner, ML and compute

### Tasks

- Report GPU model and VRAM, then install Python 3.12, CUDA PyTorch and TripoSR.
- Generate a mesh from one car photo and record the time.
- Write and run `notebooks/01_classifier_training.ipynb`, covering all 19 items in the brief.
- Export the checkpoint to ONNX, quantize it, and record size, speed and accuracy.
- Provide `reconstruct(image) → mesh` for Person C to call.
- Run the backend and tunnel on the laptop during judging.

### Hands over

- Checkpoint and class list to B and C.
- Metrics JSON and plots to B and D.
- Real sample meshes to D for viewer testing.

### Done when

- The notebook runs top to bottom with outputs saved.
- The app loads that notebook's checkpoint.
- The laptop stays plugged in, with sleep turned off.

## Person B: Classical CV and evaluation

### Tasks

- Download the dataset, check the files and test labels, then write the data loader.
- Build CLAHE, Canny, contours, HSV, HOG, LBP, ROI and geometric features in `backend/classical_cv/`.
- Measure detector IoU and mAP on the Stanford boxes.
- Train the classical baseline and test hybrid fusion on one split.
- Run the ablation study, robustness tests and threshold analysis.
- Pick and explain at least three failure cases.

### Hands over

- Data loader to A within the first hour.
- `classical_cv` functions to C.
- Comparison, ablation and robustness JSON to D.

### Done when

- `eval_P5_<TeamID>.md` follows P5's metric order, with USNs on the first line.
- Every number traces back to a saved notebook output.
- If fusion doesn't help, the report says so.

## Person C: Backend and retrieval

### Tasks

- Create the repo structure, `.gitignore` and API contract in the first hour.
- Build the FastAPI service with detection, classification, Top-5 and confidence states.
- Add background 3D jobs with real status, stored in SQLite.
- Export GLB, OBJ and STL, and validate meshes with trimesh.
- Build Wikimedia Commons retrieval: search, licence capture, deduplication, classifier filtering and viewpoint grouping.
- Handle bad input: wrong file types, oversized files, no car found.

### Hands over

- API contract to D in the first hour.
- Working endpoints to D as each one is ready.
- One-command startup to A for hosting.

### Done when

- Recognition responds within about 10 seconds through the tunnel.
- Every failure returns a clear message.
- A fresh clone runs from the README.

## Person D: Frontend and presentation

### Tasks

- Set up React, TypeScript, Vite and Tailwind with the dark, minimal visual style.
- Build Home, Upload and Vehicle Identity, with the bounding box and expandable Top-5.
- Add the classical CV view, with loading and error states on every page.
- Build 3D Studio with React Three Fiber: rotate, zoom, pan, reset, metadata and downloads.
- Build Reference Studio and Research Lab, reading real files only.
- Write the design doc, architecture diagram and demo script, and record a backup demo video.

### Hands over

- Built frontend to C, served by FastAPI through the same tunnel.
- Bug list from phone testing to C and A.
- Demo script to everyone for rehearsal.

### Done when

- The full flow works on a phone and a laptop.
- No test data or dead buttons remain in the build.
- The viewer loads a generated file, not a stored stand-in.

## API contract to agree in the first hour

B, C and D can work in parallel once this is fixed. C pushes the exact response formats. D may use test responses while building, and removes them before the freeze.

```
POST /api/recognize            box, Top-5, confidence state, timings
POST /api/classical            intermediate images and features
GET  /api/references?class_id  candidates with source, licence, viewpoint, match
POST /api/references/upload    user's own reference images
POST /api/reconstruct          job id
GET  /api/jobs/{id}            real status, logs, errors
GET  /api/assets/{id}/{file}   GLB, OBJ, STL, validation report
GET  /api/metrics              saved notebook results, or "not available"
```

## Team rules

| Rule | Detail |
|---|---|
| **Folders** | A: `notebooks/01*`, `backend/reconstruction/`. B: `notebooks/02–03*`, `backend/classical_cv/`, `backend/detection/` evaluation. C: rest of `backend/`. D: `frontend/`, `docs/` design files. |
| **Git** | One branch per person, merged to `main` at each checkpoint. Add all four as collaborators first. |
| **Large files** | Keep the dataset, weights and meshes out of Git. Share them through a team drive. |
| **Honesty** | No hardcoded metrics, fake progress or stored results shown as live. Anything unfinished is labelled in the app. |
| **Viva** | Each person explains their part to one teammate before the code freeze, and everyone can explain the model and one failure case. |

## Hosting risks to plan for

| Risk | Plan |
|---|---|
| Quick tunnel URLs change when restarted | Submit a simple stable page, such as GitHub Pages, that links to the current tunnel URL. Check first that the organisers accept this. |
| The GPU laptop crashes or disconnects during judging | Have a second laptop ready to run recognition, plus the backup demo video. |
| Slow or unstable venue Wi-Fi | Test through a phone hotspot as well, and keep the hotspot ready for judging. |
| Several judges generate 3D at once | Queue jobs and show each person's real place in the queue. |

The tunnel hasn't been tested yet. Person A tests it by 1h if possible, and by 6.5h at the latest.

## If time runs short, cut in this order

| Cut | Who's freed up | What remains |
|---|---|---|
| FBX export, multi-view experiment | A | GLB, OBJ and STL from a single image |
| Automatic viewpoint grouping | C | Users sort references by hand |
| Wikimedia retrieval | C, D | Manual reference upload |
| Retraining ablations | A, B | Inference-only ablations, disclosed in the report |
| 3D generation, only as a last resort | A, C, D | The full P5 system, with 3D labelled as unavailable |

**Never cut:** the trained classifier, classical CV, detection, the comparison, ablation, optimisation, three failure cases, the public URL and the required documents.

## Needed before you start

| From | Item |
|---|---|
| **A** | GPU model and VRAM, from running `nvidia-smi`. Under 6 GB of VRAM may mean changing the 3D settings or model. |
| **Team** | Exact deadline, Team ID and all USNs. |
| **Repo owner** | Collaborator access to `sinchanaa23/Apex3D` for all four people. |
| **Organisers** | Design-document template, correct rubric weights, and whether a laptop tunnel counts as a live hosted URL. |
