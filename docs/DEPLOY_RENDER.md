# Deploying Apex3D on Render

The repository has what Render needs: `Dockerfile` (the **lean** image: the React app and the FastAPI server, without the 3D mesh builder), `deploy/Dockerfile.full` (the same plus TripoSR and its weights), `render.yaml` (the Blueprint, which uses the lean image) and `.dockerignore`. Nothing here has been run on Render yet, and the Docker images were not test-built (Docker Desktop was not running). What *was* tested is the same code from a fresh clone of GitHub in a brand-new Python 3.11 environment, run as described below.

## Read this first: the free plan is probably too small

Measured on the lean app from a clean clone (Windows, Python 3.11, one CPU thread):

| What | Measured |
|---|---|
| Memory right after start-up, with the detector and classifier loaded | about 580 MB |
| Committed (private) memory while serving | about 680 MB |
| Highest memory seen (peak working set) | about 816 MB |
| Where the memory goes (private, in isolation) | classifier 75 MB, PyTorch about 165 MB, detector weights about 160 MB, trimesh about 50 MB, the rest server and libraries |
| One recognition | about 1.2 s on this laptop with all its cores |

Render's free plan has 512 MB of memory and a small share of one CPU (Render's figures, from memory, not checked). On those numbers the lean app is expected to **run out of memory on the free plan**, and even if it started, a share of one CPU would make each recognition far slower than the 1.2 s measured here. An earlier version of this guide said the server used about 330 MB. That figure was wrong: it was read while Windows had shrunk the idle process's memory.

What works for sure: **a plan with at least 1 GB, realistically 2 GB (`standard`)**, which needs a card. If you try the free plan anyway, the sign that it failed is "Out of memory" in the service's logs.

## What the lean image does without TripoSR

- Everything works except building a mesh from a photo: login and sign-up, upload and identification (make, model, year, confidence, top 5), the classical CV stages, references, the **showcase 3D car** with its GLB, OBJ and STL downloads, and the Research Lab.
- The server reports that it has no mesh builder (`mesh_builder: false` in `/api/health`). The 3D Studio then shows a heads-up on the experimental-mesh tab instead of the build button: building needs TripoSR, which isn't installed on this server; run Apex3D on your own computer with `scripts/setup_triposr.ps1` to try it; the showcase model works here. `POST /api/reconstruct` answers `503` with the same message instead of creating a job that could only fail.
- To get the mesh builder online, build `deploy/Dockerfile.full` on a plan with 2 GB or more (a mesh build peaked near 1.25 GB when measured, and took about 4.8 minutes with 2 CPU threads).

## Setup

1. Push the code you want live to `main` on GitHub (`sinchanaa23/Apex3D`). Render deploys from GitHub.
2. In the Render dashboard: **New > Blueprint**, connect your GitHub account if asked, and pick the `Apex3D` repository and the `main` branch.
3. Render reads `render.yaml` and shows one web service, `apex3d`, with its monthly price. Check the plan, then **Apply**.
4. The lean build should take roughly 10 minutes (PyTorch and the detector weights are downloaded into the image). When the logs show the server running and `/api/health` passes, the service is **Live**.
5. Open the `https://apex3d-xxxx.onrender.com` address Render shows. You land on the login page. **Sign up**, then try Upload > Identity > 3D Studio.

`autoDeploy` is off on purpose, so a teammate's push can't change the live site. To deploy new code: merge it to `main`, then **Manual Deploy > Deploy latest commit** on the service.

## Settings in `render.yaml`

| Setting | Why |
|---|---|
| `plan: free` (currently) | Chosen because no card is available. See the warning above: expected to be too small. For a working deployment use `standard` (2 GB). |
| no `disk` | Persistent disks aren't allowed on the free plan, so accounts are lost every time the service restarts or goes back to sleep. With a paid plan, restore the commented `disk` block so accounts survive. |
| `OMP_NUM_THREADS=1` | A small plan has one CPU, so PyTorch should not start more threads than that. |
| `APEX3D_REQUIRE_LOGIN=1`, `APEX3D_DISABLE_DOCS=1` | Everything needs a login, and the API documentation pages are hidden. |

Check the current prices on Render's pricing page before you apply; I did not look them up.

## Other things to know

- A free service goes to sleep after about 15 minutes without visitors, and the next visit waits about a minute while it wakes and loads its models.
- Anyone who has the address can create an account (there is no email check). Sign-ups are limited to 30 per hour, and a wrong password is blocked after 5 tries per email for 10 minutes.
- Uploaded photos are deleted automatically after 2 hours (`APEX3D_RETENTION_HOURS`).

## The clean-clone test

Done from a fresh `git clone` of `main`, with a new Python 3.11 virtual environment and Node 22, following the Quickstart:

| Step | Result |
|---|---|
| `pip install -r requirements.txt` | worked |
| `cd frontend && npm ci && npm run build` | worked |
| `python start_server.py`, then the whole flow over HTTP | worked: protected pages and files refuse without a login, sign-up, 4 of 4 recognitions correct and confident, the classical view, references, and a polite 503 for a mesh build |

Only Python 3.11 was available, so Python 3.12 and 3.13 (which the README names) were not tested.

## Running the same image locally

```bash
docker build -t apex3d .                                     # lean image
docker run -p 10000:10000 -v apex3d-data:/app/backend/storage apex3d
docker build -f deploy/Dockerfile.full -t apex3d-full .      # with the mesh builder
```

Then open http://localhost:10000.
