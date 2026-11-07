# Deploying Apex3D on Render

> **Current state of `render.yaml`: free plan, no disk.** It was set that way because a payment card isn't available. On the free plan (512 MB of memory, Render's limit from memory) the app is **not expected to work fully**: the 3D mesh builder needs about 1.25 GB and will crash the service, identification may also run out of memory (the server uses 330 to 580 MB with its models loaded), the service sleeps after about 15 minutes idle, and every account is lost whenever it restarts. For the complete app on Render use `plan: standard` and restore the `disk` block (the commented lines in `render.yaml`). Everything below describes that full setup.

The repository has what Render needs: `Dockerfile` (builds the React app and runs the FastAPI server and the TripoSR mesh builder in one container), `render.yaml` (the Blueprint) and `.dockerignore`. Nothing here has been run on Render yet, and the Docker build was not test-built locally (Docker Desktop was not running), so the first deploy is the real test.

## One-time setup

1. Push the code you want live to `main` on GitHub (`sinchanaa23/Apex3D`). Render deploys from GitHub.
2. In the Render dashboard: **New > Blueprint**, connect your GitHub account if asked, and pick the `Apex3D` repository and the `main` branch.
3. Render reads `render.yaml` and shows one web service, `apex3d`, with its monthly price. Check the plan and the disk, then **Apply**.
4. The first build is long (PyTorch, the detector weights, and about 2 GB of 3D model weights are downloaded into the image). Allow 20 to 30 minutes. When the logs show the server running and `/api/health` passes, the service is **Live**.
5. Open the `https://apex3d-xxxx.onrender.com` address Render shows. You land on the login page. **Sign up**, then try Upload > Identity > 3D Studio.

`autoDeploy` is off on purpose, so a teammate's push can't change the live site. To deploy new code: merge it to `main`, then **Manual Deploy > Deploy latest commit** on the service.

## Why this plan

All numbers were measured on the real code, on a laptop.

| What | Measured |
|---|---|
| Server with the detector and classifier loaded, after a few recognitions | about 330 MB |
| One 3D mesh build, with the server limited to 2 CPU threads | 287 s (4.8 min), whole server peaking near 1.1 GB |
| The same mesh build with 16 CPU threads | 247 s (4.1 min), whole server peaking near 1.25 GB |
| One recognition | about 1.2 s |

So 512 MB is too small once a mesh is built, and `standard` (2 GB, 1 CPU) fits with room to spare. More CPUs barely speed the mesh up (4.1 minutes with 16 threads against 4.8 with 2), so a bigger plan is only for memory. Expect a mesh to take **5 to 8 minutes on Render's single CPU**. That is a guess, because I could only test 2 threads, so watch the first one. The page shows progress while it builds, and only one mesh builds at a time; others wait in the queue.

| Setting in `render.yaml` | Why |
|---|---|
| `plan: standard` | 2 GB of memory, for the mesh build. If the logs ever show the service restarting for lack of memory, use `pro` (4 GB). |
| `disk` 1 GB at `/app/backend/storage` | Accounts, uploads and job records are stored here. Without a persistent disk they vanish on every restart and deploy, and everyone has to sign up again. |
| `OMP_NUM_THREADS=1` | Standard has one CPU, so PyTorch should not start more threads than that. |

Check the current prices on Render's pricing page before you apply. I did not look them up. From memory, a 2 GB instance costs a few times what the 512 MB one does (roughly 25 US dollars a month against 7), plus a few cents for the disk, but treat that as unverified.

**The free plan cannot run this app.** It has 512 MB of memory, which a mesh build exceeds, and it wipes accounts on every restart.

## What works on the live site

Everything: login and sign-up, upload and identification (make, model, year, confidence, top 5), the classical CV stages, references, the showcase 3D car with its GLB, OBJ and STL downloads, **Build experimental mesh** (TripoSR, with background removal), and the Research Lab. The one difference from the laptop is speed: identification takes a second or two and a mesh takes several minutes.

The first request after a deploy or a restart is slower while the models load.

## Before a demo

- Open the live site once, sign up, identify one photo and build one mesh, so the models are warm and you know the timing.
- Anyone who has the address can create an account (there is no email check or approval). Sign-ups are limited to 30 per hour, and a wrong password is blocked after 5 tries per email for 10 minutes.

## Settings (environment variables)

| Name | Set to | What it does |
|---|---|---|
| `APEX3D_REQUIRE_LOGIN` | `1` (default) | Everything needs a login. `0` turns it off, for local work only. |
| `APEX3D_DISABLE_DOCS` | `1` on Render | Hides `/docs`, `/redoc` and `/openapi.json`. |
| `APEX3D_CORS_ORIGINS` | not needed | Extra origins allowed to call the API. The app calls it from its own address. |
| `OMP_NUM_THREADS` | `1` on Render | CPU threads PyTorch may use. |

## Running the same image locally

```bash
docker build -t apex3d .
docker run -p 10000:10000 -v apex3d-data:/app/backend/storage apex3d
```

Then open http://localhost:10000.
