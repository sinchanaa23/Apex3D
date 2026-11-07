# Deploying Apex3D on Render (free plan)

The repository has what Render needs: `render.yaml` (the Blueprint), `Dockerfile` (the lean image) and `deploy/Dockerfile.full` (the same plus the TripoSR mesh builder, for a paid plan with 2 GB or more). **Nothing here has been run on Render yet, and the Docker image was never built** (Docker Desktop wasn't running). What was tested is the same code, with PyTorch and trimesh blocked so that the lean server's needs are real: it starts, answers, and gives the same results.

## Start it on Render

1. Push to `main` on GitHub (done: `sinchanaa23/Apex3D`).
2. Log in to https://dashboard.render.com. Click **New +**, then **Blueprint**.
3. **Connect GitHub** if asked. If `Apex3D` isn't listed, choose **Configure account** and allow Render access to that repository.
4. Pick `sinchanaa23/Apex3D`, branch `main`. Render reads `render.yaml` and shows one web service, **apex3d**, on the **Free** plan. No card should be needed.
5. Click **Apply** (or **Deploy Blueprint**). Don't enter any secrets.
6. Open the service and its **Logs** tab. The build takes roughly 10 to 15 minutes (it installs PyTorch once, in a build stage, to export the detector). When the logs show the server running and the service says **Live**, it works.
7. Open the `https://apex3d-xxxx.onrender.com` address at the top of the page. The landing page is public. Click **Sign up**, make an account, then try Upload and Identify.
8. If the build fails, copy the last 40 lines of the log and send them to me.

To redeploy after a change: merge it to `main`, then on the service click **Manual Deploy > Deploy latest commit**. Pushes don't deploy by themselves.

## What was measured

| | Before (PyTorch detector) | Now (ONNX detector, no PyTorch) |
|---|---|---|
| Server memory after start-up | 536 MB | **207 MB** |
| Server memory while serving (peak) | 816 MB | **257 MB** |
| One recognition, one CPU thread, on this laptop | not measured | **0.64 to 1.1 s** |

Render's free plan has 512 MB of memory and a small share of one CPU (Render's figures, from memory, not checked), so the memory fits with room to spare. On a share of one CPU each recognition will be slower than on a laptop. I'd guess several seconds, but it was not measured on Render.

The detector is the same Faster R-CNN network, exported to ONNX (`scripts/export_detector_onnx.py`) and run with ONNX Runtime. It is not bit-identical to PyTorch. On 200 random test photos (`scripts/check_detector_onnx.py`, results in `results/detector_onnx_parity.json`): the chosen box matches PyTorch's with a mean IoU of 0.997, its mean IoU with the true box is 0.9200 against 0.9209, 199 of 200 photos get the same final car class, and 188 against 189 are correct.

## What to expect on the live site

- Works: login and sign-up, upload and identification (make, model, year, confidence, top 5), the classical stages, references, the showcase 3D car with GLB, OBJ and STL downloads, and the Research Lab.
- **Build experimental mesh is off.** The 3D Studio shows a heads-up that building needs TripoSR, which isn't installed on the server, and points to running Apex3D on your own computer. The showcase car works.
- **Accounts are lost when the service restarts or wakes from sleep.** The free plan has no disk. Sign up again before a demo.
- **It sleeps after about 15 minutes without visitors.** The next visit waits roughly a minute while it wakes and loads. Open the site a few minutes before judging, and sign up on it.
- Anyone with the address can create an account. Sign-ups are limited to 30 per hour, wrong passwords to 5 per email per 10 minutes, and uploaded photos are deleted after 2 hours.

## Settings (environment variables)

| Name | Value in `render.yaml` | What it does |
|---|---|---|
| `APEX3D_REQUIRE_LOGIN` | `1` | Everything except the landing page needs a login |
| `APEX3D_DISABLE_DOCS` | `1` | Hides `/docs`, `/redoc` and `/openapi.json` |
| `APEX3D_ORT_THREADS` | `1` | CPU threads for the models (the free plan has one small CPU) |
| `OMP_NUM_THREADS` | `1` | Same, for the libraries that read it |

## If you can use a paid plan later

Keep accounts across restarts by adding a disk (the commented `disk` block in `render.yaml`), and for the mesh builder use `deploy/Dockerfile.full` on a plan with at least 2 GB of memory. A mesh build peaked near 1.25 GB and took about 4.8 minutes on two CPU threads when measured.

## Running the same image locally

```bash
docker build -t apex3d .
docker run -p 10000:10000 -v apex3d-data:/app/backend/storage apex3d
```
