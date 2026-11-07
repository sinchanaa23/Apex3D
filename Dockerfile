# Apex3D for Render (or any Docker host): the React app, the FastAPI backend and the TripoSR mesh builder in one container.
#   docker build -t apex3d .
#   docker run -p 10000:10000 -v apex3d-data:/app/backend/storage apex3d     then open http://localhost:10000
# The image is large (about 5 GB) because the 3D model weights are baked in, so nobody waits for a download on the first mesh.

# --- 1. Build the frontend -------------------------------------------------------------------------------------------
FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
# vite.config.ts copies these saved result files into the build for the Research Lab (it reads ../results).
COPY results/ /results/
RUN npm run build

# --- 2. The server ---------------------------------------------------------------------------------------------------
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
# OpenCV needs the two libraries; git fetches the TripoSR source below.
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 git && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# CPU-only PyTorch first (the default wheel is several times larger and drags in CUDA), then the rest.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt ./
RUN pip install -r requirements.txt

# The detector's weights are fetched once here, so the first request after a deploy doesn't wait for a download.
RUN python -c "from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn as m, FasterRCNN_MobileNet_V3_Large_FPN_Weights as W; m(weights=W.DEFAULT)"

# --- 3D mesh from a photo: TripoSR (MIT) at the commit Apex3D was tested with (same as scripts/setup_triposr.ps1) ----------
RUN pip install omegaconf einops "transformers<5" huggingface-hub rembg xatlas imageio
RUN mkdir -p third_party/TripoSR && cd third_party/TripoSR \
    && git init -q && git remote add origin https://github.com/VAST-AI-Research/TripoSR.git \
    && git fetch -q --depth 1 origin 107cefdc244c39106fa830359024f6a2f1c78871 && git checkout -q FETCH_HEAD \
    && rm -rf .git examples figures
# Weights baked in: TripoSR (about 1.7 GB), the DINO config it builds on, and the u2net background remover (about 176 MB).
RUN python -c "from huggingface_hub import snapshot_download, hf_hub_download; snapshot_download('stabilityai/TripoSR'); hf_hub_download('facebook/dino-vitb16', 'config.json')" \
    && python -c "import rembg; rembg.new_session('u2net')"

COPY backend/ backend/
COPY checkpoints/ checkpoints/
COPY results/ results/
COPY catalog/ catalog/
COPY eval_P5_metrics.json ./
COPY --from=web /web/dist frontend/dist

# Accounts, uploads and jobs live here. On Render this folder is a persistent disk, so accounts survive restarts.
VOLUME ["/app/backend/storage"]

ENV PORT=10000
EXPOSE 10000
CMD ["sh", "-c", "exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
