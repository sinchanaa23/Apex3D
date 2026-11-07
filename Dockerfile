# Apex3D, lean image for Render (or any Docker host): the React app and the FastAPI backend in one container.
#   docker build -t apex3d .
#   docker run -p 10000:10000 -v apex3d-data:/app/backend/storage apex3d     then open http://localhost:10000
# It does NOT include TripoSR (the experimental photo-to-mesh builder: 1.7 GB of weights and about 1.25 GB of memory while running).
# The server notices, the 3D Studio says so, and the showcase car still works. For the full image use deploy/Dockerfile.full.

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
# OpenCV needs these two system libraries.
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# CPU-only PyTorch first (the default wheel is several times larger and drags in CUDA), then the rest.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt ./
RUN pip install -r requirements.txt

# The detector's weights are fetched once here, so the first request after a deploy doesn't wait for a download.
RUN python -c "from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn as m, FasterRCNN_MobileNet_V3_Large_FPN_Weights as W; m(weights=W.DEFAULT)"

COPY backend/ backend/
COPY checkpoints/ checkpoints/
COPY results/ results/
COPY catalog/ catalog/
COPY eval_P5_metrics.json ./
COPY --from=web /web/dist frontend/dist

# Accounts, uploads and jobs live here. Mount a persistent disk on this folder so accounts survive restarts. Render's free plan has no disk,
# so there accounts are lost whenever the service restarts.
VOLUME ["/app/backend/storage"]

ENV PORT=10000
EXPOSE 10000
CMD ["sh", "-c", "exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
