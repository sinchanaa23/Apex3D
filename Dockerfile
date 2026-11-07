# Apex3D, lean image for Render (or any Docker host): the React app and the FastAPI backend in one container.
#   docker build -t apex3d .
#   docker run -p 10000:10000 -v apex3d-data:/app/backend/storage apex3d     then open http://localhost:10000
# The server in this image does NOT contain PyTorch. The vehicle detector is exported to ONNX during the build (stage 2) and run with ONNX
# Runtime, like the classifier, which keeps the server near 260 MB of memory (measured) instead of over 500 MB.
# It also does NOT include TripoSR (the experimental photo-to-mesh builder: 1.7 GB of weights and about 1.25 GB of memory while running).
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

# --- 2. Export the detector to ONNX (PyTorch is used here only, and is thrown away) -----------------------------------
FROM python:3.11-slim AS detector
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install onnx onnxscript
WORKDIR /build
COPY scripts/export_detector_onnx.py scripts/
RUN python scripts/export_detector_onnx.py

# --- 3. The server ---------------------------------------------------------------------------------------------------
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY requirements-serve.txt ./
RUN pip install -r requirements-serve.txt

COPY backend/ backend/
COPY checkpoints/ checkpoints/
COPY --from=detector /build/checkpoints/detector_mobilenet.onnx checkpoints/detector_mobilenet.onnx
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
