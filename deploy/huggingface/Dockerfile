# Apex3D for Hugging Face Spaces (Docker SDK). scripts/make_hf_space.py copies this to the Space as its Dockerfile.
# Same app as the root Dockerfile, with the changes Spaces needs:
#   - the container runs as user 1000, not root, so every cache and writable folder belongs to that user;
#   - the app listens on port 7860 (the app_port in the Space's README header);
#   - a free CPU Space has 2 vCPUs and about 16 GB of memory, which fits the 3D mesh builder (peak about 1.25 GB).

# --- 1. Build the frontend -------------------------------------------------------------------------------------------
FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
COPY results/ /results/
RUN npm run build

# --- 2. The server ---------------------------------------------------------------------------------------------------
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 git && rm -rf /var/lib/apt/lists/*

# Python packages go in as root. CPU-only PyTorch first (the default wheel is much larger and pulls in CUDA).
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
    && pip install omegaconf einops "transformers<5" huggingface-hub rembg xatlas imageio

# Everything after this belongs to user 1000, which is who Spaces runs the container as.
RUN useradd -m -u 1000 user && mkdir -p /app/.cache /app/third_party /app/backend/storage && chown -R user:user /app
USER user
WORKDIR /app
ENV HF_HOME=/app/.cache/huggingface TORCH_HOME=/app/.cache/torch U2NET_HOME=/app/.cache/u2net \
    PORT=7860 OMP_NUM_THREADS=2 APEX3D_REQUIRE_LOGIN=1 APEX3D_DISABLE_DOCS=1

# Weights are fetched once, here, so the first photo and the first mesh don't wait for downloads.
RUN python -c "from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn as m, FasterRCNN_MobileNet_V3_Large_FPN_Weights as W; m(weights=W.DEFAULT)"
# TripoSR (MIT) at the commit Apex3D was tested with (same as scripts/setup_triposr.ps1).
RUN cd third_party && mkdir TripoSR && cd TripoSR \
    && git init -q && git remote add origin https://github.com/VAST-AI-Research/TripoSR.git \
    && git fetch -q --depth 1 origin 107cefdc244c39106fa830359024f6a2f1c78871 && git checkout -q FETCH_HEAD \
    && rm -rf .git examples figures
RUN python -c "from huggingface_hub import snapshot_download, hf_hub_download; snapshot_download('stabilityai/TripoSR'); hf_hub_download('facebook/dino-vitb16', 'config.json')" \
    && python -c "import rembg; rembg.new_session('u2net')"

COPY --chown=user:user backend/ backend/
COPY --chown=user:user checkpoints/ checkpoints/
COPY --chown=user:user results/ results/
COPY --chown=user:user catalog/ catalog/
COPY --chown=user:user eval_P5_metrics.json ./
COPY --chown=user:user --from=web /web/dist frontend/dist

EXPOSE 7860
CMD ["sh", "-c", "exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
