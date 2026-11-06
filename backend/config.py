"""
Configuration and environment settings for Apex3D Backend.
"""

from pathlib import Path
import os
import torch

# Base paths
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

STORAGE_DIR = BACKEND_DIR / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
REFERENCES_DIR = STORAGE_DIR / "references"
JOBS_DIR = STORAGE_DIR / "jobs"
DB_PATH = STORAGE_DIR / "apex3d.db"

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DOCS_DIR = PROJECT_ROOT / "docs"
FRONTEND_DIST_DIR = PROJECT_ROOT / "frontend" / "dist"

# Candidate metrics files to check (in order of priority)
METRICS_CANDIDATE_PATHS = [
    NOTEBOOKS_DIR / "metrics.json",
    PROJECT_ROOT / "eval_P5_metrics.json",
    DOCS_DIR / "metrics.json",
]

# Create necessary directories
for path in [STORAGE_DIR, UPLOADS_DIR, REFERENCES_DIR, JOBS_DIR, CHECKPOINTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Hardware / Device configuration
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"

# Classification confidence & decision thresholds
# Top-1 prob >= CONFIDENT_THRESHOLD AND margin >= MARGIN_THRESHOLD -> confident
CONFIDENT_THRESHOLD = float(os.getenv("APEX3D_CONFIDENT_THRESHOLD", "0.50"))
AMBIGUOUS_MARGIN_THRESHOLD = float(os.getenv("APEX3D_MARGIN_THRESHOLD", "0.34"))
REJECTED_THRESHOLD = float(os.getenv("APEX3D_REJECTED_THRESHOLD", "0.52"))

# Request constraints
MAX_UPLOAD_SIZE = 20 * 1024 * 1024  # 20 MB
ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/jpg",
}

# Wikimedia Commons API settings
WIKIMEDIA_API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "Apex3D-AcademicResearchBot/1.0 (https://github.com/sinchanaa23/Apex3D; contact: hackathon-team@apex3d.ai)"

# Reconstruction queue settings
MAX_CONCURRENT_3D_JOBS = 1  # Exactly 1 active 3D job at a time to respect single GPU
JOB_TIMEOUT_SECONDS = 300   # 5 minutes

