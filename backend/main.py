"""
Apex3D Main FastAPI Application Entrypoint.
Serves recognition, classical CV, reference retrieval, 3D reconstruction, and static frontend assets.
"""

import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from backend.config import (
    FRONTEND_DIST_DIR,
    STORAGE_DIR,
    DEVICE,
    CUDA_AVAILABLE,
)
from backend.database import init_db, get_db_connection
from backend.models import HealthResponse
from backend.api import (
    recognition_router,
    classical_router,
    references_router,
    reconstruct_router,
    metrics_router,
)
from backend.services.detector_service import VehicleDetector
from backend.services.classifier_service import VehicleClassifier
from backend.services.reconstruction_service import ReconstructionManager
from backend.services.cleanup_service import cleanup_old_storage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("apex3d")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle startup and shutdown management."""
    logger.info("Initializing Apex3D Backend...")
    # Initialize SQLite schema
    init_db()

    # Warm up background services
    ReconstructionManager.get_instance()
    VehicleDetector.get_instance()
    VehicleClassifier.get_instance()

    logger.info("Apex3D Backend initialized on device: %s", DEVICE)
    yield
    logger.info("Shutting down Apex3D Backend.")


app = FastAPI(
    title="Apex3D — Vehicle Recognition & 3D Reconstruction API",
    description=(
        "Production backend for P5 Intelligent Fine-Grained Vehicle Recognition System.\n\n"
        "- **Detection**: Faster R-CNN vehicle localization\n"
        "- **Classification**: 196 Stanford Cars fine-grained recognition\n"
        "- **Classical CV**: CLAHE, Canny, Contours, HOG, LBP, and Geometric analysis\n"
        "- **Retrieval**: Wikimedia Commons CC-licensed references with deduplication & viewpoint grouping\n"
        "- **3D Reconstruction**: Background job queue, TripoSR bridge, and Trimesh multi-format export"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration allowing local dev and tunnel access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(recognition_router)
app.include_router(classical_router)
app.include_router(references_router)
app.include_router(reconstruct_router)
app.include_router(metrics_router)


@app.get(
    "/api/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="System Health & Hardware Readiness",
)
async def health_check():
    """Returns GPU status, device, and active reconstruction queue size."""
    conn = get_db_connection()
    queue_count = 0
    try:
        cur = conn.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued', 'running')")
        queue_count = cur.fetchone()[0]
    except Exception:
        pass
    finally:
        conn.close()

    return HealthResponse(
        status="ok",
        version="1.0.0",
        service="Apex3D Backend",
        gpu_available=CUDA_AVAILABLE,
        device=DEVICE,
        queue_size=queue_count,
    )


@app.post(
    "/api/maintenance/cleanup",
    tags=["System"],
    summary="Purge Stale Temporary Files",
    description="Purges uploaded images and generated 3D files older than max_age_hours (default 1.0 hr). Preserves SQLite metadata.",
)
async def run_storage_cleanup(max_age_hours: float = 1.0):
    report = cleanup_old_storage(max_age_hours=max_age_hours)
    return {"status": "success", "cleanup_report": report}


# -------------------------------------------------------------------------
# Static File & Single Page Application (SPA) Serving
# -------------------------------------------------------------------------

# Serve built frontend if dist exists (Person D handover)
if FRONTEND_DIST_DIR.exists() and FRONTEND_DIST_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIST_DIR)), name="static")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        # Allow API routes to pass through
        if full_path.startswith("api/") or full_path == "api" or full_path.startswith("docs") or full_path.startswith("openapi.json"):
            return JSONResponse(status_code=404, content={"detail": "API endpoint not found"})
        
        file_path = FRONTEND_DIST_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        
        index_html = FRONTEND_DIST_DIR / "index.html"
        if index_html.exists():
            return FileResponse(str(index_html))
        
        return JSONResponse(status_code=404, content={"detail": "Frontend build not found"})
else:
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "name": "Apex3D Backend API",
            "version": "1.0.0",
            "status": "online",
            "documentation": "/docs",
            "endpoints": [
                "/api/recognize",
                "/api/classical",
                "/api/references",
                "/api/reconstruct",
                "/api/jobs/{id}",
                "/api/assets/{id}/{file}",
                "/api/metrics",
                "/api/health",
            ],
        }


# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Global exception on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Internal Server Error: {str(exc)}"},
    )

