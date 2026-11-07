"""
Apex3D Main FastAPI Application Entrypoint.
Serves recognition, classical CV, reference retrieval, 3D reconstruction, and static frontend assets.
"""

import asyncio
import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import Depends, FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from backend.config import (
    FRONTEND_DIST_DIR,
    STORAGE_DIR,
    DEVICE,
    CUDA_AVAILABLE,
)
from backend.auth import login_required, require_user, router as auth_router, session_user
from backend.database import init_db, get_db_connection
from backend.models import HealthResponse
from backend.api import (
    recognition_router,
    classical_router,
    references_router,
    reconstruct_router,
    metrics_router,
)
from backend.reconstruction.triposr_backend import available as mesh_builder_available
from backend.services.detector_service import VehicleDetector
from backend.services.classifier_service import VehicleClassifier
from backend.services.reconstruction_service import ReconstructionManager
from backend.services.cleanup_service import cleanup_old_storage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("apex3d")


RETENTION_HOURS = float(os.getenv("APEX3D_RETENTION_HOURS", "2"))
CLEANUP_EVERY_SECONDS = 15 * 60


async def _cleanup_loop():
    """Deletes users' uploaded photos and old results once they are older than RETENTION_HOURS. Runs now, then every 15 minutes."""
    while True:
        try:
            await asyncio.to_thread(cleanup_old_storage, RETENTION_HOURS)
        except Exception:  # noqa: BLE001
            logger.exception("Scheduled storage cleanup failed")
        await asyncio.sleep(CLEANUP_EVERY_SECONDS)


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
    cleanup_task = asyncio.create_task(_cleanup_loop())
    logger.info("Uploaded photos and results are deleted after %.1f hours.", RETENTION_HOURS)
    yield
    cleanup_task.cancel()
    logger.info("Shutting down Apex3D Backend.")


# On a public deployment set APEX3D_DISABLE_DOCS=1, so the interactive API pages (/docs, /redoc, /openapi.json) are not published.
_docs = os.getenv("APEX3D_DISABLE_DOCS", "").strip().lower() not in {"1", "true", "yes", "on"}
app = FastAPI(
    docs_url="/docs" if _docs else None,
    redoc_url="/redoc" if _docs else None,
    openapi_url="/openapi.json" if _docs else None,
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

# The app is served from the same origin as the API, so no other website needs to call it. Allowing every origin together
# with login cookies would let any website read a logged-in user's responses, so only the dev servers are listed.
# Add more with APEX3D_CORS_ORIGINS="https://a.example,https://b.example".
_extra_origins = [o.strip() for o in os.getenv("APEX3D_CORS_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8000", "http://127.0.0.1:8000", *_extra_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
# Only the account routes and the health check are public. Everything else needs a login.
app.include_router(auth_router)
_login = [Depends(require_user)]
app.include_router(metrics_router, dependencies=_login)
app.include_router(recognition_router, dependencies=_login)
app.include_router(classical_router, dependencies=_login)
app.include_router(references_router, dependencies=_login)
app.include_router(reconstruct_router, dependencies=_login)


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
        mesh_builder=mesh_builder_available(),
    )


@app.post(
    "/api/maintenance/cleanup",
    tags=["System"],
    summary="Purge Stale Temporary Files",
    dependencies=[Depends(require_user)],
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
    async def serve_spa(full_path: str, request: Request):
        # Allow API routes to pass through
        if full_path.startswith("api/") or full_path == "api" or full_path.startswith("docs") or full_path.startswith("openapi.json"):
            return JSONResponse(status_code=404, content={"detail": "API endpoint not found"})
        
        # The saved result files and the 3D models are data, not app code, so they need a login like the API does.
        if full_path.startswith(("results/", "models/")) and login_required() and session_user(request) is None:
            return JSONResponse(status_code=401, content={"detail": "Please log in to use this."})

        file_path = (FRONTEND_DIST_DIR / full_path).resolve()
        if FRONTEND_DIST_DIR.resolve() in file_path.parents and file_path.is_file():
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

