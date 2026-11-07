"""
API Router for 3D Reconstruction Jobs and Mesh Assets.
POST /api/reconstruct
GET  /api/jobs/{id}
GET  /api/assets/{id}/{file}
"""

import io
import uuid
import logging
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, Body, HTTPException, status
from fastapi.responses import FileResponse
from PIL import Image

from backend.config import UPLOADS_DIR, JOBS_DIR, MAX_UPLOAD_SIZE, ALLOWED_MIME_TYPES
from backend.reconstruction.triposr_backend import available as mesh_builder_available
from backend.models import (
    ReconstructionJobResponse,
    ReconstructionJobCreate,
    ReconstructionOptions,
)
from backend.database import get_job
from backend.services.mesh_library import cache_dir, catalog_glb, instant_job
from backend.services.reconstruction_service import ReconstructionManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["3D Reconstruction"])

MIME_TYPE_MAP = {
    "mesh.glb": "model/gltf-binary",
    "mesh.obj": "text/plain",
    "mesh.stl": "model/stl",
    "validation.json": "application/json",
}


def _class_name(class_id) -> Optional[str]:
    from backend.data.stanford_cars_classes import STANFORD_CARS_CLASSES

    if isinstance(class_id, int) and 0 <= class_id < len(STANFORD_CARS_CLASSES):
        return STANFORD_CARS_CLASSES[class_id]
    return None


def _enqueued(job: dict) -> dict:
    return {
        "job_id": job["id"],
        "status": job["status"],
        "queue_position": 0,
        "created_at": job["created_at"],
        "message": "A ready-made mesh was found, no reconstruction needed.",
    }


@router.post(
    "/reconstruct",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Enqueue 3D Mesh Reconstruction Job",
    description="Submits an uploaded image to the background TripoSR queue. Respects single GPU laptop queue. Web addresses are not fetched by the server.",
)
async def create_reconstruction_job(
    file: Optional[UploadFile] = File(None, description="Optional image upload"),
    image_url: Optional[str] = Form(None, description="Not supported. Kept only so old clients get a clear error."),
    remove_background: bool = Form(True),
    mc_resolution: int = Form(256),
    class_id: Optional[int] = Form(None, description="Confirmed Stanford Cars class, if the user matched one"),
    source: str = Form("photo", description="'photo' (the user's own) or 'reference' (a reference photo of the matched car)"),
):
    # A licensed catalog model for the matched car, or a mesh already built from a reference photo, is returned at once.
    base_options = {"model": "triposr", "remove_background": remove_background, "mc_resolution": mc_resolution}
    if class_id is not None:
        if catalog_glb(class_id):
            return _enqueued(instant_job(class_id, "catalog", base_options))
        if source == "reference" and cache_dir(class_id):
            return _enqueued(instant_job(class_id, "reference", base_options))

    # No ready-made mesh, and this server has no TripoSR: say so now, instead of creating a job that can only fail.
    if not mesh_builder_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Building a mesh from a photo needs TripoSR, which is not installed on this server. Run Apex3D on your own computer with TripoSR installed to use it. The showcase model still works here.",
        )

    upload_id = uuid.uuid4().hex[:10]
    target_path = None

    # Handle direct multipart upload
    if file is not None:
        contents = await file.read()
        if len(contents) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Uploaded file exceeds maximum size limit.",
            )
        ext = file.filename.split(".")[-1].lower() if "." in file.filename else "jpg"
        target_path = UPLOADS_DIR / f"{upload_id}.{ext}"
        with open(target_path, "wb") as f:
            f.write(contents)

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Upload the image as a 'file'. The server does not fetch web addresses.",
        )

    # Options
    options = {**base_options, "mesh_source": "reference" if (source == "reference" and class_id is not None) else "photo", "class_id": class_id}

    mgr = ReconstructionManager.get_instance()
    job_data = mgr.submit_job(target_path, options=options)

    return {
        "job_id": job_data["id"],
        "status": job_data["status"],
        "queue_position": job_data.get("queue_position", 1),
        "created_at": job_data["created_at"],
        "message": "Reconstruction job enqueued successfully.",
    }


@router.get(
    "/jobs/{id}",
    response_model=ReconstructionJobResponse,
    summary="Poll 3D Reconstruction Job State & Logs",
    description="Returns real job status ('queued', 'running', 'completed', 'failed'), real-time stage, progress 0-1, execution logs, and output asset URLs.",
)
async def get_job_status(id: str):
    job = get_job(id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{id}' not found.",
        )

    return ReconstructionJobResponse(
        job_id=job["id"],
        status=job["status"],
        stage=job.get("stage", "waiting"),
        progress=float(job.get("progress", 0.0)),
        queue_position=int(job.get("queue_position", 0)),
        created_at=job["created_at"],
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        logs=job.get("logs") or [],
        error=job.get("error"),
        assets=job.get("assets"),
        validation=job.get("validation"),
        mesh_source=(job.get("options") or {}).get("mesh_source"),
        class_id=(job.get("options") or {}).get("class_id"),
        class_name=_class_name((job.get("options") or {}).get("class_id")),
    )


@router.get(
    "/assets/{id}/{file}",
    summary="Download Generated 3D Asset or Validation Telemetry",
    description="Streams generated GLB, OBJ, STL, or validation.json for a completed reconstruction job.",
)
async def download_asset(id: str, file: str):
    if file not in MIME_TYPE_MAP:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported asset requested. Supported: {list(MIME_TYPE_MAP.keys())}",
        )

    asset_path = JOBS_DIR / id / file
    if not asset_path.exists() or not asset_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Asset '{file}' for job '{id}' does not exist or has not finished generating.",
        )

    media_type = MIME_TYPE_MAP[file]
    return FileResponse(
        str(asset_path),
        media_type=media_type,
        filename=f"apex3d_{id[:8]}_{file}",
    )

