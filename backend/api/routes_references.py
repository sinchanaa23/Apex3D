"""
API Router for Reference Image Retrieval and Uploads.
GET /api/references
POST /api/references/upload
GET /api/references/file/{filename}
"""

import io
import logging
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, Query, HTTPException, status
from fastapi.responses import FileResponse
from pathlib import Path

from backend.config import MAX_UPLOAD_SIZE, ALLOWED_MIME_TYPES, REFERENCES_DIR
from backend.models import ReferencesResponse, ReferenceUploadResponse
from backend.services.retrieval_service import ReferenceRetrievalService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["References"])


@router.get(
    "/references",
    response_model=ReferencesResponse,
    summary="Search & Retrieve Reference Vehicle Images",
    description="Queries Wikimedia Commons for openly licensed images, deduplicates via perceptual hashing, and categorizes by viewpoint.",
)
async def get_references(
    class_id: Optional[int] = Query(None, description="Stanford Cars class index (0-195)"),
    query: Optional[str] = Query(None, description="Custom search query (e.g. 'Audi A5 2012')"),
    limit: int = Query(12, ge=1, le=30, description="Max reference candidates to return"),
):
    service = ReferenceRetrievalService.get_instance()
    response = service.retrieve(class_id=class_id, query=query, limit=limit)
    return response


@router.post(
    "/references/upload",
    response_model=ReferenceUploadResponse,
    summary="Upload User Reference Photograph",
    description="Allows users to upload their own reference angles (front, side, rear, 3/4) to build reference packs.",
)
async def upload_reference(
    file: UploadFile = File(..., description="Reference image file"),
    viewpoint: str = Form("unknown", description="front, front_three_quarter, side, rear, rear_three_quarter, or unknown"),
    class_id: Optional[int] = Form(None),
    label: Optional[str] = Form(None),
):
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Uploaded reference image exceeds size limit.",
        )

    service = ReferenceRetrievalService.get_instance()
    try:
        candidate = service.save_manual_upload(
            image_bytes=contents,
            filename=file.filename or "upload.jpg",
            viewpoint=viewpoint,
            class_id=class_id,
            label=label,
        )
        return ReferenceUploadResponse(
            success=True,
            candidate=candidate,
            message="User reference successfully indexed.",
        )
    except Exception as e:
        logger.error("Failed saving reference upload: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process uploaded reference image: {e}",
        )


@router.get(
    "/references/file/{filename}",
    summary="Serve Local Reference Image",
    description="Streams locally uploaded reference images.",
)
async def get_reference_file(filename: str):
    file_path = REFERENCES_DIR / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Reference file not found.")
    return FileResponse(str(file_path))

