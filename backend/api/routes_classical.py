"""
API Router for Classical Computer Vision Transforms and Feature Descriptors.
POST /api/classical
"""

import io
import logging
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from PIL import Image

from backend.config import MAX_UPLOAD_SIZE, ALLOWED_MIME_TYPES
from backend.models import ClassicalCVResponse
from backend.services.classical_service import ClassicalCVService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Classical CV"])


@router.post(
    "/classical",
    response_model=ClassicalCVResponse,
    summary="Classical Computer Vision Feature Extraction & Visualisation",
    description="Computes CLAHE (LAB luminance), Canny edges, contour overlay, largest contour ROI, HOG visualization, LBP, HSV histogram, and geometric moments.",
)
async def process_classical_cv(
    file: UploadFile = File(..., description="Target image file"),
    bbox_x1: Optional[float] = Form(None),
    bbox_y1: Optional[float] = Form(None),
    bbox_x2: Optional[float] = Form(None),
    bbox_y2: Optional[float] = Form(None),
):
    # 1. MIME type validation
    content_type = file.content_type or ""
    if content_type.lower() not in ALLOWED_MIME_TYPES and not any(
        file.filename.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp"]
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{content_type}'.",
        )

    # 2. File size validation
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File exceeds maximum allowed upload size.",
        )

    # 3. Decode image
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Cannot decode image: {e}",
        )

    # 4. Parse bounding box if provided
    bbox = None
    if all(v is not None for v in [bbox_x1, bbox_y1, bbox_x2, bbox_y2]):
        bbox = (bbox_x1, bbox_y1, bbox_x2, bbox_y2)

    # 5. Execute Classical CV Service
    service = ClassicalCVService.get_instance()
    response = service.process(image, bbox=bbox)
    return response

