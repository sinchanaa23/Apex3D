"""
API Router for Vehicle Recognition: Detection and Fine-Grained Classification.
POST /api/recognize
"""

import io
import logging
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from PIL import Image

from backend.config import MAX_UPLOAD_SIZE, ALLOWED_MIME_TYPES
from backend.models import RecognitionResponse
from backend.services.detector_service import VehicleDetector, get_crop
from backend.services.classifier_service import VehicleClassifier

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Recognition"])


@router.post(
    "/recognize",
    response_model=RecognitionResponse,
    summary="Vehicle Detection & Fine-Grained Classification",
    description="Detects primary vehicle via R-CNN, crops ROI, classifies into Top-5 Stanford Cars classes, and reports confidence state.",
)
async def recognize_vehicle(
    file: UploadFile = File(..., description="Vehicle photograph (JPEG, PNG, WebP)"),
    force_full_frame: bool = Form(False, description="Bypass vehicle crop and classify full frame"),
):
    # 1. Validate file content type
    content_type = file.content_type or ""
    if content_type.lower() not in ALLOWED_MIME_TYPES and not any(
        file.filename.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp"]
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{content_type}'. Please upload JPEG, PNG, or WebP.",
        )

    # 2. Read bytes with size protection
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded file exceeds {MAX_UPLOAD_SIZE // (1024 * 1024)}MB size limit.",
        )

    # 3. Decode image
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to decode image file. File may be corrupted. Error: {e}",
        )

    # 4. Run Vehicle Detection
    detector = VehicleDetector.get_instance()
    primary_box, all_boxes, det_time_ms = detector.detect(image)
    car_detected = primary_box is not None

    # 5. Extract Crop (or fallback to full frame)
    crop_img, crop_used = get_crop(
        image, bbox=primary_box, force_full_frame=force_full_frame
    )

    # 6. Run Fine-Grained Classification
    classifier = VehicleClassifier.get_instance()
    predictions, conf_state, conf_score, margin, cls_time_ms, message = classifier.classify(crop_img)

    # If no car was detected, update message and crop_used
    if not car_detected:
        crop_used = "full_frame"
        if conf_state != "rejected":
            message = "No vehicle bounding box detected. Classified full image frame."

    total_time_ms = round(det_time_ms + cls_time_ms, 2)

    return RecognitionResponse(
        success=True,
        car_detected=car_detected,
        crop_used=crop_used,
        bounding_box=primary_box,
        all_boxes=all_boxes,
        predictions=predictions,
        confidence_state=conf_state,
        confidence_score=conf_score,
        margin=margin,
        message=message,
        timings_ms={
            "detection": det_time_ms,
            "classification": cls_time_ms,
            "total": total_time_ms,
        },
    )

