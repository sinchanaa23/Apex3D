"""
Comprehensive End-to-End Test Suite for Apex3D API & Backend Services.
Validates all endpoints specified in docs/API_CONTRACT.md.
"""

import io
import time
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
import numpy as np

from backend.main import app
from backend.data.stanford_cars_classes import STANFORD_CARS_CLASSES

client = TestClient(app)


def create_sample_car_image(width=400, height=300) -> bytes:
    """Generates a realistic synthetic test car image."""
    img = Image.new("RGB", (width, height), color=(220, 225, 230))
    draw = ImageDraw.Draw(img)
    # Draw road/ground
    draw.rectangle([0, height - 70, width, height], fill=(60, 60, 65))
    # Draw car body (lower)
    draw.rectangle([50, height - 140, width - 50, height - 60], fill=(180, 20, 30))
    # Draw car cabin (upper)
    draw.polygon([
        (110, height - 140),
        (160, height - 200),
        (width - 130, height - 200),
        (width - 80, height - 140),
    ], fill=(30, 40, 50))
    # Draw wheels
    draw.ellipse([80, height - 85, 140, height - 25], fill=(20, 20, 20))
    draw.ellipse([width - 140, height - 85, width - 80, height - 25], fill=(20, 20, 20))
    # Headlight
    draw.rectangle([width - 55, height - 125, width - 50, height - 105], fill=(255, 255, 180))

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_health_endpoint():
    """Verify system health check."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "gpu_available" in data
    assert "device" in data
    assert "queue_size" in data


def test_metrics_endpoint():
    """Verify offline metrics endpoint respects honesty principle."""
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["available", "not available"]
    if data["status"] == "not available":
        assert "not finished" in data["message"].lower() or "not been generated" in data["message"].lower()


def test_recognize_endpoint_success():
    """Verify POST /api/recognize with sample car photo."""
    img_bytes = create_sample_car_image()
    files = {"file": ("test_car.jpg", img_bytes, "image/jpeg")}
    response = client.post("/api/recognize", files=files)
    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert "car_detected" in data
    assert "crop_used" in data
    assert len(data["predictions"]) == 5

    # Check Top-5 structure
    top1 = data["predictions"][0]
    assert "class_id" in top1
    assert "class_name" in top1
    assert "make" in top1
    assert "model" in top1
    assert "year" in top1
    assert "body_style" in top1
    assert 0.0 <= top1["probability"] <= 1.0

    # Check confidence state
    assert data["confidence_state"] in ["confident", "ambiguous", "rejected"]
    assert "confidence_score" in data
    assert "margin" in data
    assert "timings_ms" in data
    assert "detection" in data["timings_ms"]
    assert "classification" in data["timings_ms"]


def test_recognize_bad_file_type():
    """Verify rejection of non-image file uploads."""
    fake_txt = b"This is not a vehicle photograph."
    files = {"file": ("report.txt", fake_txt, "text/plain")}
    response = client.post("/api/recognize", files=files)
    assert response.status_code == 400
    assert "unsupported file format" in response.json()["detail"].lower()


def test_classical_cv_endpoint():
    """Verify POST /api/classical returns all image stages and geometric features."""
    img_bytes = create_sample_car_image()
    files = {"file": ("test_car.jpg", img_bytes, "image/jpeg")}
    response = client.post("/api/classical", files=files)
    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    # Verify image stages
    stages = data["intermediate_images"]
    for required in ["original", "clahe", "edges", "contours", "largest_contour_roi", "hog_visualization"]:
        assert required in stages
        assert stages[required].startswith("data:image/png;base64,")

    # Verify features
    features = data["features"]
    assert "geometric" in features
    assert "aspect_ratio" in features["geometric"]
    assert "extent" in features["geometric"]
    assert "solidity" in features["geometric"]
    assert "hu_moments" in features["geometric"]
    assert len(features["geometric"]["hu_moments"]) == 7

    assert "hog" in features
    assert features["hog"]["vector_length"] > 0

    assert "lbp" in features
    assert len(features["lbp"]["histogram"]) > 0

    assert "color_hsv" in features
    assert "hue_histogram" in features["color_hsv"]

    assert "timings_ms" in data
    assert data["timings_ms"]["total"] > 0


def test_references_endpoint():
    """Verify GET /api/references returns candidates, licenses, and viewpoints."""
    response = client.get("/api/references?class_id=7&limit=4")
    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert "query" in data
    assert "candidates" in data
    assert "viewpoint_groups" in data
    assert "honesty_notice" in data

    if data["candidates"]:
        cand = data["candidates"][0]
        assert "id" in cand
        assert "title" in cand
        assert "image_url" in cand
        assert "license" in cand
        assert "viewpoint" in cand


def test_manual_reference_upload():
    """Verify POST /api/references/upload saves user contributed references."""
    img_bytes = create_sample_car_image(200, 150)
    files = {"file": ("my_car_side.jpg", img_bytes, "image/jpeg")}
    data = {"viewpoint": "side", "label": "Custom Side Profile"}
    response = client.post("/api/references/upload", files=files, data=data)
    assert response.status_code == 200
    res = response.json()
    assert res["success"] is True
    assert res["candidate"]["viewpoint"] == "side"
    assert res["candidate"]["source"] == "manual_upload"


def test_reconstruction_workflow():
    """Verify complete 3D reconstruction lifecycle: submit -> poll -> download assets."""
    img_bytes = create_sample_car_image(250, 180)
    files = {"file": ("car_3d_input.jpg", img_bytes, "image/jpeg")}
    submit_resp = client.post("/api/reconstruct", files=files)
    assert submit_resp.status_code == 202
    submit_data = submit_resp.json()

    job_id = submit_data["job_id"]
    assert submit_data["status"] == "queued"
    assert "created_at" in submit_data

    # Poll until worker completes (timeout 20s)
    max_wait = 20
    start = time.time()
    final_status = None
    job_info = None

    while (time.time() - start) < max_wait:
        poll_resp = client.get(f"/api/jobs/{job_id}")
        assert poll_resp.status_code == 200
        job_info = poll_resp.json()
        final_status = job_info["status"]
        if final_status in ["completed", "failed"]:
            break
        time.sleep(1.0)

    assert final_status == "completed", f"Job failed or timed out: {job_info.get('error')}"
    assert job_info["progress"] == 1.0
    assert len(job_info["logs"]) >= 3

    # Check asset download links
    assets = job_info["assets"]
    assert "glb_url" in assets
    assert "obj_url" in assets
    assert "stl_url" in assets
    assert "validation_report_url" in assets

    # Check validation report
    val = job_info["validation"]
    assert val is not None
    assert val["is_watertight"] is True
    assert val["vertex_count"] > 0
    assert val["face_count"] > 0
    assert val["asset_readiness"] in ["draft", "needs_repair"]

    # Download GLB asset
    glb_resp = client.get(f"/api/assets/{job_id}/mesh.glb")
    assert glb_resp.status_code == 200
    assert glb_resp.headers["content-type"] == "model/gltf-binary"
    assert len(glb_resp.content) > 100

    # Download OBJ asset
    obj_resp = client.get(f"/api/assets/{job_id}/mesh.obj")
    assert obj_resp.status_code == 200
    assert len(obj_resp.content) > 50

    # Download STL asset
    stl_resp = client.get(f"/api/assets/{job_id}/mesh.stl")
    assert stl_resp.status_code == 200
    assert len(stl_resp.content) > 50

    # Download validation report
    val_resp = client.get(f"/api/assets/{job_id}/validation.json")
    assert val_resp.status_code == 200
    assert val_resp.json()["is_watertight"] is True
