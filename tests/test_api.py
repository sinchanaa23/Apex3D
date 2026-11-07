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
from backend.services.classifier_service import VehicleClassifier
from backend.services.reconstruction_service import ReconstructionManager
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


MODEL_LOADED = VehicleClassifier.get_instance().model_type != "none"


def test_recognize_without_a_model_is_an_error_not_invented_results(monkeypatch):
    """With no trained model the API must say so, never return made-up predictions."""
    classifier = VehicleClassifier.get_instance()
    monkeypatch.setattr(classifier, "model_type", "none")
    files = {"file": ("test_car.jpg", create_sample_car_image(), "image/jpeg")}
    response = client.post("/api/recognize", files=files)
    assert response.status_code == 503
    assert "model_quantized.onnx" in response.json()["detail"]


@pytest.mark.skipif(not MODEL_LOADED, reason="needs checkpoints/model_quantized.onnx (Person A's handover zip)")
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

    # Person B's pipeline: ten stages, each an image, then the features and timing.
    keys = [step["key"] for step in data["steps"]]
    assert keys == ["original", "clahe", "edges", "contours_roi", "roi_crop", "hsv_hue", "hsv_saturation", "hsv_value", "lbp", "hog"]
    for step in data["steps"]:
        assert step["title"] and step["caption"]
        assert step["image"].startswith("data:image/")

    features = data["features"]
    for name in ["aspect_ratio", "extent", "solidity", "mean_brightness", "low_light", "feature_vector_length"]:
        assert name in features
    assert len(features["hu_moments"]) == 7
    assert len(features["hsv_histogram"]) > 0

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


def _stub_mesh(self, image, options):
    """Stands in for TripoSR in tests only, so the queue, validation and export run in seconds."""
    import trimesh

    return trimesh.creation.icosphere(subdivisions=2)


def test_image_url_is_not_fetched_by_the_server():
    """The server must not fetch web addresses on a client's behalf."""
    response = client.post("/api/reconstruct", data={"image_url": "http://127.0.0.1:1/internal"})
    assert response.status_code == 400
    assert "does not fetch" in response.json()["detail"]


def test_reconstruction_failure_is_reported_not_replaced(monkeypatch):
    """If TripoSR fails the job fails with its message. No invented car is returned."""

    def boom(self, image, options):
        raise RuntimeError("3D reconstruction failed: test failure")

    monkeypatch.setattr(ReconstructionManager, "_run_reconstruction", boom)
    files = {"file": ("car.jpg", create_sample_car_image(250, 180), "image/jpeg")}
    job_id = client.post("/api/reconstruct", files=files).json()["job_id"]
    info = {}
    for _ in range(20):
        info = client.get(f"/api/jobs/{job_id}").json()
        if info["status"] in ("completed", "failed"):
            break
        time.sleep(0.5)
    assert info["status"] == "failed"
    assert "test failure" in info["error"]
    assert info["assets"] is None


def test_reconstruction_workflow(monkeypatch):
    """Verify complete 3D reconstruction lifecycle: submit -> poll -> download assets."""
    monkeypatch.setattr(ReconstructionManager, "_run_reconstruction", _stub_mesh)
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


def test_a_server_without_triposr_says_so_and_refuses_to_queue_a_mesh(monkeypatch):
    """The lean deployment has no TripoSR: health reports it, and /api/reconstruct refuses at once instead of creating a job that can only fail."""
    monkeypatch.setattr("backend.api.routes_reconstruct.mesh_builder_available", lambda: False)
    monkeypatch.setattr("backend.main.mesh_builder_available", lambda: False)
    assert client.get("/api/health").json()["mesh_builder"] is False
    response = client.post("/api/reconstruct", files={"file": ("car.jpg", create_sample_car_image(), "image/jpeg")})
    assert response.status_code == 503
    assert "TripoSR" in response.json()["detail"] and "showcase" in response.json()["detail"]
    monkeypatch.setattr("backend.main.mesh_builder_available", lambda: True)
    assert client.get("/api/health").json()["mesh_builder"] is True
