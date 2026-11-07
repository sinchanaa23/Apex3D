"""The server can run without PyTorch and without trimesh (the lean deployment), and its NumPy preprocessing equals torchvision's."""

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def test_the_server_starts_and_answers_with_pytorch_and_trimesh_unavailable():
    """Runs in a fresh process where importing torch, torchvision or trimesh raises ImportError, as on the lean image."""
    script = textwrap.dedent(
        """
        import io, os, sys
        os.environ["APEX3D_REQUIRE_LOGIN"] = "0"
        for name in ("torch", "torchvision", "trimesh"):
            sys.modules[name] = None
        from fastapi.testclient import TestClient
        from PIL import Image, ImageDraw
        import backend.main as main

        image = Image.new("RGB", (320, 220), (200, 205, 210))
        ImageDraw.Draw(image).rectangle([40, 90, 280, 180], fill=(180, 20, 30))
        buffer = io.BytesIO(); image.save(buffer, "JPEG")
        with TestClient(main.app) as client:
            health = client.get("/api/health").json()
            response = client.post("/api/recognize", files={"file": ("car.jpg", buffer.getvalue(), "image/jpeg")})
            assert response.status_code == 200, response.text
            body = response.json()
            assert len(body["predictions"]) == 5 and body["confidence_state"] in ("confident", "ambiguous", "rejected")
            classical = client.post("/api/classical", files={"file": ("car.jpg", buffer.getvalue(), "image/jpeg")})
            assert classical.status_code == 200, classical.text
        loaded = [name for name in ("torch", "torchvision", "trimesh") if sys.modules.get(name) is not None]
        assert not loaded, loaded
        print("OK", health["device"])
        """
    )
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0 and "OK cpu" in result.stdout, result.stderr[-1500:] + result.stdout[-500:]


def test_numpy_preprocessing_equals_torchvisions():
    transforms = pytest.importorskip("torchvision.transforms")
    torch = pytest.importorskip("torch")
    from backend.services.classifier_service import preprocess, softmax

    reference = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    rng = np.random.default_rng(3)
    for size in [(640, 360), (276, 182), (1024, 768), (224, 224), (90, 300)]:
        photo = Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))
        assert np.abs(preprocess(photo) - reference(photo).unsqueeze(0).numpy()).max() < 1e-5, size
    logits = rng.normal(size=(1, 196)).astype(np.float32) * 4
    assert np.abs(softmax(logits) - torch.softmax(torch.from_numpy(logits), dim=-1).numpy()).max() < 1e-6


def test_the_onnx_detector_agrees_with_pytorch_on_a_synthetic_scene():
    """Skipped unless both PyTorch and checkpoints/detector_mobilenet.onnx exist. scripts/check_detector_onnx.py does the real-photo comparison."""
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    if not (ROOT / "checkpoints" / "detector_mobilenet.onnx").exists():
        pytest.skip("run scripts/export_detector_onnx.py first")
    from backend.services.detector_service import VehicleDetector

    detector = VehicleDetector()
    assert detector.backend == "onnx" and detector.is_ready
    image = Image.new("RGB", (400, 300), (210, 215, 220))
    box, boxes, ms = detector.detect(image)
    assert isinstance(boxes, list) and ms > 0 and (box is None or box in boxes)
