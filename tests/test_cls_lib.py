"""Unit tests for the classifier helpers and the reconstruction error path (no GPU, no weights)."""

import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))

import cls_lib as L  # noqa: E402


@pytest.mark.parametrize(
    "name, make, model, year",
    [
        ("AM General Hummer SUV 2000", "AM General", "Hummer SUV", 2000),
        ("Aston Martin V8 Vantage Coupe 2012", "Aston Martin", "V8 Vantage Coupe", 2012),
        ("Land Rover Range Rover SUV 2012", "Land Rover", "Range Rover SUV", 2012),
        ("Rolls-Royce Phantom Sedan 2012", "Rolls-Royce", "Phantom Sedan", 2012),
        ("smart fortwo Convertible 2012", "smart", "fortwo Convertible", 2012),
        ("Audi S4 Sedan 2007", "Audi", "S4 Sedan", 2007),
    ],
)
def test_parse_class_name(name, make, model, year):
    p = L.parse_class_name(name)
    assert (p["make"], p["model"], p["year"]) == (make, model, year)


def test_confidence_state():
    th = {"confident_p1": 0.7, "confident_margin": 0.3, "reject_below_p1": 0.25}
    assert L.confidence_state([0.9, 0.05], th) == "confident"
    assert L.confidence_state([0.75, 0.6], th) == "ambiguous"   # small margin
    assert L.confidence_state([0.5, 0.1], th) == "ambiguous"    # below confident p1
    assert L.confidence_state([0.2, 0.1], th) == "rejected"


def test_crop_box_adds_margin_and_clips():
    img = Image.new("RGB", (100, 50))
    assert L.crop_box(img, (10, 10, 59, 29), margin=0.1).size == (60, 24)   # 50x20 box + 10%
    assert L.crop_box(img, (0, 0, 99, 49), margin=0.1).size == (100, 50)    # clipped to image


def test_reconstruct_rejects_unreadable_file(tmp_path):
    from backend.reconstruction import ReconstructionError, reconstruct
    from backend.reconstruction.triposr_backend import TRIPOSR_DIR

    if not (TRIPOSR_DIR / "tsr").exists():
        pytest.skip("TripoSR not set up")
    bad = tmp_path / "not_an_image.jpg"
    bad.write_bytes(b"this is not a jpeg")
    with pytest.raises(ReconstructionError, match="Could not read the input image"):
        reconstruct(bad)


def test_load_image_modes():
    import numpy as np
    from backend.reconstruction.triposr_backend import _load_image

    # 16-bit greyscale: scaled to 8-bit, not clipped to white
    img16 = Image.new("I;16", (8, 8), 32768)
    out = _load_image(img16)
    assert out.mode == "RGB" and 120 <= np.asarray(out)[0, 0, 0] <= 135

    # palette image with transparency keeps its alpha
    p = Image.new("P", (4, 4), 0)
    p.putpalette([0, 0, 0, 255, 0, 0] + [0] * 762)
    p.info["transparency"] = 0
    assert _load_image(p).mode == "RGBA"

    # EXIF orientation 6 (rotate 90 degrees) is applied
    exif = Image.Exif()
    exif[0x0112] = 6
    src = Image.new("RGB", (40, 20))
    import io

    buf = io.BytesIO()
    src.save(buf, format="JPEG", exif=exif.tobytes())
    buf.seek(0)
    assert _load_image(Image.open(buf)).size == (20, 40)
