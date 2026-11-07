"""Export the vehicle detector to ONNX, so the server can run it without PyTorch.

    python scripts/export_detector_onnx.py                    writes checkpoints/detector_mobilenet.onnx (about 78 MB) and checks it
    python scripts/check_detector_onnx.py 200                 then compare it with PyTorch on 200 Stanford Cars test photos

This is the same network the app has always used (torchvision's Faster R-CNN MobileNetV3 FPN with its COCO weights), written to a file.
PyTorch and torchvision are needed only to run this script, once. A small server then needs only ONNX Runtime, which saves a few hundred
megabytes of memory. The server uses the file automatically when it exists (backend/services/detector_service.py).

Two things this script does on purpose, because the first version of it silently produced a broken file in a clean environment:
  * It traces the network with a REAL photo (from results/figures/classifier_failures/), not random noise. Tracing a detector with noise
    can freeze data-dependent shapes into the graph, and the file then fails on real photos.
  * It checks the finished file against PyTorch on every sample photo, at full size and at 60% size (the app sends photos of every size),
    and exits with an error if they disagree. The Docker build runs this script, so a bad file stops the build instead of reaching the server.
The Docker image runs this during its build (see Dockerfile), so the file is not kept in git.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "checkpoints" / "detector_mobilenet.onnx"
SAMPLES = ROOT / "results" / "figures" / "classifier_failures"
VEHICLES = (3, 4, 6, 8)  # COCO car, motorcycle, bus, truck, the labels the app keeps


def sample_photos() -> list[Path]:
    photos = sorted(SAMPLES.glob("*.jpg"))
    if not photos:
        raise SystemExit(f"No sample photos in {SAMPLES}. The export needs real photos to trace and check the detector.")
    return photos


def to_array(photo: Path, scale: float = 1.0) -> np.ndarray:
    from PIL import Image

    image = Image.open(photo).convert("RGB")
    if scale != 1.0:
        image = image.resize((max(64, int(image.width * scale)), max(64, int(image.height * scale))))
    return np.ascontiguousarray(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)


def main_box(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray):
    """The app's rule: the largest box among those scoring at least 0.5."""
    keep = [b for b, s, l in zip(boxes, scores, labels) if s >= 0.5 and int(l) in VEHICLES]
    return max(keep, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])) if keep else None


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return float(inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9))


def export_and_check(path: Path) -> None:
    import onnxruntime as ort
    import torch
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights

    warnings.filterwarnings("ignore")
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT).eval()
    photos = sample_photos()

    # Trace with a real photo in which the network finds a vehicle.
    example = None
    for photo in photos:
        tensor = torch.from_numpy(to_array(photo))
        with torch.no_grad():
            found = model([tensor])[0]
        if (found["scores"] >= 0.5).any():
            example = tensor
            break
    if example is None:
        raise SystemExit("The detector found no vehicle in any sample photo, so it can't be traced reliably.")

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, ([example],), str(path), opset_version=17, dynamo=False,
        input_names=["image"], output_names=["boxes", "labels", "scores"],
        dynamic_axes={"image": {0: "c", 1: "h", 2: "w"}, "boxes": {0: "n"}, "labels": {0: "n"}, "scores": {0: "n"}},
    )
    print(f"Wrote {path} ({path.stat().st_size / 1e6:.0f} MB)")

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    worst, checked = 1.0, 0
    for photo in photos:
        for scale in (1.0, 0.6):
            pixels = to_array(photo, scale)
            with torch.no_grad():
                ref = model([torch.from_numpy(pixels.copy())])[0]
            boxes, labels, scores = session.run(None, {"image": pixels})  # raises if the graph is broken
            a = main_box(ref["boxes"].numpy(), ref["scores"].numpy(), ref["labels"].numpy())
            b = main_box(boxes, scores, labels)
            if (a is None) != (b is None):
                raise SystemExit(f"Check failed on {photo.name} at scale {scale}: only one of PyTorch and ONNX found a vehicle.")
            if a is not None:
                worst = min(worst, iou(a, b))
            checked += 1
    print(f"Checked against PyTorch on {checked} photo sizes: the chosen boxes overlap by at least {worst:.3f} (IoU).")
    if worst < 0.95:
        raise SystemExit("The ONNX detector does not match PyTorch closely enough. Do not use it.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    export_and_check(args.out)


if __name__ == "__main__":
    sys.exit(main())
