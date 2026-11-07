"""Export the vehicle detector to ONNX, so the server can run it without PyTorch.

    python scripts/export_detector_onnx.py                    writes checkpoints/detector_mobilenet.onnx (about 78 MB)
    python scripts/check_detector_onnx.py 200                 then compare it with PyTorch on real photos (the network is the same; numbers differ in the last digits)

This is the same network the app has always used (torchvision's Faster R-CNN MobileNetV3 FPN with its COCO weights), written to a file.
PyTorch and torchvision are needed only to run this script, once. A small server then needs only ONNX Runtime, which saves a few hundred
megabytes of memory. The server uses the file automatically when it exists (backend/services/detector_service.py).

The Docker image runs this during its build (see Dockerfile), so the file is not kept in git.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "checkpoints" / "detector_mobilenet.onnx"


def export(path: Path) -> None:
    import torch
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights

    warnings.filterwarnings("ignore")
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT).eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, ([torch.rand(3, 480, 640)],), str(path), opset_version=17, dynamo=False,
        input_names=["image"], output_names=["boxes", "labels", "scores"],
        dynamic_axes={"image": {0: "c", 1: "h", 2: "w"}, "boxes": {0: "n"}, "labels": {0: "n"}, "scores": {0: "n"}},
    )
    print(f"Wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1e6:.0f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    export(args.out)


if __name__ == "__main__":
    main()
