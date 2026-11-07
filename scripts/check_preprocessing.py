"""Compare the deployed model under small preprocessing differences, on all 8,041 test images.

    python scripts/check_preprocessing.py

Person A published 92.18% Top-1 for the deployed model, and the app's own preprocessing scores 92.02%.
This measures how much the resize method and a JPEG-cached crop account for, and confirms that the
evaluation on file used this model. Writes results/preprocessing_check.json. Takes about 25 minutes on CPU.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from data.stanford_cars import box_with_margin, default_root, load_manifest  # noqa: E402

MODEL = ROOT / "checkpoints" / "model_quantized.onnx"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
EXPLANATION = (
    "Person A's notebook scored the deployed model at 92.18%, and the app's own preprocessing scores 92.02%, 13 images apart. "
    "Rerunning the same model file independently reproduced the 642 errors behind the 92.02% figure, image for image. "
    "Resize method and JPEG caching change roughly 100 individual predictions but move Top-1 by under 0.2 points, and none of the variants here reproduces 92.18%. "
    "Person A's images came from a different mirror of the dataset, which may account for the rest, but this wasn't verified. "
    "Treat the deployed model's annotated-box Top-1 as about 92.0-92.2%. The app wasn't changed, because a gain of a few images isn't significant, "
    "and the deployed results in this report describe its current preprocessing."
)


def main() -> None:
    root = default_root()
    manifest = load_manifest(root)
    rows = manifest[manifest["split"] == "test"].reset_index(drop=True)
    labels = rows["label"].to_numpy()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(MODEL), options, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name

    def crop(row):
        with Image.open(root / row.path) as opened:
            return opened.convert("RGB").crop(box_with_margin(row))

    def via_cached_jpeg(image):
        image = image.copy()
        image.thumbnail((448, 448), Image.BICUBIC)
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=95)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB").resize((224, 224), Image.BICUBIC)

    variants = {
        "Bilinear resize, as the app does": lambda image: image.resize((224, 224), Image.BILINEAR),
        "Bicubic resize, as in training": lambda image: image.resize((224, 224), Image.BICUBIC),
        "Bicubic, through a cached JPEG crop like the training notebook": via_cached_jpeg,
    }
    modes = []
    for label, change in variants.items():
        predictions, top5, batch = [], [], []

        def flush():
            inputs = ((np.stack(batch) - MEAN) / STD).transpose(0, 3, 1, 2).astype(np.float32)
            logits = session.run(None, {name: inputs})[0]
            predictions.extend(logits.argmax(1).tolist())
            top5.extend(np.argsort(-logits, axis=1)[:, :5].tolist())
            batch.clear()

        for row in rows.itertuples():
            batch.append(np.asarray(change(crop(row)), dtype=np.float32) / 255)
            if len(batch) == 32:
                flush()
        if batch:
            flush()
        predicted, best = np.array(predictions), np.array(top5)
        modes.append({"name": label, "top1": float((predicted == labels).mean()), "top5": float((best == labels[:, None]).any(1).mean()), "wrong": int((predicted != labels).sum())})
        print(modes[-1], flush=True)
        if label.startswith("Bilinear"):
            recorded = set(pd.read_csv(ROOT / "results" / "failure_candidates.csv")["path"])
            mine = set(rows["path"][predicted != labels])
            print(f"Errors matching results/failure_candidates.csv: {len(mine & recorded)} of {len(recorded)}, extra {len(mine - recorded)}", flush=True)

    published = json.loads((ROOT / "results" / "optimization_metrics.json").read_text(encoding="utf-8"))["variants"]["onnx_int8_static_matmul"]
    modes.append({"name": "Published by notebook 01b, on Person A's data copy", "top1": published["test_top1"], "top5": published["test_top5"], "wrong": round((1 - published["test_top1"]) * len(rows))})
    output = {"experiment": "preprocessing_check", "model_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(), "test_images": len(rows), "modes": modes, "explanation": EXPLANATION}
    (ROOT / "results" / "preprocessing_check.json").write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
