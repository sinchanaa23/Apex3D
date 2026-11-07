"""The deployed ONNX model, in the interface Person B's evaluation code expects.

Person B's `classifier_experiments` takes any object with `predict_proba(images)` and `embed(images)`.
`recognition.classifier.load_classifier` provides that from the PyTorch checkpoint. This does the
same from `checkpoints/model_quantized.onnx`, the file the app serves, so the results describe the
model that is actually deployed (static INT8 on the matrix multiplications).

Two details of the exported graph:
- Temperature scaling is baked in: the output is logits divided by T. `predict_proba` multiplies T
  back in, because the evaluation code applies calibration itself and expects uncalibrated softmax.
- `embed` returns the 768 values just before the classification layer. They are read by adding
  that tensor as a second output of the graph, which does not change the predictions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO / "checkpoints" / "model_quantized.onnx"
EMBEDDING_TENSOR = "/model/head/flatten/Flatten_output_0"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _temperature() -> float:
    metrics = json.loads((REPO / "results" / "classifier_metrics.json").read_text(encoding="utf-8"))
    return float(metrics["calibration"]["temperature"])


class OnnxClassifier:
    def __init__(self, path: str | Path = DEFAULT_MODEL, threads: int = 6):
        model = onnx.load(str(path))
        if EMBEDDING_TENSOR not in {output for node in model.graph.node for output in node.output}:
            raise ValueError(f"{EMBEDDING_TENSOR} is not in this model, so embeddings can't be read.")
        model.graph.output.append(onnx.helper.make_tensor_value_info(EMBEDDING_TENSOR, onnx.TensorProto.FLOAT, None))
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.temperature = _temperature()

    @staticmethod
    def _batch(images: list[Image.Image]) -> np.ndarray:
        # The same preprocessing as the app: resize to 224 x 224, scale to 0-1, normalise.
        arrays = [np.asarray(image.convert("RGB").resize((224, 224), Image.BILINEAR), dtype=np.float32) / 255 for image in images]
        return ((np.stack(arrays) - MEAN) / STD).transpose(0, 3, 1, 2).astype(np.float32, copy=False)

    def _run(self, images: list[Image.Image]) -> tuple[np.ndarray, np.ndarray]:
        logits, embedding = self.session.run(None, {self.input_name: self._batch(images)})
        return logits, embedding

    def _uncalibrated(self, logits: np.ndarray) -> np.ndarray:
        scaled = logits * self.temperature
        scaled -= scaled.max(axis=1, keepdims=True)
        values = np.exp(scaled)
        return values / values.sum(axis=1, keepdims=True)

    def predict_proba(self, images: list[Image.Image]) -> np.ndarray:
        """Uncalibrated softmax, like the PyTorch classifier. The model's output already has 1/T applied."""
        return self._uncalibrated(self._run(images)[0])

    def predict_and_embed(self, images: list[Image.Image]) -> tuple[np.ndarray, np.ndarray]:
        """Both from one pass through the model, which halves the time when both are needed."""
        logits, embedding = self._run(images)
        return self._uncalibrated(logits), embedding

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        return self._run(images)[1]
