"""Exports the classifier in the form the backend's auto-loader picks up.

    .venv\\Scripts\\python scripts\\export_app_classifier.py

Writes checkpoints/model_quantized.onnx: the deployed variant from notebook 01b
(static INT8, MatMul layers only, calibrated on 256 training-split crops), except the
logits are already divided by the calibration temperature T inside the graph. So a
plain ``softmax(output)`` gives calibrated probabilities, and the thresholds from
notebook 01 apply directly.

It then checks, on the full validation and test sets, that:
  * Top-1 matches the deployed variant in results/optimization_metrics.json
    (within 0.1 points), and
  * softmax(output) equals softmax(logits_fp32 / T) closely for the FP32 export.
Results go to results/app_classifier_export.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))

import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static  # noqa: E402
from onnxruntime.quantization.shape_inference import quant_pre_process  # noqa: E402

import cls_lib as L  # noqa: E402

CK = ROOT / "checkpoints"
RESULTS = ROOT / "results"
CACHE = ROOT / "data" / "crops"
THREADS = json.loads((RESULTS / "optimization_metrics.json").read_text())["threads"]


class Scaled(torch.nn.Module):
    def __init__(self, model, temperature: float):
        super().__init__()
        self.model = model
        self.register_buffer("inv_t", torch.tensor(1.0 / temperature))

    def forward(self, x):
        return self.model(x) * self.inv_t


def crop(name: str, split: str) -> Path:
    return CACHE / (f"cars_{split}_" + Path(name).stem + ".jpg")


def main() -> None:
    clf = L.load_classifier(CK / "apex3d_convnext_tiny.pt", "cpu")
    T = clf.temperature
    split = json.loads((RESULTS / "split.json").read_text())
    data_root = L.find_stanford_root(Path(os.environ.get("APEX3D_DATA", ROOT / "data" / "raw")))
    train_lab = {Path(r.path).name: r.label for r in L.load_split(data_root, "train")}
    test = L.load_split(data_root, "test")

    fp32 = CK / "model_calibrated_fp32.onnx"
    torch.onnx.export(Scaled(clf.model, T).eval(), torch.randn(1, 3, clf.img_size, clf.img_size), fp32,
                      input_names=["input"], output_names=["probs_logits"],
                      dynamic_axes={"input": {0: "batch"}, "probs_logits": {0: "batch"}}, opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(fp32))

    calib_files = [crop(f, "train") for f in np.random.default_rng(0).permutation(split["train"])]
    calib_files = [p for p in calib_files if p.exists()][:256]   # same 256 crops as notebook 01b

    class Calib(CalibrationDataReader):
        def __init__(self):
            self.it = iter(calib_files)

        def get_next(self):
            p = next(self.it, None)
            return None if p is None else {"input": clf.transform(Image.open(p).convert("RGB")).unsqueeze(0).numpy()}

    pre = CK / "model_calibrated_pre.onnx"
    out = CK / "model_quantized.onnx"
    quant_pre_process(str(fp32), str(pre))
    quantize_static(str(pre), str(out), Calib(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])
    pre.unlink()

    def session(p):
        so = ort.SessionOptions()
        so.intra_op_num_threads = THREADS
        return ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])

    def logits(sess, paths):
        out_ = []
        for i in range(0, len(paths), 64):
            xb = np.stack([clf.transform(Image.open(p).convert("RGB")).numpy() for p in paths[i:i + 64]])
            out_.append(sess.run(None, {"input": xb})[0])
        return np.concatenate(out_)

    va_paths = [crop(f, "train") for f in split["val"]]   # val images come from the train folder
    va_y = np.array([train_lab[f] for f in split["val"]])
    te_paths = [crop(Path(r.path).name, "test") for r in test]
    te_y = np.array([r.label for r in test])

    # 1) calibration baked in correctly: compare with the uncalibrated FP32 export from 01b
    ref = session(CK / "apex3d_convnext_tiny_fp32.onnx")
    cal = session(fp32)
    probe = va_paths[:128]
    p_ref = torch.from_numpy(logits(ref, probe) / T).softmax(-1).numpy()
    p_cal = torch.from_numpy(logits(cal, probe)).softmax(-1).numpy()
    max_prob_diff = float(np.abs(p_ref - p_cal).max())

    # 2) accuracy and latency of the file the app will load
    q = session(out)
    t0 = time.time()
    va_top1 = float((logits(q, va_paths).argmax(1) == va_y).mean())
    te_log = logits(q, te_paths)
    te_top1 = float((te_log.argmax(1) == te_y).mean())
    te_top5 = float((np.argsort(-te_log, 1)[:, :5] == te_y[:, None]).any(1).mean())
    x1 = np.random.randn(1, 3, clf.img_size, clf.img_size).astype(np.float32)
    for _ in range(10):
        q.run(None, {"input": x1})
    t1 = time.perf_counter()
    for _ in range(50):
        q.run(None, {"input": x1})
    lat = (time.perf_counter() - t1) / 50 * 1000

    deployed = json.loads((RESULTS / "optimization_metrics.json").read_text())["variants"]["onnx_int8_static_matmul"]
    rec = {
        "generated_by": "scripts/export_app_classifier.py",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "file": "checkpoints/model_quantized.onnx",
        "size_mb": round(out.stat().st_size / 1e6, 2),
        "temperature_baked_in": T,
        "output": "temperature-scaled logits; softmax gives calibrated probabilities",
        "max_prob_diff_vs_fp32_softmax_logits_over_T_128_val": max_prob_diff,
        "val_top1": va_top1, "test_top1": te_top1, "test_top5": te_top5,
        "deployed_variant_test_top1_from_01b": deployed["test_top1"],
        "cpu_latency_ms_b1": round(lat, 2), "threads": THREADS,
        "app_thresholds": {"APEX3D_REJECTED_THRESHOLD": clf.thresholds["reject_below_p1"],
                           "APEX3D_MARGIN_THRESHOLD": clf.thresholds["confident_margin"]},
        "eval_seconds": round(time.time() - t0, 1),
    }
    (RESULTS / "app_classifier_export.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))
    assert max_prob_diff < 1e-4, "temperature not baked in correctly"
    assert abs(te_top1 - deployed["test_top1"]) <= 0.001, "differs from the 01b deployed variant"
    fp32.unlink()


if __name__ == "__main__":
    main()
