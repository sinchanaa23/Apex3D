"""Does swapping the PyTorch detector for the ONNX one change what the app does?

    python scripts/check_detector_onnx.py [N]       default 200 random Stanford Cars test photos

For each photo it runs both detectors, picks the main car the way the app does (largest box scoring at least 0.5), crops it with the
app's margin, and classifies the crop with the deployed model. It reports how close the two chosen boxes are, how close each is to the
dataset's true box, and whether the final car class agrees. Needs PyTorch, torchvision and the exported checkpoints/detector_mobilenet.onnx
(scripts/export_detector_onnx.py). Writes results/detector_onnx_parity.json.
"""

import sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
import onnxruntime as ort, torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from data.stanford_cars import load_manifest, default_root
from recognition.onnx_classifier import OnnxClassifier

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
root = default_root(); m = load_manifest(root); test = m[m["split"] == "test"].sample(N, random_state=7)
model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT).eval()
det = ort.InferenceSession(str(ROOT / "checkpoints" / "detector_mobilenet.onnx"), providers=["CPUExecutionProvider"])
clf = OnnxClassifier()

def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-9)

def primary(boxes, scores, labels):
    keep = [(b, s) for b, s, l in zip(boxes, scores, labels) if s >= 0.5 and int(l) in (3, 4, 6, 8)]
    return max(keep, key=lambda t: (t[0][2]-t[0][0])*(t[0][3]-t[0][1]))[0] if keep else None

def crop(img, b):
    if b is None: return img
    w, h = img.size; pw, ph = (b[2]-b[0])*0.08, (b[3]-b[1])*0.08
    x1, y1, x2, y2 = max(0, int(b[0]-pw)), max(0, int(b[1]-ph)), min(w, int(b[2]+pw)), min(h, int(b[3]+ph))
    return img.crop((x1, y1, x2, y2)) if (x2-x1) >= 20 and (y2-y1) >= 20 else img

pri_iou, gt_t, gt_o, same_cls, correct_t, correct_o, none_t, none_o = [], [], [], 0, 0, 0, 0, 0
for r in test.itertuples():
    img = Image.open(root / r.path).convert("RGB")
    px = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    with torch.no_grad(): o = model([torch.from_numpy(px.copy())])[0]
    bt = primary(o["boxes"].numpy(), o["scores"].numpy(), o["labels"].numpy())
    boxes, labels, scores = det.run(None, {"image": np.ascontiguousarray(px)})
    bo = primary(boxes, scores, labels)
    none_t += bt is None; none_o += bo is None
    gt = (r.x1, r.y1, r.x2, r.y2)
    gt_t.append(iou(bt, gt) if bt is not None else 0.0); gt_o.append(iou(bo, gt) if bo is not None else 0.0)
    if bt is not None and bo is not None: pri_iou.append(iou(bt, bo))
    pt = clf.predict_proba([crop(img, bt)]).argmax(1)[0]; po = clf.predict_proba([crop(img, bo)]).argmax(1)[0]
    same_cls += int(pt == po); correct_t += int(pt == r.label); correct_o += int(po == r.label)
print(f"photos: {N}")
print(f"chosen box, ONNX vs PyTorch: mean IoU {np.mean(pri_iou):.4f}, min {np.min(pri_iou):.4f}, share with IoU >= 0.99: {np.mean(np.array(pri_iou) >= 0.99):.1%}")
print(f"no box found: PyTorch {none_t}, ONNX {none_o}")
print(f"mean IoU with the true box: PyTorch {np.mean(gt_t):.4f}, ONNX {np.mean(gt_o):.4f}")
print(f"same final car class: {same_cls}/{N}; correct: PyTorch {correct_t}/{N}, ONNX {correct_o}/{N}")

import json
(ROOT / "results" / "detector_onnx_parity.json").write_text(json.dumps({
    "photos": N, "sample_seed": 7,
    "chosen_box_iou_onnx_vs_pytorch": {"mean": float(np.mean(pri_iou)), "min": float(np.min(pri_iou)), "share_at_least_0.99": float(np.mean(np.array(pri_iou) >= 0.99))},
    "no_box_found": {"pytorch": int(none_t), "onnx": int(none_o)},
    "mean_iou_with_true_box": {"pytorch": float(np.mean(gt_t)), "onnx": float(np.mean(gt_o))},
    "same_final_class": int(same_cls), "correct": {"pytorch": int(correct_t), "onnx": int(correct_o)},
}, indent=2), encoding="utf-8")
print("Wrote results/detector_onnx_parity.json")
