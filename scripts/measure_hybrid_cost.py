"""Measure what the hybrid model costs: the size of its head and the time it adds to the deployed deep model.

    python scripts/measure_hybrid_cost.py

The hybrid is the deployed ONNX model's 768-value embedding plus the classical geometry features, into a linear head
(the fusion `results/hybrid.json` chose on validation). So it costs the deep model plus the geometry features plus the head.
This refits that head on train plus validation, as hybrid_experiment does for its test result, saves it to measure its size,
and times the two extra steps on 200 random test crops (batch 1, after a warm-up). Writes results/hybrid_cost.json.

The deep model's own size and latency are Person A's numbers in results/optimization_metrics.json, taken on their laptop.
The extra steps are timed here, so the total mixes two machines and the file says so.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import cv2
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from classical_cv.pipeline import GEOMETRY_NAMES, feature_blocks, geometric_features, normalize  # noqa: E402
from data.stanford_cars import box_with_margin, default_root, load_manifest  # noqa: E402
from evaluation import classifier_experiments as ex  # noqa: E402

RESULTS = ROOT / "results"
SAMPLE, SEED = 200, 42


def main() -> None:
    import platform

    hybrid = json.loads((RESULTS / "hybrid.json").read_text(encoding="utf-8"))
    optimisation = json.loads((RESULTS / "optimization_metrics.json").read_text(encoding="utf-8"))
    deployed = optimisation["variants"][optimisation["deploy_variant"]]
    name = hybrid["chosen_fusion"]
    if name != "deep_plus_geometry":
        raise SystemExit(f"This script times the geometry fusion, but the chosen fusion is {name}.")

    root = default_root()
    manifest = load_manifest(root)
    cache = root.parent / "features"
    embeddings = np.load(cache / "onnx_embeddings.npy")
    classical = np.load(cache / "classical_crop8.npy")
    start = sum(length for block, length in feature_blocks().items() if block != "geometry")
    geometry = classical[:, start:]
    assert geometry.shape[1] == len(GEOMETRY_NAMES)

    labels = manifest["label"].to_numpy()
    masks = {split: (manifest["split"] == split).to_numpy() for split in ("train", "val", "test")}
    features = np.hstack((embeddings, geometry))
    development = masks["train"] | masks["val"]
    head = ex.linear_head(hybrid["validation"][name]["C"]).fit(features[development], labels[development])
    accuracy = float((head.decision_function(features[masks["test"]]).argmax(1) == labels[masks["test"]]).mean())
    print(f"head refitted: test top-1 {accuracy:.4f} (hybrid.json: {hybrid['test'][name]['top1']:.4f})", flush=True)
    if abs(accuracy - hybrid["test"][name]["top1"]) > 0.002:
        raise SystemExit("The refitted head does not reproduce hybrid.json, so its cost would describe another model.")

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "hybrid_head.joblib"
        joblib.dump(head, path)
        head_mb = path.stat().st_size / 1e6

    rows = manifest[masks["test"]].sample(SAMPLE, random_state=SEED)
    crops = []
    for row in rows.itertuples():
        image = cv2.imread(str(root / row.path), cv2.IMREAD_COLOR)
        x1, y1, x2, y2 = box_with_margin(row)
        crops.append(image[y1:y2, x1:x2])

    cv2.setNumThreads(1)
    geometry_ms, head_ms = [], []
    for round_ in range(2):  # the first round warms caches and is not counted
        for crop in crops:
            t0 = time.perf_counter()
            vector = geometric_features(normalize(crop))
            t1 = time.perf_counter()
            head.decision_function(np.hstack((np.zeros(embeddings.shape[1], dtype=np.float32), vector))[None, :])
            t2 = time.perf_counter()
            if round_:
                geometry_ms.append((t1 - t0) * 1000)
                head_ms.append((t2 - t1) * 1000)

    extra = float(np.median(geometry_ms) + np.median(head_ms))
    result = {
        "experiment": "hybrid_cost",
        "fusion": name,
        "test_images_timed": SAMPLE,
        "deep_model": {"size_mb": deployed["size_mb"], "cpu_latency_ms_b1": deployed["cpu_latency_ms_b1"], "from": "results/optimization_metrics.json (Person A's laptop)"},
        "geometry_features_ms_median": float(np.median(geometry_ms)),
        "linear_head_ms_median": float(np.median(head_ms)),
        "linear_head_size_mb": head_mb,
        "total_size_mb": deployed["size_mb"] + head_mb,
        "total_inference_ms": deployed["cpu_latency_ms_b1"] + extra,
        "extra_steps_timed_on": f"{platform.processor() or platform.machine()}, batch 1, one thread",
        "note": "The deep model's time is from another laptop; only the geometry features and the head were timed here. The extra steps are small next to the deep model, so the total is dominated by the published figure.",
    }
    (RESULTS / "hybrid_cost.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
