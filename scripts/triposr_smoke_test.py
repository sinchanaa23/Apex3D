"""Hour-1 checkpoint: generate a real mesh from a car photo and record the time.

    .venv\\Scripts\\python scripts\\triposr_smoke_test.py path\\to\\car.jpg [more.jpg ...]

Writes meshes to outputs/triposr_test/<name>/ (gitignored; share samples through the
team drive) and appends a measured record to results/triposr_test.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from backend.reconstruction import ReconstructionSettings, reconstruct_detailed  # noqa: E402


def preview(mesh, path: Path) -> None:
    """Orthographic point previews (front / side / top) to check orientation (Y-up)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v = np.asarray(mesh.vertices)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(v), size=min(len(v), 30000), replace=False)
    c = mesh.visual.vertex_colors[idx, :3] / 255.0 if mesh.visual.kind == "vertex" else "k"
    views = [("front (X right, Y up)", 0, 1), ("side (Z right, Y up)", 2, 1), ("top (X right, Z down)", 0, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, a, b) in zip(axes, views):
        ax.scatter(v[idx, a], v[idx, b], s=0.5, c=c)
        ax.set_aspect("equal")
        ax.set_title(title)
        if title.startswith("top"):
            ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--chunk-size", type=int, default=ReconstructionSettings.chunk_size)
    ap.add_argument("--mc-resolution", type=int, default=ReconstructionSettings.mc_resolution)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    out_root = ROOT / "outputs" / "triposr_test"
    log_path = ROOT / "results" / "triposr_test.json"
    log_path.parent.mkdir(exist_ok=True)
    log = json.loads(log_path.read_text()) if log_path.exists() else []

    settings = ReconstructionSettings(device=args.device, chunk_size=args.chunk_size, mc_resolution=args.mc_resolution)
    for img in args.images:
        name = Path(img).stem
        out = out_root / name
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        res = reconstruct_detailed(img, settings)
        wall = time.perf_counter() - t0
        res.processed_image.save(out / "input_processed.png")
        sizes = {}
        for ext in ("glb", "obj", "stl"):
            p = out / f"mesh.{ext}"
            res.mesh.export(p)
            sizes[ext] = round(p.stat().st_size / 1e6, 2)
        preview(res.mesh, out / "preview.png")
        rec = {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "image": Path(img).name,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2) if torch.cuda.is_available() else None,
            "cpu": platform.processor(),
            "torch": torch.__version__,
            "wall_seconds_incl_first_load": round(wall, 2),
            **res.summary(),
            "watertight": bool(res.mesh.is_watertight),
            "file_mb": sizes,
        }
        log.append(rec)
        print(json.dumps(rec, indent=2), flush=True)
    log_path.write_text(json.dumps(log, indent=2))
    print("appended to", log_path)


if __name__ == "__main__":
    main()
