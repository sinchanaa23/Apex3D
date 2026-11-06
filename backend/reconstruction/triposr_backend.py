"""Single-image 3D reconstruction with TripoSR (MIT licence, code and weights).

Owner: Person A. Person C calls :func:`reconstruct` from the background job runner,
then exports (GLB/OBJ/STL) and validates the returned ``trimesh.Trimesh``.

Honesty notes that the app must surface:
* TripoSR takes ONE image. Unseen sides are inferred, not observed.
* The output has no real-world scale (normalised to TripoSR's unit cube, radius ~0.87).

GPU notes (team laptop: RTX 3050 Laptop, 4 GB VRAM, below TripoSR's ~6 GB default):
* ``chunk_size`` is lowered from TripoSR's 8192 default to cut peak VRAM.
* If CUDA runs out of memory the job is retried once on the CPU (slow but works).
* :func:`release_gpu` frees the model so classifier training can use the GPU.
"""

from __future__ import annotations

import gc
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Union

import numpy as np
from PIL import Image, ImageOps

from . import mcubes_shim

REPO_ROOT = Path(__file__).resolve().parents[2]
TRIPOSR_DIR = Path(os.environ.get("APEX3D_TRIPOSR_DIR", REPO_ROOT / "third_party" / "TripoSR"))
# Hugging Face repo id, or a local folder holding config.yaml + model.ckpt.
TRIPOSR_WEIGHTS = os.environ.get("APEX3D_TRIPOSR_WEIGHTS", "stabilityai/TripoSR")
REMBG_MODEL = "u2net"

ImageInput = Union[str, Path, Image.Image]


@dataclass
class ReconstructionSettings:
    device: str = "auto"          # "auto" | "cuda" | "cpu"
    chunk_size: int = 2048        # TripoSR default 8192; lower = less VRAM, slower
    mc_resolution: int = 256      # marching-cubes grid; 256 is TripoSR's default
    mc_threshold: float = 25.0    # TripoSR default density threshold
    foreground_ratio: float = 0.85
    remove_background: bool = True
    vertex_colors: bool = True
    cpu_fallback_on_oom: bool = True


@dataclass
class ReconstructionResult:
    mesh: Any                                   # trimesh.Trimesh
    processed_image: Image.Image                # exact image fed to TripoSR
    device: str                                 # device actually used
    timings_s: dict[str, float] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    isosurface_backend: str = ""
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """JSON-safe metadata for the job record (mesh excluded)."""
        return {
            "device": self.device,
            "timings_s": self.timings_s,
            "settings": self.settings,
            "isosurface_backend": self.isosurface_backend,
            "vertices": int(len(self.mesh.vertices)),
            "faces": int(len(self.mesh.faces)),
            "has_vertex_colors": bool(self.mesh.visual.kind == "vertex"),
            "notes": self.notes,
        }


class ReconstructionError(RuntimeError):
    """Raised with a message that is safe to show to the user."""


_lock = threading.Lock()          # one reconstruction at a time on the single GPU
_rembg_lock = threading.Lock()
_model = None
_model_device: str | None = None
_rembg_session = None
_iso_backend = ""


def _import_tsr():
    global _iso_backend
    if not (TRIPOSR_DIR / "tsr" / "system.py").exists():
        raise ReconstructionError(
            f"TripoSR source not found at {TRIPOSR_DIR}. Run scripts/setup_triposr.ps1."
        )
    _iso_backend = mcubes_shim.install()
    if str(TRIPOSR_DIR) not in sys.path:
        sys.path.insert(0, str(TRIPOSR_DIR))
    from tsr.system import TSR  # noqa: WPS433
    from tsr import utils as tsr_utils

    return TSR, tsr_utils


def _resolve_device(requested: str) -> str:
    import torch

    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda:0"
    if requested == "cuda":
        raise ReconstructionError("CUDA was requested but no CUDA GPU is available.")
    return "cpu"


def _get_model(device: str, chunk_size: int):
    global _model, _model_device
    TSR, _ = _import_tsr()
    if _model is None:
        try:
            _model = TSR.from_pretrained(TRIPOSR_WEIGHTS, config_name="config.yaml", weight_name="model.ckpt")
        except Exception as exc:  # noqa: BLE001 (download / file errors)
            raise ReconstructionError(f"Could not load the TripoSR weights: {exc}") from exc
        _model.eval()
        _model_device = "cpu"
    if _model_device != device:
        _model_device = None          # unknown until the move completes (it may OOM midway)
        _model.to(device)
        _model_device = device
    _model.renderer.set_chunk_size(chunk_size)
    return _model


def _load_image(image: ImageInput) -> Image.Image:
    try:
        img = image if isinstance(image, Image.Image) else Image.open(image)
        img.load()
        img = ImageOps.exif_transpose(img)          # phone photos: apply EXIF rotation
    except Exception as exc:  # noqa: BLE001
        raise ReconstructionError(f"Could not read the input image: {exc}") from exc
    if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        arr = np.asarray(img, dtype=np.float64)
        hi = float(arr.max()) if arr.size else 0.0
        scale = 257.0 if (img.mode.startswith("I;16") and hi > 255) else max(hi / 255.0, 1.0)
        img = Image.fromarray(np.clip(arr / scale, 0, 255).astype(np.uint8)).convert("RGB")
    elif img.mode == "P" and "transparency" in img.info:
        img = img.convert("RGBA")
    elif img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
    return img


def _composite_on_grey(rgba: Image.Image) -> Image.Image:
    arr = np.asarray(rgba).astype(np.float32) / 255.0
    arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5   # TripoSR's grey background
    return Image.fromarray((arr * 255.0).astype(np.uint8))


def preprocess(image: ImageInput, settings: ReconstructionSettings | None = None) -> Image.Image:
    """Background removal and recentring, exactly as TripoSR's run.py does it."""
    global _rembg_session
    settings = settings or ReconstructionSettings()
    _, tsr_utils = _import_tsr()
    img = _load_image(image)
    if not settings.remove_background:
        if img.mode == "RGBA" and np.asarray(img)[..., 3].min() < 255:
            # caller supplied a cut-out: recentre it and use TripoSR's grey background
            return _composite_on_grey(tsr_utils.resize_foreground(img, settings.foreground_ratio))
        return img.convert("RGB")
    with _rembg_lock:
        if _rembg_session is None:
            try:
                import rembg

                # Pin u2net (Apache-2.0), the model TripoSR was released with. Recent rembg
                # releases default to BRIA RMBG-2.0, which is CC BY-NC (non-commercial).
                _rembg_session = rembg.new_session(REMBG_MODEL)
            except Exception as exc:  # noqa: BLE001
                raise ReconstructionError(f"Background removal is unavailable: {exc}") from exc
        rgba = tsr_utils.remove_background(img, _rembg_session)
    if np.asarray(rgba)[..., 3].max() == 0:
        raise ReconstructionError("Background removal found no foreground object in the image.")
    return _composite_on_grey(tsr_utils.resize_foreground(rgba, settings.foreground_ratio))


def _is_oom(exc: BaseException) -> bool:
    import torch

    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc)
    return isinstance(exc, RuntimeError) and ("out of memory" in msg or "ALLOC_FAILED" in msg)


def _run(processed: Image.Image, settings: ReconstructionSettings, device: str):
    import torch

    timings: dict[str, float] = {}
    t = time.perf_counter()
    model = _get_model(device, settings.chunk_size)
    timings["model_load"] = time.perf_counter() - t

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    with torch.no_grad():
        scene_codes = model([processed], device=device)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    timings["inference"] = time.perf_counter() - t

    t = time.perf_counter()
    meshes = model.extract_mesh(
        scene_codes, settings.vertex_colors, resolution=settings.mc_resolution, threshold=settings.mc_threshold
    )
    timings["mesh_extraction"] = time.perf_counter() - t
    peak = torch.cuda.max_memory_allocated() / 2**30 if device.startswith("cuda") else None
    del scene_codes
    return meshes[0], timings, peak


def _run_with_fallback(processed: Image.Image, settings: ReconstructionSettings, device: str, notes: list[str]):
    try:
        return (*_run(processed, settings, device), device)
    except ReconstructionError:
        raise
    except Exception as exc:  # noqa: BLE001
        if not _is_oom(exc):
            raise ReconstructionError(f"Reconstruction failed: {type(exc).__name__}: {exc}") from exc
        if not (settings.cpu_fallback_on_oom and device != "cpu"):
            raise ReconstructionError("The GPU ran out of memory during reconstruction.") from None
    # Only an OOM with fallback enabled reaches here. The retry runs outside the except
    # block so the failed attempt's GPU tensors (held by the traceback) can be freed.
    _release_gpu_unlocked()
    notes.append("GPU ran out of memory; this mesh was generated on the CPU.")
    try:
        return (*_run(processed, settings, "cpu"), "cpu")
    except ReconstructionError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ReconstructionError(f"Reconstruction failed on the CPU too: {type(exc).__name__}: {exc}") from exc


def reconstruct(image: ImageInput, settings: ReconstructionSettings | None = None) -> ReconstructionResult:
    """Generate a mesh from ONE image. Blocking; call it from a background worker.

    Returns a :class:`ReconstructionResult` whose ``mesh`` is a ``trimesh.Trimesh``
    (Y-up, unitless, roughly centred at the origin). Raises
    :class:`ReconstructionError` with a user-safe message on any failure.
    """
    settings = settings or ReconstructionSettings()
    with _lock:
        total = time.perf_counter()
        notes: list[str] = ["Single-view reconstruction: unseen sides are inferred.", "Scale is unknown."]
        t = time.perf_counter()
        processed = preprocess(image, settings)
        pre_s = time.perf_counter() - t

        mesh, timings, peak, device = _run_with_fallback(processed, settings, _resolve_device(settings.device), notes)
        if len(mesh.faces) == 0:
            raise ReconstructionError("TripoSR produced an empty mesh for this image.")

        # TripoSR's frame is Z-up; rotate to Y-up for glTF/Three.js (same transform as
        # TripoSR's to_gradio_3d_orientation).
        import trimesh

        mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))

        timings = {"preprocess": pre_s, **timings, "total": time.perf_counter() - total}
        timings = {k: round(v, 3) for k, v in timings.items()}
        used = asdict(settings)
        if peak is not None:
            used["peak_vram_gib"] = round(peak, 3)
        return ReconstructionResult(
            mesh=mesh,
            processed_image=processed,
            device=device,
            timings_s=timings,
            settings=used,
            isosurface_backend=_iso_backend,
            notes=notes,
        )


def release_gpu() -> None:
    """Move the model off the GPU and free cached VRAM (e.g. before training).

    Waits for any running reconstruction to finish first.
    """
    with _lock:
        _release_gpu_unlocked()


def _release_gpu_unlocked() -> None:
    global _model_device
    if _model is not None and _model_device != "cpu":
        _model.to("cpu")            # also repairs a model left half-moved by an OOM
        _model_device = "cpu"
    gc.collect()
    if "torch" in sys.modules:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
