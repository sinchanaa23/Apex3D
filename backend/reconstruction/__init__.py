"""3D reconstruction interface (owner: Person A).

Usage (Person C, from a background job)::

    from backend.reconstruction import reconstruct, ReconstructionError

    result = reconstruct("upload.jpg")          # or a PIL.Image
    result.mesh.export("mesh.glb")              # trimesh.Trimesh, Y-up, unitless
    job_meta = result.summary()                 # JSON-safe timings/settings/notes
"""

from .triposr_backend import (
    ReconstructionError,
    ReconstructionResult,
    ReconstructionSettings,
    preprocess,
    reconstruct,
    release_gpu,
)

__all__ = [
    "ReconstructionError",
    "ReconstructionResult",
    "ReconstructionSettings",
    "preprocess",
    "reconstruct",
    "release_gpu",
]
