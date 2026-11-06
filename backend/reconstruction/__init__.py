"""3D reconstruction interface (owner: Person A).

Usage (Person C, from a background job)::

    from backend.reconstruction import reconstruct, ReconstructionError

    mesh = reconstruct(pil_image)               # trimesh.Trimesh, Y-up, unitless
    job_meta = mesh.metadata["apex3d"]          # JSON-safe timings/settings/notes

    # or, with the processed input image as well:
    result = reconstruct_detailed(pil_image)    # .mesh, .processed_image, .summary()
"""

from .triposr_backend import (
    ReconstructionError,
    ReconstructionResult,
    ReconstructionSettings,
    preprocess,
    reconstruct,
    reconstruct_detailed,
    release_gpu,
)

__all__ = [
    "ReconstructionError",
    "ReconstructionResult",
    "ReconstructionSettings",
    "preprocess",
    "reconstruct",
    "reconstruct_detailed",
    "release_gpu",
]
