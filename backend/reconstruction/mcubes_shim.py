"""Drop-in replacement for ``torchmcubes.marching_cubes`` built on scikit-image.

TripoSR imports ``torchmcubes``, which must be compiled from source. On Windows that
needs MSVC plus CMake and often fails, which would block Person C or the backup
laptop. scikit-image ships prebuilt wheels and implements the same algorithm.

Contract matched (from ``tsr/models/isosurface.py``)::

    v_pos, t_pos_idx = marching_cubes(level, 0.0)   # level: (R, R, R) float tensor
    v_pos = v_pos[..., [2, 1, 0]]                    # caller swaps axis order back
    v_pos = v_pos / (R - 1)

torchmcubes returns vertices in (x, y, z) = (dim2, dim1, dim0) index order, so the
caller's ``[2, 1, 0]`` swap restores (dim0, dim1, dim2). scikit-image returns
(dim0, dim1, dim2), so we reverse the columns here to match.

Triangle winding is chosen so that the final TripoSR mesh has outward-facing normals
(positive signed volume). This is checked by ``tests/test_mcubes_shim.py``.
"""

from __future__ import annotations

import sys
import types

import numpy as np
from skimage import measure


def marching_cubes(volume: "torch.Tensor", threshold: float) -> "tuple[torch.Tensor, torch.Tensor]":
    import torch  # imported here so that loading this module (the server does) doesn't load PyTorch

    vol = volume.detach().float().cpu().numpy()
    if not (vol.min() < threshold < vol.max()):
        # No surface crosses the threshold: return an empty mesh, as torchmcubes does.
        return torch.zeros((0, 3), dtype=torch.float32), torch.zeros((0, 3), dtype=torch.int64)
    verts, faces, _normals, _values = measure.marching_cubes(vol, level=threshold)
    verts = np.ascontiguousarray(verts[:, ::-1], dtype=np.float32)
    # TripoSR's helper negates -(density - t) again, so values here are positive inside
    # the object. With that sign and the axis reversal above (which mirrors the mesh),
    # skimage's winding faces inward, so we flip each triangle. Checked by the tests.
    faces = np.ascontiguousarray(faces[:, ::-1], dtype=np.int64)
    return torch.from_numpy(verts), torch.from_numpy(faces)


SHIM_NAME = "scikit-image shim"


def install() -> str:
    """Make ``import torchmcubes`` resolve. Returns which implementation is used."""
    try:
        import torchmcubes  # (a real compiled build takes priority)

        if getattr(torchmcubes, "__apex3d_shim__", False):
            return SHIM_NAME
        return "torchmcubes"
    except ImportError:
        pass
    module = types.ModuleType("torchmcubes")
    module.marching_cubes = marching_cubes  # type: ignore[attr-defined]
    module.__apex3d_shim__ = True  # type: ignore[attr-defined]
    sys.modules["torchmcubes"] = module
    return SHIM_NAME
