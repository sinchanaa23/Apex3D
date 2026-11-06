"""Checks the scikit-image marching-cubes shim against TripoSR's own helper.

Runs TripoSR's real ``MarchingCubeHelper`` on an analytic sphere and checks the
mesh position, size and normal orientation. Needs no model weights and no GPU.

    .venv\\Scripts\\python -m pytest tests/test_reconstruction_shim.py -q
"""

import sys

import numpy as np
import pytest
import torch
import trimesh

from backend.reconstruction import mcubes_shim
from backend.reconstruction.triposr_backend import TRIPOSR_DIR

pytestmark = pytest.mark.skipif(not (TRIPOSR_DIR / "tsr").exists(), reason="TripoSR not set up")


def _helper(resolution):
    mcubes_shim.install()
    sys.path.insert(0, str(TRIPOSR_DIR))
    from tsr.models.isosurface import MarchingCubeHelper

    return MarchingCubeHelper(resolution)


def test_sphere_matches_expected_geometry():
    res = 64
    helper = _helper(res)
    grid = helper.grid_vertices                        # (res^3, 3) in [0, 1]
    centre = torch.tensor([0.6, 0.5, 0.4])             # off-centre to catch axis swaps
    radius = 0.25
    dist = (grid - centre).norm(dim=-1)
    density = 50.0 * (radius - dist) + 25.0            # > 25 (threshold) inside the sphere
    threshold = 25.0
    v, f = helper(-(density - threshold))              # exactly how TSR.extract_mesh calls it
    mesh = trimesh.Trimesh(v.numpy(), f.numpy(), process=True)

    np.testing.assert_allclose(mesh.vertices.mean(axis=0), centre.numpy(), atol=0.01)
    r = np.linalg.norm(mesh.vertices - centre.numpy(), axis=1)
    np.testing.assert_allclose(r.mean(), radius, atol=0.01)
    assert mesh.is_watertight
    assert mesh.volume > 0, "normals point inward: flip the winding in mcubes_shim"
    np.testing.assert_allclose(mesh.volume, 4 / 3 * np.pi * radius**3, rtol=0.05)


def test_empty_volume_returns_empty_mesh():
    v, f = mcubes_shim.marching_cubes(torch.ones(8, 8, 8), 0.0)
    assert v.shape == (0, 3) and f.shape == (0, 3)
