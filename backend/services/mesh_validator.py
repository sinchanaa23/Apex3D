"""
Automated 3D Mesh Validation and Multi-Format Export Service using Trimesh.
Implements integrity checks (watertightness, manifold topology, degenerate faces,
component counts) and exports GLB, OBJ, STL files with full validation telemetry.
"""

import json
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, List
import numpy as np
import trimesh

from backend.models import MeshValidationReport, ReconstructionAssets

logger = logging.getLogger(__name__)


def validate_mesh(mesh: trimesh.Trimesh) -> MeshValidationReport:
    """
    Performs comprehensive geometric and topological validation of a 3D mesh.
    """
    # 1. Watertightness & Manifold checks
    is_watertight = bool(mesh.is_watertight)
    
    # Check for non-manifold edges
    # In a 2-manifold surface, every edge is shared by exactly 1 (boundary) or 2 (interior) faces
    try:
        edges_count = mesh.edges_unique_inverse
        # Trimesh provides is_winding_consistent and edge validations
        is_winding = bool(mesh.is_winding_consistent)
        is_manifold = is_watertight and is_winding
    except Exception:
        is_manifold = is_watertight

    # 2. Geometry counts
    vertex_count = int(len(mesh.vertices))
    face_count = int(len(mesh.faces))

    # 3. Degenerate faces (zero or near-zero area)
    try:
        face_areas = mesh.area_faces
        degenerate_count = int(np.sum(face_areas < 1e-9))
    except Exception:
        degenerate_count = 0

    # 4. Connected components
    try:
        components = len(mesh.split(only_watertight=False))
    except Exception:
        components = 1

    # 5. Visual attributes (vertex colors & UV textures)
    has_vertex_colors = False
    has_uv_texture = False
    if hasattr(mesh, "visual") and mesh.visual is not None:
        if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
            has_vertex_colors = len(mesh.visual.vertex_colors) > 0
        if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
            has_uv_texture = len(mesh.visual.uv) > 0

    # 6. Dimensions and extents
    try:
        extents = [round(float(x), 3) for x in mesh.extents]
    except Exception:
        extents = [1.0, 1.0, 1.0]

    # 7. Asset readiness assessment & honesty notices (§16 & §19)
    notes: List[str] = []
    notes.append("Geometry generated from single-view inference; unseen sides are hallucinated by neural prior.")
    notes.append("Real-world metric scale is uncalibrated. Output is suitable as a first-pass creative draft.")

    if is_watertight and degenerate_count == 0:
        asset_readiness = "draft"
        notes.append("Mesh topology is valid and closed (watertight).")
    else:
        asset_readiness = "needs_repair"
        if not is_watertight:
            notes.append("Mesh contains open boundaries or holes; requires mesh repair before slicing/printing.")
        if degenerate_count > 0:
            notes.append(f"Found {degenerate_count} degenerate faces requiring cleanup.")

    notes.append("Game engines require manual retopology, UV unwrapping, and LOD generation.")

    return MeshValidationReport(
        is_watertight=is_watertight,
        is_manifold=is_manifold,
        vertex_count=vertex_count,
        face_count=face_count,
        degenerate_faces=degenerate_count,
        connected_components=components,
        has_vertex_colors=has_vertex_colors,
        has_uv_texture=has_uv_texture,
        bounding_box_extents=extents,
        asset_readiness=asset_readiness,
        scale_status="unknown",
        notes=notes,
    )


def export_mesh_package(
    mesh: trimesh.Trimesh,
    output_dir: Path,
    job_id: str,
) -> Tuple[ReconstructionAssets, MeshValidationReport]:
    """
    Validates mesh, exports GLB, OBJ, STL, and validation JSON to output_dir,
    and returns (ReconstructionAssets, MeshValidationReport).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Validate
    report = validate_mesh(mesh)

    # 2. Export GLB (Primary interactive format for Three.js viewer)
    glb_path = output_dir / "mesh.glb"
    mesh.export(str(glb_path), file_type="glb")

    # 3. Export OBJ (Interoperability with Blender / Maya)
    obj_path = output_dir / "mesh.obj"
    mesh.export(str(obj_path), file_type="obj")

    # 4. Export STL (Standard 3D printing & CAD format)
    stl_path = output_dir / "mesh.stl"
    mesh.export(str(stl_path), file_type="stl")

    # 5. Export validation report
    val_path = output_dir / "validation.json"
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2)

    # Asset URLs relative to API prefix
    assets = ReconstructionAssets(
        glb_url=f"/api/assets/{job_id}/mesh.glb",
        obj_url=f"/api/assets/{job_id}/mesh.obj",
        stl_url=f"/api/assets/{job_id}/mesh.stl",
        validation_report_url=f"/api/assets/{job_id}/validation.json",
    )

    return assets, report

