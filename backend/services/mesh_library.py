"""Ready-made meshes for a matched car: a hand-placed catalog and a cache of meshes built from reference photos.

catalog/<class_id>.glb        licensed models dropped in by the team. Shown instantly and labelled "catalog model".
storage/mesh_cache/<class_id> a mesh built earlier from a reference photo of the same car. Reused instead of waiting again.
"""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from backend.config import JOBS_DIR, PROJECT_ROOT, STORAGE_DIR
from backend.database import create_job, update_job_status
from backend.services.mesh_validator import export_mesh_package

logger = logging.getLogger(__name__)

CATALOG_DIR = PROJECT_ROOT / "catalog"
MESH_CACHE_DIR = STORAGE_DIR / "mesh_cache"
CACHE_FILES = ("mesh.glb", "mesh.obj", "mesh.stl", "validation.json")


def catalog_glb(class_id: int) -> Optional[Path]:
    path = CATALOG_DIR / f"{class_id}.glb"
    return path if path.is_file() else None


def cache_dir(class_id: int) -> Optional[Path]:
    folder = MESH_CACHE_DIR / str(class_id)
    return folder if all((folder / name).is_file() for name in CACHE_FILES) else None


def store_in_cache(class_id: int, job_dir: Path) -> None:
    """Keeps a finished reference-photo job's files for the next person who matches this car."""
    if not all((job_dir / name).is_file() for name in CACHE_FILES):
        return
    target = MESH_CACHE_DIR / str(class_id)
    target.mkdir(parents=True, exist_ok=True)
    for name in CACHE_FILES:
        shutil.copy2(job_dir / name, target / name)
    logger.info("Cached the mesh for class %s", class_id)


def _asset_urls(job_id: str) -> Dict[str, str]:
    return {
        "glb_url": f"/api/assets/{job_id}/mesh.glb",
        "obj_url": f"/api/assets/{job_id}/mesh.obj",
        "stl_url": f"/api/assets/{job_id}/mesh.stl",
        "validation_report_url": f"/api/assets/{job_id}/validation.json",
    }


def instant_job(class_id: int, mesh_source: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """Creates an already-finished job from the catalog ('catalog') or the cache ('reference'). Never touches the GPU queue."""
    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    create_job(job_id, "", {**options, "mesh_source": mesh_source, "class_id": class_id})

    if mesh_source == "catalog":
        source = catalog_glb(class_id)
        assert source is not None
        import trimesh  # loaded only when a mesh is read, so a server that never builds one doesn't pay for it

        loaded = trimesh.load(str(source), force="mesh")
        _, report = export_mesh_package(loaded, job_dir, job_id)
        shutil.copy2(source, job_dir / "mesh.glb")  # keeps the model's own materials
        validation = report.model_dump()
        message = f"Catalog model for class {class_id}."
    else:
        cached = cache_dir(class_id)
        assert cached is not None
        for name in CACHE_FILES:
            shutil.copy2(cached / name, job_dir / name)
        validation = json.loads((job_dir / "validation.json").read_text(encoding="utf-8"))
        message = f"Reused the mesh built earlier for class {class_id}."

    return update_job_status(
        job_id,
        status="completed",
        stage="done",
        progress=1.0,
        new_log=message,
        assets=_asset_urls(job_id),
        validation=validation,
    )
