"""
Reconstruction Service and Background Queue Worker for Apex3D.
Manages single-GPU job execution, status updates in SQLite, TripoSR inference bridge,
trimesh validation, and multi-format asset exports.
"""

import time
import uuid
import queue
import threading
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image
import numpy as np

from backend.config import JOBS_DIR, CUDA_AVAILABLE, DEVICE
from backend.database import (
    create_job,
    get_job,
    update_job_status,
    get_next_queued_job,
)
from backend.services.mesh_library import store_in_cache
from backend.services.mesh_validator import export_mesh_package
from backend.models import ReconstructionJobResponse

logger = logging.getLogger(__name__)


class ReconstructionManager:
    _instance = None

    def __init__(self):
        self.job_queue: queue.Queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info("Reconstruction background worker initialized.")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ReconstructionManager()
        return cls._instance

    def submit_job(self, input_image_path: Path, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Registers job in SQLite and enqueues for sequential execution.
        """
        job_id = str(uuid.uuid4())
        job_data = create_job(job_id, str(input_image_path), options)
        self.job_queue.put(job_id)
        logger.info("Enqueued 3D reconstruction job %s", job_id)
        return job_data

    def _worker_loop(self):
        """Sequential single-job execution loop respecting GPU resource queue."""
        while True:
            try:
                # Check next job from in-memory queue or SQLite
                job_id = None
                try:
                    job_id = self.job_queue.get(timeout=3.0)
                except queue.Empty:
                    next_job = get_next_queued_job()
                    if next_job:
                        job_id = next_job["id"]

                if not job_id:
                    continue

                self._process_single_job(job_id)

            except Exception as e:
                logger.error("Unhandled exception in reconstruction worker loop: %s", e)
                time.sleep(1.0)

    def _process_single_job(self, job_id: str):
        """Executes full pipeline for a single reconstruction job."""
        job = get_job(job_id)
        if not job:
            return

        job_dir = JOBS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        input_image_path = Path(job["input_image_path"])

        try:
            update_job_status(
                job_id,
                status="running",
                stage="preprocessing",
                progress=0.10,
                new_log="Worker started. Loading and pre-processing input image...",
            )

            if not input_image_path.exists():
                raise FileNotFoundError(f"Source image not found: {input_image_path}")

            img = Image.open(input_image_path).convert("RGBA")
            time.sleep(0.5)

            # Stage 2: Background Removal
            update_job_status(
                job_id,
                stage="background_removal",
                progress=0.30,
                new_log="Removing background and isolating vehicle silhouette...",
            )
            segmented_img = self._remove_background(img)
            time.sleep(0.5)

            # Stage 3: 3D Reconstruction
            update_job_status(
                job_id,
                stage="triposr_inference",
                progress=0.60,
                new_log="Executing neural 3D reconstruction...",
            )

            mesh = self._run_reconstruction(segmented_img, job.get("options") or {})
            time.sleep(0.5)

            # Stage 4: Mesh Validation
            update_job_status(
                job_id,
                stage="mesh_validation",
                progress=0.85,
                new_log="Validating geometric topology and watertightness with Trimesh...",
            )

            assets, val_report = export_mesh_package(mesh, job_dir, job_id)

            options = job.get("options") or {}
            if options.get("mesh_source") == "reference" and options.get("class_id") is not None:
                store_in_cache(int(options["class_id"]), job_dir)

            # Stage 5: Completed
            update_job_status(
                job_id,
                status="completed",
                stage="done",
                progress=1.0,
                new_log=f"Mesh generated successfully: {val_report.vertex_count} vertices, {val_report.face_count} faces. Formats exported: GLB, OBJ, STL.",
                assets=assets.model_dump(),
                validation=val_report.model_dump(),
            )
            logger.info("Reconstruction job %s completed successfully.", job_id)

        except Exception as e:
            logger.exception("Reconstruction job %s failed: %s", job_id, e)
            update_job_status(
                job_id,
                status="failed",
                stage="error",
                progress=0.0,
                new_log=f"Job failed with error: {str(e)}",
                error=str(e),
            )

    def _remove_background(self, img: Image.Image) -> Image.Image:
        """Background removal happens inside backend.reconstruction (TripoSR's own preprocessing), so nothing runs twice."""
        return img

    def _run_reconstruction(self, image: Image.Image, options: Dict[str, Any]) -> "trimesh.Trimesh":
        """Runs Person A's TripoSR reconstruction. A failure is reported to the user, never replaced by an invented car."""
        from backend.reconstruction import ReconstructionError, reconstruct as triposr_reconstruct

        try:
            mesh = triposr_reconstruct(image)
        except ReconstructionError as e:
            raise RuntimeError(f"3D reconstruction failed: {e}") from e
        import trimesh

        if not isinstance(mesh, trimesh.Trimesh):
            raise RuntimeError("3D reconstruction did not return a mesh.")
        return mesh
