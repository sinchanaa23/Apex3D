"""
Storage Cleanup Service for Apex3D.
Purges temporary uploaded images and old generated assets older than a specified age (default: 1 hour)
to ensure disk space management, while preserving job metadata and status in SQLite.
"""

import time
import logging
from pathlib import Path
from backend.config import UPLOADS_DIR, JOBS_DIR

logger = logging.getLogger(__name__)


def cleanup_old_storage(max_age_hours: float = 1.0) -> dict:
    """
    Deletes upload and job asset files older than max_age_hours.
    Returns summary of cleaned files and freed bytes.
    """
    now = time.time()
    max_age_seconds = max_age_hours * 3600.0

    deleted_files = 0
    bytes_freed = 0
    errors = 0

    # 1. Clean old uploads
    if UPLOADS_DIR.exists():
        for item in UPLOADS_DIR.iterdir():
            if item.is_file():
                try:
                    file_age = now - item.stat().st_mtime
                    if file_age > max_age_seconds:
                        size = item.stat().st_size
                        item.unlink()
                        deleted_files += 1
                        bytes_freed += size
                except Exception as e:
                    logger.debug("Failed removing old upload %s: %s", item, e)
                    errors += 1

    # 2. Clean old job assets
    if JOBS_DIR.exists():
        for job_folder in JOBS_DIR.iterdir():
            if job_folder.is_dir():
                try:
                    folder_age = now - job_folder.stat().st_mtime
                    if folder_age > max_age_seconds:
                        for asset_file in job_folder.iterdir():
                            if asset_file.is_file():
                                bytes_freed += asset_file.stat().st_size
                                asset_file.unlink()
                                deleted_files += 1
                        # Remove folder if empty
                        try:
                            job_folder.rmdir()
                        except Exception:
                            pass
                except Exception as e:
                    logger.debug("Failed cleaning job folder %s: %s", job_folder, e)
                    errors += 1

    logger.info(
        "Storage cleanup completed: removed %d files, freed %.2f MB.",
        deleted_files,
        bytes_freed / (1024 * 1024),
    )

    return {
        "deleted_files": deleted_files,
        "bytes_freed_mb": round(bytes_freed / (1024 * 1024), 2),
        "errors": errors,
    }

