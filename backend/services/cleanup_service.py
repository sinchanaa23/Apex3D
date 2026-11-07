"""
Storage Cleanup Service for Apex3D.
Purges temporary uploaded images, photos uploaded as references, and old generated assets older than a specified age,
to ensure disk space management and so that users' photos are not kept, while preserving job metadata and status in SQLite.
The server runs this at start-up and every 15 minutes (see backend/main.py); APEX3D_RETENTION_HOURS sets the age.
"""

import time
import logging
from pathlib import Path
from backend import config
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

    # A photo that a queued or running job still needs is kept, however old.
    in_use: set[str] = set()
    try:
        from backend.database import get_db_connection

        conn = get_db_connection()
        try:
            in_use = {Path(r[0]).name for r in conn.execute("SELECT input_image_path FROM jobs WHERE status IN ('queued', 'running')")}
        finally:
            conn.close()
    except Exception as e:  # noqa: BLE001
        logger.debug("Could not read active jobs: %s", e)

    # 1. Clean old uploads
    if UPLOADS_DIR.exists():
        for item in UPLOADS_DIR.iterdir():
            if item.is_file() and item.name not in in_use:
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

    # 1b. Photos a user uploaded as their own reference (named upload_*) and their index rows. Downloaded Wikimedia references stay.
    refs_dir = config.REFERENCES_DIR
    if refs_dir.exists():
        for item in refs_dir.glob("upload_*"):
            if item.is_file():
                try:
                    if now - item.stat().st_mtime > max_age_seconds:
                        size = item.stat().st_size
                        item.unlink()
                        deleted_files += 1
                        bytes_freed += size
                        try:
                            from backend.database import get_db_connection

                            conn = get_db_connection()
                            try:
                                with conn:
                                    conn.execute("DELETE FROM references_cache WHERE id = ?", (item.stem,))
                            finally:
                                conn.close()
                        except Exception as e:  # noqa: BLE001
                            logger.debug("Could not drop the index row for %s: %s", item.name, e)
                except Exception as e:  # noqa: BLE001
                    logger.debug("Failed removing old reference upload %s: %s", item, e)
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

