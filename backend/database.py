"""
SQLite database management for Apex3D background jobs and reference cache.
"""

import sqlite3
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pathlib import Path
from backend.config import DB_PATH


def get_db_connection() -> sqlite3.Connection:
    """Creates a thread-safe connection to the SQLite database."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes database tables if they do not exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = get_db_connection()
    try:
        with conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                stage TEXT NOT NULL,
                progress REAL NOT NULL DEFAULT 0.0,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                input_image_path TEXT NOT NULL,
                options_json TEXT,
                logs_json TEXT NOT NULL DEFAULT '[]',
                error TEXT,
                assets_json TEXT,
                validation_json TEXT
            );
            """)

            conn.execute("""
            CREATE TABLE IF NOT EXISTS references_cache (
                id TEXT PRIMARY KEY,
                class_id INTEGER,
                title TEXT NOT NULL,
                image_url TEXT NOT NULL,
                thumb_url TEXT NOT NULL,
                source TEXT NOT NULL,
                license TEXT NOT NULL,
                license_url TEXT,
                author TEXT,
                viewpoint TEXT NOT NULL DEFAULT 'unknown',
                viewpoint_confidence REAL NOT NULL DEFAULT 0.0,
                match_score REAL NOT NULL DEFAULT 0.0,
                is_approximate INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            """)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_refs_class ON references_cache(class_id);")
    finally:
        conn.close()


def create_job(job_id: str, input_image_path: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Creates and inserts a new job into the queue."""
    now_iso = datetime.now(timezone.utc).isoformat()
    initial_log = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Job registered and added to GPU queue."
    logs_json = json.dumps([initial_log])
    options_json = json.dumps(options or {})

    conn = get_db_connection()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO jobs (id, status, stage, progress, created_at, input_image_path, options_json, logs_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, "queued", "waiting", 0.0, now_iso, str(input_image_path), options_json, logs_json),
            )
    finally:
        conn.close()

    return get_job(job_id)


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Fetches a single job and calculates its queue position."""
    conn = get_db_connection()
    try:
        cursor = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = cursor.fetchone()
        if not row:
            return None

        job_data = dict(row)
        job_data["logs"] = json.loads(job_data.get("logs_json") or "[]")
        job_data["options"] = json.loads(job_data.get("options_json") or "{}")
        job_data["assets"] = json.loads(job_data["assets_json"]) if job_data.get("assets_json") else None
        job_data["validation"] = json.loads(job_data["validation_json"]) if job_data.get("validation_json") else None

        # Calculate real position in queue
        if job_data["status"] == "queued":
            pos_cursor = conn.execute(
                """
                SELECT COUNT(*) FROM jobs 
                WHERE status = 'queued' AND created_at < ?
                """,
                (job_data["created_at"],),
            )
            # 1-indexed queue position (pos 1 means next up)
            job_data["queue_position"] = pos_cursor.fetchone()[0] + 1
        elif job_data["status"] == "running":
            job_data["queue_position"] = 0
        else:
            job_data["queue_position"] = 0

        return job_data
    finally:
        conn.close()


def update_job_status(
    job_id: str,
    status: Optional[str] = None,
    stage: Optional[str] = None,
    progress: Optional[float] = None,
    new_log: Optional[str] = None,
    error: Optional[str] = None,
    assets: Optional[Dict[str, Any]] = None,
    validation: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Updates job progress, status, logs, or completion assets."""
    conn = get_db_connection()
    try:
        cursor = conn.execute("SELECT logs_json, status, started_at, completed_at FROM jobs WHERE id = ?", (job_id,))
        row = cursor.fetchone()
        if not row:
            return None

        current_logs = json.loads(row["logs_json"] or "[]")
        if new_log:
            time_str = datetime.now(timezone.utc).strftime("%H:%M:%S")
            current_logs.append(f"[{time_str}] {new_log}")

        updates = ["logs_json = ?"]
        params = [json.dumps(current_logs)]

        now_iso = datetime.now(timezone.utc).isoformat()

        if status is not None:
            updates.append("status = ?")
            params.append(status)
            if status == "running" and not row["started_at"]:
                updates.append("started_at = ?")
                params.append(now_iso)
            elif status in ("completed", "failed") and not row["completed_at"]:
                updates.append("completed_at = ?")
                params.append(now_iso)

        if stage is not None:
            updates.append("stage = ?")
            params.append(stage)

        if progress is not None:
            updates.append("progress = ?")
            params.append(progress)

        if error is not None:
            updates.append("error = ?")
            params.append(error)

        if assets is not None:
            updates.append("assets_json = ?")
            params.append(json.dumps(assets))

        if validation is not None:
            updates.append("validation_json = ?")
            params.append(json.dumps(validation))

        params.append(job_id)
        sql = f"UPDATE jobs SET {', '.join(updates)} WHERE id = ?"

        with conn:
            conn.execute(sql, params)

    finally:
        conn.close()

    return get_job(job_id)


def get_next_queued_job() -> Optional[Dict[str, Any]]:
    """Gets the oldest queued job waiting to be processed."""
    conn = get_db_connection()
    try:
        cursor = conn.execute(
            """
            SELECT id FROM jobs 
            WHERE status = 'queued' 
            ORDER BY created_at ASC 
            LIMIT 1
            """
        )
        row = cursor.fetchone()
        if row:
            return get_job(row["id"])
        return None
    finally:
        conn.close()


def save_cached_reference(candidate_data: Dict[str, Any]):
    """Stores a reference candidate in SQLite cache."""
    conn = get_db_connection()
    try:
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO references_cache 
                (id, class_id, title, image_url, thumb_url, source, license, license_url, author, viewpoint, viewpoint_confidence, match_score, is_approximate, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate_data["id"],
                    candidate_data.get("class_id"),
                    candidate_data["title"],
                    candidate_data["image_url"],
                    candidate_data["thumb_url"],
                    candidate_data.get("source", "wikimedia_commons"),
                    candidate_data["license"],
                    candidate_data.get("license_url"),
                    candidate_data.get("author"),
                    candidate_data.get("viewpoint", "unknown"),
                    candidate_data.get("viewpoint_confidence", 0.0),
                    candidate_data.get("match_score", 0.0),
                    1 if candidate_data.get("is_approximate") else 0,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
    finally:
        conn.close()


def get_cached_references(class_id: Optional[int] = None, limit: int = 12) -> List[Dict[str, Any]]:
    """Fetches cached references for a vehicle class."""
    conn = get_db_connection()
    try:
        if class_id is not None:
            cursor = conn.execute(
                """
                SELECT * FROM references_cache 
                WHERE class_id = ? 
                ORDER BY match_score DESC, created_at DESC 
                LIMIT ?
                """,
                (class_id, limit),
            )
        else:
            cursor = conn.execute(
                """
                SELECT * FROM references_cache 
                ORDER BY created_at DESC 
                LIMIT ?
                """,
                (limit,),
            )

        results = []
        for r in cursor.fetchall():
            d = dict(r)
            d["is_approximate"] = bool(d["is_approximate"])
            results.append(d)
        return results
    finally:
        conn.close()

