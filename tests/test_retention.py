"""Uploaded photos and old results are deleted after the retention period, but never while a job still needs them."""

import os
import time

from backend import database
from backend.services import cleanup_service


def _age(path, hours):
    old = time.time() - hours * 3600
    os.utime(path, (old, old))


def test_old_photos_are_deleted_and_recent_or_active_ones_are_kept(tmp_path, monkeypatch):
    uploads, jobs, refs = tmp_path / "uploads", tmp_path / "jobs", tmp_path / "references"
    for folder in (uploads, jobs, refs):
        folder.mkdir()
    monkeypatch.setattr(cleanup_service, "UPLOADS_DIR", uploads)
    monkeypatch.setattr(cleanup_service, "JOBS_DIR", jobs)
    monkeypatch.setattr(cleanup_service.config, "REFERENCES_DIR", refs)
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    database.init_db()

    old_photo, new_photo, busy_photo = uploads / "old.jpg", uploads / "new.jpg", uploads / "busy.jpg"
    for photo in (old_photo, new_photo, busy_photo):
        photo.write_bytes(b"x")
    _age(old_photo, 5)
    _age(busy_photo, 5)
    database.create_job("job-busy", str(busy_photo))  # queued: its photo must survive

    old_ref, new_ref, wiki_ref = refs / "upload_aaaa1111.jpg", refs / "upload_bbbb2222.jpg", refs / "wikimedia_cc.jpg"
    for ref in (old_ref, new_ref, wiki_ref):
        ref.write_bytes(b"x")
    _age(old_ref, 5)
    _age(wiki_ref, 5)  # downloaded references are not user photos, so they stay
    database.save_cached_reference({"id": "upload_aaaa1111", "class_id": None, "title": "t", "image_url": "u", "thumb_url": "u", "source": "manual_upload", "license": "l", "license_url": None,
                                    "author": "a", "viewpoint": "front", "viewpoint_confidence": 1.0, "match_score": 1.0, "is_approximate": False})

    old_job = jobs / "oldjob"
    old_job.mkdir()
    (old_job / "mesh.glb").write_bytes(b"x")
    _age(old_job, 5)

    report = cleanup_service.cleanup_old_storage(max_age_hours=2)

    assert not old_photo.exists() and new_photo.exists() and busy_photo.exists()
    assert not old_ref.exists() and new_ref.exists() and wiki_ref.exists()
    assert not old_job.exists()
    assert report["deleted_files"] == 3
    conn = database.get_db_connection()
    try:
        assert conn.execute("SELECT COUNT(*) FROM references_cache WHERE id = 'upload_aaaa1111'").fetchone()[0] == 0
    finally:
        conn.close()
