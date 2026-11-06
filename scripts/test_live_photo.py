"""
End-to-End Teammate Verification Script for Apex3D.
Usage:
    python scripts/test_live_photo.py
    python scripts/test_live_photo.py path/to/car.jpg
"""

import sys
import os
import io
import time
from pathlib import Path
from PIL import Image, ImageDraw

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def generate_synthetic_car() -> bytes:
    """Generates a synthetic car photograph for testing."""
    img = Image.new("RGB", (640, 480), color=(210, 215, 225))
    draw = ImageDraw.Draw(img)
    # Ground
    draw.rectangle([0, 360, 640, 480], fill=(50, 52, 55))
    # Car body
    draw.rectangle([80, 260, 560, 370], fill=(210, 35, 45))
    # Greenhouse / Cabin
    draw.polygon([(180, 260), (250, 180), (450, 180), (500, 260)], fill=(35, 45, 55))
    # Wheels
    draw.ellipse([130, 330, 220, 420], fill=(25, 25, 25))
    draw.ellipse([420, 330, 510, 420], fill=(25, 25, 25))
    # Headlight
    draw.rectangle([545, 280, 558, 310], fill=(255, 255, 200))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def run_e2e_verification(image_path: str = None):
    print("=" * 65)
    print("      Apex3D — End-to-End Pipeline Verification")
    print("=" * 65)

    if image_path and os.path.exists(image_path):
        print(f"[*] Reading image from: {image_path}")
        with open(image_path, "rb") as f:
            image_bytes = f.read()
    else:
        print("[*] Generating synthetic high-resolution test car image...")
        image_bytes = generate_synthetic_car()

    # 1. Health check
    print("\n[1/6] Testing Service Health (/api/health)...")
    res = client.get("/api/health")
    assert res.status_code == 200
    h = res.json()
    print(f"      Status: {h['status']} | Device: {h['device']} | Queue Size: {h['queue_size']}")

    # 2. Recognition
    print("\n[2/6] Running Vehicle Recognition (/api/recognize)...")
    files = {"file": ("test_car.jpg", image_bytes, "image/jpeg")}
    rec_res = client.post("/api/recognize", files=files)
    assert rec_res.status_code == 200
    rec = rec_res.json()
    print(f"      Vehicle Detected: {rec['car_detected']}")
    if rec["bounding_box"]:
        bb = rec["bounding_box"]
        print(f"      Bounding Box: [{bb['x1']}, {bb['y1']}, {bb['x2']}, {bb['y2']}] (conf: {bb['confidence']})")
    print(f"      Crop Mode: {rec['crop_used']}")
    print(f"      Confidence State: {rec['confidence_state'].upper()} (Score: {rec['confidence_score']}, Margin: {rec['margin']})")
    print("      Top-3 Predictions:")
    for i, p in enumerate(rec["predictions"][:3], 1):
        print(f"        {i}. {p['class_name']} ({p['probability']*100:.1f}%)")
    print(f"      Latency: Total {rec['timings_ms']['total']} ms (Detection: {rec['timings_ms']['detection']} ms, Classification: {rec['timings_ms']['classification']} ms)")

    top_class_id = rec["predictions"][0]["class_id"]

    # 3. Classical CV
    print("\n[3/6] Running Classical CV Pipeline (/api/classical)...")
    files = {"file": ("test_car.jpg", image_bytes, "image/jpeg")}
    cls_res = client.post("/api/classical", files=files)
    assert cls_res.status_code == 200
    ccv = cls_res.json()
    stages = list(ccv["intermediate_images"].keys())
    print(f"      Visual Stages Generated ({len(stages)}): {', '.join(stages)}")
    geo = ccv["features"]["geometric"]
    print(f"      Geometric Features: Aspect={geo['aspect_ratio']}, Extent={geo['extent']}, Solidity={geo['solidity']}")
    print(f"      HOG Vector Length: {ccv['features']['hog']['vector_length']} dims")
    print(f"      Classical Latency: {ccv['timings_ms']['total']} ms")

    # 4. Reference Retrieval
    print(f"\n[4/6] Retrieving Reference Viewpoints (/api/references?class_id={top_class_id})...")
    ref_res = client.get(f"/api/references?class_id={top_class_id}&limit=4")
    assert ref_res.status_code == 200
    refs = ref_res.json()
    print(f"      Search Query: '{refs['query']}'")
    print(f"      Candidates Found: {refs['total_candidates']}")
    for cand in refs["candidates"][:3]:
        print(f"        - [{cand['viewpoint'].upper()}] {cand['title'][:45]}... (Lic: {cand['license']})")
    print(f"      Viewpoint Groups: {', '.join([f'{k}: {len(v)}' for k, v in refs['viewpoint_groups'].items() if len(v)>0])}")

    # 5. 3D Reconstruction
    print("\n[5/6] Enqueuing 3D Reconstruction Job (/api/reconstruct)...")
    files = {"file": ("reconstruct_input.jpg", image_bytes, "image/jpeg")}
    rec3d_res = client.post("/api/reconstruct", files=files)
    assert rec3d_res.status_code == 202
    job_id = rec3d_res.json()["job_id"]
    print(f"      Job ID: {job_id} enqueued. Polling worker state...")

    # Poll
    max_retries = 25
    for attempt in range(max_retries):
        time.sleep(1.0)
        poll = client.get(f"/api/jobs/{job_id}").json()
        print(f"      [{attempt+1}s] Stage: {poll['stage']} ({int(poll['progress']*100)}%) | Status: {poll['status']}")
        if poll["status"] in ["completed", "failed"]:
            break

    assert poll["status"] == "completed", f"Job failed: {poll.get('error')}"
    val = poll["validation"]
    print(f"\n      [+] Trimesh Validation Report:")
    print(f"          - Watertight: {val['is_watertight']}")
    print(f"          - 2-Manifold: {val['is_manifold']}")
    print(f"          - Vertices: {val['vertex_count']:,} | Faces: {val['face_count']:,}")
    print(f"          - Degenerate Faces: {val['degenerate_faces']}")
    print(f"          - Asset Readiness: {val['asset_readiness'].upper()}")
    print(f"          - Scale Disclaimer: {val['scale_status']}")

    # 6. Verify Asset Downloads
    print("\n[6/6] Verifying 3D Asset Downloads (/api/assets/{id}/{file})...")
    glb_res = client.get(f"/api/assets/{job_id}/mesh.glb")
    obj_res = client.get(f"/api/assets/{job_id}/mesh.obj")
    stl_res = client.get(f"/api/assets/{job_id}/mesh.stl")
    assert glb_res.status_code == 200
    assert obj_res.status_code == 200
    assert stl_res.status_code == 200
    print(f"      GLB: {len(glb_res.content):,} bytes [OK]")
    print(f"      OBJ: {len(obj_res.content):,} bytes [OK]")
    print(f"      STL: {len(stl_res.content):,} bytes [OK]")

    print("\n" + "=" * 65)
    print("      ALL END-TO-END CHECKS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    img_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_e2e_verification(img_arg)

