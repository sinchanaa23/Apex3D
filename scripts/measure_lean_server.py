"""Measure the lean server's memory with PyTorch, torchvision and trimesh blocked, as on a server that doesn't have them.

    python scripts/measure_lean_server.py        (needs checkpoints/detector_mobilenet.onnx: python scripts/export_detector_onnx.py)

It starts the real app in-process, sends 6 Stanford Cars test photos to /api/recognize and one to /api/classical, and prints the memory
(private bytes) after each step and the time per recognition, with one CPU thread. Needs psutil and the dataset's test photos.
"""
import os, sys, glob, time, psutil
os.environ["APEX3D_ORT_THREADS"]="1"; os.environ["OMP_NUM_THREADS"]="1"; os.environ["APEX3D_REQUIRE_LOGIN"]="0"
for blocked in ("torch","torchvision","trimesh"):   # importing any of these now raises ImportError, as on a server that doesn't have them
    sys.modules[blocked] = None
sys.path.insert(0, os.getcwd())
p = psutil.Process()
def mb(): return round(p.memory_full_info().uss/1e6)
def step(label): print(f"{label:<54} {mb():>4} MB", flush=True)
step("python started")
import backend.main as m; step("import backend.main")
from fastapi.testclient import TestClient
with TestClient(m.app) as c:
    step("lifespan start-up (detector + classifier loaded)")
    print("   health:", c.get("/api/health").json())
    imgs = sorted(glob.glob(os.path.expanduser("~/.apex3d/data/stanford_cars/images/test/0000*.jpg")))[:6]
    peak = mb(); times = []
    for i, img in enumerate(imgs):
        t = time.time(); r = c.post("/api/recognize", files={"file": ("c.jpg", open(img,"rb"), "image/jpeg")}); times.append(time.time()-t)
        j = r.json(); top = j["predictions"][0]
        print(f"   recognition {i+1}: {r.status_code} {times[-1]:.2f}s  {top['class_name']} {top['probability']} [{j['confidence_state']}] crop={j['crop_used']}")
        peak = max(peak, mb())
    step("after 6 recognitions"); r = c.post("/api/classical", files={"file": ("c.jpg", open(imgs[0],"rb"), "image/jpeg")}); step(f"after classical ({r.status_code})")
    print("   peak seen:", max(peak, mb()), "MB | torch loaded:", sys.modules.get("torch") is not None, "| trimesh loaded:", sys.modules.get("trimesh") is not None)
