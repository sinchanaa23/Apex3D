"""The Hugging Face Space folder: what goes in, what stays out, and the header Spaces needs."""

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_hf_space", ROOT / "scripts" / "make_hf_space.py")
make_hf_space = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_hf_space)


def test_space_folder_has_what_the_build_needs_and_nothing_private(tmp_path):
    out = make_hf_space.assemble(tmp_path / "space")
    for needed in ["Dockerfile", "README.md", "requirements.txt", "backend/main.py", "checkpoints/model_quantized.onnx", "results/class_names.json", "frontend/package-lock.json", "eval_P5_metrics.json"]:
        assert (out / needed).is_file(), needed
    names = {p.name for p in out.rglob("*")}
    assert "node_modules" not in names and "__pycache__" not in names and "storage" not in names and "dist" not in names
    assert not list(out.rglob("*.mp4")) and not list(out.rglob("apex3d.db"))


def test_readme_header_is_what_spaces_reads(tmp_path):
    readme = (make_hf_space.assemble(tmp_path / "space") / "README.md").read_text(encoding="utf-8")
    header = re.match(r"---\n(.*?)\n---\n", readme, re.S).group(1)
    assert "sdk: docker" in header and "app_port: 7860" in header
    short = re.search(r"short_description: (.*)", header).group(1)
    assert len(short) <= 60


def test_space_dockerfile_runs_as_user_1000_on_the_port_the_header_names():
    text = (ROOT / "deploy" / "huggingface" / "Dockerfile").read_text(encoding="utf-8")
    assert "useradd -m -u 1000 user" in text and "\nUSER user" in text
    assert "PORT=7860" in text and "EXPOSE 7860" in text
    # Weights are downloaded after switching to that user, into folders it owns.
    assert text.index("USER user") < text.index("snapshot_download")
    assert "APEX3D_REQUIRE_LOGIN=1" in text
