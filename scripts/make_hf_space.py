"""Assemble (and optionally upload) the Hugging Face Space for Apex3D.

    python scripts/make_hf_space.py                         build the Space folder in build/hf_space/
    python scripts/make_hf_space.py --upload USER/apex3d    build it, create the Space if needed, and upload it

The folder holds only what the Docker build needs: the app source, the model, the saved results, the Space's own Dockerfile
(deploy/huggingface/Dockerfile) and a README.md with the header Spaces reads. Uploading uses huggingface_hub, which sends
large files like the 34 MB model the way Hugging Face wants them, so no git-lfs is needed.

Your Hugging Face token is never stored by this script. It comes from the HF_TOKEN environment variable or from a login you
did yourself with `hf auth login`. Create a token with "Write" access at https://huggingface.co/settings/tokens.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (source, destination) relative to the repository root. Directories are copied whole, minus the ignored names below.
COPY = [
    ("backend", "backend"),
    ("checkpoints", "checkpoints"),
    ("results", "results"),
    ("catalog", "catalog"),
    ("frontend", "frontend"),
    ("requirements.txt", "requirements.txt"),
    ("eval_P5_metrics.json", "eval_P5_metrics.json"),
    ("deploy/huggingface/Dockerfile", "Dockerfile"),
]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "node_modules", "dist", "storage", ".venv", "*.mp4")

HEADER = """---
title: Apex3D
emoji: 🚗
colorFrom: orange
colorTo: red
sdk: docker
app_port: 7860
pinned: false
short_description: Recognise a car from a photo and build a 3D model
---

"""

BODY = """# Apex3D

Upload a photo of a car. Apex3D finds the car, names its manufacturer, model and year (196 Stanford Cars classes), shows how
sure it is, and builds a draft 3D model. Log in or sign up to use it.

- Open the app at its direct address, `https://<user>-<space>.hf.space`. The page embedded on huggingface.co can't keep you
  logged in, because browsers block login cookies inside another site's frame.
- Accounts are kept only while the Space is running. If it restarts, sign up again.
- Source code, evaluation results and the report: https://github.com/sinchanaa23/Apex3D
"""


def assemble(out: Path) -> Path:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for source, destination in COPY:
        src, dst = ROOT / source, out / destination
        if not src.exists():
            raise SystemExit(f"Missing {source}. Run this from a complete checkout.")
        if src.is_dir():
            shutil.copytree(src, dst, ignore=IGNORE)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    (out / "README.md").write_text(HEADER + BODY, encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "hf_space")
    parser.add_argument("--upload", metavar="USER/SPACE", help="create the Space if needed and upload the folder to it")
    args = parser.parse_args()

    folder = assemble(args.out)
    size = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()) / 1e6
    print(f"Built {folder} ({size:.0f} MB, {sum(1 for p in folder.rglob('*') if p.is_file())} files).")
    if not args.upload:
        print("Not uploaded. Run again with --upload USER/apex3d to publish it.")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise SystemExit("Install the uploader first:  pip install huggingface_hub")
    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    try:
        who = api.whoami()["name"]
    except Exception:
        raise SystemExit("No Hugging Face login found. Set HF_TOKEN to a token with Write access, or run `hf auth login`, then try again.")
    print(f"Logged in to Hugging Face as {who}.")
    api.create_repo(args.upload, repo_type="space", space_sdk="docker", exist_ok=True, private=False)
    api.upload_folder(folder_path=str(folder), repo_id=args.upload, repo_type="space", commit_message="Apex3D")
    owner, name = args.upload.split("/")
    print(f"Uploaded. Watch the build at https://huggingface.co/spaces/{args.upload}")
    print(f"When it says Running, open https://{owner.lower()}-{name.lower()}.hf.space")


if __name__ == "__main__":
    sys.exit(main())
