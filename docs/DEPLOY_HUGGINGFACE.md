# Deploying Apex3D on Hugging Face Spaces

> **Correction: this is not free.** When the upload was tried, Hugging Face answered `402 Payment Required`: "Static Spaces are free for everyone, but hosting Gradio and Docker Spaces on free cpu-basic requires a PRO subscription." Apex3D needs a Docker Space, so it needs a **Hugging Face PRO subscription**, which is paid and needs a card. Earlier versions of this guide said it was free; that was wrong, written from memory without checking. Nothing was created on Hugging Face. The rest of this guide is correct for an account that has PRO.

A basic CPU Space gives the app about 2 CPUs and 16 GB of memory (Hugging Face's figures, from memory, not checked). That is enough for everything, including the 3D mesh builder, which peaked near 1.25 GB when measured here. Nothing in this guide has been run on Hugging Face yet, and the Docker build was never test-built locally, so the first upload is the real test.

## What is prepared

| File | What it is |
|---|---|
| `deploy/huggingface/Dockerfile` | The Space's Dockerfile. It runs as user 1000, as Spaces requires, listens on port 7860, installs CPU-only PyTorch, and bakes in the detector weights, TripoSR (about 1.7 GB) and the background remover. |
| `scripts/make_hf_space.py` | Builds a clean folder (`build/hf_space/`, about 42 MB) with the app, the model, the saved results, the Dockerfile and a `README.md` with the header Spaces reads. With `--upload` it also creates the Space and uploads the folder. |
| `tests/test_hf_space.py` | Checks the folder has what the build needs, leaves out private and heavy files, and has a valid header. |

## Steps (about 10 minutes of your time, then a long build)

1. **Make an account** at https://huggingface.co/join, confirm your email and subscribe to PRO at https://huggingface.co/pro (check the price there; you can cancel afterwards).
2. **Make a token.** Go to https://huggingface.co/settings/tokens, choose **Create new token**, type **Write**, and copy it. Treat it like a password: paste it only into your own terminal, never into a chat or a file.
3. **Upload.** In PowerShell, in the Apex3D folder:
   ```powershell
   pip install huggingface_hub
   $env:HF_TOKEN = "paste-your-token-here"
   python scripts\make_hf_space.py --upload YOUR_USERNAME/apex3d
   ```
   The script creates a **public** Docker Space named `apex3d` on free CPU hardware if it isn't there yet, and uploads the folder. To use another name, change `apex3d` in the command.
4. **Watch the build.** Open `https://huggingface.co/spaces/YOUR_USERNAME/apex3d` and the **Logs** tab. The first build is long (PyTorch, then about 2 GB of 3D model weights), allow 20 to 30 minutes. The status turns to **Running** when it is done.
5. **Open the direct address** `https://YOUR_USERNAME-apex3d.hf.space` (the script prints it). Sign up, identify a photo, then build a mesh in the 3D Studio and time it. Measured here with 2 CPU threads it took about 5 minutes.

**Use the direct `.hf.space` address, not the page on huggingface.co.** That page shows the app inside a frame, and browsers refuse to keep a login cookie inside another site's frame, so signing in would appear to do nothing.

To publish a new version: merge to `main`, run the same `python scripts\make_hf_space.py --upload ...` command again. It rebuilds the Space.

## What to expect

- **Accounts are lost when the Space restarts.** The free tier has no persistent storage (a paid add-on). Restarts happen when the Space is rebuilt, or after it sleeps. Plan on signing up again before a demo, and open the site a few minutes early.
- **Sleeping.** A free Space sleeps after a long time without visitors (I believe about 2 days) and wakes when someone opens it, which takes a minute or two.
- **Public means public.** The Space's files can be read by anyone, like the GitHub repository. Anyone with the address can create an account. Sign-ups are limited to 30 per hour, and a wrong password is blocked after 5 tries per email for 10 minutes. A private Space would need every visitor to have a Hugging Face login, which defeats a link for judges.
- **Speed.** Identification takes a second or two. A mesh takes about 5 minutes; one builds at a time and the others queue.
- **Revoking access.** If you want, delete the token at https://huggingface.co/settings/tokens after uploading.

## If the build fails

Open the Space's **Logs**, copy the last 40 or so lines, and send them to me. The likely places are a package download inside the Docker build, or fetching TripoSR from GitHub.
