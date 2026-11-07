"""Render the Mermaid diagrams in docs/ARCHITECTURE.md to docs/architecture_system.{svg,png} and docs/architecture_sequence.{svg,png}.

    python scripts/render_diagrams.py

GitHub draws the Mermaid blocks by itself, but the report needs a picture. Each diagram is drawn by headless Edge or Chrome, the drawing is
read back from the page, and then screenshotted at exactly its own size at double resolution, so there is no empty margin and the
text stays sharp. It needs internet access while it runs (it loads mermaid 10.9.1 from the jsDelivr CDN); nothing is installed.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "ARCHITECTURE.md"
NAMES = ["architecture_system", "architecture_sequence"]
BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "/usr/bin/google-chrome", "/usr/bin/chromium",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]
DRAW = """<!doctype html><html><head><meta charset="utf-8"><style>body{{margin:0;background:#fff;font-family:Segoe UI,Arial,sans-serif}}</style></head>
<body><pre class="mermaid">
{code}
</pre><script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"></script>
<script>mermaid.initialize({{startOnLoad:true,theme:"neutral",flowchart:{{htmlLabels:true,useMaxWidth:false}},sequence:{{useMaxWidth:false}}}});</script></body></html>"""
SHOT = """<!doctype html><html><head><meta charset="utf-8"><style>html,body{{margin:0;background:#fff}}svg{{display:block;width:{w}px;height:{h}px;max-width:none!important}}</style></head><body>{svg}</body></html>"""


def run(browser: str, args: list[str], tmp: str) -> subprocess.CompletedProcess:
    return subprocess.run([browser, "--headless=new", "--disable-gpu", f"--user-data-dir={tmp}/profile", *args], check=True, capture_output=True, timeout=180)


def main() -> None:
    blocks = re.findall(r"```mermaid\n(.*?)```", SOURCE.read_text(encoding="utf-8"), re.S)
    if len(blocks) != len(NAMES):
        raise SystemExit(f"Expected {len(NAMES)} Mermaid blocks in {SOURCE.name}, found {len(blocks)}.")
    browser = next((b for b in BROWSERS if Path(b).exists()), None)
    if not browser:
        raise SystemExit("No Edge or Chrome found.")
    for name, code in zip(NAMES, blocks):
        with tempfile.TemporaryDirectory() as tmp:
            draw = Path(tmp) / "draw.html"
            draw.write_text(DRAW.format(code=code), encoding="utf-8")
            dom = run(browser, ["--virtual-time-budget=25000", "--dump-dom", draw.as_uri()], tmp).stdout.decode("utf-8", "replace")
            svg = re.search(r"<svg[^>]*id=\"mermaid-[^\"]*\".*?</svg>", dom, re.S)
            if not svg:
                raise SystemExit(f"{name}: the diagram did not draw (is the internet reachable?).")
            svg_text = svg.group(0)
            box = re.search(r'viewBox="[-\d.]+ [-\d.]+ ([\d.]+) ([\d.]+)"', svg_text)
            width, height = (int(float(box.group(1))) + 1, int(float(box.group(2))) + 1) if box else (1600, 1200)
            (ROOT / "docs" / f"{name}.svg").write_text(svg_text.replace("<svg ", '<svg xmlns:xlink="http://www.w3.org/1999/xlink" ', 1) if "xmlns:xlink" not in svg_text else svg_text, encoding="utf-8")
            shot = Path(tmp) / "shot.html"
            shot.write_text(SHOT.format(w=width, h=height, svg=svg_text), encoding="utf-8")
            out = ROOT / "docs" / f"{name}.png"
            run(browser, [f"--window-size={width},{height}", "--force-device-scale-factor=2", "--hide-scrollbars", "--virtual-time-budget=5000", f"--screenshot={out}", shot.as_uri()], tmp)
        print(f"Wrote docs/{name}.svg and {out.relative_to(ROOT)} ({width}x{height} at 2x, {out.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    sys.exit(main())
