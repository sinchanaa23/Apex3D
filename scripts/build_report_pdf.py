"""Render the report to PDF, after checking it against the template.

    python scripts/build_report_pdf.py                 checks, then writes report/report_P5_T25.pdf
    python scripts/build_report_pdf.py --draft         builds the PDF even if slots are unfinished (marked as a draft)
    python scripts/build_report_pdf.py --check-only    only run the checks

Checks, in order:
  1. No [WRITE: ...] or [TODO: ...] slot left (skipped with --draft).
  2. Every section the template requires is present.
  3. No long code blocks (the template says: do not paste raw code). The quickstart block is allowed.
  4. Every percentage written in the team's own text also appears in the result files. A number that doesn't is listed so it can be
     checked by hand. It is a net for typos and stale figures, not a proof.
The PDF is made by Microsoft Edge or Chrome in headless mode, so nothing else needs installing except the `markdown` package
(`pip install markdown`).
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report" / "report_P5_T25.md"
PDF = ROOT / "report" / "report_P5_T25.pdf"

REQUIRED = [
    "1. Problem Understanding & Solution Strategy",
    "2. Technical Architecture & Implementation",
    "2.1 Classical Algorithmic Baseline",
    "2.2 Deep Learning Pipeline",
    "2.3 Hybrid Integration & Adaptive Mechanism",
    "3. Experimental Benchmarks & Evaluation",
    "3.1 Quantitative Benchmarks",
    "3.2 Visual Outputs & Intermediate Representations",
    "3.3 Ablation Study",
    "4. Failure Modes & Limitations",
    "5. Deployment & Quickstart Guide",
    "Student Submission Checklist",
]
BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
]
AUTO = re.compile(r"<!-- AUTO:[\w-]+ -->.*?<!-- /AUTO -->", re.S)
CSS = """
@page { size: A4; margin: 16mm 15mm; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 10pt; line-height: 1.45; color: #111; }
h1 { font-size: 20pt; margin: 0 0 8pt; } h2 { font-size: 14pt; margin: 16pt 0 6pt; border-bottom: 1px solid #bbb; padding-bottom: 2pt; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 9pt; page-break-inside: avoid; }
th, td { border: 1px solid #999; padding: 3pt 5pt; vertical-align: top; } th { background: #eee; }
img { max-width: 100%; max-height: 95mm; display: block; margin: 6pt auto 2pt; page-break-inside: avoid; }
em { color: #333; } blockquote { margin: 6pt 0; padding-left: 8pt; border-left: 3px solid #bbb; color: #333; }
pre { background: #f4f4f4; padding: 6pt; font-size: 8.5pt; white-space: pre-wrap; } code { font-family: Consolas, monospace; font-size: 9pt; }
hr { border: 0; border-top: 1px solid #ccc; margin: 12pt 0; } .draft { color: #b00; font-weight: bold; }
"""


def team_text(markdown_text: str) -> str:
    """The report without the auto-filled tables: what the team wrote (and the template's own wording)."""
    return AUTO.sub("", markdown_text)


def check(markdown_text: str, draft: bool) -> tuple[list[str], list[str]]:
    problems, notes = [], []
    slots = re.findall(r"\[(?:WRITE|TODO)[^\]]*\]", markdown_text)
    if slots and not draft:
        problems.append(f"{len(slots)} unfinished slot(s), for example: {slots[0][:90]}")
    headings = " ".join(re.findall(r"^#{1,4} (.+)$", markdown_text, re.M))
    for name in REQUIRED:
        if name not in headings:
            problems.append(f"Missing template section: {name}")
    for block in re.findall(r"```(\w*)\n(.*?)```", markdown_text, re.S):
        lines = [l for l in block[1].splitlines() if l.strip()]
        if len(lines) > 12 and "git clone" not in block[1]:
            problems.append(f"A code block with {len(lines)} lines. The template says not to paste raw code.")

    allowed: set[str] = set()
    for path in list((ROOT / "results").glob("*.json")) + [ROOT / "eval_P5_metrics.json"]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        stack = [data]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                stack.extend(item.values())
            elif isinstance(item, list):
                stack.extend(item)
            elif isinstance(item, (int, float)) and not isinstance(item, bool):
                for scale in (100, 1):
                    for digits in (0, 1, 2, 3):
                        allowed.add(f"{item * scale:.{digits}f}")
    own = team_text(markdown_text)
    unknown = sorted({m for m in re.findall(r"(\d+(?:\.\d+)?)\s*%", own) if m not in allowed and m.rstrip("0").rstrip(".") not in allowed})
    if unknown:
        notes.append("Percentages in your text that are in no result file (check each by hand): " + ", ".join(m + "%" for m in unknown))
    return problems, notes


def to_html(markdown_text: str, draft: bool) -> str:
    import markdown

    body = markdown.markdown(markdown_text, extensions=["tables", "fenced_code", "sane_lists"])
    banner = '<p class="draft">DRAFT: unfinished slots remain.</p>' if draft else ""
    base = (ROOT / "report").as_uri() + "/"
    return f'<!doctype html><html><head><meta charset="utf-8"><base href="{base}"><title>Hackathon Technical Solution Report: P5, T25</title><style>{CSS}</style></head><body>{banner}{body}</body></html>'


def find_browser() -> str:
    for path in BROWSERS:
        if Path(path).exists():
            return path
    raise SystemExit("No Edge or Chrome found. Install one, or open report/report_P5_T25.html in any browser and print it to PDF.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draft", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if not REPORT.exists():
        raise SystemExit("report/report_P5_T25.md doesn't exist. Run: python scripts/build_report_facts.py")
    text = REPORT.read_text(encoding="utf-8")
    problems, notes = check(text, args.draft)
    for note in notes:
        print("NOTE:", note)
    for problem in problems:
        print("PROBLEM:", problem)
    if problems:
        print("Not built. Fix the problems above, or use --draft to preview.")
        sys.exit(1)
    if args.check_only:
        print("Checks passed.")
        return

    page = PDF.with_suffix(".html")
    page.write_text(to_html(text, args.draft), encoding="utf-8")
    with tempfile.TemporaryDirectory() as profile:  # a throwaway profile, so an Edge you have open doesn't swallow the command
        subprocess.run([find_browser(), "--headless=new", "--disable-gpu", f"--user-data-dir={profile}", "--no-pdf-header-footer", "--allow-file-access-from-files",
                        "--virtual-time-budget=20000", f"--print-to-pdf={PDF}", page.as_uri()], check=True, capture_output=True, timeout=180)
    if not PDF.exists():
        raise SystemExit("The browser did not produce a PDF.")
    pages = ""
    try:
        from pypdf import PdfReader

        pages = f", {len(PdfReader(str(PDF)).pages)} pages"
    except Exception:
        pass
    print(f"Wrote {PDF.relative_to(ROOT)} ({PDF.stat().st_size / 1e3:.0f} KB{pages})" + (" [DRAFT]" if args.draft else ""))


if __name__ == "__main__":
    main()
