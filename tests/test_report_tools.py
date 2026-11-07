"""The report tools: the skeleton follows the template, re-running never touches the team's prose, and the checks catch problems."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


facts = load("build_report_facts")
pdf = load("build_report_pdf")


def test_skeleton_has_every_template_section_and_the_header_fields():
    text = facts.skeleton()
    assert text.splitlines()[0] == "# Hackathon Technical Solution Report"
    for field in ("Problem ID", "Problem Title", "Team Name", "Team Members", "GitHub Repository", "Live Demo / Deployment"):
        assert f"**{field}**" in text
    problems, _ = pdf.check(facts.refresh(text), draft=True)
    assert problems == []
    assert "1RUA24CSE0461" in text and "1RUA24CSE0453" in text


def test_skeleton_writes_no_explanation_for_the_team():
    text = facts.skeleton()
    for label in ("Core Technical Challenge", "Proposed Methodology", "Algorithms Utilized", "Model Architecture", "Fusion Strategy", "Root Cause & Mitigation"):
        line = next(l for l in text.splitlines() if label in l)
        assert "[WRITE" in line, label


def test_refreshing_changes_the_tables_and_never_the_teams_words():
    mine = "# Title\n\nMy own paragraph about the work, with 12 words.\n\n<!-- AUTO:table-main -->\nOLD\n<!-- /AUTO -->\n\nMy closing words.\n"
    out = facts.refresh(mine)
    assert "My own paragraph about the work, with 12 words." in out and "My closing words." in out
    assert "OLD" not in out and "| Approach / Model Variant |" in out
    assert facts.refresh(out) == out  # a second run changes nothing


def test_the_main_table_numbers_come_from_the_result_files():
    table = facts.table_main()
    assert "17.32%" in table and "92.18%" in table and "92.09%" in table


def test_build_refuses_unfinished_slots_but_allows_a_draft():
    text = facts.refresh(facts.skeleton())
    problems, _ = pdf.check(text, draft=False)
    assert any("unfinished slot" in p for p in problems)
    assert not any("unfinished" in p for p in pdf.check(text, draft=True)[0])


def test_a_missing_template_section_is_reported():
    text = facts.skeleton().replace("## 3. Experimental Benchmarks & Evaluation", "## Results")
    assert any("Experimental Benchmarks" in p for p in pdf.check(text, draft=True)[0])


def test_percentages_not_in_any_result_file_are_listed():
    base = facts.refresh(facts.skeleton())
    _, notes = pdf.check(base + "\nThe deployed model reaches 92.18% and a made-up 93.77%.\n", draft=True)
    joined = " ".join(notes)
    assert "93.77%" in joined and "92.18%" not in joined


def test_long_code_blocks_are_reported_but_the_quickstart_is_allowed():
    long_block = "```python\n" + "\n".join(f"x = {i}" for i in range(20)) + "\n```\n"
    assert any("code block" in p for p in pdf.check(facts.refresh(facts.skeleton()) + long_block, draft=True)[0])
    assert not any("code block" in p for p in pdf.check(facts.refresh(facts.skeleton()), draft=True)[0])
