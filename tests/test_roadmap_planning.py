from src.core.roadmap.planning import build_fallback_roadmap, build_issue_body

CONTEXT_MD = """### Open pull requests
- #101 Fix the thing [fix/101-thing -> main; draft=false; merge=CLEAN; updated=2026-01-01; labels=]

### Open issues
- #202 Some backlog issue [updated=2026-01-01; labels=; assignees=]

### Anvil roadmap status
- Open milestone: #5 Milestone: Phase M
"""


def test_fallback_prefers_anvil_target_when_present():
    roadmap = build_fallback_roadmap(
        context_md=CONTEXT_MD,
        anvil_target="11|next up",
        rc="1",
        invalid_reason="",
        output_bytes="0",
        console_bytes="10",
        error_bytes="20",
    )
    assert roadmap.startswith("# Advance Anvil roadmap sub-issue 11\n")
    assert "Anvil roadmap sub-issue #11: next up" in roadmap
    assert "SparkleForge CLI fallback metadata" in roadmap
    assert "## Why now" in roadmap
    assert "## Proposed change" in roadmap
    assert "## Implementation notes" in roadmap
    assert "## Acceptance criteria" in roadmap
    assert "## Validation" in roadmap


def test_fallback_falls_back_to_open_pr_when_no_anvil_target():
    roadmap = build_fallback_roadmap(
        context_md=CONTEXT_MD,
        anvil_target="",
        rc="0",
        invalid_reason="CLI output was already a failure report",
        output_bytes="5",
        console_bytes="5",
        error_bytes="5",
    )
    assert roadmap.startswith("# Unblock pull request 101\n")
    assert "PR #101: Fix the thing" in roadmap
    assert "Invalid output reason: CLI output was already a failure report." in roadmap


def test_fallback_falls_back_to_generic_when_nothing_open():
    roadmap = build_fallback_roadmap(
        context_md="### Open pull requests\n\n### Open issues\n\n### Anvil roadmap status\n",
        anvil_target="",
        rc="1",
        invalid_reason="",
        output_bytes="0",
        console_bytes="0",
        error_bytes="0",
    )
    assert roadmap.startswith("# Triage issue 13\n")


def test_issue_body_generated_status_has_no_occurrence_log():
    body = build_issue_body(today="2026-01-02", status="generated", roadmap_text="# Roadmap\n", previous_body="")
    assert "## Occurrence log" not in body
    assert "<!-- sparkleforge-daily-roadmap:2026-01-02 -->" in body
    assert body.strip().endswith("# Roadmap")


def test_issue_body_fallback_status_appends_and_dedups_occurrence_log():
    previous = "## Occurrence log\n- 2026-01-01\n- 2025-12-31\n"
    body = build_issue_body(today="2026-01-02", status="fallback", roadmap_text="# Fallback\n", previous_body=previous)
    assert "## Occurrence log" in body
    assert "- 2026-01-01" in body
    assert "- 2025-12-31" in body
    assert "- 2026-01-02" in body


def test_issue_body_fallback_status_does_not_duplicate_todays_entry():
    previous = "## Occurrence log\n- 2026-01-02\n"
    body = build_issue_body(today="2026-01-02", status="fallback", roadmap_text="# Fallback\n", previous_body=previous)
    assert body.count("- 2026-01-02") == 1


def test_issue_body_fallback_caps_occurrence_log_to_last_ten():
    previous_days = [f"2025-12-{d:02d}" for d in range(1, 21)]
    previous = "## Occurrence log\n" + "\n".join(f"- {d}" for d in previous_days) + "\n"
    body = build_issue_body(today="2026-01-02", status="fallback", roadmap_text="# Fallback\n", previous_body=previous)
    occurrence_lines = [line for line in body.splitlines() if line.startswith("- 2")]
    assert len(occurrence_lines) == 10
    assert occurrence_lines[-1] == "- 2026-01-02"
