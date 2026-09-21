from src.core.roadmap.cli_ux_audit import build_cli_ux_audit_prompt


def test_prompt_contains_required_sections_today_and_transcripts() -> None:
    transcripts = {"sparkleforge --help": "usage: sparkleforge ..."}
    prompt = build_cli_ux_audit_prompt("2026-09-21", transcripts)

    assert "2026-09-21" in prompt
    assert "sparkleforge --help" in prompt
    assert "usage: sparkleforge ..." in prompt
    assert "NO_ACTIONABLE_CLI_UX_ISSUES" in prompt
    for section in (
        "## Why now",
        "## Proposed change",
        "## Implementation notes",
        "## Acceptance criteria",
        "## Validation",
    ):
        assert section in prompt


def test_prompt_renders_multiple_transcripts_in_order() -> None:
    transcripts = {"cmd-a": "output-a", "cmd-b": "output-b"}
    prompt = build_cli_ux_audit_prompt("2026-09-21", transcripts)

    assert prompt.index("cmd-a") < prompt.index("cmd-b")
    assert "output-a" in prompt
    assert "output-b" in prompt
