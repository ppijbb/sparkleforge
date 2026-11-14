from src.core.roadmap.cli_ux_audit import (
    DEFAULT_AUDIT_COMMANDS,
    E2E_AUDIT_COMMANDS,
    build_cli_ux_audit_prompt,
    get_audit_commands,
)


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


def test_get_audit_commands_controls_e2e_query_by_flag_and_env() -> None:
    assert get_audit_commands(include_e2e=False) == DEFAULT_AUDIT_COMMANDS
    with_e2e = get_audit_commands(include_e2e=True)
    assert with_e2e == DEFAULT_AUDIT_COMMANDS + E2E_AUDIT_COMMANDS
    assert any("--query" in cmd for cmd in with_e2e)

    assert get_audit_commands(env={}) == DEFAULT_AUDIT_COMMANDS
    assert get_audit_commands(env={"GOOGLE_API_KEY": "test-key"}) == with_e2e


def test_prompt_flags_runtime_crashes_and_provider_fallback_spam() -> None:
    prompt = build_cli_ux_audit_prompt("2026-11-12", {"sparkleforge --query hi": "[exit 1]"})
    assert "[exit != 0]" in prompt
    assert "[FAIL]" in prompt
    assert "402" in prompt and "404" in prompt and "429" in prompt
