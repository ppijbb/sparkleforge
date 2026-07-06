"""Tests for the OpenCode GitHub worker prompt construction."""

from scripts.opencode_github_worker import build_prompt


def _common_kwargs(**overrides):
    base = {
        "snapshot": "src/foo.py",
        "status": "",
        "issue_context": "Fix the bug",
        "file_contents_str": "",
        "extra_context": "",
        "tool_context": "",
    }
    base.update(overrides)
    return base


def test_build_prompt_allows_file_request_when_not_forced():
    prompt = build_prompt(**_common_kwargs(force_diff=False))
    assert "file request" in prompt


def test_build_prompt_omits_file_request_when_force_diff_true():
    prompt = build_prompt(**_common_kwargs(force_diff=True))
    assert "file request" not in prompt
    assert "unified git diff" in prompt
