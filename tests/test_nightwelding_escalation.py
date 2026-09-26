"""Tests for the Nightwelding repeated-failure escalation queue (issue #1615)."""

from __future__ import annotations

from src.core.nightwelding import escalation, github_adapter
from src.core.nightwelding.local_adapter import LocalGitAdapter
from src.core.nightwelding.models import NightweldingItem, NightweldingStatus


def _failed_item(issue_number="42", consecutive_failures=3) -> NightweldingItem:
    return NightweldingItem(
        issue_number=issue_number,
        status=NightweldingStatus.FAILED,
        failure_reason="boom",
        consecutive_failures=consecutive_failures,
    )


def test_should_escalate_below_threshold_is_false():
    assert escalation.should_escalate(_failed_item(consecutive_failures=2)) is False


def test_should_escalate_at_threshold_is_true():
    assert escalation.should_escalate(_failed_item(consecutive_failures=3)) is True


def test_should_escalate_custom_threshold():
    assert escalation.should_escalate(_failed_item(consecutive_failures=5), threshold=5) is True
    assert escalation.should_escalate(_failed_item(consecutive_failures=4), threshold=5) is False


def test_escalate_issue_labels_and_comments_on_github_adapter(monkeypatch):
    calls: list[tuple[str, tuple]] = []
    monkeypatch.setattr(escalation, "ensure_label", lambda *a: calls.append(("ensure_label", a)))
    monkeypatch.setattr(escalation, "add_labels", lambda *a: calls.append(("add_labels", a)))
    monkeypatch.setattr(escalation, "comment_on_issue", lambda *a: calls.append(("comment_on_issue", a)))

    adapter = github_adapter.GitHubAdapter(repo="owner/repo")
    escalation.escalate_issue(adapter, 42, _failed_item(issue_number=42))

    kinds = [name for name, _ in calls]
    assert kinds.count("ensure_label") == 2
    assert "add_labels" in kinds
    labels_call = calls[kinds.index("add_labels")][1]
    assert labels_call == (
        "owner/repo",
        42,
        [github_adapter.HUMAN_REVIEW_NEEDED_LABEL[0], github_adapter.AUTO_FIX_BACKOFF_LABEL[0]],
    )
    assert "comment_on_issue" in kinds


def test_escalate_issue_is_a_noop_for_local_adapter(tmp_path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(escalation, "add_labels", lambda *a: calls.append("add_labels"))

    adapter = LocalGitAdapter(repo_root=tmp_path)
    escalation.escalate_issue(adapter, 42, _failed_item(issue_number=42))

    assert calls == []


def test_escalate_issue_is_a_noop_for_non_numeric_issue_ref(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(escalation, "add_labels", lambda *a: calls.append("add_labels"))

    adapter = github_adapter.GitHubAdapter(repo="owner/repo")
    escalation.escalate_issue(adapter, "local://some-file.md", _failed_item(issue_number="local://some-file.md"))

    assert calls == []


def test_escalate_issue_never_raises_on_underlying_failure(monkeypatch):
    def _boom(*a):
        raise RuntimeError("gh is down")

    monkeypatch.setattr(escalation, "ensure_label", _boom)

    adapter = github_adapter.GitHubAdapter(repo="owner/repo")
    escalation.escalate_issue(adapter, 42, _failed_item(issue_number=42))  # must not raise
