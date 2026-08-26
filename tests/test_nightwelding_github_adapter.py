"""GitHubAdapter.report_failure must never post failure detail as an issue comment.

Regression test for the incident where a Nightwelding run's internal failure
reason ("Reproduction diff touched no files.") was posted verbatim as a public
comment on a real, external repo's issue.
"""

from __future__ import annotations

from src.core.nightwelding import github_adapter


def test_report_failure_labels_but_never_comments(monkeypatch):
    calls: list[tuple[str, tuple]] = []

    monkeypatch.setattr(github_adapter, "ensure_label", lambda *a: calls.append(("ensure_label", a)))
    monkeypatch.setattr(github_adapter, "add_labels", lambda *a: calls.append(("add_labels", a)))
    monkeypatch.setattr(github_adapter, "remove_labels", lambda *a: calls.append(("remove_labels", a)))
    monkeypatch.setattr(github_adapter, "comment_on_issue", lambda *a: calls.append(("comment_on_issue", a)))

    adapter = github_adapter.GitHubAdapter(repo="qwp0905/lfdb")
    adapter.report_failure(293, "Reproduction diff touched no files.", log="some internal trace")

    kinds = [name for name, _ in calls]
    assert "comment_on_issue" not in kinds
    assert "add_labels" in kinds
    assert calls[kinds.index("add_labels")][1] == ("qwp0905/lfdb", 293, [github_adapter.NIGHTWELDING_FAILED_LABEL[0]])
