"""Unit tests for src/core/nightwelding's issue selection and PR-safety invariants.

These mock github_adapter's subprocess boundary (gh/git) so they run without
network access or a real LLM.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.validate_commit_messages import normalize_title, validate_subject
from src.core.nightwelding import github_adapter


class _FakeCompleted(SimpleNamespace):
    pass


def test_repro_commit_subject_passes_commit_message_policy() -> None:
    # Regression test: runner.py used to commit the reproduction test with
    # `f"test: add reproduction test for #{issue_number} (nightwelding)"`,
    # which embeds a bare '#<digits>' and trips the repo's own commit-msg
    # hook ("issue or PR numbers are not allowed in commit subjects"),
    # making every Nightwelding run fail right after writing a passing
    # reproduction test. The subject must use the issue number without a
    # leading '#'.
    subject = "test: add reproduction test for issue 539 (nightwelding)"

    assert validate_subject(subject, "reproduction commit") == []


def test_normalize_title_lowercases_natural_language_capitalization() -> None:
    # Regression test: GitHub issue titles routinely capitalize env var names,
    # exception classes, and acronyms (e.g. "TODAY", "KeyError"), but the
    # commit-message policy requires an all-lowercase summary. normalize_title
    # used to preserve the issue title's casing verbatim and then reject it,
    # so nightwelding could never derive a commit title for issues like #550.
    title = "fix: missing TODAY and STATUS env vars cause KeyError in roadmap issue workflow"

    normalized = normalize_title(title)

    assert normalized == "fix: missing today and status env vars cause keyerror in roadmap issue workflow"
    assert validate_subject(normalized, "normalized title") == []


def test_list_candidate_issues_filters_by_label_and_open_pr(monkeypatch) -> None:
    issues = [
        {"number": 1, "labels": [{"name": "auto-fix-failed"}]},
        {"number": 2, "labels": [{"name": "auto-fix-failed"}, {"name": "nightwelding-failed"}]},
        {"number": 3, "labels": [{"name": "enhancement"}]},
        {"number": 4, "labels": [{"name": "auto-fix-failed"}]},
    ]
    open_prs_by_issue = {
        4: [{"headRefName": "nightwelding/4-12345"}],
    }

    def fake_run(cmd, cwd=None, check=True):
        if cmd[:3] == ["gh", "issue", "list"]:
            return _FakeCompleted(returncode=0, stdout=json.dumps(issues), stderr="")
        if cmd[:3] == ["gh", "pr", "list"]:
            # The adapter re-queries PRs per candidate; return the same full set
            # each time and let the caller filter by branch prefix.
            all_prs = [pr for prs in open_prs_by_issue.values() for pr in prs]
            return _FakeCompleted(returncode=0, stdout=json.dumps(all_prs), stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(github_adapter, "_run", fake_run)

    candidates = github_adapter.list_candidate_issues(
        repo="acme/widgets",
        backlog_label="auto-fix-failed",
        exclude_labels=["nightwelding-draft-opened", "nightwelding-failed"],
    )

    # #1: has the backlog label, no exclude label, no open PR -> eligible.
    # #2: excluded via nightwelding-failed label.
    # #3: doesn't carry the backlog label at all.
    # #4: has the backlog label but already has an open nightwelding/4-... PR.
    assert candidates == [1]


def test_open_draft_pr_requires_opencode_generated_marker(monkeypatch) -> None:
    monkeypatch.setattr(github_adapter, "find_open_pr", lambda repo, branch, base: None)

    with pytest.raises(github_adapter.GitHubAdapterError):
        github_adapter.open_draft_pr(
            repo="acme/widgets",
            base_branch="main",
            branch="nightwelding/1-123",
            title="fix: something",
            body="This PR body is missing the required marker.",
        )


def test_create_worktree_checks_out_branch_outside_repo_root(monkeypatch, tmp_path) -> None:
    # Regression test for #574: nightwelding used to `git checkout -B` directly
    # in the invoking working tree, so a failed/overlapping run left it dirty
    # and two concurrent runs would stomp on each other's checkout. It must
    # instead create an isolated `git worktree` whose path lives outside
    # `repo_root` (under ~/.sparkleforge/nightwelding-worktrees), so the
    # caller's own working tree is never touched.
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    calls = []

    def fake_run(cmd, cwd=None, check=True):
        calls.append((cmd, cwd))
        return _FakeCompleted(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(github_adapter, "_run", fake_run)

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    worktree_dir = github_adapter.create_worktree(repo_root, "nightwelding/1-123", "main")

    assert repo_root not in worktree_dir.parents
    assert worktree_dir == tmp_path / ".sparkleforge" / "nightwelding-worktrees" / "nightwelding-1-123"
    worktree_add_calls = [cmd for cmd, _ in calls if cmd[:3] == ["git", "worktree", "add"]]
    assert worktree_add_calls == [
        ["git", "worktree", "add", "-B", "nightwelding/1-123", str(worktree_dir), "origin/main"]
    ]
    # git worktree add/fetch must run against repo_root, never the new worktree.
    assert all(cwd == repo_root for _, cwd in calls)


def test_remove_worktree_never_raises_on_failure(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        github_adapter, "_run", lambda cmd, cwd=None, check=True: _FakeCompleted(returncode=1, stdout="", stderr="boom")
    )

    # Should not raise even though the underlying command "fails" (check=False).
    github_adapter.remove_worktree(tmp_path, tmp_path / "some-worktree")


def test_open_draft_pr_returns_existing_pr_without_creating_a_new_one(monkeypatch) -> None:
    monkeypatch.setattr(github_adapter, "find_open_pr", lambda repo, branch, base: "https://github.com/acme/widgets/pull/99")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("gh pr create should not be called when a PR already exists")

    monkeypatch.setattr(github_adapter, "_run", fail_if_called)

    url = github_adapter.open_draft_pr(
        repo="acme/widgets",
        base_branch="main",
        branch="nightwelding/1-123",
        title="fix: something",
        body="OpenCode-generated fix.",
    )

    assert url == "https://github.com/acme/widgets/pull/99"
