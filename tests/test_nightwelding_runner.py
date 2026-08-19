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
    # caller's own working tree is never touched. The path also carries a
    # random suffix so concurrent runs handling the same branch slug resolve
    # to distinct directories instead of colliding (#1091).
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
    assert worktree_dir.parent == tmp_path / ".sparkleforge" / "nightwelding-worktrees"
    assert worktree_dir.name.startswith("nightwelding-1-123-")
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


def test_publish_draft_change_targets_upstream_parent_when_repo_is_a_fork(monkeypatch) -> None:
    # Nightwelding used to open the Draft PR against `self.repo` unconditionally,
    # so dogfooding a fork (e.g. ppijbb/lfdb, forked from qwp0905/lfdb) only ever
    # produced PRs inside the fork itself -- never a contribution upstream. A PR
    # against a fork must instead target the fork's parent, with the fork owner
    # prefixed onto --head, and a bare "Closes #N" rewritten to a fully-qualified
    # cross-repo reference so it can't accidentally reference an unrelated issue
    # number in the upstream repo.
    monkeypatch.setattr(github_adapter, "find_open_pr", lambda repo, branch, base: None)

    calls = []

    def fake_run(cmd, cwd=None, check=True):
        calls.append(cmd)
        if cmd[:3] == ["gh", "repo", "view"]:
            return _FakeCompleted(returncode=0, stdout="qwp0905/lfdb\n", stderr="")
        if cmd[:3] == ["gh", "pr", "create"]:
            return _FakeCompleted(returncode=0, stdout="https://github.com/qwp0905/lfdb/pull/300\n", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(github_adapter, "_run", fake_run)

    adapter = github_adapter.GitHubAdapter(repo="ppijbb/lfdb")
    url = adapter.publish_draft_change(
        repo_root=Path("/repo"),
        base_branch="main",
        branch="nightwelding/42-abc",
        title="fix: something",
        body="OpenCode-generated fix.\n\nCloses #42",
        issue_ref=42,
    )

    assert url == "https://github.com/qwp0905/lfdb/pull/300"
    create_cmd = next(cmd for cmd in calls if cmd[:3] == ["gh", "pr", "create"])
    assert "--repo" in create_cmd and create_cmd[create_cmd.index("--repo") + 1] == "qwp0905/lfdb"
    assert "--head" in create_cmd and create_cmd[create_cmd.index("--head") + 1] == "ppijbb:nightwelding/42-abc"
    body_arg = create_cmd[create_cmd.index("--body") + 1]
    assert "Closes ppijbb/lfdb#42" in body_arg
    assert "Closes #42" not in body_arg
    # issue_number must not be forwarded cross-repo: labels/milestones live on
    # the fork, not the upstream repo, and `gh issue view 42 --repo qwp0905/lfdb`
    # would look up the wrong issue entirely.
    assert "--label" not in create_cmd


def test_publish_draft_change_stays_same_repo_when_not_a_fork(monkeypatch) -> None:
    monkeypatch.setattr(github_adapter, "find_open_pr", lambda repo, branch, base: None)

    calls = []

    def fake_run(cmd, cwd=None, check=True):
        calls.append(cmd)
        if cmd[:3] == ["gh", "repo", "view"]:
            return _FakeCompleted(returncode=0, stdout="\n", stderr="")  # not a fork -> empty
        if cmd[:3] == ["gh", "issue", "view"]:
            return _FakeCompleted(returncode=0, stdout="{}", stderr="")
        if cmd[:3] == ["gh", "pr", "create"]:
            return _FakeCompleted(returncode=0, stdout="https://github.com/acme/widgets/pull/1\n", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(github_adapter, "_run", fake_run)

    adapter = github_adapter.GitHubAdapter(repo="acme/widgets")
    adapter.publish_draft_change(
        repo_root=Path("/repo"),
        base_branch="main",
        branch="nightwelding/7-abc",
        title="fix: something",
        body="OpenCode-generated fix.\n\nCloses #7",
        issue_ref=7,
    )

    create_cmd = next(cmd for cmd in calls if cmd[:3] == ["gh", "pr", "create"])
    assert create_cmd[create_cmd.index("--head") + 1] == "nightwelding/7-abc"
    assert "Closes #7" in create_cmd[create_cmd.index("--body") + 1]


def test_run_nightwelding_issue_accepts_provider_kwarg(monkeypatch, tmp_path) -> None:
    # Regression test for #1431: run_nightwelding_issue() must accept a
    # `provider` keyword argument so the sweep call site
    # `run_nightwelding_issue(..., provider=provider)` does not raise
    # `TypeError: ... got an unexpected keyword argument 'provider'`.
    import asyncio
    import inspect

    from src.core.nightwelding import runner
    from src.core.nightwelding.adapter import IssueContext

    sig = inspect.signature(runner.run_nightwelding_issue)
    assert "provider" in sig.parameters
    assert sig.parameters["provider"].default is None

    captured: dict[str, object] = {}

    class _StubAdapter:
        def fetch_issue_context(self, issue_ref):
            captured["issue_ref"] = issue_ref
            return IssueContext(
                number=issue_ref,
                title="fix: sample",
                url="local://sample",
                markdown="# fix: sample",
            )

    monkeypatch.setattr(runner, "_resolve_adapter", lambda *a, **kw: _StubAdapter())
    monkeypatch.setattr(
        runner.gate, "is_reproducible_bug_eligible", lambda md: (False, "stub-ineligible")
    )

    item = asyncio.run(
        runner.run_nightwelding_issue(
            "local-issue", repo_root=tmp_path, provider="local", adapter=_StubAdapter()
        )
    )
    assert captured["issue_ref"] == "local-issue"
    assert item.failure_reason == "stub-ineligible"
