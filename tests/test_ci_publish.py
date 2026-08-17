import subprocess

import pytest

from src.core.ci.publish import PublishResult, commit_push_and_open_pr


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _git("init", "-q", "-b", "main", cwd=repo_root)
    _git("config", "user.email", "test@test.com", cwd=repo_root)
    _git("config", "user.name", "test", cwd=repo_root)
    (repo_root / "README.md").write_text("hello\n")
    _git("add", "README.md", cwd=repo_root)
    _git("commit", "-q", "-m", "fix: initial commit", cwd=repo_root)
    return repo_root


def test_no_op_diff_skips_commit_push_and_pr(repo, monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("push/gh must not run when there is nothing to commit")

    monkeypatch.setattr("src.core.ci.publish.find_open_pr", fail_if_called)

    result = commit_push_and_open_pr(
        repo="acme/widgets",
        repo_root=repo,
        branch="chore/no-op",
        base_branch="main",
        commit_title="chore: nothing changed",
        paths=["README.md"],
        pr_title="chore: nothing changed",
        pr_body="body",
    )

    assert result == PublishResult(committed=False, pushed=False, pr_url=None, skipped_reason="nothing to commit")


def test_commit_pushes_and_opens_new_pr(repo, monkeypatch):
    (repo / "README.md").write_text("hello world\n")
    gh_calls = []

    def fake_run(cmd, cwd, check=True):
        if cmd[:2] == ["git", "push"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[0] == "gh":
            gh_calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, "https://github.com/acme/widgets/pull/1", "")
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=check)

    monkeypatch.setattr("src.core.ci.publish._run", fake_run)
    monkeypatch.setattr("src.core.ci.publish.find_open_pr", lambda repo, branch, base_branch: None)

    result = commit_push_and_open_pr(
        repo="acme/widgets",
        repo_root=repo,
        branch="chore/publish-test",
        base_branch="main",
        commit_title="chore: update readme",
        paths=["README.md"],
        pr_title="chore: update readme",
        pr_body="body",
        labels=["automated"],
    )

    assert result.committed is True
    assert result.pushed is True
    assert result.pr_url == "https://github.com/acme/widgets/pull/1"
    assert gh_calls, "expected gh pr create to run"
    assert "--label" in gh_calls[0] and "automated" in gh_calls[0]


def test_reuses_already_open_pr_instead_of_creating_duplicate(repo, monkeypatch):
    (repo / "README.md").write_text("hello world\n")

    def fake_run(cmd, cwd, check=True):
        if cmd[:2] == ["git", "push"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[0] == "gh":
            raise AssertionError("gh pr create must not run when a PR is already open")
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=check)

    monkeypatch.setattr("src.core.ci.publish._run", fake_run)
    monkeypatch.setattr(
        "src.core.ci.publish.find_open_pr",
        lambda repo, branch, base_branch: "https://github.com/acme/widgets/pull/42",
    )

    result = commit_push_and_open_pr(
        repo="acme/widgets",
        repo_root=repo,
        branch="chore/publish-test",
        base_branch="main",
        commit_title="chore: update readme",
        paths=["README.md"],
        pr_title="chore: update readme",
        pr_body="body",
    )

    assert result.pr_url == "https://github.com/acme/widgets/pull/42"
