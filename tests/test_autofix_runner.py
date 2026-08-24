import subprocess
from pathlib import Path

import pytest

from src.core.autofix import runner as autofix_runner
from src.core.autofix.runner import run_autofix_repair_loop


def _proc(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


class ScriptedRun:
    """Fake for src.core.autofix.runner._run that dispatches on the command shape.

    `fix_issue_outcomes` / `verify_outcomes` / `self_verify_outcomes` are consumed
    one-per-call (in order), so tests can script a different result per attempt.
    """

    def __init__(self, fix_issue_outcomes=None, verify_outcomes=None, self_verify_outcomes=None):
        self.fix_issue_outcomes = list(fix_issue_outcomes or [_proc(0)])
        self.verify_outcomes = list(verify_outcomes or [_proc(0)])
        self.self_verify_outcomes = list(self_verify_outcomes or [_proc(0)])
        self.calls = []

    def __call__(self, cmd, cwd, timeout=None):
        self.calls.append(cmd)
        if "fix-issue" in cmd:
            return self.fix_issue_outcomes.pop(0) if len(self.fix_issue_outcomes) > 1 else self.fix_issue_outcomes[0]
        if cmd[:2] == ["git", "add"]:
            return _proc(0)
        if cmd[:3] == ["git", "ls-files", "--others"]:
            return _proc(0, stdout="")
        if cmd[:3] == ["git", "diff", "--cached"]:
            return _proc(1)  # non-zero => there IS a staged diff
        if any("validate_commit_messages.py" in part for part in cmd):
            return _proc(0)
        if cmd[:2] == ["git", "commit"]:
            return _proc(0)
        if cmd[:2] == ["bash", "-lc"] and cmd[2] == "self-verify":
            return self.self_verify_outcomes.pop(0) if len(self.self_verify_outcomes) > 1 else self.self_verify_outcomes[0]
        if cmd[:2] == ["bash", "-lc"]:
            return self.verify_outcomes.pop(0) if len(self.verify_outcomes) > 1 else self.verify_outcomes[0]
        raise AssertionError(f"unexpected command in test: {cmd}")


@pytest.fixture
def issue_context(tmp_path):
    path = tmp_path / "issue-context.md"
    path.write_text("fix the bug", encoding="utf-8")
    return path


def test_self_verify_failure_aborts_immediately_without_retry(tmp_path, issue_context, monkeypatch):
    scripted = ScriptedRun(self_verify_outcomes=[_proc(1, stderr="tests failed")])
    monkeypatch.setattr(autofix_runner, "_run", scripted)
    monkeypatch.setattr(autofix_runner.patch_ops, "repository_change_signature", lambda: ("M foo.py",))

    result = run_autofix_repair_loop(
        issue_context_path=issue_context,
        repo_root=tmp_path,
        commit_title="fix: something",
        max_iterations=3,
        self_verify_command="self-verify",
    )

    assert result.success is False
    assert result.attempts == 1
    fix_issue_calls = [c for c in scripted.calls if "fix-issue" in c]
    assert len(fix_issue_calls) == 1


def test_verify_command_failure_retries_then_fails_at_max_iterations(tmp_path, issue_context, monkeypatch):
    scripted = ScriptedRun(verify_outcomes=[_proc(1, stdout="still broken")] * 3)
    monkeypatch.setattr(autofix_runner, "_run", scripted)
    monkeypatch.setattr(autofix_runner.patch_ops, "repository_change_signature", lambda: ("M foo.py",))

    result = run_autofix_repair_loop(
        issue_context_path=issue_context,
        repo_root=tmp_path,
        commit_title="fix: something",
        max_iterations=3,
        verify_command="verify",
    )

    assert result.success is False
    assert result.attempts == 3
    fix_issue_calls = [c for c in scripted.calls if "fix-issue" in c]
    assert len(fix_issue_calls) == 3


def test_success_short_circuits_before_max_iterations(tmp_path, issue_context, monkeypatch):
    scripted = ScriptedRun(verify_outcomes=[_proc(0)])
    monkeypatch.setattr(autofix_runner, "_run", scripted)
    monkeypatch.setattr(autofix_runner.patch_ops, "repository_change_signature", lambda: ("M foo.py",))

    result = run_autofix_repair_loop(
        issue_context_path=issue_context,
        repo_root=tmp_path,
        commit_title="fix: something",
        max_iterations=5,
        verify_command="verify",
    )

    assert result.success is True
    assert result.attempts == 1


def test_writes_worker_error_and_verify_logs_to_disk(tmp_path, issue_context, monkeypatch):
    scripted = ScriptedRun(verify_outcomes=[_proc(0, stdout="all good")])
    monkeypatch.setattr(autofix_runner, "_run", scripted)
    monkeypatch.setattr(autofix_runner.patch_ops, "repository_change_signature", lambda: ("M foo.py",))

    result = run_autofix_repair_loop(
        issue_context_path=issue_context,
        repo_root=tmp_path,
        commit_title="fix: something",
        max_iterations=3,
        verify_command="verify",
    )

    assert result.success is True
    assert (tmp_path / "opencode-worker-error.log").exists()
    assert (tmp_path / "opencode-verify.log").exists()
    assert "all good" in (tmp_path / "opencode-verify.log").read_text(encoding="utf-8")


def test_silent_commit_failure_is_reported_not_swallowed(tmp_path, issue_context, monkeypatch):
    """git commit can fail after staging succeeds (pre-commit hook rejection,
    identity misconfiguration, etc.). Regression test for the loop treating
    that as success and letting a later step discover "no commits" instead."""

    class CommitFailingRun(ScriptedRun):
        def __call__(self, cmd, cwd, timeout=None):
            if cmd[:2] == ["git", "commit"]:
                return _proc(1, stderr="pre-commit hook rejected commit")
            return super().__call__(cmd, cwd, timeout=timeout)

    scripted = CommitFailingRun(verify_outcomes=[_proc(0)])
    monkeypatch.setattr(autofix_runner, "_run", scripted)
    monkeypatch.setattr(autofix_runner.patch_ops, "repository_change_signature", lambda: ("M foo.py",))

    result = run_autofix_repair_loop(
        issue_context_path=issue_context,
        repo_root=tmp_path,
        commit_title="fix: something",
        max_iterations=3,
        verify_command="verify",
    )

    assert result.success is False
    assert "pre-commit hook rejected commit" in result.reason
    # No verification should run against a commit that was never made.
    verify_calls = [c for c in scripted.calls if c[:2] == ["bash", "-lc"] and c[2] == "verify"]
    assert verify_calls == []
