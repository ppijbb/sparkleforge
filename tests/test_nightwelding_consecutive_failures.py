"""Tests that consecutive-failure tracking persists across runs and triggers
escalation at the threshold (issue #1615)."""

from __future__ import annotations

import asyncio

from src.core.nightwelding import escalation, runner
from src.core.nightwelding.adapter import IssueContext
from src.core.nightwelding.models import NightweldingItem, NightweldingQueue, NightweldingStatus


class _AlwaysFailsAdapter:
    """Fails at the commit-title step -- the earliest failure point in
    run_nightwelding_issue that doesn't need a real worktree/git checkout."""

    def fetch_issue_context(self, issue_ref):
        return IssueContext(number=issue_ref, title="fix: x", url="local://x", markdown="# fix: x")

    def normalize_commit_title(self, title, repo_root):
        return None

    def report_failure(self, issue_ref, reason, log=""):
        pass


def test_fail_increments_consecutive_failures(tmp_path):
    queue = NightweldingQueue(storage_path=tmp_path)
    adapter = _AlwaysFailsAdapter()

    item = asyncio.run(
        runner.run_nightwelding_issue("77", repo_root=tmp_path, queue=queue, adapter=adapter)
    )

    assert item.status == NightweldingStatus.FAILED
    assert item.consecutive_failures == 1
    assert queue.get("77").consecutive_failures == 1


def test_consecutive_failures_carries_over_and_escalates_at_threshold(tmp_path, monkeypatch):
    queue = NightweldingQueue(storage_path=tmp_path)
    queue.upsert(
        NightweldingItem(
            issue_number="77", status=NightweldingStatus.FAILED, consecutive_failures=2
        )
    )
    escalated: list[int] = []
    monkeypatch.setattr(
        escalation, "escalate_issue", lambda adapter, num, item: escalated.append(item.consecutive_failures)
    )
    adapter = _AlwaysFailsAdapter()

    item = asyncio.run(
        runner.run_nightwelding_issue("77", repo_root=tmp_path, queue=queue, adapter=adapter)
    )

    assert item.consecutive_failures == 3
    assert escalated == [3]


class _AlwaysSucceedsAdapter:
    """Stubs every step of run_nightwelding_issue's happy path."""

    def fetch_issue_context(self, issue_ref):
        return IssueContext(number=issue_ref, title="fix: x", url="local://x", markdown="# fix: x")

    def normalize_commit_title(self, title, repo_root):
        return "fix: x"

    def default_base_branch(self):
        return "main"

    def create_worktree(self, repo_root, branch, base_branch):
        return repo_root

    def commit_changes(self, repo_root, message):
        pass

    def push_branch(self, repo_root, branch, base_branch):
        return True

    def publish_draft_change(self, **kwargs):
        return "https://example.com/pr/1"

    def report_success(self, issue_ref, pr_or_patch_ref):
        pass

    def report_failure(self, issue_ref, reason, log=""):
        pass

    def remove_worktree(self, main_repo_root, worktree_dir):
        pass


def test_success_resets_consecutive_failures(tmp_path, monkeypatch):
    from src.core.nightwelding.gate import ReproResult
    from src.core.nightwelding.implement import ImplementResult

    monkeypatch.setattr(
        runner.gate,
        "is_reproducible_bug_eligible",
        lambda md: (True, ""),
    )

    async def _fake_write_repro(*a, **kw):
        return ReproResult(success=True, test_files=["tests/test_x.py"], red_output="red")

    monkeypatch.setattr(runner.gate, "write_reproduction_test", _fake_write_repro)
    monkeypatch.setattr(
        runner,
        "implement_until_green",
        lambda **kw: ImplementResult(success=True, green_output="green"),
    )

    queue = NightweldingQueue(storage_path=tmp_path)
    queue.upsert(
        NightweldingItem(
            issue_number="77", status=NightweldingStatus.FAILED, consecutive_failures=2
        )
    )

    item = asyncio.run(
        runner.run_nightwelding_issue(
            "77", repo_root=tmp_path, queue=queue, adapter=_AlwaysSucceedsAdapter()
        )
    )

    assert item.status == NightweldingStatus.DRAFT_OPENED
    assert item.consecutive_failures == 0
    assert queue.get("77").consecutive_failures == 0
