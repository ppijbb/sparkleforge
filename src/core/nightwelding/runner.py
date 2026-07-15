"""Nightwelding orchestration: fetch issue -> gate -> implement -> draft PR.

Two entry points:
  run_nightwelding_issue(issue_number)  -- full pipeline for one issue.
  run_nightwelding_sweep(backlog_label) -- find eligible issues and run each.

The `auto-fix-failed` backlog is the default sweep target: it's the set of
issues the daytime opencode-auto-fix.yml pipeline already tried and gave up
on (and which its own 30-minute sweep explicitly excludes from ever being
retried automatically) -- confirmed non-overlapping with existing automation.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import List, Optional

from src.core.nightwelding import gate, github_adapter
from src.core.nightwelding.implement import implement_until_green
from src.core.nightwelding.models import (
    NightweldingItem,
    NightweldingQueue,
    NightweldingStatus,
)

logger = logging.getLogger(__name__)

DEFAULT_BACKLOG_LABEL = "auto-fix-failed"
DEFAULT_MAX_ITERATIONS = 4
DEFAULT_MAX_PER_RUN = 3


def _repo_slug() -> str:
    repo = os.getenv("GITHUB_REPOSITORY")
    if repo:
        return repo
    proc = github_adapter._run(["gh", "repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"], check=False)
    if proc.returncode == 0 and proc.stdout.strip():
        return proc.stdout.strip()
    raise github_adapter.GitHubAdapterError(
        "Could not determine the GitHub repository (set GITHUB_REPOSITORY or run inside a repo with `gh` configured)."
    )


async def run_nightwelding_issue(
    issue_number: int,
    repo_root: Path | None = None,
    repo: str | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    queue: NightweldingQueue | None = None,
) -> NightweldingItem:
    main_repo_root = repo_root or Path.cwd()
    repo_root = main_repo_root
    repo = repo or _repo_slug()
    queue = queue or NightweldingQueue()

    item = NightweldingItem(issue_number=issue_number, status=NightweldingStatus.WRITING_TEST)
    queue.upsert(item)

    worktree_dir: Path | None = None
    try:
        issue = github_adapter.fetch_issue_context(repo, issue_number)

        commit_title_impl = github_adapter.normalize_commit_title(issue.title, main_repo_root)
        if not commit_title_impl:
            return _fail(queue, item, repo, issue_number, f"Could not derive a valid commit title from issue title: {issue.title!r}")

        eligible, eligibility_reason = gate.is_reproducible_bug_eligible(issue.markdown)
        if not eligible:
            return _fail(queue, item, repo, issue_number, eligibility_reason)

        base_branch = os.getenv("NIGHTWELDING_BASE_BRANCH") or os.getenv("NIGHTSHIFT_BASE_BRANCH") or os.getenv("NIGHTSHIFT_BASE_BRANCH") or github_adapter.default_base_branch(repo)
        branch = f"nightwelding/{issue_number}-{int(time.time())}"
        worktree_dir = github_adapter.create_worktree(main_repo_root, branch, base_branch)
        repo_root = worktree_dir
        item.branch = branch
        queue.upsert(item)

        repro = await gate.write_reproduction_test(issue.markdown, repo_root=repo_root)
        if not repro.success:
            return _fail(queue, item, repo, issue_number, repro.reason, log=repro.red_output)

        item.status = NightweldingStatus.RED
        item.repro_test_files = repro.test_files
        queue.upsert(item)

        # Commit the reproduction test itself before attempting an implementation.
        github_adapter._run(["git", "add", "-u"], cwd=repo_root, check=False)
        untracked = github_adapter._run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=repo_root, check=False
        ).stdout.split("\0")
        untracked = [p for p in untracked if p]
        if untracked:
            github_adapter._run(["git", "add", "--", *untracked], cwd=repo_root, check=False)
        github_adapter._run(
            ["git", "commit", "-m", f"test: add reproduction test for issue {issue_number} (nightwelding)"],
            cwd=repo_root,
        )

        item.status = NightweldingStatus.IMPLEMENTING
        queue.upsert(item)

        implement_result = implement_until_green(
            issue_context=issue.markdown,
            repro_test_files=repro.test_files,
            repo_root=repo_root,
            commit_title=commit_title_impl,
            max_iterations=max_iterations,
        )
        if not implement_result.success:
            return _fail(queue, item, repo, issue_number, implement_result.reason, log=implement_result.log)

        item.status = NightweldingStatus.GREEN
        queue.upsert(item)

        pushed = github_adapter.push_branch(repo_root, branch, base_branch)
        if not pushed:
            return _fail(queue, item, repo, issue_number, "Nightwelding completed without commits.")

        body = (
            f"OpenCode-generated Nightwelding fix for {issue.url}.\n\n"
            "This PR was opened by Nightwelding, an overnight autonomous-implementation "
            "pipeline. It is intentionally a **Draft** and requires a human to review it "
            "and mark it ready before it can merge.\n\n"
            "## Reproduction\n\n"
            f"Test file(s): `{', '.join(repro.test_files)}`\n\n"
            "Red (before fix):\n```text\n" + repro.red_output[-2000:] + "\n```\n\n"
            "Green (after fix):\n```text\n" + implement_result.green_output[-2000:] + "\n```\n\n"
            f"Closes #{issue_number}"
        )
        pr_url = github_adapter.open_draft_pr(repo, base_branch, branch, commit_title_impl, body)
        item.pr_url = pr_url
        item.status = NightweldingStatus.DRAFT_OPENED
        queue.upsert(item)

        github_adapter.ensure_label(repo, *github_adapter.NIGHTWELDING_DRAFT_LABEL)
        github_adapter.add_labels(repo, issue_number, [github_adapter.NIGHTWELDING_DRAFT_LABEL[0]])
        github_adapter.remove_labels(repo, issue_number, [github_adapter.NIGHTWELDING_QUEUE_LABEL[0]])
        github_adapter.comment_on_issue(
            repo, issue_number,
            f"Nightwelding opened a Draft PR: {pr_url}. It requires human review — mark it ready for review, then merge manually.",
        )
        return item
    except Exception as exc:  # noqa: BLE001 - surfaced via the failure report below
        logger.exception("Nightwelding run failed for issue #%s", issue_number)
        return _fail(queue, item, repo, issue_number, str(exc))
    finally:
        if worktree_dir is not None:
            github_adapter.remove_worktree(main_repo_root, worktree_dir)


def _fail(
    queue: NightweldingQueue,
    item: NightweldingItem,
    repo: str,
    issue_number: int,
    reason: str,
    log: str = "",
) -> NightweldingItem:
    item.status = NightweldingStatus.FAILED
    item.failure_reason = reason
    queue.upsert(item)
    try:
        github_adapter.ensure_label(repo, *github_adapter.NIGHTWELDING_FAILED_LABEL)
        github_adapter.add_labels(repo, issue_number, [github_adapter.NIGHTWELDING_FAILED_LABEL[0]])
        github_adapter.remove_labels(repo, issue_number, [github_adapter.NIGHTWELDING_QUEUE_LABEL[0]])
        comment = (
            "Nightwelding could not complete this issue overnight.\n\n"
            f"Reason: {reason}\n\n"
        )
        if log:
            comment += f"Log (tail):\n```text\n{log[-3000:]}\n```\n\n"
        comment += "Re-add the `nightwelding-queue` label to retry after addressing the root cause."
        github_adapter.comment_on_issue(repo, issue_number, comment)
    except Exception:
        logger.exception("Nightwelding: failed to report failure for issue #%s", issue_number)
    return item


async def run_nightwelding_sweep(
    backlog_label: str = DEFAULT_BACKLOG_LABEL,
    limit: int = 100,
    max_per_run: int = DEFAULT_MAX_PER_RUN,
    repo_root: Path | None = None,
    repo: str | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> List[NightweldingItem]:
    repo_root = repo_root or Path.cwd()
    repo = repo or _repo_slug()
    queue = NightweldingQueue()

    candidates = github_adapter.list_candidate_issues(
        repo,
        backlog_label=backlog_label,
        exclude_labels=[
            github_adapter.NIGHTWELDING_DRAFT_LABEL[0],
            github_adapter.NIGHTWELDING_FAILED_LABEL[0],
        ],
        limit=limit,
    )

    results: List[NightweldingItem] = []
    for issue_number in candidates[:max_per_run]:
        result = await run_nightwelding_issue(
            issue_number,
            repo_root=repo_root,
            repo=repo,
            max_iterations=max_iterations,
            queue=queue,
        )
        results.append(result)
    return results
