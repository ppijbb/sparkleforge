"""Nightwelding orchestration: fetch issue -> gate -> implement -> draft change.

Two entry points:
  run_nightwelding_issue(issue_ref)   -- full pipeline for one issue.
  run_nightwelding_sweep(backlog_label) -- find eligible issues and run each.

Pluggable adapter architecture: works with GitHub (`gh` CLI) as well as Local Git/Files.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import List

from src.core.nightwelding import gate, github_adapter
from src.core.nightwelding.adapter import BaseNightweldingAdapter
from src.core.nightwelding.github_adapter import GitHubAdapter
from src.core.nightwelding.implement import implement_until_green
from src.core.nightwelding.local_adapter import LocalGitAdapter
from src.core.nightwelding.models import (
    NightweldingItem,
    NightweldingQueue,
    NightweldingStatus,
)

logger = logging.getLogger(__name__)

DEFAULT_BACKLOG_LABEL = "auto-fix-failed"
DEFAULT_MAX_ITERATIONS = 4
DEFAULT_MAX_PER_RUN = 3


def _resolve_adapter(
    issue_ref: int | str | None,
    repo_root: Path,
    repo: str | None = None,
    explicit_adapter: BaseNightweldingAdapter | None = None,
    provider: str | None = None,
) -> BaseNightweldingAdapter:
    """Resolve the appropriate adapter for Nightwelding."""
    if explicit_adapter is not None:
        return explicit_adapter

    if provider == "local":
        return LocalGitAdapter(repo_root=repo_root)
    elif provider == "github":
        return GitHubAdapter(repo=repo, repo_root=repo_root)

    # Auto-detection heuristic
    if issue_ref is not None:
        ref_str = str(issue_ref).strip()
        if (
            Path(ref_str).is_file()
            or ref_str.startswith("file://")
            or ref_str.startswith("local://")
        ):
            return LocalGitAdapter(repo_root=repo_root)

    try:
        gh_adapter = GitHubAdapter(repo=repo, repo_root=repo_root)
        gh_adapter._get_repo()
        return gh_adapter
    except Exception:
        logger.debug("GitHub adapter unavailable, falling back to LocalGitAdapter")
        return LocalGitAdapter(repo_root=repo_root)


async def run_nightwelding_issue(
    issue_number: int | str,
    repo_root: Path | None = None,
    repo: str | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    queue: NightweldingQueue | None = None,
    adapter: BaseNightweldingAdapter | None = None,
    provider: str | None = None,
) -> NightweldingItem:
    main_repo_root = repo_root or Path.cwd()
    repo_root = main_repo_root
    queue = queue or NightweldingQueue()
    active_adapter = _resolve_adapter(
        issue_number, main_repo_root, repo=repo, explicit_adapter=adapter, provider=provider
    )

    item = NightweldingItem(issue_number=issue_number, status=NightweldingStatus.WRITING_TEST)
    queue.upsert(item)

    worktree_dir: Path | None = None
    try:
        issue = active_adapter.fetch_issue_context(issue_number)

        commit_title_impl = active_adapter.normalize_commit_title(issue.title, main_repo_root)
        if not commit_title_impl:
            msg = f"Could not derive a valid commit title from issue title: {issue.title!r}"
            return _fail(queue, item, active_adapter, issue_number, msg)

        eligible, eligibility_reason = gate.is_reproducible_bug_eligible(issue.markdown)
        if not eligible:
            return _fail(queue, item, active_adapter, issue_number, eligibility_reason)

        base_branch = (
            os.getenv("NIGHTWELDING_BASE_BRANCH")
            or os.getenv("NIGHTSHIFT_BASE_BRANCH")
            or active_adapter.default_base_branch()
        )
        safe_num = str(issue_number).replace("/", "-").replace("\\", "-")
        branch = f"nightwelding/{safe_num}-{int(time.time())}-{os.urandom(2).hex()}"
        worktree_dir = active_adapter.create_worktree(main_repo_root, branch, base_branch)
        repo_root = worktree_dir
        item.branch = branch
        queue.upsert(item)

        repro = await gate.write_reproduction_test(issue.markdown, repo_root=repo_root)
        if not repro.success:
            return _fail(
                queue, item, active_adapter, issue_number, repro.reason, log=repro.red_output
            )

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
            [
                "git", "commit", "-m",
                f"test: add reproduction test for issue {issue_number} (nightwelding)",
            ],
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
            return _fail(
                queue,
                item,
                active_adapter,
                issue_number,
                implement_result.reason,
                log=implement_result.log,
            )

        item.status = NightweldingStatus.GREEN
        queue.upsert(item)

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

        # Push branch if remote is configured
        push_ok = active_adapter.push_branch(repo_root, branch, base_branch)
        if not push_ok:
            return _fail(queue, item, active_adapter, issue_number, "Nightwelding completed without commits.")

        published_ref = active_adapter.publish_draft_change(
            repo_root=repo_root,
            base_branch=base_branch,
            branch=branch,
            title=commit_title_impl,
            body=body,
            issue_ref=issue_number,
        )
        item.pr_url = published_ref
        item.status = NightweldingStatus.DRAFT_OPENED
        queue.upsert(item)

        active_adapter.report_success(issue_number, published_ref)
        return item
    except Exception as exc:  # noqa: BLE001 - surfaced via the failure report below
        logger.exception("Nightwelding run failed for issue #%s", issue_number)
        return _fail(queue, item, active_adapter, issue_number, str(exc))
    finally:
        if worktree_dir is not None:
            active_adapter.remove_worktree(main_repo_root, worktree_dir)


def _fail(
    queue: NightweldingQueue,
    item: NightweldingItem,
    adapter: BaseNightweldingAdapter,
    issue_number: int | str,
    reason: str,
    log: str = "",
) -> NightweldingItem:
    item.status = NightweldingStatus.FAILED
    item.failure_reason = reason
    queue.upsert(item)
    try:
        adapter.report_failure(issue_number, reason, log=log)
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
    adapter: BaseNightweldingAdapter | None = None,
    provider: str | None = None,
) -> List[NightweldingItem]:
    repo_root = repo_root or Path.cwd()
    queue = NightweldingQueue()
    active_adapter = _resolve_adapter(
        None, repo_root, repo=repo, explicit_adapter=adapter, provider=provider
    )

    candidates = active_adapter.list_candidate_issues(
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
            adapter=active_adapter,
        )
        results.append(result)
    return results
