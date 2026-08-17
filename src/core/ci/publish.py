"""Shared "commit a result, push a branch, open a PR" helper for CI report jobs.

Before this existed, sparkleforge-daily-roadmap.yml's Anvil doc sync,
scenario-eval.yml's record-history job, and swebench-weekly.yml's
record-report job each hand-rolled this sequence independently, and drifted:
different git identity strategies, only one of the three guarded against an
empty diff, only one force-pushed, only one checked for an already-open PR
before creating a duplicate. This consolidates the sequence to one behavior:
fixed bot identity, always guard the no-op-diff case, always force-with-lease
push (safe for these branches -- they're either freshly derived per run via a
SHA/date suffix, or intentionally reused across reruns), always reuse an
already-open PR for the same branch instead of erroring on a duplicate.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from src.core.nightwelding.github_adapter import find_open_pr

_BOT_NAME = "sparkleforge-bot"
_BOT_EMAIL = "actions@github.com"


@dataclass
class PublishResult:
    committed: bool
    pushed: bool
    pr_url: str | None
    skipped_reason: str | None = None


def _run(cmd: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    if check and proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed ({proc.returncode}): {proc.stderr.strip()}")
    return proc


def commit_push_and_open_pr(
    *,
    repo: str,
    repo_root: Path,
    branch: str,
    base_branch: str,
    commit_title: str,
    paths: list[str],
    pr_title: str,
    pr_body: str,
    commit_body: str | None = None,
    labels: list[str] | None = None,
) -> PublishResult:
    """Commit `paths` on `branch`, push it, and open (or reuse) a PR into `base_branch`."""
    _run(["git", "config", "user.name", _BOT_NAME], cwd=repo_root)
    _run(["git", "config", "user.email", _BOT_EMAIL], cwd=repo_root)
    _run(["git", "checkout", "-B", branch], cwd=repo_root)
    _run(["git", "add", "--", *paths], cwd=repo_root)

    diff_check = _run(["git", "diff", "--cached", "--quiet"], cwd=repo_root, check=False)
    if diff_check.returncode == 0:
        return PublishResult(committed=False, pushed=False, pr_url=None, skipped_reason="nothing to commit")

    message = commit_title if not commit_body else f"{commit_title}\n\n{commit_body}"
    _run(["git", "commit", "-m", message], cwd=repo_root)
    _run(["git", "push", "--force-with-lease", "origin", branch], cwd=repo_root)

    pr_url = find_open_pr(repo, branch, base_branch)
    if not pr_url:
        cmd = [
            "gh", "pr", "create",
            "--repo", repo,
            "--base", base_branch,
            "--head", branch,
            "--title", pr_title,
            "--body", pr_body,
        ]
        if labels:
            cmd.extend(["--label", ",".join(labels)])
        proc = _run(cmd, cwd=repo_root)
        pr_url = proc.stdout.strip()

    return PublishResult(committed=True, pushed=True, pr_url=pr_url)
