"""Nightwelding abstract adapter interface and shared Git utilities.

Decouples Nightwelding from GitHub/gh CLI specifics, allowing different issue
trackers and VCS backends (GitHub, Local Git/Files, GitLab, etc.) to plug in.
"""

from __future__ import annotations

import secrets
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class IssueContext:
    number: int | str
    title: str
    url: str
    markdown: str


class NightweldingAdapterError(RuntimeError):
    """Raised when an adapter operation fails."""
    pass


class BaseNightweldingAdapter(ABC):
    """Abstract base adapter for Nightwelding issue tracking and change publication."""

    @abstractmethod
    def fetch_issue_context(self, issue_ref: int | str) -> IssueContext:
        """Retrieve issue title, description, and discussion as IssueContext."""
        pass

    @abstractmethod
    def default_base_branch(self) -> str:
        """Return the default base branch (e.g., 'main')."""
        pass

    @abstractmethod
    def list_candidate_issues(
        self,
        backlog_label: str,
        exclude_labels: List[str],
        limit: int = 100,
    ) -> List[int | str]:
        """Return a list of eligible issue identifiers."""
        pass

    @abstractmethod
    def publish_draft_change(
        self,
        repo_root: Path,
        base_branch: str,
        branch: str,
        title: str,
        body: str,
        issue_ref: int | str,
    ) -> str:
        """Publish the fix (e.g. open a Draft PR, or record a local branch/patch).

        Returns:
            URL or reference path of the published change.
        """
        pass

    @abstractmethod
    def report_success(
        self,
        issue_ref: int | str,
        pr_or_patch_ref: str,
    ) -> None:
        """Update issue state/labels/comments upon successful fix."""
        pass

    @abstractmethod
    def report_failure(
        self,
        issue_ref: int | str,
        reason: str,
        log: str = "",
    ) -> None:
        """Update issue state/labels/comments upon failure."""
        pass

    def normalize_commit_title(self, issue_title: str, repo_root: Path) -> str | None:
        """Derive a conventional-commit title from an issue title."""
        scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
        validator_script = scripts_dir / "validate_commit_messages.py"
        if not validator_script.exists():
            validator_script = repo_root / "scripts" / "validate_commit_messages.py"

        proc = subprocess.run(
            [sys.executable, str(validator_script), "--normalize-title", issue_title],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return None
        return proc.stdout.strip()

    def create_worktree(self, repo_root: Path, branch: str, base_branch: str) -> Path:
        """Create a disposable git worktree outside repo_root."""
        parent_dir = Path.home() / ".sparkleforge" / "nightwelding-worktrees"
        parent_dir.mkdir(parents=True, exist_ok=True)
        safe_branch = branch.replace("/", "-")
        worktree_dir = parent_dir / f"{safe_branch}-{secrets.token_hex(4)}"

        subprocess.run(
            ["git", "fetch", "origin", f"{base_branch}:{base_branch}"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        start_point = f"origin/{base_branch}"
        check = subprocess.run(
            ["git", "rev-parse", "--verify", start_point],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if check.returncode != 0:
            start_point = base_branch

        proc = subprocess.run(
            ["git", "worktree", "add", "-B", branch, str(worktree_dir), start_point],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise NightweldingAdapterError(
                f"git worktree add failed ({proc.returncode}): {proc.stderr.strip()}"
            )
        return worktree_dir

    def remove_worktree(self, repo_root: Path, worktree_dir: Path) -> None:
        """Cleanly remove a git worktree."""
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree_dir)],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        subprocess.run(
            ["git", "worktree", "prune"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
