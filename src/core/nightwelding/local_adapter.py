"""Local Git & File-based adapter for Nightwelding.

Allows Nightwelding to run entirely locally without requiring GitHub CLI (`gh`),
GH_TOKEN, or network access. Issues can be supplied via local Markdown/text
files, directories, or direct text input.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import List
import yaml

from src.core.nightwelding.adapter import (
    BaseNightweldingAdapter,
    IssueContext,
)

logger = logging.getLogger(__name__)


class LocalGitAdapter(BaseNightweldingAdapter):
    """Local Git and file-based adapter for offline/non-GitHub environments."""

    def __init__(self, issues_dir: Path | str | None = None, repo_root: Path | None = None):
        self.repo_root = Path(repo_root or Path.cwd())
        if issues_dir:
            self.issues_dir = Path(issues_dir)
        else:
            self.issues_dir = self.repo_root / ".sparkleforge" / "issues"

    def default_base_branch(self) -> str:
        """Determine default branch from local git configuration."""
        proc = subprocess.run(
            ["git", "symbolic-ref", "--short", "refs/remotes/origin/HEAD"],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            ref = proc.stdout.strip()
            return ref.replace("origin/", "")

        # Fall back to current branch or main
        proc2 = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc2.returncode == 0 and proc2.stdout.strip():
            return proc2.stdout.strip()
        return "main"

    def fetch_issue_context(self, issue_ref: int | str) -> IssueContext:
        """Read issue context from local file or inline description."""
        ref_str = str(issue_ref).strip()
        path = Path(ref_str)

        # 1. Direct file path check
        if not path.is_file() and self.issues_dir.is_dir():
            candidate = self.issues_dir / ref_str
            if candidate.is_file():
                path = candidate
            elif (self.issues_dir / f"{ref_str}.md").is_file():
                path = self.issues_dir / f"{ref_str}.md"

        if path.is_file():
            content = path.read_text(encoding="utf-8")
            title = self._extract_title(content, default=path.stem)
            return IssueContext(
                number=ref_str,
                title=title,
                url=f"file://{path.resolve()}",
                markdown=content,
            )

        # 2. Raw text prompt / issue description
        title = self._extract_title(ref_str, default="local-issue")
        return IssueContext(
            number=ref_str,
            title=title,
            url=f"local://{title.replace(' ', '_')}",
            markdown=ref_str,
        )

    def _extract_title(self, text: str, default: str) -> str:
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("#"):
                clean = line.lstrip("#").strip()
                if clean:
                    return clean
            elif line and not line.startswith("```"):
                return line[:100]
        return default

    def list_candidate_issues(
        self,
        backlog_label: str = "auto-fix-failed",
        exclude_labels: List[str] | None = None,
        limit: int = 100,
    ) -> List[int | str]:
        """List local issue files from issues directory with label filtering."""
        if not self.issues_dir.is_dir():
            return []

        exclude_labels = exclude_labels or []
        candidates: List[int | str] = []
        for file_path in sorted(self.issues_dir.glob("*.md")):
            content = file_path.read_text(encoding="utf-8")
            labels = None
            if content.startswith("---"):
                parts = content.split("---")
                if len(parts) > 2:
                    meta = yaml.safe_load(parts[1]) or {}
                    if "labels" in meta:
                        labels = meta.get("labels") or []
            # Local issue files aren't required to declare frontmatter labels
            # at all -- only apply backlog_label/exclude_labels filtering to
            # files that opt in by declaring a `labels:` field; undecorated
            # files are candidates by default.
            if labels is None:
                candidates.append(str(file_path.resolve()))
            elif backlog_label in labels and not any(l in labels for l in exclude_labels):
                candidates.append(str(file_path.resolve()))
            if len(candidates) >= limit:
                break
        return candidates

    def push_branch(self, repo_root: Path, branch: str, base_branch: str) -> bool:
        return True

    def commit_changes(self, repo_root: Path, message: str) -> None:
        subprocess.run(["git", "add", "-u"], cwd=repo_root, capture_output=True, text=True, check=False)
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=repo_root, capture_output=True, text=True, check=False,
        ).stdout.split("\0")
        untracked = [p for p in untracked if p]
        if untracked:
            subprocess.run(["git", "add", "--", *untracked], cwd=repo_root, capture_output=True, text=True, check=False)
        subprocess.run(["git", "commit", "-m", message], cwd=repo_root, capture_output=True, text=True, check=False)

    def publish_draft_change(
        self,
        repo_root: Path,
        base_branch: str,
        branch: str,
        title: str,
        body: str,
        issue_ref: int | str,
    ) -> str:
        """Record the local fix branch and export a patch file."""
        patches_dir = Path.home() / ".sparkleforge" / "nightwelding" / "patches"
        patches_dir.mkdir(parents=True, exist_ok=True)

        safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(issue_ref))
        patch_file = patches_dir / f"fix_{safe_name}.patch"

        diff_proc = subprocess.run(
            ["git", "diff", f"{base_branch}..HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if diff_proc.stdout:
            patch_file.write_text(diff_proc.stdout, encoding="utf-8")

        summary_file = patches_dir / f"fix_{safe_name}_summary.md"
        summary_file.write_text(f"# {title}\n\nBranch: `{branch}`\n\n{body}", encoding="utf-8")

        logger.info("Local fix generated on branch '%s'. Patch saved: %s", branch, patch_file)
        return f"local://branch/{branch} (patch: {patch_file})"

    def report_success(
        self,
        issue_ref: int | str,
        pr_or_patch_ref: str,
    ) -> None:
        logger.info("Nightwelding succeeded for local issue %s: %s", issue_ref, pr_or_patch_ref)

    def report_failure(
        self,
        issue_ref: int | str,
        reason: str,
        log: str = "",
    ) -> None:
        logger.warning("Nightwelding failed for local issue %s: %s", issue_ref, reason)
