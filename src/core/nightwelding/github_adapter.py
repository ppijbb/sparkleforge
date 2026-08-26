"""Thin `gh`/`git` subprocess adapter for Nightwelding.

Consistent with the rest of this project: no GitHub API client is built here,
just subprocess calls to the `gh` CLI and `git`, exactly like the existing
GitHub Actions workflows and src/core/ci/fix_issue.py already do.

The PR body always contains the literal substring "OpenCode-generated" and
branches always use the "nightwelding/" prefix — both are load-bearing for
staying Draft-only forever: gemini-assistant.yml's `code-review` job (and,
transitively, `merge-decision`, which needs `code-review` to succeed) skips
any PR whose body contains that substring, and its separate
`auto-merge-ready-fix-prs` scheduled sweep only matches `fix/`/`chore/`
branch prefixes.
"""

from __future__ import annotations

import json
import logging
import secrets
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core import patch_ops
from src.core.nightwelding.adapter import (
    BaseNightweldingAdapter,
    IssueContext,
    NightweldingAdapterError,
)

logger = logging.getLogger(__name__)

NIGHTWELDING_DRAFT_LABEL = ("nightwelding-draft-opened", "5319E7", "Nightwelding opened a Draft PR; human must review and mark it ready before it can merge.")
NIGHTWELDING_FAILED_LABEL = ("nightwelding-failed", "B60205", "Nightwelding could not reproduce the issue, or could not make the reproduction test pass.")
NIGHTWELDING_QUEUE_LABEL = ("nightwelding-queue", "1D76DB", "Queued for Nightwelding's overnight autonomous-implementation pipeline.")


class GitHubAdapterError(NightweldingAdapterError):
    pass


class GitHubAdapter(BaseNightweldingAdapter):
    """GitHub & `gh` CLI implementation of BaseNightweldingAdapter."""

    def __init__(self, repo: str | None = None, repo_root: Path | None = None):
        self.repo = repo
        self.repo_root = Path(repo_root or Path.cwd())

    def _get_repo(self) -> str:
        if self.repo:
            return self.repo
        import os
        env_repo = os.getenv("GITHUB_REPOSITORY")
        if env_repo:
            return env_repo
        proc = _run(["gh", "repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"], check=False)
        if proc.returncode == 0 and proc.stdout.strip():
            self.repo = proc.stdout.strip()
            return self.repo
        raise GitHubAdapterError(
            "Could not determine the GitHub repository (set GITHUB_REPOSITORY or run inside a repo with `gh` configured)."
        )

    def fetch_issue_context(self, issue_ref: int | str) -> IssueContext:
        issue_num = int(issue_ref) if str(issue_ref).isdigit() else issue_ref
        return fetch_issue_context(self._get_repo(), issue_num)

    def _get_pr_repo(self) -> str:
        """Repo a Draft PR should target: the upstream parent if `_get_repo()`
        is a fork, otherwise the same repo. Detected via `gh`'s own fork
        metadata (`isFork`/`parent`), so this isn't specific to any one repo.
        """
        repo = self._get_repo()
        proc = _run(
            ["gh", "repo", "view", repo, "--json", "isFork,parent",
             "--jq", 'if .isFork then .parent.owner.login + "/" + .parent.name else empty end'],
            check=False,
        )
        parent = proc.stdout.strip() if proc.returncode == 0 else ""
        return parent or repo

    def default_base_branch(self) -> str:
        return default_base_branch(self._get_pr_repo())

    def list_candidate_issues(
        self,
        backlog_label: str = "auto-fix-failed",
        exclude_labels: List[str] | None = None,
        limit: int = 100,
    ) -> List[int | str]:
        return list_candidate_issues(
            self._get_repo(),
            backlog_label=backlog_label,
            exclude_labels=exclude_labels or [NIGHTWELDING_DRAFT_LABEL[0], NIGHTWELDING_FAILED_LABEL[0]],
            limit=limit,
        )

    def push_branch(self, repo_root: Path, branch: str, base_branch: str) -> bool:
        return push_branch(repo_root, branch, base_branch)

    def commit_changes(self, repo_root: Path, message: str) -> None:
        commit_changes(repo_root, message)

    def publish_draft_change(
        self,
        repo_root: Path,
        base_branch: str,
        branch: str,
        title: str,
        body: str,
        issue_ref: int | str,
    ) -> str:
        issue_num = int(issue_ref) if str(issue_ref).isdigit() else None
        if issue_ref is not None and issue_num is None:
            raise GitHubAdapterError(
                f"issue_ref {issue_ref!r} is not a numeric GitHub issue number; "
                "cannot open a PR with a 'Closes #N' keyword or apply issue labels."
            )
        fork_repo = self._get_repo()
        pr_repo = self._get_pr_repo()
        head = branch
        if pr_repo != fork_repo:
            # Cross-repo PR (fork -> upstream): head needs the fork owner
            # prefix, and a bare "Closes #N" would target the wrong repo's
            # issue N -- qualify it so it links instead of misfiring, and
            # skip label/milestone lookup since those live on fork_repo.
            head = f"{fork_repo.split('/')[0]}:{branch}"
            if issue_num is not None:
                body = body.replace(f"Closes #{issue_num}", f"Closes {fork_repo}#{issue_num}")
            issue_num = None
        return open_draft_pr(
            pr_repo,
            base_branch,
            head,
            title,
            body,
            issue_number=issue_num,
        )

    def report_success(
        self,
        issue_ref: int | str,
        pr_or_patch_ref: str,
    ) -> None:
        if str(issue_ref).isdigit():
            repo = self._get_repo()
            num = int(issue_ref)
            ensure_label(repo, NIGHTWELDING_DRAFT_LABEL[0], NIGHTWELDING_DRAFT_LABEL[1], NIGHTWELDING_DRAFT_LABEL[2])
            add_labels(repo, num, [NIGHTWELDING_DRAFT_LABEL[0]])
            remove_labels(repo, num, [NIGHTWELDING_QUEUE_LABEL[0]])
            comment_on_issue(
                repo,
                num,
                f"Nightwelding opened a Draft PR: {pr_or_patch_ref}. It requires human review — mark it ready for review, then merge manually.",
            )

    def report_failure(
        self,
        issue_ref: int | str,
        reason: str,
        log: str = "",
    ) -> None:
        # Failure/error detail is never posted to the issue -- it's an internal
        # diagnostic, not something to surface on someone else's public repo.
        # It stays in our own logs (and the local NightweldingQueue) only; only
        # the `nightwelding-failed` label (state, not text) is visible on GitHub.
        logger.warning("Nightwelding failed for issue #%s: %s", issue_ref, reason)
        if log:
            logger.debug("Nightwelding failure log for issue #%s:\n%s", issue_ref, log)
        if str(issue_ref).isdigit():
            repo = self._get_repo()
            num = int(issue_ref)
            ensure_label(repo, NIGHTWELDING_FAILED_LABEL[0], NIGHTWELDING_FAILED_LABEL[1], NIGHTWELDING_FAILED_LABEL[2])
            add_labels(repo, num, [NIGHTWELDING_FAILED_LABEL[0]])
            remove_labels(repo, num, [NIGHTWELDING_QUEUE_LABEL[0]])


def _run(cmd: List[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    if check and proc.returncode != 0:
        raise GitHubAdapterError(f"{' '.join(cmd)} failed ({proc.returncode}): {proc.stderr.strip()}")
    return proc


def default_base_branch(repo: str) -> str:
    proc = _run(["gh", "repo", "view", repo, "--json", "defaultBranchRef", "--jq", ".defaultBranchRef.name"])
    return proc.stdout.strip()


def normalize_commit_title(issue_title: str, repo_root: Path) -> str | None:
    """Derive a conventional-commit title from an issue title.

    Returns None if the title can't be normalized into a valid commit
    subject (reuses scripts/validate_commit_messages.py's existing
    --normalize-title logic rather than reimplementing it).
    """
    proc = subprocess.run(
        [sys.executable, "scripts/validate_commit_messages.py", "--normalize-title", issue_title],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def fetch_issue_context(repo: str, issue_number: int | str) -> IssueContext:
    title = _run(["gh", "issue", "view", str(issue_number), "--repo", repo, "--json", "title", "--jq", ".title"]).stdout.strip()
    url = _run(["gh", "issue", "view", str(issue_number), "--repo", repo, "--json", "url", "--jq", ".url"]).stdout.strip()
    jq = (
        '"# " + .title + "\\n\\n" + (.body // "") + "\\n\\n" + .url + '
        '"\\n\\n## Recent issue comments\\n\\n" + '
        "([(.comments // [])[-8:][]? | \"### \" + .author.login + \" at \" + .createdAt + \"\\n\" + (.body // \"\")] | join(\"\\n\\n\"))"
    )
    markdown = _run(
        ["gh", "issue", "view", str(issue_number), "--repo", repo, "--json", "title,body,url,comments", "--jq", jq]
    ).stdout
    return IssueContext(number=issue_number, title=title, url=url, markdown=markdown)


def list_candidate_issues(
    repo: str,
    backlog_label: str,
    exclude_labels: List[str],
    limit: int = 100,
) -> List[int]:
    """Open issues carrying `backlog_label`, not carrying any of `exclude_labels`,
    and with no existing open PR from a `nightwelding/$N-` branch.
    """
    proc = _run(
        ["gh", "issue", "list", "--repo", repo, "--state", "open", "--limit", str(limit), "--json", "number,labels"]
    )
    issues = json.loads(proc.stdout or "[]")
    candidates: List[int] = []
    for issue in issues:
        labels = {label["name"] for label in issue.get("labels", [])}
        if backlog_label not in labels and NIGHTWELDING_QUEUE_LABEL[0] not in labels:
            continue
        if labels & set(exclude_labels):
            continue
        candidates.append(issue["number"])

    eligible: List[int] = []
    for number in candidates:
        pr_proc = _run(
            ["gh", "pr", "list", "--repo", repo, "--state", "open", "--limit", str(limit), "--json", "headRefName"]
        )
        open_prs = json.loads(pr_proc.stdout or "[]")
        if any(
            pr["headRefName"].startswith(f"nightshift/{number}-")
            or pr["headRefName"].startswith(f"nightwelding/{number}-")
            for pr in open_prs
        ):
            continue
        eligible.append(number)
    return eligible


def create_branch(repo_root: Path, branch: str, base_branch: str) -> None:
    _run(["git", "fetch", "origin", base_branch, "--depth=1"], cwd=repo_root)
    _run(["git", "checkout", "-B", branch, f"origin/{base_branch}"], cwd=repo_root)


def create_worktree(repo_root: Path, branch: str, base_branch: str) -> Path:
    """Create a git worktree for `branch` and return its path.

    The worktree directory lives under ~/.sparkleforge/nightwelding-worktrees,
    outside repo_root, so caller's working tree is never touched. A random
    suffix prevents concurrent runs handling the same branch slug from
    colliding on the same filesystem path.
    """
    slug = branch.replace("/", "-")
    worktree_dir = Path.home() / ".sparkleforge" / "nightwelding-worktrees" / f"{slug}-{secrets.token_hex(2)}"
    worktree_dir.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "fetch", "origin", base_branch, "--depth=1"], cwd=repo_root)
    _run(["git", "worktree", "add", "-B", branch, str(worktree_dir), f"origin/{base_branch}"], cwd=repo_root)
    return worktree_dir


def remove_worktree(repo_root: Path | str, worktree_dir: Path | None = None) -> None:
    """Remove a git worktree created by `create_worktree`.

    The path is derived from the branch name plus a random suffix, so each
    run gets a unique worktree directory; this only removes the specific
    worktree passed in.
    """
    if worktree_dir is None:
        worktree_dir = Path(repo_root)
        repo_root = Path.cwd()
    proc = _run(["git", "worktree", "remove", "--force", str(worktree_dir)], cwd=repo_root, check=False)
    if proc.returncode != 0:
        print(f"Warning: git worktree remove failed for {worktree_dir}: {proc.stderr.strip()}")
    try:
        shutil.rmtree(worktree_dir)
        _run(["git", "worktree", "prune"], cwd=Path.cwd(), check=False)
    except OSError as e:
        print(f"Error: Failed to clean up worktree directory {worktree_dir}: {e}")


def commit_changes(repo_root: Path, message: str) -> None:
    """Stage all changes (tracked modifications + untracked files) and commit.

    Never stages SparkleForge's own runtime scratch files (issue-context.md,
    opencode.patch, logs, ...) -- they must not leak into a real repo's history.
    """
    _run(["git", "add", "-u"], cwd=repo_root, check=False)
    untracked = _run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=repo_root, check=False
    ).stdout.split("\0")
    untracked = [p for p in untracked if p and not patch_ops.is_runtime_scratch_path(p)]
    if untracked:
        _run(["git", "add", "--", *untracked], cwd=repo_root, check=False)
    _run(["git", "commit", "-m", message], cwd=repo_root)


def push_branch(repo_root: Path, branch: str, base_branch: str) -> bool:
    proc = subprocess.run(
        ["git", "log", "--oneline", f"origin/{base_branch}..HEAD"],
        cwd=repo_root, capture_output=True, text=True, check=False,
    )
    if not proc.stdout.strip():
        return False
    _run(["git", "push", "--set-upstream", "origin", branch], cwd=repo_root)
    return True


def ensure_label(repo: str, name: str, color: str, description: str) -> None:
    subprocess.run(
        ["gh", "label", "create", name, "--repo", repo, "--color", color, "--description", description, "--force"],
        capture_output=True, text=True, check=False,
    )


def add_labels(repo: str, issue_number: int, labels: List[str]) -> None:
    subprocess.run(
        ["gh", "issue", "edit", str(issue_number), "--repo", repo, "--add-label", ",".join(labels)],
        capture_output=True, text=True, check=False,
    )


def remove_labels(repo: str, issue_number: int, labels: List[str]) -> None:
    subprocess.run(
        ["gh", "issue", "edit", str(issue_number), "--repo", repo, "--remove-label", ",".join(labels)],
        capture_output=True, text=True, check=False,
    )


def comment_on_issue(repo: str, issue_number: int, body: str) -> None:
    subprocess.run(
        ["gh", "issue", "comment", str(issue_number), "--repo", repo, "--body", body],
        capture_output=True, text=True, check=False,
    )


def find_open_pr(repo: str, branch: str, base_branch: str) -> str | None:
    proc = _run(
        ["gh", "pr", "list", "--repo", repo, "--head", branch, "--base", base_branch, "--state", "open", "--json", "url"],
        check=False,
    )
    items = json.loads(proc.stdout or "[]")
    return items[0]["url"] if items else None


def _validate_issue_metadata_schema(meta: Dict[str, Any]) -> None:
    """Validate the shape of issue metadata returned by `gh issue view`.

    Nightwelding's auto-fix pipeline (issue #917) was failing because callers
    assumed `labels`, `milestone`, and `projectItems` always had the expected
    nested shape. This guard rejects malformed payloads early with a clear
    error instead of letting a downstream KeyError/TypeError abort the run.
    """
    if not isinstance(meta, dict):
        raise GitHubAdapterError("Issue metadata payload must be a JSON object.")

    labels = meta.get("labels", [])
    if labels is not None and not isinstance(labels, list):
        raise GitHubAdapterError("Issue metadata 'labels' must be a list.")
    for label in labels or []:
        if not isinstance(label, dict) or "name" not in label:
            raise GitHubAdapterError("Issue metadata label entries must be objects with a 'name' field.")

    milestone = meta.get("milestone")
    if milestone is not None and not isinstance(milestone, dict):
        raise GitHubAdapterError("Issue metadata 'milestone' must be an object or null.")

    project_items = meta.get("projectItems", [])
    if project_items is not None and not isinstance(project_items, list):
        raise GitHubAdapterError("Issue metadata 'projectItems' must be a list.")
    for item in project_items or []:
        if not isinstance(item, dict):
            raise GitHubAdapterError("Issue metadata projectItems entries must be objects.")


def fetch_issue_metadata(repo: str, issue_number: int) -> Dict[str, Any]:
    """Fetch labels, milestone, and projectItems for an issue."""
    try:
        proc = _run([
            "gh", "issue", "view", str(issue_number),
            "--repo", repo,
            "--json", "labels,milestone,projectItems"
        ])
        meta = json.loads(proc.stdout or "{}")
        _validate_issue_metadata_schema(meta)
        return meta
    except Exception:
        try:
            # Fallback if projectItems triggers permission/scope errors
            proc = _run([
                "gh", "issue", "view", str(issue_number),
                "--repo", repo,
                "--json", "labels,milestone"
            ])
            meta = json.loads(proc.stdout or "{}")
            _validate_issue_metadata_schema(meta)
            return meta
        except Exception as ex:
            import logging
            logging.getLogger(__name__).warning(f"Failed to fetch issue metadata: {ex}")
            return {}


def open_draft_pr(
    repo: str,
    base_branch: str,
    branch: str,
    title: str,
    body: str,
    issue_number: Optional[int] = None
) -> str:
    """Open a Draft PR. `body` MUST already contain the literal substring
    'OpenCode-generated' — callers are responsible for that (see module
    docstring for why).
    """
    if "OpenCode-generated" not in body:
        raise GitHubAdapterError("PR body must contain the literal substring 'OpenCode-generated'.")

    existing = find_open_pr(repo, branch, base_branch)
    if existing:
        return existing

    cmd = [
        "gh", "pr", "create",
        "--repo", repo,
        "--draft",
        "--base", base_branch,
        "--head", branch,
        "--title", title,
        "--body", body
    ]

    if issue_number:
        meta = fetch_issue_metadata(repo, issue_number)
        
        # 1. Add Labels
        labels = [label["name"] for label in meta.get("labels", []) if label.get("name")]
        if labels:
            cmd.extend(["--label", ",".join(labels)])
            
        # 2. Add Milestone
        milestone = meta.get("milestone")
        if milestone and isinstance(milestone, dict) and milestone.get("title"):
            cmd.extend(["--milestone", milestone["title"]])
            
        # 3. Add Projects
        project_items = meta.get("projectItems", [])
        for item in project_items:
            project = item.get("project")
            if project and isinstance(project, dict) and project.get("title"):
                cmd.extend(["--project", project["title"]])

    proc = _run(cmd)
    return proc.stdout.strip()

async def create_subissues(parent_issue_number: str, sub_issues: list[dict]) -> None:
    """Create sub-issues and link them to the parent."""
    created_numbers = []
    for sub in sub_issues[:6]: # Limit to 6
        title = sub["title"]
        body = sub["body"]
        # Check for existing sub-issue to avoid duplicates
        existing = subprocess.run(
            ["gh", "issue", "list", "--search", f"{title} in:title", "--json", "number"],
            capture_output=True, text=True
        ).stdout
        if existing and json.loads(existing):
            continue
            
        cmd = ["gh", "issue", "create", "--title", title, "--body", body]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        url = proc.stdout.strip()
        num = url.split("/")[-1]
        created_numbers.append(num)

    if created_numbers:
        comment = "Decomposed into sub-issues: " + ", ".join([f"#{n}" for n in created_numbers])
        subprocess.run(["gh", "issue", "comment", parent_issue_number, "--body", comment], check=True)
        subprocess.run(["gh", "issue", "edit", parent_issue_number, "--add-label", "roadmap-decomposed"], check=True)
