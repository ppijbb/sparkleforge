"""Thin `gh`/`git` subprocess adapter for Nightwelding.
import json

Consistent with the rest of this project: no GitHub API client is built here,
just subprocess calls to the `gh` CLI and `git`, exactly like the existing
GitHub Actions workflows and scripts/opencode_github_worker.py already do.

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
import subprocess
import shutil
import sys
import time
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

NIGHTWELDING_DRAFT_LABEL = ("nightwelding-draft-opened", "5319E7", "Nightwelding opened a Draft PR; human must review and mark it ready before it can merge.")
NIGHTWELDING_FAILED_LABEL = ("nightwelding-failed", "B60205", "Nightwelding could not reproduce the issue, or could not make the reproduction test pass.")
NIGHTWELDING_QUEUE_LABEL = ("nightwelding-queue", "1D76DB", "Queued for Nightwelding's overnight autonomous-implementation pipeline.")


class GitHubAdapterError(RuntimeError):
    pass


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


@dataclass
class IssueContext:
    number: int
    title: str
    url: str
    markdown: str


def fetch_issue_context(repo: str, issue_number: int) -> IssueContext:
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

    The worktree directory is derived from the branch name (with slashes
    replaced by dashes) plus a short random suffix, so concurrent runs for
    the same issue within the same second do not collide on the same path.
    """
    slug = branch.replace("/", "-")
    suffix = secrets.token_hex(2)
    worktree_dir = repo_root / ".worktrees" / f"{slug}-{suffix}"
    worktree_dir.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "fetch", "origin", base_branch, "--depth=1"], cwd=repo_root)
    _run(["git", "worktree", "add", "--no-checkout", str(worktree_dir), f"origin/{base_branch}"], cwd=repo_root)
    _run(["git", "checkout", "-B", branch], cwd=worktree_dir)
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


def fetch_issue_metadata(repo: str, issue_number: int) -> Dict[str, Any]:
    """Fetch labels, milestone, and projectItems for an issue."""
    try:
        proc = _run([
            "gh", "issue", "view", str(issue_number),
            "--repo", repo,
            "--json", "labels,milestone,projectItems"
        ])
        return json.loads(proc.stdout or "{}")
    except Exception:
        try:
            # Fallback if projectItems triggers permission/scope errors
            proc = _run([
                "gh", "issue", "view", str(issue_number),
                "--repo", repo,
                "--json", "labels,milestone"
            ])
            return json.loads(proc.stdout or "{}")
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
