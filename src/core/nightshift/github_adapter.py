"""Thin `gh`/`git` subprocess adapter for Nightshift.

Consistent with the rest of this project: no GitHub API client is built here,
just subprocess calls to the `gh` CLI and `git`, exactly like the existing
GitHub Actions workflows and scripts/opencode_github_worker.py already do.

The PR body always contains the literal substring "OpenCode-generated" and
branches always use the "nightshift/" prefix — both are load-bearing for
staying Draft-only forever: gemini-assistant.yml's `code-review` job (and,
transitively, `merge-decision`, which needs `code-review` to succeed) skips
any PR whose body contains that substring, and its separate
`auto-merge-ready-fix-prs` scheduled sweep only matches `fix/`/`chore/`
branch prefixes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

NIGHTSHIFT_DRAFT_LABEL = ("nightshift-draft-opened", "5319E7", "Nightshift opened a Draft PR; human must review and mark it ready before it can merge.")
NIGHTSHIFT_FAILED_LABEL = ("nightshift-failed", "B60205", "Nightshift could not reproduce the issue, or could not make the reproduction test pass.")
NIGHTSHIFT_QUEUE_LABEL = ("nightshift-queue", "1D76DB", "Queued for Nightshift's overnight autonomous-implementation pipeline.")


class GitHubAdapterError(RuntimeError):
    pass


def _run(cmd: List[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    if check and proc.returncode != 0:
        raise GitHubAdapterError(f"{' '.join(cmd)} failed ({proc.returncode}): {proc.stderr.strip()}")
    return proc


def default_base_branch(repo: str) -> str:
    proc = _run(["gh", "repo", "view", repo, "--json", "defaultBranchRef", "--jq", ".defaultBranchRef.name"])
    return proc.stdout.strip()


def normalize_commit_title(issue_title: str, repo_root: Path) -> Optional[str]:
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
    and with no existing open PR from a `nightshift/$N-` branch."""
    proc = _run(
        ["gh", "issue", "list", "--repo", repo, "--state", "open", "--limit", str(limit), "--json", "number,labels"]
    )
    issues = json.loads(proc.stdout or "[]")
    candidates: List[int] = []
    for issue in issues:
        labels = {label["name"] for label in issue.get("labels", [])}
        if backlog_label not in labels:
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
        if any(pr["headRefName"].startswith(f"nightshift/{number}-") for pr in open_prs):
            continue
        eligible.append(number)
    return eligible


def create_branch(repo_root: Path, branch: str, base_branch: str) -> None:
    _run(["git", "fetch", "origin", base_branch, "--depth=1"], cwd=repo_root)
    _run(["git", "checkout", "-B", branch, f"origin/{base_branch}"], cwd=repo_root)


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


def find_open_pr(repo: str, branch: str, base_branch: str) -> Optional[str]:
    proc = _run(
        ["gh", "pr", "list", "--repo", repo, "--head", branch, "--base", base_branch, "--state", "open", "--json", "url"],
        check=False,
    )
    items = json.loads(proc.stdout or "[]")
    return items[0]["url"] if items else None


def open_draft_pr(repo: str, base_branch: str, branch: str, title: str, body: str) -> str:
    """Open a Draft PR. `body` MUST already contain the literal substring
    'OpenCode-generated' — callers are responsible for that (see module
    docstring for why)."""
    if "OpenCode-generated" not in body:
        raise GitHubAdapterError("PR body must contain the literal substring 'OpenCode-generated'.")

    existing = find_open_pr(repo, branch, base_branch)
    if existing:
        return existing

    proc = _run(
        ["gh", "pr", "create", "--repo", repo, "--draft", "--base", base_branch, "--head", branch, "--title", title, "--body", body]
    )
    return proc.stdout.strip()
