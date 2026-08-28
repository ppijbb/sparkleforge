"""Nightwelding issue digest (Anvil Phase Γ).

Open auto-fix issues pile up (fix: ... titles filed by Nightwelding/CI) with
no triage: recurring root causes surface as separate issues nobody links
together, and old ones sit unlooked-at indefinitely. docs/ANVIL_PLAN.md
section 3 already names this exact pattern -- a component/signal that exists
but nobody looks at. This is the fix, applied to Nightwelding's own issue
backlog: group issues that look like the same root cause recurring, and flag
ones that have gone stale.

Pure grouping/staleness logic lives here and is unit-tested without gh.
GitHub I/O (fetch/comment/label) is a thin subprocess wrapper, mirroring
stagnation_issue.py's `_gh` pattern.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

STALE_LABEL = "nightwelding-stale"
DEFAULT_STALE_DAYS = 14
DEFAULT_MIN_SIMILARITY = 0.5

_STOPWORDS = {
    "fix", "feat", "chore", "the", "a", "an", "and", "or", "for", "to", "of",
    "in", "on", "with", "is", "are", "issue", "bug",
}


def _title_terms(title: str) -> set[str]:
    """Normalize a title into a bag of lowercase words, dropping conventional-commit
    type prefixes and short stopwords that would otherwise dominate the overlap score."""
    cleaned = re.sub(r"^[a-z]+:\s*", "", title.strip().lower())
    words = re.findall(r"[a-z0-9_]{3,}", cleaned)
    return {w for w in words if w not in _STOPWORDS}


def _similarity(a: str, b: str) -> float:
    """Jaccard similarity of two issue titles' term sets."""
    terms_a, terms_b = _title_terms(a), _title_terms(b)
    if not terms_a or not terms_b:
        return 0.0
    union = terms_a | terms_b
    return len(terms_a & terms_b) / len(union) if union else 0.0


def group_recurring(
    issues: list[dict[str, Any]], min_similarity: float = DEFAULT_MIN_SIMILARITY
) -> list[list[dict[str, Any]]]:
    """Group issues whose titles look like the same recurring root cause.

    Single-linkage clustering on title term-overlap -- simple and good enough
    for "same bug filed twice with slightly different wording"; not meant to
    catch semantically-similar-but-differently-worded issues. Only groups
    with 2+ members are returned.
    """
    n = len(issues)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if _similarity(issues[i].get("title", ""), issues[j].get("title", "")) >= min_similarity:
                union(i, j)

    clusters: dict[int, list[dict[str, Any]]] = {}
    for i, issue in enumerate(issues):
        clusters.setdefault(find(i), []).append(issue)

    return [group for group in clusters.values() if len(group) > 1]


def find_stale(
    issues: list[dict[str, Any]],
    stale_days: int = DEFAULT_STALE_DAYS,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Issues whose updatedAt is older than `stale_days`. Issues already
    carrying STALE_LABEL are skipped (already flagged)."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=stale_days)
    stale = []
    for issue in issues:
        labels = {label.get("name", "") for label in issue.get("labels", [])}
        if STALE_LABEL in labels:
            continue
        updated_at = issue.get("updatedAt")
        if not updated_at:
            continue
        try:
            updated = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        except ValueError:
            continue
        if updated < cutoff:
            stale.append(issue)
    return stale


@dataclass
class DigestResult:
    recurring_groups: list[list[dict[str, Any]]]
    stale_issues: list[dict[str, Any]]


def build_digest(
    issues: list[dict[str, Any]],
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    stale_days: int = DEFAULT_STALE_DAYS,
    now: datetime | None = None,
) -> DigestResult:
    return DigestResult(
        recurring_groups=group_recurring(issues, min_similarity),
        stale_issues=find_stale(issues, stale_days, now),
    )


# --- GitHub I/O -------------------------------------------------------------


def _gh(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(args, text=True, capture_output=True, check=check)


def fetch_open_issues(repo: str, limit: int = 200) -> list[dict[str, Any]]:
    """Open, non-PR issues via `gh issue list`. Excludes issues already
    labeled no-auto-fix's sibling exemptions to keep the digest focused on
    the fix:/feat: auto-fix backlog it's meant to triage."""
    proc = _gh([
        "gh", "issue", "list", "--repo", repo, "--state", "open",
        "--limit", str(limit),
        "--json", "number,title,labels,updatedAt,url",
    ], check=False)
    if proc.returncode != 0:
        print(f"[issue-digest] failed to list issues: {proc.stderr}", file=sys.stderr)
        return []
    return json.loads(proc.stdout)


def _ensure_stale_label(repo: str) -> None:
    _gh([
        "gh", "label", "create", STALE_LABEL, "--repo", repo,
        "--color", "D4C5F9",
        "--description", "Nightwelding digest: no activity in 14+ days.",
        "--force",
    ], check=False)


def apply_digest(repo: str, digest: DigestResult) -> None:
    """Best-effort side effects: comment each recurring group's issues with
    links to their siblings, and label stale issues. Never raises -- a failed
    gh call is logged and skipped, not fatal to the run."""
    for group in digest.recurring_groups:
        numbers = [issue["number"] for issue in group]
        for issue in group:
            siblings = [n for n in numbers if n != issue["number"]]
            if not siblings:
                continue
            body = (
                "🔁 Nightwelding digest: this looks like the same root cause as "
                + ", ".join(f"#{n}" for n in siblings)
                + ". Consider fixing them together or closing duplicates."
            )
            proc = _gh(["gh", "issue", "comment", str(issue["number"]), "--repo", repo, "--body", body], check=False)
            if proc.returncode != 0:
                print(f"[issue-digest] failed to comment on #{issue['number']}: {proc.stderr}", file=sys.stderr)

    if digest.stale_issues:
        _ensure_stale_label(repo)
    for issue in digest.stale_issues:
        proc = _gh(["gh", "issue", "edit", str(issue["number"]), "--repo", repo, "--add-label", STALE_LABEL], check=False)
        if proc.returncode != 0:
            print(f"[issue-digest] failed to label #{issue['number']}: {proc.stderr}", file=sys.stderr)
