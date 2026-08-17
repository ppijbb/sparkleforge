"""Pick the next open issue for opencode-auto-fix.yml's scheduled sweep.

Moved verbatim from the workflow's "Resolve issue and branch" step: skip any
open issue already carrying an auto-fix-pr-opened/auto-fix-merged/
auto-fix-failed label, or that already has an open PR linked to it (by
branch-name convention or a closing-keyword reference in the PR body).

Simplification the relocation enables, not silent: the original bash ran
`gh pr list` once per remaining candidate inside the loop; this fetches the
open-PR list once and reuses it for every candidate, since it doesn't change
within a single run.
"""

from __future__ import annotations

import re

_SKIP_LABELS = {"auto-fix-pr-opened", "auto-fix-merged", "auto-fix-failed"}
_ISSUE_REF_PATTERN = r"(close[sd]?|fix(e[sd])?|resolve[sd]?) #{number}( |$|[^0-9])"


def _pr_references_issue(pr: dict, issue_number: int) -> bool:
    head_ref = pr.get("headRefName", "")
    if head_ref.startswith(f"fix/{issue_number}-"):
        return True
    body = pr.get("body") or ""
    return re.search(_ISSUE_REF_PATTERN.format(number=issue_number), body, re.IGNORECASE) is not None


def select_fixable_issue(issues: list[dict], open_prs: list[dict]) -> int | None:
    """issues: [{"number": int, "labels": [{"name": str}, ...]}, ...]
    open_prs: [{"headRefName": str, "body": str}, ...]

    First issue (in the given order) with none of the skip labels and no
    already-open linked fix PR.
    """
    for issue in issues:
        labels = {label["name"] for label in issue.get("labels", [])}
        if labels & _SKIP_LABELS:
            continue
        number = issue["number"]
        if not any(_pr_references_issue(pr, number) for pr in open_prs):
            return number
    return None
