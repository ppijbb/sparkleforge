"""Plan which TODO-debt inventory items become new GitHub issues.

Moved verbatim from todo-issue-sync.yml's "Plan new todo-debt issues"
heredoc: filters the inventory to Critical/High priority items, dedups
against already-tracked items via an anchor comment embedded in existing
issue bodies, and templates the issue title/body.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_ANCHOR_RE = re.compile(r"<!-- todo-debt:(.+?) -->")


@dataclass
class PlannedIssue:
    anchor: str
    title: str
    body: str


def known_anchors(existing_issues: list[dict]) -> set[str]:
    anchors = set()
    for issue in existing_issues:
        match = _ANCHOR_RE.search(issue.get("body") or "")
        if match:
            anchors.add(match.group(1))
    return anchors


def plan_todo_issues(inventory: dict, existing_issues: list[dict]) -> list[PlannedIssue]:
    anchors = known_anchors(existing_issues)
    plan: list[PlannedIssue] = []

    for item in inventory["all_items"]:
        if item["priority"] not in ("Critical", "High"):
            continue

        anchor = f'{item["file"]}:{item["line"]}'
        if anchor in anchors:
            continue

        # Lowercased so the generated title reliably passes the repo's
        # Conventional Commits subject policy (scripts/validate_commit_messages.py);
        # the body below keeps the real-case path for humans/agents.
        title = (
            f'chore: resolve {item["issue_type"].lower()} at '
            f'{item["file"].lower()}:{item["line"]}'
        )

        body_lines = [
            f"<!-- todo-debt:{anchor} -->",
            "",
            "Auto-registered from a TODO/FIXME code comment scan (`sparkleforge ci collect-todos`),",
            "scoped for the existing issue-driven auto-fix pipeline.",
            "",
            "## Location",
            f"- File: `{item['file']}`",
            f"- Line: {item['line']}",
            f"- Category: {item['category']}",
            f"- Priority: {item['priority']}",
            "",
            "## Comment",
            f"> {item['content']}",
            "",
            "## Expected fix",
            "Resolve the referenced TODO/FIXME with the smallest correct change, "
            "or replace it with an accurate comment if it depends on context that "
            "isn't available yet.",
            "",
            "## Acceptance criteria",
            "- [ ] The TODO/FIXME comment at the referenced location is resolved "
            "or replaced with an accurate comment.",
            "- [ ] Existing tests pass; add coverage if the fix changes behavior.",
            "",
        ]

        plan.append(PlannedIssue(anchor=anchor, title=title, body="\n".join(body_lines)))

    return plan
