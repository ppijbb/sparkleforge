"""Escalate Nightwelding issues that fail repeatedly to human review (issue #1615).

GitHub-only: labels/comments need a real issue tracker, which LocalGitAdapter
doesn't have. Never raises -- a failure to escalate must not crash the sweep,
matching GitHubAdapter.report_failure's own error-isolation convention.
"""

from __future__ import annotations

import logging

from src.core.nightwelding.github_adapter import (
    AUTO_FIX_BACKOFF_LABEL,
    HUMAN_REVIEW_NEEDED_LABEL,
    GitHubAdapter,
    add_labels,
    comment_on_issue,
    ensure_label,
)
from src.core.nightwelding.adapter import BaseNightweldingAdapter
from src.core.nightwelding.models import NightweldingItem

logger = logging.getLogger(__name__)

DEFAULT_ESCALATION_THRESHOLD = 3


def should_escalate(item: NightweldingItem, threshold: int = DEFAULT_ESCALATION_THRESHOLD) -> bool:
    """True once an issue has failed `threshold` consecutive Nightwelding attempts."""
    return item.consecutive_failures >= threshold


def escalate_issue(adapter: BaseNightweldingAdapter, issue_number: int | str, item: NightweldingItem) -> None:
    """Label a repeatedly-failing issue for human review and post a diagnostic summary.

    No-op for non-GitHub adapters and non-numeric issue refs (local issue
    files have no label/comment concept to escalate onto).
    """
    if not isinstance(adapter, GitHubAdapter) or not str(issue_number).isdigit():
        return
    try:
        repo = adapter._get_repo()
        num = int(issue_number)
        ensure_label(repo, *HUMAN_REVIEW_NEEDED_LABEL)
        ensure_label(repo, *AUTO_FIX_BACKOFF_LABEL)
        add_labels(repo, num, [HUMAN_REVIEW_NEEDED_LABEL[0], AUTO_FIX_BACKOFF_LABEL[0]])
        comment_on_issue(
            repo,
            num,
            f"Nightwelding has failed {item.consecutive_failures} consecutive time(s) on this "
            "issue and is backing off further automatic attempts.\n\n"
            f"Last failure: {item.failure_reason or 'unknown'}\n\n"
            f"Flagged for human review (`{HUMAN_REVIEW_NEEDED_LABEL[0]}`); Nightwelding sweeps "
            f"will skip this issue (`{AUTO_FIX_BACKOFF_LABEL[0]}`) until that label is removed.",
        )
    except Exception:
        logger.exception("Nightwelding: failed to escalate issue #%s", issue_number)
