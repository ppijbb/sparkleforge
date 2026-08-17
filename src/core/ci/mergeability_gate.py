"""Mechanical mergeability gate shared by gemini-assistant.yml's
merge-decision and auto-merge-ready-fix-prs jobs.

Before this, both jobs independently computed "is this PR actually ready to
merge right now" (draft / mergeable / checks-settled) with slightly
different bash, and had drifted: merge-decision never checked the raw
`mergeable` field (only mergeStateStatus), auto-merge-ready-fix-prs did.
This consolidates both to the stricter check (mergeable field required),
a deliberate tightening, not silent.
"""

from __future__ import annotations

from dataclasses import dataclass

_NOT_CLEAN_STATES = ("DIRTY", "BLOCKED", "DRAFT")


@dataclass
class MergeabilityVerdict:
    ready: bool
    reason: str


def _not_ready_check_count(status_check_rollup: list[dict]) -> int:
    count = 0
    for check in status_check_rollup or []:
        status = check.get("status")
        conclusion = check.get("conclusion")
        if status != "COMPLETED" or (conclusion is not None and conclusion not in ("SUCCESS", "SKIPPED")):
            count += 1
    return count


def check_mechanical_mergeability(pr: dict) -> MergeabilityVerdict:
    """pr: a `gh pr view`/`gh pr list` JSON object with isDraft (or draft),
    mergeable, mergeStateStatus, and statusCheckRollup fields."""
    is_draft = pr.get("isDraft", pr.get("draft", False))
    if is_draft:
        return MergeabilityVerdict(False, "draft PR")

    mergeable = pr.get("mergeable")
    if mergeable is not None and mergeable != "MERGEABLE":
        return MergeabilityVerdict(False, f"not mergeable: mergeable={mergeable}")

    merge_state = pr.get("mergeStateStatus", "UNKNOWN")
    if merge_state in _NOT_CLEAN_STATES:
        return MergeabilityVerdict(False, f"PR is not currently mergeable: {merge_state}")

    not_ready = _not_ready_check_count(pr.get("statusCheckRollup") or [])
    if not_ready != 0 or merge_state != "CLEAN":
        return MergeabilityVerdict(
            False, f"checks are still settling or merge state is {merge_state}"
        )

    return MergeabilityVerdict(True, "")
