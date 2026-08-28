"""Anvil Phase Γ: recurring-issue grouping + stale-issue detection (pure logic, no gh)."""

from datetime import datetime, timedelta, timezone

from src.core.ci.issue_digest import STALE_LABEL, build_digest, find_stale, group_recurring


def _issue(number, title, updated_days_ago=0, labels=None):
    updated = datetime.now(timezone.utc) - timedelta(days=updated_days_ago)
    return {
        "number": number,
        "title": title,
        "updatedAt": updated.isoformat().replace("+00:00", "Z"),
        "labels": [{"name": n} for n in (labels or [])],
    }


def test_group_recurring_clusters_similar_titles():
    issues = [
        _issue(1, "fix: thread-unsafe global config initialization race condition"),
        _issue(2, "fix: global config initialization race in SDK bootstrap"),
        _issue(3, "feat: expose sparkleforge externally as an MCP server"),
    ]

    groups = group_recurring(issues, min_similarity=0.3)

    assert len(groups) == 1
    numbers = {i["number"] for i in groups[0]}
    assert numbers == {1, 2}


def test_group_recurring_returns_nothing_for_all_distinct_titles():
    issues = [
        _issue(1, "fix: A completely different bug about disk space"),
        _issue(2, "feat: add token-budget guardrails for review prompts"),
    ]

    assert group_recurring(issues) == []


def test_find_stale_flags_old_issues_and_skips_already_labeled():
    issues = [
        _issue(1, "fresh issue", updated_days_ago=1),
        _issue(2, "old issue", updated_days_ago=30),
        _issue(3, "old but already flagged", updated_days_ago=30, labels=[STALE_LABEL]),
    ]

    stale = find_stale(issues, stale_days=14)

    assert [i["number"] for i in stale] == [2]


def test_build_digest_combines_both():
    issues = [
        _issue(1, "fix: X race condition bug", updated_days_ago=1),
        _issue(2, "fix: X race condition in worker", updated_days_ago=1),
        _issue(3, "feat: unrelated old thing", updated_days_ago=40),
    ]

    digest = build_digest(issues, min_similarity=0.3, stale_days=14)

    assert len(digest.recurring_groups) == 1
    assert len(digest.stale_issues) == 1
    assert digest.stale_issues[0]["number"] == 3


if __name__ == "__main__":
    test_group_recurring_clusters_similar_titles()
    test_group_recurring_returns_nothing_for_all_distinct_titles()
    test_find_stale_flags_old_issues_and_skips_already_labeled()
    test_build_digest_combines_both()
    print("ok")
