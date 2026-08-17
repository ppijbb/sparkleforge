from src.core.ci.issue_selection import select_fixable_issue


def test_skips_issues_with_skip_labels():
    issues = [
        {"number": 1, "labels": [{"name": "auto-fix-pr-opened"}]},
        {"number": 2, "labels": [{"name": "auto-fix-merged"}]},
        {"number": 3, "labels": [{"name": "auto-fix-failed"}]},
        {"number": 4, "labels": []},
    ]
    assert select_fixable_issue(issues, []) == 4


def test_skips_issue_with_open_pr_by_branch_name():
    issues = [{"number": 5, "labels": []}]
    open_prs = [{"headRefName": "fix/5-12345", "body": ""}]
    assert select_fixable_issue(issues, open_prs) is None


def test_skips_issue_with_open_pr_referenced_in_body():
    issues = [{"number": 6, "labels": []}]
    open_prs = [{"headRefName": "some-other-branch", "body": "This closes #6 for good."}]
    assert select_fixable_issue(issues, open_prs) is None


def test_body_reference_requires_word_boundary_after_number():
    # "#6" inside "#67" must not count as a reference to issue 6.
    issues = [{"number": 6, "labels": []}]
    open_prs = [{"headRefName": "unrelated", "body": "Fixes #67 only."}]
    assert select_fixable_issue(issues, open_prs) == 6


def test_returns_first_eligible_issue_in_order():
    issues = [
        {"number": 1, "labels": []},
        {"number": 2, "labels": []},
    ]
    open_prs = [{"headRefName": "fix/1-999", "body": ""}]
    assert select_fixable_issue(issues, open_prs) == 2


def test_no_eligible_issue_returns_none():
    issues = [{"number": 1, "labels": [{"name": "auto-fix-failed"}]}]
    assert select_fixable_issue(issues, []) is None
