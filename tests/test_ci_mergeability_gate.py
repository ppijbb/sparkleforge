from src.core.ci.mergeability_gate import check_mechanical_mergeability


def _pr(**overrides):
    base = {
        "isDraft": False,
        "mergeable": "MERGEABLE",
        "mergeStateStatus": "CLEAN",
        "statusCheckRollup": [],
    }
    base.update(overrides)
    return base


def test_draft_pr_is_not_ready():
    verdict = check_mechanical_mergeability(_pr(isDraft=True))
    assert verdict.ready is False
    assert "draft" in verdict.reason


def test_draft_field_variant_also_respected():
    verdict = check_mechanical_mergeability({"draft": True, "mergeStateStatus": "CLEAN"})
    assert verdict.ready is False


def test_not_mergeable_field_blocks():
    verdict = check_mechanical_mergeability(_pr(mergeable="CONFLICTING"))
    assert verdict.ready is False
    assert "mergeable=CONFLICTING" in verdict.reason


def test_missing_mergeable_field_does_not_block():
    pr = _pr()
    del pr["mergeable"]
    verdict = check_mechanical_mergeability(pr)
    assert verdict.ready is True


def test_dirty_blocked_draft_state_blocks():
    for state in ("DIRTY", "BLOCKED", "DRAFT"):
        verdict = check_mechanical_mergeability(_pr(mergeStateStatus=state))
        assert verdict.ready is False, state


def test_pending_check_blocks():
    verdict = check_mechanical_mergeability(
        _pr(statusCheckRollup=[{"status": "IN_PROGRESS", "conclusion": None}])
    )
    assert verdict.ready is False


def test_failed_check_blocks():
    verdict = check_mechanical_mergeability(
        _pr(statusCheckRollup=[{"status": "COMPLETED", "conclusion": "FAILURE"}])
    )
    assert verdict.ready is False


def test_success_and_skipped_checks_do_not_block():
    verdict = check_mechanical_mergeability(
        _pr(
            statusCheckRollup=[
                {"status": "COMPLETED", "conclusion": "SUCCESS"},
                {"status": "COMPLETED", "conclusion": "SKIPPED"},
            ]
        )
    )
    assert verdict.ready is True


def test_all_clear_is_ready():
    verdict = check_mechanical_mergeability(_pr())
    assert verdict.ready is True
    assert verdict.reason == ""


def test_non_clean_merge_state_with_no_checks_still_blocks():
    verdict = check_mechanical_mergeability(_pr(mergeStateStatus="UNSTABLE"))
    assert verdict.ready is False
