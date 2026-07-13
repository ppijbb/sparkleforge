import time

import pytest

from src.core.session_control import SessionControl, SessionStatus


def test_session_quota_initialization():
    """
    Verify that active sessions are initialized with a quota schema.
    This test fails if SessionControl does not track per-session quotas.
    """
    controller = SessionControl()
    session_id = "test_session_123"

    # Register a session
    controller.register_active_session(session_id, user_query="test query")

    # Check if quota exists in the session state
    session_data = controller.get_session_state(session_id)
    assert "quota" in session_data, "Session state must contain a 'quota' field"
    assert session_data["quota"].get("max_tokens") is not None, "Quota should have a max_tokens"


def test_concurrent_session_cap_exceeded_raises() -> None:
    controller = SessionControl()
    controller.DEFAULT_MAX_CONCURRENT_SESSIONS = 1

    controller.register_active_session("session-a", user_query="first")

    with pytest.raises(RuntimeError, match="Active session quota reached"):
        controller.register_active_session("session-b", user_query="second")


def test_budget_exhausted_cancels_session() -> None:
    controller = SessionControl()
    session_id = "session-budget"
    controller.register_active_session(session_id, user_query="q")

    controller._session_quotas[session_id]["cost_incurred"] = 999.0
    controller._session_quotas[session_id]["budget"] = 1.0

    assert controller.check_quotas(session_id) is False
    assert controller.active_sessions[session_id]["status"] == SessionStatus.CANCELLED


def test_timeout_exceeded_cancels_session() -> None:
    controller = SessionControl()
    session_id = "session-timeout"
    controller.register_active_session(session_id, user_query="q")

    controller._session_quotas[session_id]["start_time"] = time.time() - 999999
    controller._session_quotas[session_id]["timeout"] = 1

    assert controller.check_quotas(session_id) is False
    assert controller.active_sessions[session_id]["status"] == SessionStatus.CANCELLED


def test_within_quota_is_not_cancelled() -> None:
    controller = SessionControl()
    session_id = "session-ok"
    controller.register_active_session(session_id, user_query="q")

    assert controller.check_quotas(session_id) is True
    assert controller.active_sessions[session_id]["status"] == SessionStatus.ACTIVE


def test_get_quota_usage_reports_remaining_and_pct() -> None:
    controller = SessionControl()
    session_id = "session-usage"
    controller.register_active_session(session_id, user_query="q")
    controller._session_quotas[session_id].update(
        {"max_tokens": 100, "tokens_used": 25, "budget": 10.0, "cost_incurred": 2.5}
    )

    usage = controller.get_quota_usage(session_id)

    assert usage["tokens"] == {"used": 25, "limit": 100, "remaining": 75, "pct_used": 25.0}
    assert usage["cost"]["remaining"] == pytest.approx(7.5)
    assert usage["cost"]["pct_used"] == pytest.approx(25.0)


def test_get_quota_usage_returns_none_for_untracked_session() -> None:
    controller = SessionControl()

    assert controller.get_quota_usage("does-not-exist") is None


def test_update_session_quota_changes_limits() -> None:
    controller = SessionControl()
    session_id = "session-update"
    controller.register_active_session(session_id, user_query="q")

    updated = controller.update_session_quota(session_id, max_tokens=5000, budget=2.5, timeout=60)

    assert updated is True
    usage = controller.get_quota_usage(session_id)
    assert usage["tokens"]["limit"] == 5000
    assert usage["cost"]["limit"] == 2.5
    assert usage["time"]["limit"] == 60


def test_update_session_quota_returns_false_for_untracked_session() -> None:
    controller = SessionControl()

    assert controller.update_session_quota("does-not-exist", max_tokens=1) is False
