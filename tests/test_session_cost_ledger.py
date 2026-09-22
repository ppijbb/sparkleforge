"""Anvil Phase A-1: session-level $ cost ledger.

Before this, SessionQuota tracked cost_incurred/tokens_used fields but nothing
ever wrote to them (check_quotas' cost/token thresholds could never trip).
record_usage() is the write side; this checks it actually accumulates and
that check_quotas/get_quota_usage reflect it.
"""

from src.core.session_control import SessionControl, SessionStatus


def _fresh_control() -> SessionControl:
    control = SessionControl.__new__(SessionControl)
    control._session_quotas = {}
    control.active_sessions = {}
    control.session_controls = {}
    control.session_tasks = {}
    from src.core.session_control import SessionQuota

    control.default_quota = SessionQuota(max_cost_per_session=1.0, max_tokens_per_session=1000)
    return control


def test_record_usage_accumulates_cost():
    control = _fresh_control()
    control.register_active_session("s1", "do something")

    control.record_usage("s1", cost=0.2)
    control.record_usage("s1", cost=0.3)

    usage = control.get_quota_usage("s1")
    assert usage["cost"]["used"] == 0.5


def test_record_usage_noop_for_unknown_session():
    control = _fresh_control()
    control.record_usage("no-such-session", cost=5.0)  # must not raise


def test_check_quotas_trips_once_cost_exceeds_budget():
    control = _fresh_control()
    control.register_active_session("s2", "expensive task")

    assert control.check_quotas("s2") is True
    control.record_usage("s2", cost=1.5)  # over the 1.0 default budget
    assert control.check_quotas("s2") is False


def test_check_quotas_marks_session_quota_exceeded_not_generic_cancelled(monkeypatch):
    monkeypatch.setattr("src.core.session_control.start_history_session", lambda *a, **kw: "h1")
    monkeypatch.setattr("src.core.session_control.log_history_event", lambda *a, **kw: None)
    monkeypatch.setattr("src.core.session_control.end_history_session", lambda *a, **kw: None)

    control = _fresh_control()
    control.register_active_session("s7", "expensive task")
    control.record_usage("s7", cost=1.5)

    assert control.check_quotas("s7") is False
    assert control.active_sessions["s7"]["status"] == SessionStatus.QUOTA_EXCEEDED


def test_check_quotas_logs_budget_warning_at_80_percent(monkeypatch):
    logged = []
    monkeypatch.setattr("src.core.session_control.start_history_session", lambda *a, **kw: "h2")
    monkeypatch.setattr(
        "src.core.session_control.log_history_event",
        lambda *a, **kw: logged.append((a, kw)),
    )

    control = _fresh_control()
    control.register_active_session("s8", "near-budget task")
    control.record_usage("s8", cost=0.85)  # 85% of the 1.0 default budget
    control.check_quotas("s8")

    assert any(call[0][1] == "budget_warning" for call in logged)


def test_check_quotas_logs_budget_exceeded_and_ends_history_session(monkeypatch):
    exceeded = []
    ended = []
    monkeypatch.setattr("src.core.session_control.start_history_session", lambda *a, **kw: "h3")
    monkeypatch.setattr(
        "src.core.session_control.log_history_event",
        lambda *a, **kw: exceeded.append((a, kw)),
    )
    monkeypatch.setattr(
        "src.core.session_control.end_history_session",
        lambda *a, **kw: ended.append((a, kw)),
    )

    control = _fresh_control()
    control.register_active_session("s9", "over-budget task")
    control.record_usage("s9", cost=1.5)
    control.check_quotas("s9")

    assert any(call[0][1] == "budget_exceeded" for call in exceeded)
    assert len(ended) == 1
    assert ended[0][1]["status"] == "quota_exceeded"


def test_get_quota_usage_reports_quota_event(monkeypatch):
    monkeypatch.setattr("src.core.session_control.start_history_session", lambda *a, **kw: "h4")
    monkeypatch.setattr("src.core.session_control.log_history_event", lambda *a, **kw: None)
    monkeypatch.setattr("src.core.session_control.end_history_session", lambda *a, **kw: None)

    control = _fresh_control()
    control.register_active_session("s10", "task")
    assert control.get_quota_usage("s10")["quota_event"] == "none"

    control.record_usage("s10", cost=1.5)
    control.check_quotas("s10")
    assert control.get_quota_usage("s10")["quota_event"] == "exceeded"


if __name__ == "__main__":
    test_record_usage_accumulates_cost()
    test_record_usage_noop_for_unknown_session()
    test_check_quotas_trips_once_cost_exceeds_budget()
    print("ok")
