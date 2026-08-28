"""Anvil Phase A-1: session-level $ cost ledger.

Before this, SessionQuota tracked cost_incurred/tokens_used fields but nothing
ever wrote to them (check_quotas' cost/token thresholds could never trip).
record_usage() is the write side; this checks it actually accumulates and
that check_quotas/get_quota_usage reflect it.
"""

from src.core.session_control import SessionControl


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


if __name__ == "__main__":
    test_record_usage_accumulates_cost()
    test_record_usage_noop_for_unknown_session()
    test_check_quotas_trips_once_cost_exceeds_budget()
    print("ok")
