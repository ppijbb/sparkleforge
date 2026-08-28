"""Anvil Phase A-2: `session cost` CLI surface for the ledger written in Phase A-1."""

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


def test_get_all_quota_usage_aggregates_every_tracked_session():
    control = _fresh_control()
    control.register_active_session("s1", "task one")
    control.register_active_session("s2", "task two")
    control.record_usage("s1", cost=0.1)
    control.record_usage("s2", cost=0.4)

    all_usage = control.get_all_quota_usage()

    assert set(all_usage.keys()) == {"s1", "s2"}
    assert all_usage["s1"]["cost"]["used"] == 0.1
    assert all_usage["s2"]["cost"]["used"] == 0.4


class _FakeConsole:
    def __init__(self):
        self.printed = []

    def print(self, *args, **kwargs):
        self.printed.append(args)


class _FakeCli:
    def __init__(self, session_control):
        self.session_control = session_control
        self.console = _FakeConsole()


async def _run_cost_command(control):
    from src.cli.commands.session import session_cost_command

    cli = _FakeCli(control)
    await session_cost_command(cli, [])
    return cli.console.printed


def test_session_cost_command_prints_total(anyio_backend=None):
    import asyncio

    control = _fresh_control()
    control.register_active_session("s1", "task")
    control.record_usage("s1", cost=0.25)

    printed = asyncio.run(_run_cost_command(control))
    assert any("Total across all tracked sessions" in str(p) for p in printed)


if __name__ == "__main__":
    test_get_all_quota_usage_aggregates_every_tracked_session()
    test_session_cost_command_prints_total()
    print("ok")
