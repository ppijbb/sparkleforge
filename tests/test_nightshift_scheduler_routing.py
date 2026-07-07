"""Regression test for Nightshift's scheduler sentinel routing.

AutomationEngine._wrapped_execution_callback is the interception point that
is actually live once BootstrapGraph runs (it unconditionally constructs
AutomationEngine, which claims scheduler.execution_callback before
configure_scheduler_execution() ever gets a chance to install the research
callback -- see src/core/nightshift/runner.py's module docstring / the
Nightshift plan for the full trace). This test locks in that a schedule
tagged metadata={"nightshift_sweep": True} short-circuits straight to
run_nightshift_sweep() *before* route_task()/run_research() are ever reached
-- the exact thing that's easy to silently regress since the dead-code path
(configure_scheduler_execution's own closure) looks superficially correct on
its own.
"""

from __future__ import annotations

import pytest

from src.core.automation.automation_engine import AutomationEngine
from src.core.scheduler import Scheduler


@pytest.fixture
def scheduler(tmp_path):
    return Scheduler(storage_path=tmp_path / "schedules")


@pytest.fixture
def automation_engine(scheduler):
    # AutomationEngine is a singleton; reset it so each test starts fresh and
    # binds to this test's own Scheduler instance instead of a prior test's.
    AutomationEngine._instance = None
    engine = AutomationEngine(scheduler=scheduler)
    yield engine
    AutomationEngine._instance = None


async def test_nightshift_sweep_schedule_short_circuits_before_route_task(
    automation_engine, scheduler, monkeypatch
) -> None:
    schedule = scheduler.create_schedule(
        name="nightshift",
        cron_expression="30 16 * * *",
        user_query="__nightshift_sweep__",
        metadata={"nightshift_sweep": True},
    )

    sweep_called = []

    async def fake_sweep():
        sweep_called.append(True)
        return ["fake-result"]

    monkeypatch.setattr(
        "src.core.nightshift.runner.run_nightshift_sweep", fake_sweep
    )

    route_task_called = []
    monkeypatch.setattr(
        automation_engine,
        "route_task",
        lambda *a, **k: route_task_called.append(True),
    )

    result = await automation_engine._wrapped_execution_callback(
        schedule.user_query, "session-1"
    )

    assert sweep_called == [True]
    assert route_task_called == []  # never reached -- short-circuited first
    assert result == ["fake-result"]


async def test_non_nightshift_schedule_still_routes_normally(
    automation_engine, scheduler, monkeypatch
) -> None:
    schedule = scheduler.create_schedule(
        name="research",
        cron_expression="0 9 * * *",
        user_query="summarize today's news",
    )

    sweep_called = []
    monkeypatch.setattr(
        "src.core.nightshift.runner.run_nightshift_sweep",
        lambda: sweep_called.append(True),
    )

    route_task_called = []

    def fake_route_task(query, metadata):
        route_task_called.append(query)
        return query

    monkeypatch.setattr(automation_engine, "route_task", fake_route_task)
    automation_engine._orig_callback = None  # avoid touching a real orchestrator

    result = await automation_engine._wrapped_execution_callback(
        schedule.user_query, "session-2"
    )

    assert sweep_called == []
    assert route_task_called == [schedule.user_query]
    assert result == {"status": "skipped", "reason": "no callback"}
