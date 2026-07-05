"""Regression tests for AutomationEngine singleton coordinator handling."""

import pytest

from src.core.automation.automation_engine import AutomationEngine


class _FakeScheduler:
    def __init__(self):
        self.schedules = {}
        self.execution_callback = None
        self.coordinator = None

    def set_execution_callback(self, callback):
        self.execution_callback = callback


@pytest.fixture(autouse=True)
def reset_automation_singleton():
    AutomationEngine._instance = None
    yield
    AutomationEngine._instance = None


def test_singleton_preserves_coordinator_when_reinstantiated_without_argument():
    sentinel_coordinator = object()
    scheduler = _FakeScheduler()

    engine = AutomationEngine(scheduler=scheduler, coordinator=sentinel_coordinator)
    assert engine.coordinator is sentinel_coordinator

    second = AutomationEngine()
    assert second is engine
    assert second.coordinator is sentinel_coordinator


def test_explicit_none_does_not_clobber_existing_coordinator():
    sentinel_coordinator = object()
    scheduler = _FakeScheduler()

    engine = AutomationEngine(scheduler=scheduler, coordinator=sentinel_coordinator)
    assert engine.coordinator is sentinel_coordinator

    second = AutomationEngine(scheduler=scheduler, coordinator=None)
    assert second is engine
    assert second.coordinator is sentinel_coordinator
