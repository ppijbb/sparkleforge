"""Regression tests for src/sdk.py, the in-process headless run entrypoint."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import AsyncMock, patch

from src.core import researcher_config
import src.sdk as sdk


def test_run_loads_config_when_not_loaded(monkeypatch):
    monkeypatch.setattr(researcher_config, "config", None)
    load_calls = []
    monkeypatch.setattr(researcher_config, "load_config_from_env", lambda: load_calls.append(1))

    with patch("src.core.autonomous_orchestrator.AutonomousOrchestrator") as MockOrch:
        MockOrch.return_value.run_research = AsyncMock(return_value={"content": "ok"})
        result = asyncio.run(sdk.run("test prompt"))

    assert load_calls == [1]
    assert result == {"content": "ok"}


def test_run_skips_loading_when_config_already_set(monkeypatch):
    monkeypatch.setattr(researcher_config, "config", object())
    load_calls = []
    monkeypatch.setattr(researcher_config, "load_config_from_env", lambda: load_calls.append(1))

    with patch("src.core.autonomous_orchestrator.AutonomousOrchestrator") as MockOrch:
        MockOrch.return_value.run_research = AsyncMock(return_value={"content": "ok"})
        asyncio.run(sdk.run("test prompt"))

    assert load_calls == []


def test_run_passes_prompt_to_orchestrator(monkeypatch):
    monkeypatch.setattr(researcher_config, "config", object())

    with patch("src.core.autonomous_orchestrator.AutonomousOrchestrator") as MockOrch:
        MockOrch.return_value.run_research = AsyncMock(return_value={"content": "ok"})
        asyncio.run(sdk.run("quantum ML review"))

        MockOrch.return_value.run_research.assert_awaited_once_with("quantum ML review")


@dataclass
class _FakeEventType:
    value: str


@dataclass
class _FakeStreamingEvent:
    workflow_id: str
    event_type: _FakeEventType
    agent_id: str
    data: Dict[str, Any]
    timestamp: datetime = datetime.now(timezone.utc)


class _FakeStreamingManager:
    def __init__(self):
        self.listeners = []

    def add_local_listener(self, callback):
        self.listeners.append(callback)

    def remove_local_listener(self, callback):
        self.listeners.remove(callback)


def test_run_with_on_progress_calls_execute_with_a_fresh_objective_id(monkeypatch):
    monkeypatch.setattr(researcher_config, "config", object())
    fake_manager = _FakeStreamingManager()
    monkeypatch.setattr(
        "src.core.streaming_manager.get_streaming_manager", lambda: fake_manager
    )

    with patch("src.core.autonomous_orchestrator.AutonomousOrchestrator") as MockOrch:
        MockOrch.return_value.execute = AsyncMock(return_value={"content": "ok"})
        result = asyncio.run(sdk.run("prompt", on_progress=AsyncMock()))

    assert result == {"content": "ok"}
    MockOrch.return_value.execute.assert_awaited_once()
    _args, kwargs = MockOrch.return_value.execute.call_args
    assert kwargs["objective_id"].startswith("sdk_")


def test_run_with_on_progress_unregisters_listener_after_completion(monkeypatch):
    monkeypatch.setattr(researcher_config, "config", object())
    fake_manager = _FakeStreamingManager()
    monkeypatch.setattr(
        "src.core.streaming_manager.get_streaming_manager", lambda: fake_manager
    )

    with patch("src.core.autonomous_orchestrator.AutonomousOrchestrator") as MockOrch:
        MockOrch.return_value.execute = AsyncMock(return_value={"content": "ok"})
        asyncio.run(sdk.run("prompt", on_progress=AsyncMock()))

    assert fake_manager.listeners == []


def test_run_relays_matching_workflow_events_as_progress_event(monkeypatch):
    monkeypatch.setattr(researcher_config, "config", object())
    fake_manager = _FakeStreamingManager()
    monkeypatch.setattr(
        "src.core.streaming_manager.get_streaming_manager", lambda: fake_manager
    )
    received = []

    async def on_progress(event):
        received.append(event)

    with patch("src.core.autonomous_orchestrator.AutonomousOrchestrator") as MockOrch:
        async def fake_execute(_prompt, objective_id=None, **_kw):
            listener = fake_manager.listeners[0]
            await listener(
                _FakeStreamingEvent(
                    workflow_id=objective_id,
                    event_type=_FakeEventType("workflow_start"),
                    agent_id="orchestrator",
                    data={"message": "Starting objective analysis"},
                )
            )
            return {"content": "ok"}

        MockOrch.return_value.execute = fake_execute
        asyncio.run(sdk.run("prompt", on_progress=on_progress))

    assert len(received) == 1
    assert isinstance(received[0], sdk.ProgressEvent)
    assert received[0].phase == "workflow_start"
    assert received[0].message == "Starting objective analysis"


def test_run_ignores_events_from_other_workflow_ids(monkeypatch):
    monkeypatch.setattr(researcher_config, "config", object())
    fake_manager = _FakeStreamingManager()
    monkeypatch.setattr(
        "src.core.streaming_manager.get_streaming_manager", lambda: fake_manager
    )
    received = []

    async def on_progress(event):
        received.append(event)

    with patch("src.core.autonomous_orchestrator.AutonomousOrchestrator") as MockOrch:
        async def fake_execute(_prompt, objective_id=None, **_kw):
            listener = fake_manager.listeners[0]
            await listener(
                _FakeStreamingEvent(
                    workflow_id="some-other-concurrent-run",
                    event_type=_FakeEventType("workflow_start"),
                    agent_id="orchestrator",
                    data={"message": "unrelated"},
                )
            )
            return {"content": "ok"}

        MockOrch.return_value.execute = fake_execute
        asyncio.run(sdk.run("prompt", on_progress=on_progress))

    assert received == []


def test_run_swallows_on_progress_exceptions(monkeypatch):
    monkeypatch.setattr(researcher_config, "config", object())
    fake_manager = _FakeStreamingManager()
    monkeypatch.setattr(
        "src.core.streaming_manager.get_streaming_manager", lambda: fake_manager
    )

    async def raising_on_progress(_event):
        raise RuntimeError("boom")

    with patch("src.core.autonomous_orchestrator.AutonomousOrchestrator") as MockOrch:
        async def fake_execute(_prompt, objective_id=None, **_kw):
            listener = fake_manager.listeners[0]
            await listener(
                _FakeStreamingEvent(
                    workflow_id=objective_id,
                    event_type=_FakeEventType("workflow_start"),
                    agent_id="orchestrator",
                    data={"message": "hi"},
                )
            )
            return {"content": "ok"}

        MockOrch.return_value.execute = fake_execute
        result = asyncio.run(sdk.run("prompt", on_progress=raising_on_progress))

    assert result == {"content": "ok"}
