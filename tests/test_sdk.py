"""Regression tests for src/sdk.py, the in-process headless run entrypoint."""

import asyncio
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
