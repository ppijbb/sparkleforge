"""Issue #641: ObservationPlane/WindowTracker wiring into the live AgentHarness runtime.

Verifies the executor node records an observation snapshot into meta state,
and that the snapshot capture degrades gracefully instead of failing the run.
"""

import asyncio

from src.core.agent_harness import AgentHarness
from src.core.harness_state import create_initial_harness_state


def test_initial_harness_state_has_observation_snapshot_slot():
    state = create_initial_harness_state("session-1", "do the thing")
    assert state["meta"]["observation_snapshot"] == {}


def test_capture_observation_snapshot_returns_integrated_state():
    async def run_test():
        harness = object.__new__(AgentHarness)  # skip __init__'s heavy tool registration
        snapshot = await harness._capture_observation_snapshot()

        assert "error" not in snapshot
        assert "metrics" in snapshot
        assert "active_window" in snapshot

    asyncio.run(run_test())


def test_capture_observation_snapshot_swallows_failures(monkeypatch):
    async def run_test():
        harness = object.__new__(AgentHarness)

        class _BoomObservationPlane:
            async def get_integrated_state(self):
                raise RuntimeError("boom")

        monkeypatch.setattr(
            "src.core.observe.observation_plane.ObservationPlane",
            _BoomObservationPlane,
        )

        snapshot = await harness._capture_observation_snapshot()
        assert snapshot == {"error": "boom"}

    asyncio.run(run_test())
