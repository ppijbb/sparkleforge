"""Regression for issues #1768/#1773/#1784: a stage whose coroutine never
resolves must fail fast, not hang `BootstrapGraph.run()` forever, and the
`_stage_observation_plane` dead-code/cleanup path must not blow up."""

import asyncio

import pytest

from src.core import bootstrap_graph
from src.core.bootstrap_graph import BootstrapGraph, BootstrapStage


class _HangingBootstrapGraph(BootstrapGraph):
    def _default_stages(self):
        async def _hang():
            await asyncio.Event().wait()

        return [BootstrapStage("hangs_forever", _hang)]


@pytest.mark.asyncio
async def test_hung_stage_times_out_instead_of_hanging(monkeypatch):
    monkeypatch.setattr(bootstrap_graph, "DEFAULT_STAGE_TIMEOUT_SECONDS", 0.05)

    result = await asyncio.wait_for(_HangingBootstrapGraph().run(), timeout=5.0)

    assert result.ok is False
    assert len(result.stages) == 1
    assert result.stages[0].name == "hangs_forever"
    assert "timed out" in result.stages[0].error


@pytest.mark.asyncio
async def test_non_critical_hung_stage_does_not_abort_the_run(monkeypatch):
    monkeypatch.setattr(bootstrap_graph, "DEFAULT_STAGE_TIMEOUT_SECONDS", 0.05)

    class _NonCriticalHang(BootstrapGraph):
        def _default_stages(self):
            async def _hang():
                await asyncio.Event().wait()

            async def _ok():
                return {"done": True}

            return [
                BootstrapStage("hangs_forever", _hang, critical=False),
                BootstrapStage("later_stage", _ok),
            ]

    result = await asyncio.wait_for(_NonCriticalHang().run(), timeout=5.0)

    assert result.ok is True
    assert [s.name for s in result.stages] == ["hangs_forever", "later_stage"]
    assert result.stages[0].ok is False
    assert "timed out" in result.stages[0].error
