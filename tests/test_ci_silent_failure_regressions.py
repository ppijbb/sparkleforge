"""Regression guards for the 'silently succeeds despite a real failure' bug
class (issues behind PRs #1507, #1511, #1513, #1518, #1520): each of those
fixed one call site that swallowed a failure and reported success anyway.
This covers the two fixed sites that had no test yet.
"""

import asyncio
from unittest.mock import patch

from src.core.agent_harness import AgentHarness


def test_execute_autonomous_propagates_failure_from_run_conversation(monkeypatch):
    async def run_test():
        harness = object.__new__(AgentHarness)  # skip __init__'s heavy tool registration
        harness.orchestrator = None

        class _FakeLoop:
            def __init__(self, orchestrator, plan_first=False):
                pass

            async def run_conversation(self, **kwargs):
                return {
                    "success": False,
                    "error": "boom",
                    "content": "",
                    "metadata": {},
                    "iterations": 1,
                }

        monkeypatch.setattr("src.core.agent_loop.AgentLoop", _FakeLoop)

        result = await harness.execute(session_id="s1", request="do the thing", mode="autonomous")

        assert result["success"] is False
        assert result["error"] == "boom"

    asyncio.run(run_test())


def test_main_entry_loads_dotenv_before_dispatch():
    import main as main_module  # import first so patching main_entry below doesn't trigger main.py's own module-level config load

    calls = []

    with patch("dotenv.load_dotenv", side_effect=lambda *a, **kw: calls.append(("load_dotenv", a, kw))), \
         patch.object(main_module, "main_entry", side_effect=lambda: calls.append(("dispatch", (), {}))), \
         patch("sys.argv", ["sparkleforge", "--help"]):
        import src.cli.entry as entry

        entry.main_entry()

    assert calls[0] == ("load_dotenv", (), {"override": True})
    assert calls[-1] == ("dispatch", (), {})
