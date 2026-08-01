"""Issue #1183: batch dispatch results must go to the log file, not flood
stdout, and the REPL's forge-batch command must only print one summary line
per task (never the raw per-task response)."""

import json
import logging
from unittest.mock import AsyncMock, patch

import pytest

from src.cli.commands.forge_master import forge_master_batch_command
from src.core.forge_master.tools import _log_batch_manifest


class _FakeConsole:
    def __init__(self):
        self.lines: list[str] = []

    def print(self, *args, **kwargs):
        self.lines.append(" ".join(str(a) for a in args))

    def status(self, *args, **kwargs):
        class _Ctx:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                return False

        return _Ctx()


class _FakeCLI:
    def __init__(self):
        self.console = _FakeConsole()


def test_log_batch_manifest_writes_one_structured_line(caplog):
    results = [
        {"success": True, "agent_used": "codex", "master_verdict": "PASSED",
         "adversarial_audit": {"skepticism_score": 0.2}},
        {"success": False, "last_agent_used": "gemini_cli", "master_verdict": "REJECTED"},
    ]

    with caplog.at_level(logging.INFO, logger="src.core.forge_master.tools"):
        _log_batch_manifest(results)

    assert len(caplog.records) == 1
    manifest = json.loads(caplog.records[0].message.split("forge_master batch manifest: ", 1)[1])
    assert manifest[0]["agent_used"] == "codex"
    assert manifest[0]["success"] is True
    assert manifest[1]["agent_used"] == "gemini_cli"
    assert manifest[1]["success"] is False


@pytest.mark.asyncio
async def test_forge_batch_command_prints_one_summary_line_per_task_not_raw_response():
    cli = _FakeCLI()
    fake_batch_result = {
        "success": True,
        "total": 2,
        "succeeded": 2,
        "results": [
            {"success": True, "master_verdict": "PASSED", "response": "x" * 5000},
            {"success": True, "master_verdict": "PASSED", "response": "y" * 5000},
        ],
    }

    with patch(
        "src.core.forge_master.tools._dispatch_batch_to_forge_master_tool",
        new=AsyncMock(return_value=fake_batch_result),
    ):
        await forge_master_batch_command(cli, ["task one", "|||", "task two", "--agent", "codex"])

    joined = "\n".join(cli.console.lines)
    assert "2/2 succeeded" in joined
    # The 5000-char raw responses must never reach the console.
    assert "x" * 5000 not in joined
    assert "y" * 5000 not in joined
