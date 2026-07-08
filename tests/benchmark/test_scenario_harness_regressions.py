from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tests.benchmark import run_scenarios
from tests.benchmark.scenario_fixtures import scheduled_summary


def test_require_openrouter_api_key_fails_closed_when_missing(monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    assert run_scenarios.require_openrouter_api_key() is False

    captured = capsys.readouterr()
    assert "OPENROUTER_API_KEY is required" in captured.err


def test_require_openrouter_api_key_accepts_configured_secret(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    assert run_scenarios.require_openrouter_api_key() is True


def test_run_agent_timeout_returns_text_streams(monkeypatch, tmp_path):
    def timeout_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=kwargs.get("cmd") or args[0],
            timeout=kwargs.get("timeout", 1),
            output="partial stdout",
            stderr="partial stderr",
        )

    monkeypatch.setattr(run_scenarios.subprocess, "run", timeout_run)

    result = run_scenarios.run_agent("do work", tmp_path, timeout_s=1)

    assert result["timed_out"] is True
    assert result["stdout"] == "partial stdout"
    assert result["stderr"] == "partial stderr"


def test_scheduled_summary_detects_standard_nine_am_crons():
    assert scheduled_summary._is_nine_am_cron("0 9 * * *") is True
    assert scheduled_summary._is_nine_am_cron("0 9 * * 1") is True
    assert scheduled_summary._is_nine_am_cron("*/30 9 * * *") is True
    assert scheduled_summary._is_nine_am_cron("0 8 * * *") is False
    assert scheduled_summary._is_nine_am_cron("15 9 * * *") is False


def test_crontab_line_detection_parses_only_cron_fields():
    line = "0 9 * * * /usr/local/bin/sparkleforge-summary"

    assert scheduled_summary._is_nine_am_cron(scheduled_summary._crontab_cron_expr(line)) is True


@pytest.mark.asyncio
async def test_run_scenario_injects_yaml_judge_rubric(monkeypatch):
    captured = {}

    async def fake_grade(workspace, ctx, stdout):
        captured["judge_rubric"] = ctx["judge_rubric"]
        return {"judge_quality": (1.0, "ok")}

    fake_fixture = SimpleNamespace(
        build=lambda workspace: {},
        grade=fake_grade,
    )
    monkeypatch.setattr(run_scenarios.importlib, "import_module", lambda name: fake_fixture)
    monkeypatch.setattr(
        run_scenarios,
        "run_agent",
        lambda user_query, workspace, timeout_s: {
            "returncode": 0,
            "stdout": "done",
            "stderr": "",
            "timed_out": False,
            "duration_s": 0.01,
        },
    )

    await run_scenarios.run_scenario(
        {
            "id": "custom",
            "name": "Custom",
            "fixture": "custom_fixture",
            "user_query": "do work",
            "timeout_s": 1,
            "weights": {"judge_quality": 1.0},
            "judge_rubric": "custom rubric from yaml",
        }
    )

    assert captured["judge_rubric"] == "custom rubric from yaml"


@pytest.mark.asyncio
async def test_scheduled_summary_grade_passes_yaml_rubric_to_judge(monkeypatch, tmp_path):
    captured = {}

    async def fake_judge_score(rubric, transcript, context=""):
        captured["rubric"] = rubric
        return 1.0, "ok"

    monkeypatch.setattr(scheduled_summary, "judge_score", fake_judge_score)
    ctx = scheduled_summary.build(tmp_path)
    ctx["judge_rubric"] = "custom scheduled-summary rubric"

    await scheduled_summary.grade(tmp_path, ctx, stdout="summary")

    assert captured["rubric"] == "custom scheduled-summary rubric"
