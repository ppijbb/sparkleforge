from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tests.benchmark import run_scenarios
from tests.benchmark.scenario_fixtures import build_env, scheduled_summary


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


@pytest.mark.asyncio
async def test_build_env_grade_reports_env_setup_failure_when_unconfigured(monkeypatch, tmp_path):
    async def fake_judge_score(rubric, transcript, context=""):
        return 1.0, "ok"

    monkeypatch.setattr(build_env, "judge_score", fake_judge_score)
    monkeypatch.delenv("APP_ENV", raising=False)
    ctx = build_env.build(tmp_path)

    scores = await build_env.grade(tmp_path, ctx, stdout="")

    assert scores["env_setup"][0] == 0.0
    assert scores["verify_runs"][0] == 0.0


@pytest.mark.asyncio
async def test_build_env_grade_reports_env_setup_success_via_dotenv_file(monkeypatch, tmp_path):
    async def fake_judge_score(rubric, transcript, context=""):
        return 1.0, "ok"

    monkeypatch.setattr(build_env, "judge_score", fake_judge_score)
    monkeypatch.delenv("APP_ENV", raising=False)
    ctx = build_env.build(tmp_path)
    project = tmp_path / ctx["project_dir"]
    (project / ".env").write_text("APP_ENV=development\n", encoding="utf-8")

    scores = await build_env.grade(tmp_path, ctx, stdout="")

    assert scores["env_setup"][0] == 1.0


def test_compare_scenarios_skips_rule_based_fallback():
    current = {
        "s1": {
            "total": 0.0,
            "adjusted_total": 0.0,
            "breakdown": {
                "judge_quality": {"score": 0.0, "reason": "empty output", "inconclusive": False}
            },
        }
    }
    baseline = {
        "s1": {
            "total": 0.5,
            "adjusted_total": 0.5,
            "breakdown": {
                "judge_quality": {
                    "score": 0.5,
                    "reason": "rule-based fallback judge: matched 1 signal group(s) (LLM judge unavailable)",
                    "inconclusive": False,
                }
            },
        }
    }
    # Should not report regression because baseline was an infra rule-based fallback
    exit_code = run_scenarios._compare_scenarios(current, baseline)
    assert exit_code == 0


def test_compare_to_history_resolves_stagnation_on_improvement(tmp_path):
    history_file = tmp_path / "history.jsonl"
    import json

    # 5 history entries with 0.24 score (0 internal improvements)
    with history_file.open("w", encoding="utf-8") as f:
        for _ in range(5):
            entry = {
                "overall_score_adjusted": 0.24,
                "inconclusive_checks": 0,
                "generated_at": "2026-08-08T00:00:00Z",
                "scenarios": {},
            }
            f.write(json.dumps(entry) + "\n")

    current_report = {
        "overall_score_adjusted": 0.55,
        "scenarios": {},
    }

    exit_code = run_scenarios.compare_to_history(current_report, history_file)
    assert exit_code == 0
    assert current_report["stagnation_detected"] is False
    assert current_report["stagnation_improvements"] == 1


def test_compare_to_history_does_not_block_merge_when_no_regression(tmp_path):
    history_file = tmp_path / "history.jsonl"
    import json

    with history_file.open("w", encoding="utf-8") as f:
        for _ in range(5):
            entry = {
                "overall_score_adjusted": 0.24,
                "inconclusive_checks": 0,
                "generated_at": "2026-08-08T00:00:00Z",
                "scenarios": {},
            }
            f.write(json.dumps(entry) + "\n")

    current_report = {
        "overall_score_adjusted": 0.24,
        "scenarios": {},
    }

    exit_code = run_scenarios.compare_to_history(current_report, history_file)
    # Stagnation is detected because no improvement, but exit code is 0 (no regression) so PR is not deadlocked
    assert exit_code == 0
    assert current_report["stagnation_detected"] is True

