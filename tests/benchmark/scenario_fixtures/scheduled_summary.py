"""Fixture + grader for scenario 2: "매일 아침 9시에 어제 한 일 요약해서 노션에 올려줘."

This is the scenario most likely to score low today: nothing in the NL
execution graph (AgentHarness/AgentLoop) currently calls into
src/core/automation/automation_engine.py or src/core/scheduler.py, so a
generic-tool-only agent has no direct path to register a real automation.
That's a real, useful finding, not a fixture bug — see
tests/benchmark/baselines/scenario_history.jsonl and issue #330/#329 for the
wiring gap this exposes.

Grading therefore checks two independently-achievable things:
  1. was any 9am-ish recurring trigger registered (Scheduler store or real crontab)
  2. did the agent at least perform the summarization step right now
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict

from tests.benchmark.scenario_grading import (
    concat_new_file_text,
    judge_score,
    keyword_hit,
    new_files,
    rubric_from_context,
    snapshot_tree,
)

WORKLOG_CONTENT = (
    "- Fixed auth session timeout bug (#412)\n"
    "- Reviewed PR #418 (rate limiter)\n"
    "- Wrote onboarding docs draft for the billing module\n"
)
WORKLOG_KEYWORDS = ["auth", "412", "418", "rate limiter", "billing", "onboarding"]


def _schedules_file(workspace: Path) -> Path:
    # run_scenarios.py pins HOME/USERPROFILE to `workspace` for the agent
    # subprocess, so src/core/scheduler.py's default `Path.home()`-relative
    # store resolves here too — reading the real developer/CI home dir would
    # both miss what the agent wrote and leak state across scenario runs.
    return workspace / ".sparkleforge" / "schedules" / "schedules.json"


def _read_schedule_records(workspace: Path) -> Dict[str, Any]:
    # Stored as {"schedules": [{"schedule_id": ..., "cron_expression": ..., ...}, ...],
    # "updated_at": ...} per src/core/scheduler.py._save_schedules -- keyed by
    # schedule_id here for lookup convenience, dropping any entry missing an id.
    schedules_file = _schedules_file(workspace)
    if not schedules_file.exists():
        return {}
    try:
        import json

        data = json.loads(schedules_file.read_text(encoding="utf-8"))
        entries = data.get("schedules", []) if isinstance(data, dict) else []
        return {
            entry["schedule_id"]: entry
            for entry in entries
            if isinstance(entry, dict) and "schedule_id" in entry
        }
    except Exception:
        return {}


def _read_schedule_ids(workspace: Path) -> set[str]:
    return set(_read_schedule_records(workspace).keys())


def _crontab_lines() -> list[str]:
    try:
        result = subprocess.run(["crontab", "-l"], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return []
        return [ln for ln in result.stdout.splitlines() if ln.strip() and not ln.startswith("#")]
    except (OSError, subprocess.TimeoutExpired):
        return []


def _is_nine_am_cron(expr: str) -> bool:
    parts = expr.split()
    if len(parts) < 5:
        return False
    minute, hour = parts[0], parts[1]
    return hour == "9" and (minute == "0" or minute.startswith("*/"))


def _crontab_cron_expr(line: str) -> str:
    return " ".join(line.split()[:5])


def build(workspace: Path) -> Dict[str, Any]:
    (workspace / "worklog.md").write_text(WORKLOG_CONTENT, encoding="utf-8")

    return {
        "workspace": str(workspace),
        "worklog_file": "worklog.md",
        "before_snapshot": snapshot_tree(workspace),
        "schedule_ids_before": _read_schedule_ids(workspace),
        "crontab_lines_before": _crontab_lines(),
    }


async def grade(workspace: Path, ctx: Dict[str, Any], stdout: str) -> Dict[str, tuple[float, str]]:
    new_schedule_ids = _read_schedule_ids(workspace) - ctx["schedule_ids_before"]
    records = _read_schedule_records(workspace)
    nine_am_schedules = []
    for sid in new_schedule_ids:
        record = records.get(sid, {})
        cron_expr = str(record.get("cron_expression", ""))
        if _is_nine_am_cron(cron_expr):
            nine_am_schedules.append(sid)

    new_crontab = [ln for ln in _crontab_lines() if ln not in ctx["crontab_lines_before"]]
    crontab_nine_am = [ln for ln in new_crontab if _is_nine_am_cron(_crontab_cron_expr(ln))]

    if nine_am_schedules or crontab_nine_am:
        automation_registered = (
            1.0,
            f"registered trigger(s): scheduler={nine_am_schedules or None}, crontab={crontab_nine_am or None}",
        )
    elif new_schedule_ids or new_crontab:
        automation_registered = (
            0.4,
            f"a trigger was registered but not clearly at 9am: scheduler={new_schedule_ids}, crontab={new_crontab}",
        )
    else:
        automation_registered = (
            0.0,
            "no automation/cron trigger registered — generic tool set has no path to "
            "AutomationEngine/Scheduler from the NL execution graph (known gap, see #330)",
        )

    before = ctx["before_snapshot"]
    after = snapshot_tree(workspace)
    created = new_files(before, after)
    summary_text = concat_new_file_text(workspace, created) + "\n" + stdout
    if keyword_hit(summary_text, WORKLOG_KEYWORDS):
        summary_produced = (1.0, "a summary referencing worklog content was produced")
    else:
        summary_produced = (0.0, "no summary of worklog.md content found in new files or stdout")

    judge = await judge_score(
        rubric=rubric_from_context(
            ctx,
            "Is the produced summary a faithful, coherent recap of yesterday's worklog entries?",
        ),
        transcript=summary_text[:4000],
        context=f"worklog.md content: {WORKLOG_CONTENT}",
    )

    return {
        "automation_registered": automation_registered,
        "summary_produced": summary_produced,
        "judge_summary_quality": judge,
    }
