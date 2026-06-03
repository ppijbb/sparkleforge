import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_module_entrypoint_prints_help() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    result = subprocess.run(
        [sys.executable, "-m", "src.cli.entry", "--help"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )

    assert result.returncode == 0
    assert "SparkleForge - Autonomous Multi-Agent Research System" in result.stdout
    assert "run" in result.stdout


def test_run_entrypoint_accepts_workflow_max_tokens_flag() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    result = subprocess.run(
        [sys.executable, "-m", "src.cli.entry", "run", "--help"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )

    assert result.returncode == 0
    assert "--max-tokens" in result.stdout


def test_daily_roadmap_workflow_uses_supported_cli_command() -> None:
    workflow = (
        PROJECT_ROOT / ".github" / "workflows" / "sparkleforge-daily-roadmap.yml"
    ).read_text(encoding="utf-8")

    assert "uv run python -m src.cli.entry run" in workflow
    assert "timeout 25m uv run python -m src.cli.entry run" in workflow
    assert "timeout-minutes: 35" in workflow
    assert "uv run python -m src.cli.entry research" not in workflow
    assert "--no-interactive" not in workflow
    assert "Generated roadmap based on:" in workflow
    assert "CLI output is missing required section" in workflow


def test_module_entrypoint_delegates_to_real_repository_cli() -> None:
    entrypoint = (PROJECT_ROOT / "src" / "cli" / "entry.py").read_text(
        encoding="utf-8"
    )

    assert "from main import main_entry as repository_main_entry" in entrypoint
    assert "repository_main_entry()" in entrypoint
    assert "Generated roadmap based on:" not in entrypoint
