import os
import json
import subprocess
import sys
from io import StringIO
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
    assert "--model" in result.stdout


def test_daily_roadmap_workflow_uses_supported_cli_command() -> None:
    workflow = (
        PROJECT_ROOT / ".github" / "workflows" / "sparkleforge-daily-roadmap.yml"
    ).read_text(encoding="utf-8")

    assert "uv run python -m src.cli.entry run" in workflow
    assert "timeout 10m uv run python -m src.cli.entry run" in workflow
    assert "timeout-minutes: 35" in workflow
    assert "set +e\n          timeout 10m uv run python -m src.cli.entry run" in workflow
    assert "RC=$?\n          set -e" in workflow
    assert "--model google/gemini-3.5-flash-lite" in workflow
    assert "Collect GitHub planning context" in workflow
    assert "sparkleforge report roadmap-fallback-issue" in workflow
    assert "gh pr list" in workflow
    assert "gh issue list" in workflow
    assert "github-planning-context.md" in workflow
    assert "gh issue edit \"$EXISTING\"" in workflow
    assert "uv run python -m src.cli.entry research" not in workflow
    assert "--no-interactive" not in workflow
    assert "Generated roadmap based on:" in workflow
    assert "CLI output is missing required section" in workflow


def test_module_entrypoint_injects_piped_run_query(monkeypatch) -> None:
    from src.cli import entry

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "entry.py",
            "run",
            "--max-tokens",
            "64",
            "--model",
            "google/gemini-3.5-flash-lite",
            "--output",
            "roadmap.md",
            "--format",
            "markdown",
        ],
    )
    monkeypatch.setattr(sys, "stdin", StringIO("daily roadmap prompt\n"))

    entry._inject_stdin_query_for_run()

    assert sys.argv[2] == "daily roadmap prompt"


def test_module_entrypoint_delegates_to_real_repository_cli() -> None:
    entrypoint = (PROJECT_ROOT / "src" / "cli" / "entry.py").read_text(
        encoding="utf-8"
    )

    assert "from main import main_entry as repository_main_entry" in entrypoint
    assert "_inject_stdin_query_for_run()" in entrypoint
    assert "repository_main_entry()" in entrypoint
    assert "Generated roadmap based on:" not in entrypoint


def test_main_entry_does_not_chdir_to_the_package_install_dir(tmp_path, monkeypatch, capsys) -> None:
    # Regression test for #1222: the installed `sparkleforge`/`sparkle` command
    # used to unconditionally os.chdir() to its own package install directory
    # before dispatching, so running it from another project's directory
    # silently operated on SparkleForge's own repo instead -- unlike git, npx,
    # or eslint, which respect the caller's cwd. `main_entry()` chdir's (if at
    # all) before any argparse dispatch, so even `--help` exercises the bug.
    from src.cli import entry

    other_project = tmp_path / "other-project"
    other_project.mkdir()
    monkeypatch.chdir(other_project)
    monkeypatch.setattr(sys, "argv", ["sparkleforge", "--help"])

    try:
        entry.main_entry()
    except SystemExit:
        pass  # argparse --help exits 0 after printing

    assert Path.cwd() == other_project


def test_report_generator_history_prunes_to_cap(tmp_path, monkeypatch) -> None:
    from src.generation import report_generator

    monkeypatch.setattr(report_generator, "MAX_HISTORY_ENTRIES", 5)

    history_path = tmp_path / "history.json"
    history_path.write_text("[]", encoding="utf-8")

    for i in range(10):
        history = json.loads(history_path.read_text(encoding="utf-8"))
        history.append({"index": i})
        history = history[-report_generator.MAX_HISTORY_ENTRIES:]
        history_path.write_text(json.dumps(history), encoding="utf-8")

    final_history = json.loads(history_path.read_text(encoding="utf-8"))
    assert len(final_history) == report_generator.MAX_HISTORY_ENTRIES
    assert final_history[0]["index"] == 5
    assert final_history[-1]["index"] == 9
