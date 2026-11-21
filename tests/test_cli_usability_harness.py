from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import tests.benchmark.cli_usability_harness as harness
from src.cli import entry
from src.cli.ui import logging_policy

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_entry_sets_quiet_boot_env_vars_before_importing_main(monkeypatch) -> None:
    for key in entry.QUIET_BOOT_ENV_VARS:
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setattr(sys, "argv", ["sparkleforge", "--help"])
    seen_env: dict[str, str | None] = {}

    import main as main_module

    def fake_repo_main() -> None:
        for key in entry.QUIET_BOOT_ENV_VARS:
            seen_env[key] = os.environ.get(key)

    with (
        patch("dotenv.load_dotenv"),
        patch.object(main_module, "main_entry", side_effect=fake_repo_main),
    ):
        entry.main_entry()

    assert seen_env == {
        "TQDM_DISABLE": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TRANSFORMERS_VERBOSITY": "error",
        "TOKENIZERS_PARALLELISM": "false",
        "SPARKLEFORGE_QUIET_BOOT": "1",
    }


def test_entry_env_configured_prior_to_main_import_subprocess() -> None:
    snippet = (
        "import os, sys, types\n"
        "captured = {}\n"
        "class MainProbe(types.ModuleType):\n"
        "    def main_entry(self):\n"
        "        pass\n"
        "real_import = __import__\n"
        "def spy_import(name, globals=None, locals=None, fromlist=(), level=0):\n"
        "    if name == 'main' and level == 0:\n"
        "        for k in ('TQDM_DISABLE', 'HF_HUB_DISABLE_PROGRESS_BARS', "
        "'TRANSFORMERS_VERBOSITY', 'TOKENIZERS_PARALLELISM', 'SPARKLEFORGE_QUIET_BOOT'):\n"
        "            captured[k] = os.environ.get(k)\n"
        "        return MainProbe('main')\n"
        "    return real_import(name, globals, locals, fromlist, level)\n"
        "import builtins\n"
        "builtins.__import__ = spy_import\n"
        "sys.argv = ['sparkleforge', 'health']\n"
        "import src.cli.entry as e\n"
        "e.main_entry()\n"
        "import json; print(json.dumps(captured))\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    for key in entry.QUIET_BOOT_ENV_VARS:
        env.pop(key, None)

    proc = subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    captured = json.loads(proc.stdout.strip().splitlines()[-1])
    assert captured == entry.QUIET_BOOT_ENV_VARS


def test_entry_preserves_verbose_without_setting_quiet_boot_env(monkeypatch) -> None:
    for flag in ("--verbose", "-v"):
        for key in entry.QUIET_BOOT_ENV_VARS:
            monkeypatch.delenv(key, raising=False)

        monkeypatch.setattr(sys, "argv", ["sparkleforge", flag, "health"])
        import main as main_module

        with (
            patch("dotenv.load_dotenv"),
            patch.object(main_module, "main_entry", return_value=None),
        ):
            entry.main_entry()

        for key in entry.QUIET_BOOT_ENV_VARS:
            assert key not in os.environ


def test_quiet_boot_routes_startup_warnings_to_log_file_only(tmp_path) -> None:
    root = logging.getLogger()
    orig_handlers = list(root.handlers)
    orig_level = root.level
    logger_names = (
        "google_genai._api_client",
        "src.core.skill_tree",
        "src.core.llm_manager.model_registry",
        "src.core.llm_manager.cascade",
        "src.cli.commands.health",
        "src.core.agent_harness",
    )
    orig_levels = {name: logging.getLogger(name).level for name in logger_names}

    log_file = tmp_path / "sparkleforge.log"
    console_stream = io.StringIO()
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(console_stream)
    console_handler.setLevel(logging.INFO)

    try:
        root.handlers.clear()
        root.setLevel(logging.INFO)
        root.addHandler(file_handler)
        root.addHandler(console_handler)

        logging_policy.apply_quiet_boot_logging(suppress_warnings=False)

        logging.getLogger("google_genai._api_client").warning(
            "Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY."
        )
        logging.getLogger("src.core.skill_tree").warning(
            "FlashRank Ranker unavailable: No module named 'flashrank'"
        )
        logging.getLogger("src.core.llm_manager.model_registry").warning(
            "OPENROUTER_API_KEY not found - OpenRouter models will be unavailable"
        )
        logging.getLogger("src.core.llm_manager.cascade").warning(
            "Cascade execution failed: 429 rate limited, falling back to single model..."
        )
        logging.getLogger("src.cli.commands.health").info("System is healthy")
        logging.getLogger("src.core.agent_harness").error("Fatal harness error")

        file_handler.flush()
        console_output = console_stream.getvalue()
        file_output = log_file.read_text(encoding="utf-8")

        assert "Both GOOGLE_API_KEY and GEMINI_API_KEY" not in console_output
        assert "FlashRank Ranker unavailable" not in console_output
        assert "OPENROUTER_API_KEY not found" not in console_output
        assert "Cascade execution failed" not in console_output

        assert "System is healthy" in console_output
        assert "Fatal harness error" in console_output

        assert "Both GOOGLE_API_KEY and GEMINI_API_KEY" in file_output
        assert "FlashRank Ranker unavailable" in file_output
        assert "OPENROUTER_API_KEY not found" in file_output
        assert "Cascade execution failed" in file_output
    finally:
        logging_policy.remove_startup_noise_filters()
        file_handler.close()
        root.handlers = orig_handlers
        root.setLevel(orig_level)
        for name, level in orig_levels.items():
            logging.getLogger(name).setLevel(level)


def test_remove_startup_noise_filters_restores_verbose_console_output() -> None:
    root = logging.getLogger()
    orig_handlers = list(root.handlers)
    orig_level = root.level
    console_stream = io.StringIO()
    console_handler = logging.StreamHandler(console_stream)
    console_handler.setLevel(logging.WARNING)

    try:
        root.handlers.clear()
        root.setLevel(logging.INFO)
        root.addHandler(console_handler)

        logging_policy.apply_quiet_boot_logging(suppress_warnings=False)
        logging.getLogger("google_genai._api_client").warning(
            "Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY."
        )
        assert console_stream.getvalue() == ""

        logging_policy.remove_startup_noise_filters()
        logging.getLogger("google_genai._api_client").warning(
            "Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY."
        )
        assert "Both GOOGLE_API_KEY and GEMINI_API_KEY" in console_stream.getvalue()
    finally:
        logging_policy.remove_startup_noise_filters()
        root.handlers = orig_handlers
        root.setLevel(orig_level)


def test_cli_entry_health_suppresses_duplicate_key_and_startup_warn_noise() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["GOOGLE_API_KEY"] = "test-google-key"
    env["GEMINI_API_KEY"] = "test-gemini-key"

    proc = subprocess.run(
        [sys.executable, "-m", "src.cli.entry", "health"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert proc.returncode == 0
    combined = f"{proc.stdout}\n{proc.stderr}"
    assert "Both GOOGLE_API_KEY and GEMINI_API_KEY" not in combined
    assert "FlashRank Ranker unavailable" not in combined
    assert "[WARN]" not in combined


def test_evaluate_latency_dimension() -> None:
    fast = harness.ProbeObservation(
        name="fast",
        command=["sparkleforge", "--help"],
        returncode=0,
        duration_ms=250.0,
        cold_start_ms=250.0,
    )
    fast_score = harness.evaluate_latency(fast, default_threshold_ms=1000.0)
    assert fast_score.score == 1.0
    assert fast_score.passed is True

    slow = harness.ProbeObservation(
        name="slow",
        command=["sparkleforge", "--help"],
        returncode=0,
        duration_ms=1500.0,
    )
    slow_score = harness.evaluate_latency(slow, default_threshold_ms=1000.0)
    assert slow_score.score == 0.5
    assert slow_score.passed is False

    timed_out = harness.ProbeObservation(
        name="timeout",
        command=["sparkleforge", "run", "x"],
        returncode=-1,
        duration_ms=5000.0,
        timed_out=True,
    )
    timeout_score = harness.evaluate_latency(timed_out, default_threshold_ms=1000.0)
    assert timeout_score.score == 0.0
    assert timeout_score.passed is False


def test_evaluate_output_hygiene_dimension() -> None:
    clean = harness.ProbeObservation(
        name="clean",
        command=["sparkleforge", "health"],
        returncode=0,
        duration_ms=100.0,
        stdout="[OK] System is healthy\n",
        stderr="",
    )
    clean_res = harness.evaluate_output_hygiene(clean)
    assert clean_res.score == 1.0
    assert clean_res.passed is True

    one_warn = harness.ProbeObservation(
        name="one_warn",
        command=["sparkleforge", "health"],
        returncode=0,
        duration_ms=100.0,
        stdout="[WARN] FlashRank Ranker unavailable\n[OK] healthy\n",
    )
    one_warn_res = harness.evaluate_output_hygiene(one_warn)
    assert one_warn_res.score == 0.8
    assert one_warn_res.passed is False

    all_bad = harness.ProbeObservation(
        name="all_bad",
        command=["sparkleforge", "run", "x"],
        returncode=1,
        duration_ms=100.0,
        stdout="Loading weights: 100%\nBatches: 100%\n[WARN] noisy\ngarbled \ufffd\n",
        stderr="Traceback (most recent call last):\n  File 'x.py'\n",
    )
    all_bad_res = harness.evaluate_output_hygiene(all_bad)
    assert all_bad_res.score == 0.0
    assert all_bad_res.passed is False
    assert len(all_bad_res.findings) == 5


def test_evaluate_request_info_processing_dimension() -> None:
    good = harness.ProbeObservation(
        name="good_req",
        command=["sparkleforge", "run", "audit"],
        returncode=0,
        duration_ms=200.0,
        query="audit cli ux",
        routed_target="research",
        flags={
            "model": "google/gemini-3.1-flash-lite-preview",
            "depth": "quick",
            "autopilot": "true",
            "format": "json",
        },
        applied_flags={
            "model": "google/gemini-3.1-flash-lite-preview",
            "depth": "quick",
            "autopilot": True,
            "format": "json",
        },
        expected_format="json",
        stdout=json.dumps({"success": True, "summary": "ok"}),
    )
    good_res = harness.evaluate_request_info_processing(good)
    assert good_res.score == 1.0
    assert good_res.passed is True

    bad = harness.ProbeObservation(
        name="bad_req",
        command=["sparkleforge", "run", "audit"],
        returncode=2,
        duration_ms=100.0,
        query="audit cli ux",
        flags={"depth": "invalid-depth", "format": "json"},
        expected_format="json",
        stdout="not valid json",
        stderr="error: invalid choice: 'invalid-depth'",
    )
    bad_res = harness.evaluate_request_info_processing(bad)
    assert bad_res.score == 0.0
    assert bad_res.passed is False

    partial_flags = harness.ProbeObservation(
        name="help_flags",
        command=["sparkleforge", "run", "--help"],
        returncode=0,
        duration_ms=100.0,
        supported_flags=["--model", "--format"],
        stdout="usage: entry.py run",
    )
    partial_res = harness.evaluate_request_info_processing(partial_flags)
    assert 0.0 < partial_res.score < 1.0
    assert partial_res.passed is False


def test_evaluate_response_quality_dimension() -> None:
    good = harness.ProbeObservation(
        name="good_resp",
        command=["sparkleforge", "run", "explain"],
        returncode=0,
        duration_ms=150.0,
        stdout="SparkleForge routes requests cleanly through AgentOrchestrator.",
    )
    good_res = harness.evaluate_response_quality(good)
    assert good_res.score == 1.0
    assert good_res.passed is True

    empty = harness.ProbeObservation(
        name="empty_resp",
        command=["sparkleforge", "run", "explain"],
        returncode=0,
        duration_ms=150.0,
        stdout="   \n",
    )
    empty_res = harness.evaluate_response_quality(empty)
    assert empty_res.score == 0.0
    assert empty_res.passed is False

    spam = harness.ProbeObservation(
        name="spam_resp",
        command=["sparkleforge", "run", "explain"],
        returncode=0,
        duration_ms=150.0,
        stdout="Summary completed after fallback.\n",
        stderr=(
            "OpenRouter model returned HTTP 429 rate limit exceeded, trying fallback\n"
            "HTTP 404 model_not_found\n"
            "HTTP 402 Payment Required\n"
        ),
    )
    spam_res = harness.evaluate_response_quality(spam)
    assert spam_res.score < 1.0
    assert spam_res.passed is False
    assert spam_res.metrics["no_provider_spam"] is False


def test_evaluate_orchestration_monitoring_dimension() -> None:
    good_structured = harness.ProbeObservation(
        name="good_orch",
        command=["sparkleforge", "run", "q"],
        returncode=0,
        duration_ms=200.0,
        orchestration={
            "tasks": [{"id": "plan", "status": "completed", "duration_ms": 15.0}],
            "subagents": [{"name": "researcher", "status": "completed", "duration_ms": 40.0}],
            "step_latencies_ms": {"plan": 15.0, "researcher": 40.0},
        },
    )
    good_res = harness.evaluate_orchestration_monitoring(good_structured)
    assert good_res.score == 1.0
    assert good_res.passed is True

    good_bootstrap_stdout = harness.ProbeObservation(
        name="bootstrap_orch",
        command=["sparkleforge", "--debug-bootstrap"],
        returncode=0,
        duration_ms=300.0,
        stdout="[OK] config (1.4 ms)\n[OK] database (3.2 ms)\n",
    )
    boot_res = harness.evaluate_orchestration_monitoring(good_bootstrap_stdout)
    assert boot_res.score == 1.0
    assert boot_res.passed is True

    bad_orch = harness.ProbeObservation(
        name="bad_orch",
        command=["sparkleforge", "run", "q"],
        returncode=1,
        duration_ms=200.0,
        stdout="[FAIL] crashed",
    )
    bad_res = harness.evaluate_orchestration_monitoring(bad_orch)
    assert bad_res.score == 0.0
    assert bad_res.passed is False


def test_harness_cli_help_and_offline_json_out(tmp_path) -> None:
    help_proc = subprocess.run(
        [sys.executable, "tests/benchmark/cli_usability_harness.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert help_proc.returncode == 0
    assert "--json-out" in help_proc.stdout
    assert "--mode" in help_proc.stdout

    json_out = tmp_path / "usability_report.json"
    rc = harness.main(["--mode", "mock", "--json-out", str(json_out), "--min-score", "0.9"])
    assert rc == 0
    assert json_out.exists()

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["mode"] == "offline"
    assert payload["passed"] is True
    assert payload["overall_score"] == 1.0
    for dim in harness.DIMENSIONS:
        assert dim in payload["dimension_scores"]
        assert payload["dimension_scores"][dim] == 1.0


def test_harness_live_mode_with_mocked_subprocesses(monkeypatch, tmp_path) -> None:
    def fake_run(cmd, **kwargs):
        args_str = " ".join(cmd)
        if "--help" in args_str and "run" in args_str:
            out = "options: --model --depth --autopilot --format"
        elif "--debug-bootstrap" in args_str:
            out = "[OK] config (1.5 ms)\n[OK] mcp_hub (22.0 ms)\n"
        elif "health" in args_str:
            out = "Running system health check...\n[OK] System is healthy\n"
        else:
            out = "SparkleForge - Autonomous Multi-Agent Research System"
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(harness.subprocess, "run", fake_run)

    json_out = tmp_path / "live_report.json"
    rc = harness.main(
        ["--mode", "live", "--json-out", str(json_out), "--latency-threshold-ms", "2000"]
    )
    assert rc == 0
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["mode"] == "live"
    assert payload["summary"]["probe_count"] == 4
    assert payload["summary"]["clean_returncodes"] == 4
    assert payload["dimension_scores"]["output_hygiene"] == 1.0
