import json
import os
import shutil
import sqlite3
import stat
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CARGO_MANIFEST = PROJECT_ROOT / "Cargo.toml"
BIN_NAME = "sparkleforge.exe" if os.name == "nt" else "sparkleforge"
BINARY_PATH = PROJECT_ROOT / "target" / "debug" / BIN_NAME


@pytest.fixture(scope="module")
def sparkleforge_bin() -> Path:
    """Ensure the Rust CLI binary is compiled via cargo build."""
    result = subprocess.run(
        ["cargo", "build", "--manifest-path", str(CARGO_MANIFEST)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, f"cargo build failed:\n{result.stderr}\n{result.stdout}"
    assert BINARY_PATH.is_file(), f"expected compiled binary at {BINARY_PATH}"
    return BINARY_PATH


def _run_cli(
    binary: Path,
    args: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 30,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(binary), *args],
        cwd=cwd or PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def test_cargo_build_succeeds(sparkleforge_bin: Path) -> None:
    assert sparkleforge_bin.is_file()
    assert os.access(sparkleforge_bin, os.X_OK)


def test_ping_and_version_sub_5ms_response(sparkleforge_bin: Path) -> None:
    ping_text = _run_cli(sparkleforge_bin, ["ping"])
    assert ping_text.returncode == 0
    assert "pong" in ping_text.stdout.lower()

    ping_json = _run_cli(sparkleforge_bin, ["ping", "--json"])
    assert ping_json.returncode == 0
    ping_data = json.loads(ping_json.stdout)
    assert ping_data["status"] == "ok"
    assert ping_data["pong"] is True
    assert ping_data["elapsed_ms"] < 5.0

    ver_text = _run_cli(sparkleforge_bin, ["version"])
    assert ver_text.returncode == 0
    assert "sparkleforge" in ver_text.stdout.lower()

    ver_json = _run_cli(sparkleforge_bin, ["version", "--json"])
    assert ver_json.returncode == 0
    ver_data = json.loads(ver_json.stdout)
    assert ver_data["name"] == "sparkleforge"
    assert ver_data["runtime"] == "rust"


def test_health_checks_venv_keys_and_workspace_without_duplicate_warnings(
    sparkleforge_bin: Path, tmp_path: Path
) -> None:
    repo_health = _run_cli(sparkleforge_bin, ["health", "--json"])
    assert repo_health.returncode == 0
    repo_payload = json.loads(repo_health.stdout)
    assert repo_payload["binary"]["ready"] is True
    assert repo_payload["python"]["ready"] is True
    assert repo_payload["python"]["entrypoint_exists"] is True

    # Isolated workspace with mock .venv/bin/python, both Google keys, and .sparkleforge
    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    py_shim = venv_bin / "python"
    py_shim.write_text(f"#!/bin/sh\nexec {sys.executable} \"$@\"\n", encoding="utf-8")
    py_shim.chmod(py_shim.stat().st_mode | stat.S_IXUSR)

    sf_dir = tmp_path / ".sparkleforge"
    sf_dir.mkdir()
    (sf_dir / "state.json").write_text('{"status": "completed"}', encoding="utf-8")

    env = os.environ.copy()
    env["GOOGLE_API_KEY"] = "test-google-token"
    env["GEMINI_API_KEY"] = "test-gemini-token"
    env["OPENROUTER_API_KEY"] = "test-openrouter-token"

    ws_health = _run_cli(sparkleforge_bin, ["health", "--json"], cwd=tmp_path, env=env)
    assert ws_health.returncode == 0
    ws_payload = json.loads(ws_health.stdout)
    assert ws_payload["status"] == "ok"
    assert ws_payload["python"]["venv_exists"] is True
    assert ws_payload["api_keys"]["google_configured"] is True
    assert ws_payload["api_keys"]["openrouter_configured"] is True
    assert ws_payload["workspace"]["exists"] is True
    assert ws_payload["workspace"]["state"] == "active"
    assert "Both GOOGLE_API_KEY and GEMINI_API_KEY" not in ws_health.stdout
    assert "Both GOOGLE_API_KEY and GEMINI_API_KEY" not in ws_health.stderr

    ws_text = _run_cli(sparkleforge_bin, ["health"], cwd=tmp_path, env=env)
    assert ws_text.returncode == 0
    assert "[OK]" in ws_text.stdout
    assert "[WARN]" not in ws_text.stdout


def test_monitor_handles_empty_and_missing_dirs_gracefully(
    sparkleforge_bin: Path, tmp_path: Path
) -> None:
    missing = tmp_path / "does_not_exist"
    res_missing = _run_cli(
        sparkleforge_bin, ["monitor", "--session-dir", str(missing), "--json"]
    )
    assert res_missing.returncode == 0
    data_missing = json.loads(res_missing.stdout)
    assert data_missing["exists"] is False
    assert data_missing["empty"] is True
    assert data_missing["tasks"]["total"] == 0
    assert data_missing["errors"]["count"] == 0

    empty_dir = tmp_path / "empty_sf"
    empty_dir.mkdir()
    res_empty = _run_cli(sparkleforge_bin, ["monitor", "--session-dir", str(empty_dir)])
    assert res_empty.returncode == 0
    assert "empty" in res_empty.stdout.lower()


def test_monitor_aggregates_json_jsonl_and_sqlite(
    sparkleforge_bin: Path, tmp_path: Path
) -> None:
    sf_dir = tmp_path / ".sparkleforge"
    sf_dir.mkdir()

    session_payload = {
        "session_id": "sess-rust-1",
        "status": "active",
        "max_concurrency": 3,
        "error_count": 1,
        "tasks": [
            {"id": "t1", "status": "completed", "duration_ms": 150.0, "agent_id": "planner"},
            {"id": "t2", "status": "running", "agent_id": "executor"},
            {"id": "t3", "status": "failed", "error": "timeout"},
        ],
        "subagents": [
            {"agent_id": "planner", "status": "completed", "duration_ms": 150.0},
            {"agent_id": "executor", "status": "active"},
        ],
    }
    (sf_dir / "session_1.json").write_text(json.dumps(session_payload), encoding="utf-8")

    trace_lines = [
        json.dumps({"event": "step_end", "step": "search", "duration_ms": 80.0, "concurrency": 2}),
        json.dumps({"event": "step_end", "step": "verify", "duration_ms": 40.0, "concurrency": 3}),
        json.dumps({"level": "ERROR", "error": "rate limit exceeded"}),
    ]
    (sf_dir / "trace.jsonl").write_text("\n".join(trace_lines) + "\n", encoding="utf-8")

    db_path = sf_dir / "session.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE sessions (session_id TEXT, status TEXT, error_count INTEGER)")
    conn.execute("INSERT INTO sessions VALUES ('db-s1', 'completed', 0)")
    conn.execute("CREATE TABLE tasks (task_id TEXT, status TEXT, duration_ms REAL, agent_id TEXT)")
    conn.execute("INSERT INTO tasks VALUES ('db-t1', 'completed', 60.0, 'verifier')")
    conn.commit()
    conn.close()

    res = _run_cli(sparkleforge_bin, ["monitor", "--session-dir", str(sf_dir), "--json"])
    assert res.returncode == 0
    report = json.loads(res.stdout)
    assert report["empty"] is False
    assert report["sessions_count"] == 2
    assert report["tasks"]["active"] == 1
    assert report["tasks"]["completed"] == 2
    assert report["tasks"]["failed"] == 1
    assert report["subagents"]["total_unique"] == 3
    assert report["subagents"]["max_concurrency"] == 3
    assert report["step_durations"]["count"] == 4
    assert report["errors"]["count"] == 2
    assert len(report["sqlite_files"]) == 1
    assert report["sqlite_files"][0]["valid_header"] is True


def test_qa_bench_scores_clean_and_noisy_commands(sparkleforge_bin: Path) -> None:
    smoke = _run_cli(sparkleforge_bin, ["qa-bench", "--json"])
    assert smoke.returncode == 0
    smoke_data = json.loads(smoke.stdout)
    assert smoke_data["commands_run"] == 4
    assert smoke_data["passed"] is True
    assert smoke_data["overall_score"] >= 0.85
    assert smoke_data["total_anti_patterns"] == 0

    noisy_py = (
        "import sys; "
        "sys.stderr.write('Loading weights: 100%|###|\\n'); "
        "sys.stderr.write('Batches: 100%|###|\\n'); "
        "sys.stderr.write('[WARN] duplicate key\\n'); "
        "sys.stderr.write('Fallback model failed HTTP 429 rate limit\\n'); "
        "sys.stdout.write('Traceback (most recent call last):\\n\\ufffd\\n')"
    )
    noisy_cmd = f"{shutil.which(sys.executable) or sys.executable} -c {json.dumps(noisy_py)}"
    noisy_res = _run_cli(
        sparkleforge_bin,
        ["qa-bench", "--command", noisy_cmd, "--json"],
    )
    assert noisy_res.returncode == 0
    noisy_data = json.loads(noisy_res.stdout)
    assert noisy_data["passed"] is False
    assert noisy_data["overall_score"] < 0.7
    kinds = {h["kind"] for h in noisy_data["results"][0]["anti_patterns"]}
    assert {
        "loading_weights",
        "batches_progress",
        "warn_leak",
        "traceback",
        "provider_fallback_spam",
        "replacement_char",
    }.issubset(kinds)

    fail_gate = _run_cli(
        sparkleforge_bin,
        ["qa-bench", "--command", noisy_cmd, "--fail-under", "0.8", "--json"],
    )
    assert fail_gate.returncode == 1


def test_delegation_run_query_passthrough_and_quiet_boot_filtering(
    sparkleforge_bin: Path, tmp_path: Path
) -> None:
    pkg_dir = tmp_path / "src" / "cli"
    pkg_dir.mkdir(parents=True)
    (tmp_path / "src" / "__init__.py").write_text("", encoding="utf-8")
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
    (pkg_dir / "entry.py").write_text(
        "import json, os, sys\n"
        "sys.stderr.write('Loading weights: 100%|###|\\rLoading weights: 100%|###|\\n')\n"
        "sys.stderr.write('Batches: 100%|###| 1/1 [00:00<00:00, 10.0it/s]\\n')\n"
        "sys.stderr.write('[WARN] Both GOOGLE_API_KEY and GEMINI_API_KEY are set\\n')\n"
        "sys.stderr.write('2026-07-24 12:00:00,000 - src.core.db.database_driver - INFO - DatabaseDriver initialized\\n')\n"
        "payload = {\n"
        "    'argv': sys.argv[1:],\n"
        "    'env': {\n"
        "        'TQDM_DISABLE': os.environ.get('TQDM_DISABLE'),\n"
        "        'HF_HUB_DISABLE_PROGRESS_BARS': os.environ.get('HF_HUB_DISABLE_PROGRESS_BARS'),\n"
        "        'TRANSFORMERS_VERBOSITY': os.environ.get('TRANSFORMERS_VERBOSITY'),\n"
        "        'TOKENIZERS_PARALLELISM': os.environ.get('TOKENIZERS_PARALLELISM'),\n"
        "        'SPARKLEFORGE_QUIET_BOOT': os.environ.get('SPARKLEFORGE_QUIET_BOOT'),\n"
        "    },\n"
        "}\n"
        "print('RESULT_JSON:' + json.dumps(payload))\n",
        encoding="utf-8",
    )

    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    py_shim = venv_bin / "python"
    py_shim.write_text(f"#!/bin/sh\nexec {sys.executable} \"$@\"\n", encoding="utf-8")
    py_shim.chmod(py_shim.stat().st_mode | stat.S_IXUSR)

    clean_env = {k: v for k, v in os.environ.items() if not k.startswith("SPARKLEFORGE_QUIET")}
    clean_env.pop("TQDM_DISABLE", None)

    # 1. `run` with quiet boot and noise filtering
    run_res = _run_cli(
        sparkleforge_bin,
        ["run", "rust cli test", "--model", "gpt-5-nano"],
        cwd=tmp_path,
        env=clean_env,
    )
    assert run_res.returncode == 0
    assert "Loading weights" not in run_res.stdout + run_res.stderr
    assert "Batches:" not in run_res.stdout + run_res.stderr
    assert "[WARN]" not in run_res.stdout + run_res.stderr
    assert "DatabaseDriver initialized" not in run_res.stdout + run_res.stderr
    assert "[sparkleforge] orchestration=completed" in run_res.stdout
    assert "duration_ms=" in run_res.stdout

    result_line = next(
        line for line in run_res.stdout.splitlines() if line.startswith("RESULT_JSON:")
    )
    delegated = json.loads(result_line.removeprefix("RESULT_JSON:"))
    assert delegated["argv"] == ["run", "rust cli test", "--model", "gpt-5-nano"]
    assert delegated["env"] == {
        "TQDM_DISABLE": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TRANSFORMERS_VERBOSITY": "error",
        "TOKENIZERS_PARALLELISM": "false",
        "SPARKLEFORGE_QUIET_BOOT": "1",
    }

    # 2. `--verbose` bypasses filtering and passes --verbose before subcommand
    verb_res = _run_cli(
        sparkleforge_bin,
        ["run", "rust cli verbose", "--verbose"],
        cwd=tmp_path,
        env=clean_env,
    )
    assert verb_res.returncode == 0
    assert "[WARN]" in verb_res.stderr
    verb_line = next(
        line for line in verb_res.stdout.splitlines() if line.startswith("RESULT_JSON:")
    )
    verb_delegated = json.loads(verb_line.removeprefix("RESULT_JSON:"))
    assert verb_delegated["argv"] == ["--verbose", "run", "rust cli verbose"]
    assert verb_delegated["env"]["SPARKLEFORGE_QUIET_BOOT"] is None

    # 3. `query`, `--query`, and passthrough `session list`
    for cli_args, expected_argv in [
        (["query", "check query"], ["query", "check query"]),
        (["--query", "flag query"], ["--query", "flag query"]),
        (["session", "list", "--limit", "5"], ["session", "list", "--limit", "5"]),
    ]:
        res = _run_cli(sparkleforge_bin, cli_args, cwd=tmp_path, env=clean_env)
        assert res.returncode == 0
        line = next(
            l for l in res.stdout.splitlines() if l.startswith("RESULT_JSON:")
        )
        assert json.loads(line.removeprefix("RESULT_JSON:"))["argv"] == expected_argv

    # 4. Fallback to python3 when .venv/bin/python is absent
    shutil.rmtree(tmp_path / ".venv")
    fallback_res = _run_cli(
        sparkleforge_bin,
        ["run", "fallback check"],
        cwd=tmp_path,
        env=clean_env,
    )
    assert fallback_res.returncode == 0
    assert "RESULT_JSON:" in fallback_res.stdout


def test_real_repo_run_help_delegation(sparkleforge_bin: Path) -> None:
    res = _run_cli(sparkleforge_bin, ["run", "--help"])
    assert res.returncode == 0
    assert "--max-tokens" in res.stdout
    assert "--model" in res.stdout
