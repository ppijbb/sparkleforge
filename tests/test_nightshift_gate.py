"""Unit tests for src/core/nightshift/gate.py's scope guard.

These test the pure git-status-parsing guard logic without touching an LLM:
the guard must accept isolated tests/test_*.py changes and reject anything
that touches shared test infrastructure (conftest.py, tests/benchmark/,
tests/baselines/) or files outside tests/.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from src.core.nightshift.gate import _touched_test_files


def _init_repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "conftest.py").write_text("# fixtures\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "chore: initial commit"], cwd=tmp_path, check=True)
    return tmp_path


def test_guard_accepts_isolated_repro_test(tmp_path) -> None:
    repo = _init_repo(tmp_path)
    (repo / "tests" / "test_repro_issue_1.py").write_text("def test_x():\n    assert False\n", encoding="utf-8")

    all_ok, touched = _touched_test_files(repo)

    assert all_ok is True
    assert touched == ["tests/test_repro_issue_1.py"]


def test_guard_rejects_conftest_edit(tmp_path) -> None:
    repo = _init_repo(tmp_path)
    (repo / "tests" / "conftest.py").write_text("# fixtures\n# modified\n", encoding="utf-8")
    (repo / "tests" / "test_repro_issue_1.py").write_text("def test_x():\n    assert False\n", encoding="utf-8")

    all_ok, touched = _touched_test_files(repo)

    assert all_ok is False
    assert "tests/conftest.py" in touched


def test_guard_rejects_benchmark_subdir(tmp_path) -> None:
    repo = _init_repo(tmp_path)
    (repo / "tests" / "benchmark").mkdir()
    (repo / "tests" / "benchmark" / "test_helper.py").write_text("def test_x():\n    pass\n", encoding="utf-8")

    all_ok, touched = _touched_test_files(repo)

    assert all_ok is False
    assert "tests/benchmark/test_helper.py" in touched


def test_guard_rejects_src_edit(tmp_path) -> None:
    repo = _init_repo(tmp_path)
    (repo / "src" / "sneaky_fix.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "tests" / "test_repro_issue_1.py").write_text("def test_x():\n    assert False\n", encoding="utf-8")

    all_ok, touched = _touched_test_files(repo)

    assert all_ok is False
    assert "src/sneaky_fix.py" in touched


def test_guard_ignores_opencode_patch_scratch_file(tmp_path) -> None:
    repo = _init_repo(tmp_path)
    (repo / "tests" / "test_repro_issue_1.py").write_text("def test_x():\n    assert False\n", encoding="utf-8")
    (repo / "opencode.patch").write_text("diff --git a/x b/x\n", encoding="utf-8")

    all_ok, touched = _touched_test_files(repo)

    assert all_ok is True
    assert touched == ["tests/test_repro_issue_1.py"]
