import os
import stat
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_install_script_macos_path_skips_linux_gvisor_setup(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    uv = fake_bin / "uv"
    uv.write_text("#!/bin/sh\nprintf 'uv 0.0.0-test\\n'\n", encoding="utf-8")
    uv.chmod(uv.stat().st_mode | stat.S_IXUSR)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["SPARKLEFORGE_INSTALL_OS"] = "darwin"
    env["SPARKLEFORGE_SKIP_UV_SYNC"] = "1"

    result = subprocess.run(
        ["bash", "install.sh"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Detected macOS" in result.stdout
    assert "Skipping Linux-only Docker/gVisor runsc installation" in result.stdout
    assert "macOS installation completed successfully" in result.stdout


def test_install_script_has_valid_bash_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", "install.sh"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_merge_branch_main_subject_passes_validation() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "validate_commit_messages",
        PROJECT_ROOT / "scripts" / "validate_commit_messages.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    errors = module.validate_subject("Merge branch 'main'", "merge-main")
    assert errors == [], errors


def test_non_main_merge_subject_is_rejected() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "validate_commit_messages",
        PROJECT_ROOT / "scripts" / "validate_commit_messages.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    errors = module.validate_subject("Merge branch 'feature/foo'", "merge-feature")
    assert any("merge commit subjects are not allowed" in error for error in errors), errors
