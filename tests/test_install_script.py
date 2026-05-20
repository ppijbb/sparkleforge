import os
import stat
import subprocess
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
