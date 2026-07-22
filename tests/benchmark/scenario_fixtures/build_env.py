"""Fixture + grader for scenario 5: "이 프로젝트 빌드 환경 세팅해줘."

Seeds a tiny project that fails to run because a required environment
variable is unset, plus an undeclared dependency. main.py ships with a small
stdlib-only .env loader (a realistic, common pattern) so the check for
"did the agent actually fix it" can re-run the project fresh, with no
network/package installs required, and get a real pass/fail signal.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from tests.benchmark.scenario_grading import (
    judge_score,
    keyword_hit,
    read_text_safe,
    rubric_from_context,
)

REQUIRED_ENV = "APP_ENV"
REQUIRED_PACKAGE = "requests"

MAIN_PY = '''"""Tiny demo app: fails until APP_ENV is configured."""
import os
from pathlib import Path
import requests  # the undeclared dependency this scenario asks the agent to set up

_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        if "=" in _line and not _line.strip().startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

print(f"APP_ENV={os.environ.get('APP_ENV', 'development')}")
'''


def build(workspace: Path) -> Dict[str, Any]:
    project = workspace / "project"
    project.mkdir(parents=True, exist_ok=True)
    (project / "main.py").write_text(MAIN_PY, encoding="utf-8")
    (project / "requirements.in").write_text(f"{REQUIRED_PACKAGE}\n", encoding="utf-8")

    return {
        "workspace": str(workspace),
        "project_dir": "project",
        "required_env": REQUIRED_ENV,
        "required_package": REQUIRED_PACKAGE,
    }


def _env_configured(project: Path, required_env: str) -> bool:
    """Check whether the agent configured the required env var via .env or export."""
    env_file = project / ".env"
    if env_file.exists() and keyword_hit(read_text_safe(env_file), [f"{required_env}="]):
        return True
    return required_env in os.environ


async def grade(workspace: Path, ctx: Dict[str, Any], stdout: str) -> Dict[str, tuple[float, str]]:
    project = workspace / ctx["project_dir"]

    env_setup = (
        (1.0, f"{ctx['required_env']} is configured via .env file or environment")
        if _env_configured(project, ctx["required_env"])
        else (0.0, f"{ctx['required_env']} is not configured: create a .env file or export it before re-running the project")
    )

    try:
        result = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, "main.py"],
            cwd=project,
            capture_output=True,
            text=True,
            timeout=15,
        )
        runs_now = result.returncode == 0 and "APP_ENV=" in result.stdout
        verify_runs = (
            (1.0, f"project now runs successfully: {result.stdout.strip()}")
            if runs_now
            else (0.0, f"project still fails (rc={result.returncode}): {result.stderr.strip()[:300]}")
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        verify_runs = (0.0, f"could not re-run project: {e}")

    manifest_candidates = ["requirements.txt", "requirements.in", "pyproject.toml", "Pipfile"]
    declared = False
    for name in manifest_candidates:
        text = read_text_safe(project / name)
        if keyword_hit(text, [ctx["required_package"]]):
            declared = True
            break
    packages_declared = (
        (1.0, "dependency manifest declares the required package")
        if declared
        else (0.0, "no manifest found declaring the required package")
    )

    judge = await judge_score(
        rubric=rubric_from_context(
            ctx,
            "Does the agent explain the build-environment setup steps taken "
            "(packages/env vars/verification) coherently?",
        ),
        transcript=stdout[:4000],
        context=f"required env var: {ctx['required_env']}, required package: {ctx['required_package']}",
    )

    return {
        "verify_runs": verify_runs,
        "packages_declared": packages_declared,
        "env_setup": env_setup,
        "judge_quality": judge,
    }
