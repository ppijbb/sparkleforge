"""Implementation repair loop: turn a red reproduction test green.

Mirrors the retry loop in .github/workflows/opencode-auto-fix.yml (the
"OpenCode repair loop" step) but in Python, and verifies against the
reproduction test written by src/core/nightwelding/gate.py instead of a bare
`compileall` check.

Reuses scripts/opencode_github_worker.py's `fix-issue` subcommand as a
subprocess (the same, already-hardened usage pattern the existing GitHub
Actions workflow already relies on) rather than importing `fix_issue()`
in-process — see the Nightwelding plan for why.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.core import patch_ops

_FIX_ISSUE_TIMEOUT_SECONDS = 600
_PYTEST_TIMEOUT_SECONDS = 300


@dataclass
class ImplementResult:
    success: bool
    reason: str = ""
    log: str = ""
    green_output: str = ""
    attempts: int = 0


def _run(cmd: List[str], cwd: Path, timeout: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False)


def implement_until_green(
    issue_context: str,
    repro_test_files: List[str],
    repo_root: Path,
    commit_title: str,
    max_iterations: int = 4,
) -> ImplementResult:
    max_iterations = max(1, min(6, max_iterations))

    issue_context_path = repo_root / "issue-context.md"
    issue_context_path.write_text(issue_context, encoding="utf-8")
    extra_context_path = repo_root / "opencode-extra-context.md"
    extra_context_path.write_text("", encoding="utf-8")

    validate = _run(
        [sys.executable, "scripts/validate_commit_messages.py", "--message", commit_title],
        cwd=repo_root,
    )
    if validate.returncode != 0:
        return ImplementResult(
            success=False,
            reason=f"Derived commit title failed validation: {commit_title!r}",
            log=(validate.stdout + validate.stderr)[-2000:],
        )

    for attempt in range(1, max_iterations + 1):
        before_sig = patch_ops.repository_change_signature()

        proc = _run(
            [
                sys.executable,
                "scripts/opencode_github_worker.py",
                "fix-issue",
                "--issue-context",
                str(issue_context_path),
                "--extra-context",
                str(extra_context_path),
            ],
            cwd=repo_root,
            timeout=_FIX_ISSUE_TIMEOUT_SECONDS,
        )
        if proc.returncode != 0:
            extra_context_path.write_text(
                "Previous attempt failed before commit.\n\n"
                f"Worker error:\n{proc.stderr}\n\n"
                "Regenerate the patch against the current repository contents.",
                encoding="utf-8",
            )
            if attempt == max_iterations:
                return ImplementResult(
                    success=False,
                    reason="OpenCode did not produce an applicable implementation diff.",
                    log=proc.stderr[-4000:],
                    attempts=attempt,
                )
            continue

        after_sig = patch_ops.repository_change_signature()
        if after_sig == before_sig:
            extra_context_path.write_text(
                "Previous attempt produced no repository changes. The response either "
                "generated a no-op patch or changed only files ignored by git. Regenerate "
                "a minimal patch that changes tracked repository files relevant to the issue.",
                encoding="utf-8",
            )
            if attempt == max_iterations:
                return ImplementResult(
                    success=False,
                    reason="OpenCode produced no repository changes.",
                    attempts=attempt,
                )
            continue

        _run(["git", "add", "-u"], cwd=repo_root)
        ls = _run(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=repo_root)
        untracked = [p for p in ls.stdout.split("\0") if p]
        if untracked:
            _run(["git", "add", "--", *untracked], cwd=repo_root)

        commit_proc = _run(["git", "commit", "-m", commit_title], cwd=repo_root)
        if commit_proc.returncode != 0:
            return ImplementResult(
                success=False,
                reason="git commit failed after a real change was detected.",
                log=(commit_proc.stdout + commit_proc.stderr)[-2000:],
                attempts=attempt,
            )

        verify = _run(
            [sys.executable, "-m", "pytest", *repro_test_files, "-q"],
            cwd=repo_root,
            timeout=_PYTEST_TIMEOUT_SECONDS,
        )
        if verify.returncode == 0:
            return ImplementResult(
                success=True,
                green_output=(verify.stdout + verify.stderr)[-4000:],
                attempts=attempt,
            )

        extra_context_path.write_text(
            "Previous attempt did not turn the reproduction test green.\n\n"
            f"Reproduction test output:\n{(verify.stdout + verify.stderr)[-4000:]}",
            encoding="utf-8",
        )
        if attempt == max_iterations:
            return ImplementResult(
                success=False,
                reason=f"Reproduction test still failing after {max_iterations} attempts.",
                log=(verify.stdout + verify.stderr)[-4000:],
                attempts=attempt,
            )

    return ImplementResult(success=False, reason="Exhausted iterations.", attempts=max_iterations)
