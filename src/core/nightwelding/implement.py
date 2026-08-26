"""Implementation repair loop: turn a red reproduction test green.

Mirrors the retry loop in src/core/autofix/runner.py (which backs the
"OpenCode repair loop" step in .github/workflows/opencode-auto-fix.yml) but
verifies against the reproduction test written by
src/core/nightwelding/gate.py instead of a bare `compileall` check.

Reuses `sparkleforge ci fix-issue` (src/core/ci/fix_issue.py) as a subprocess
(the same, already-hardened usage pattern the autofix repair loop and the
GitHub Actions workflow already relied on) rather than importing `fix_issue()`
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

    validator_script = Path(__file__).resolve().parents[3] / "scripts" / "validate_commit_messages.py"
    if not validator_script.exists():
        validator_script = repo_root / "scripts" / "validate_commit_messages.py"

    sparkleforge_entrypoint = Path(__file__).resolve().parents[3] / "main.py"
    if not sparkleforge_entrypoint.exists():
        sparkleforge_entrypoint = repo_root / "main.py"

    validate = _run(
        [sys.executable, str(validator_script), "--message", commit_title],
        cwd=repo_root,
    )
    if validate.returncode != 0:
        return ImplementResult(
            success=False,
            reason=f"Derived commit title failed validation: {commit_title!r}",
            log=(validate.stdout + validate.stderr)[-2000:],
        )

    for attempt in range(1, max_iterations + 1):
        before_sig = patch_ops.repository_change_signature(cwd=repo_root)

        # Issue #917: validate the synthetic schema of the worker's diff before
        # attempting to apply it, so malformed LLM output fails fast with a clear
        # reason instead of an opaque `git apply` failure deep in patch_ops.
        from src.core.nightwelding.gate import _validate_repro_diff_schema

        proc = _run(
            [
                sys.executable,
                str(sparkleforge_entrypoint),
                "ci",
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
            candidate_diff = patch_ops.extract_diff(proc.stdout or "")
            if candidate_diff.strip():
                schema_ok, schema_reason = _validate_repro_diff_schema(candidate_diff)
                if not schema_ok:
                    extra_context_path.write_text(
                        f"Previous attempt produced a malformed diff: {schema_reason}\n\n"
                        "Regenerate a minimal git-apply compatible diff with "
                        "'diff --git', '--- a/'/'+++ b/' headers and 'a/'/'b/' path prefixes.",
                        encoding="utf-8",
                    )
                    if attempt == max_iterations:
                        return ImplementResult(success=False, reason=schema_reason, log=proc.stderr[-4000:], attempts=attempt)
                    continue
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

        after_sig = patch_ops.repository_change_signature(cwd=repo_root)
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
        untracked = [p for p in ls.stdout.split("\0") if p and not patch_ops.is_runtime_scratch_path(p)]
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
